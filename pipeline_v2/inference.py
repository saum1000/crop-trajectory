"""
Cube Earth Crop Classifier — Inference Pipeline
Usage: given a parcel centroid (lat, lng), returns crop prediction + confidence
"""
import ee, json, pickle, numpy as np
from catboost import CatBoostClassifier

# Load model artifacts
clf = CatBoostClassifier()
clf.load_model('models/crop_classifier_v1.cbm')
with open('models/label_encoder.pkl','rb') as f:
    le = pickle.load(f)
with open('models/feature_columns.json') as f:
    feat_cols = json.load(f)

METRICS = ['NDVI','NDRE','EVI','NDWI','NDII']
YEARS = [2022,2023,2024,2025]

def init_gee():
    creds = ee.ServiceAccountCredentials(
        'id-cube-earth-gee@ireland-mrv-prototype.iam.gserviceaccount.com',
        '/tmp/gee_key.json'
    )
    ee.Initialize(credentials=creds, project='ireland-mrv-prototype')

def mask_s2_scl(img):
    scl = img.select('SCL')
    mask = scl.neq(0).And(scl.neq(1)).And(scl.neq(2)).And(scl.neq(3)) \
              .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(mask).divide(10000)

def get_dekads(year):
    dekads = []
    for m in range(1,12):
        for d,d2 in [(1,11),(11,21),(21,None)]:
            s=f'{year}-{m:02d}-{d:02d}'
            e=f'{year}-{m:02d}-{d2:02d}' if d2 else \
              (f'{year}-{m+1:02d}-01' if m<11 else f'{year}-12-01')
            dekads.append((s,e))
    return dekads

def extract_parcel_features(lat, lng):
    """Extract 4-year S2 time series for a single parcel centroid."""
    point = ee.Geometry.Point([lng, lat]).buffer(50)
    features = {}

    for year in YEARS:
        for i,(s,e) in enumerate(get_dekads(year)):
            s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(s,e) \
                .map(mask_s2_scl) \
                .median()

            b2=s2.select('B2'); b3=s2.select('B3'); b4=s2.select('B4')
            b5=s2.select('B5'); b8=s2.select('B8')
            b8a=s2.select('B8A'); b11=s2.select('B11')

            ndvi=b8.subtract(b4).divide(b8.add(b4).add(1e-8))
            ndre=b8.subtract(b5).divide(b8.add(b5).add(1e-8))
            evi=b8.subtract(b4).multiply(2.5).divide(
                b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
            ndwi=b3.subtract(b8).divide(b3.add(b8).add(1e-8))
            ndii=b8a.subtract(b11).divide(b8a.add(b11).add(1e-8))

            img = ndvi.rename('NDVI').addBands([
                ndre.rename('NDRE'), evi.rename('EVI'),
                ndwi.rename('NDWI'), ndii.rename('NDII')
            ])

            vals = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point, scale=10, maxPixels=1e6
            ).getInfo()

            for metric in METRICS:
                features[f'y{year}_d{i:02d}_{metric}'] = vals.get(metric)

    return features

def interpolate_features(features):
    """Fill NaN gaps with linear interpolation per metric per year."""
    for year in YEARS:
        for metric in METRICS:
            cols = sorted([k for k in features if
                           k.startswith(f'y{year}') and k.endswith(metric)])
            vals = [features[c] for c in cols]
            # Linear interpolation
            xs = [i for i,v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            if len(xs) < 2:
                for c in cols:
                    if features[c] is None:
                        features[c] = 0.4  # fallback mean
            else:
                for i,c in enumerate(cols):
                    if features[c] is None:
                        features[c] = float(np.interp(i, xs, ys))
    return features

def predict(lat, lng):
    """
    Predict crop type for a parcel centroid.
    Returns: dict with crop, confidence, and all class probabilities.
    """
    print(f"Extracting features for ({lat:.4f}, {lng:.4f})...")
    raw = extract_parcel_features(lat, lng)
    filled = interpolate_features(raw)

    # Build feature vector in correct order
    X = np.array([[filled.get(c, 0.4) for c in feat_cols]])

    probs = clf.predict_proba(X)[0]
    pred_idx = np.argmax(probs)
    crop = le.inverse_transform([pred_idx])[0]
    confidence = round(float(probs[pred_idx]) * 100, 1)

    result = {
        'crop': crop,
        'confidence': confidence,
        'lat': lat,
        'lng': lng,
        'probabilities': {
            le.classes_[i]: round(float(p)*100,1)
            for i,p in enumerate(probs)
        }
    }
    return result

if __name__ == '__main__':
    init_gee()
    # Test on a known Spring Barley location in Carlow
    result = predict(52.65, -6.92)
    print(f"\nPrediction:")
    print(f"  Crop:       {result['crop']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"\nAll probabilities:")
    for crop,prob in sorted(result['probabilities'].items(),
                            key=lambda x: -x[1]):
        bar = '█' * int(prob/5)
        print(f"  {crop:<20} {prob:5.1f}% {bar}")
