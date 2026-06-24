"""
Cube Earth Crop Classifier — Inference Pipeline v2
S1+S2 fusion: 1152 features, 81% cross-year balanced accuracy
"""
import ee, json, pickle, numpy as np
from catboost import CatBoostClassifier

# Load v2 model artifacts
clf = CatBoostClassifier()
clf.load_model('models/crop_classifier_v2.cbm')
with open('models/label_encoder_v2.pkl','rb') as f:
    le = pickle.load(f)
with open('models/feature_columns_v2.json') as f:
    feat_cols = json.load(f)

S2_METRICS = ['NDVI','NDRE','EVI','NDWI','NDII']
S1_METRICS = ['VV','VH','RVI','VHVV']
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
    dekads=[]
    for m in range(1,12):
        for d,d2 in [(1,11),(11,21),(21,None)]:
            s=f'{year}-{m:02d}-{d:02d}'
            e=f'{year}-{m:02d}-{d2:02d}' if d2 else \
              (f'{year}-{m+1:02d}-01' if m<11 else f'{year}-12-01')
            dekads.append((s,e))
    return dekads

def extract_s2(point, year):
    features = {}
    for i,(s,e) in enumerate(get_dekads(year)):
        s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(s,e).map(mask_s2_scl).median()
        b2=s2.select('B2'); b3=s2.select('B3'); b4=s2.select('B4')
        b5=s2.select('B5'); b8=s2.select('B8')
        b8a=s2.select('B8A'); b11=s2.select('B11')
        ndvi=b8.subtract(b4).divide(b8.add(b4).add(1e-8))
        ndre=b8.subtract(b5).divide(b8.add(b5).add(1e-8))
        evi=b8.subtract(b4).multiply(2.5).divide(
            b8.add(b4.multiply(6)).subtract(b2.multiply(7.5)).add(1))
        ndwi=b3.subtract(b8).divide(b3.add(b8).add(1e-8))
        ndii=b8a.subtract(b11).divide(b8a.add(b11).add(1e-8))
        img=ndvi.rename('NDVI').addBands([ndre.rename('NDRE'),
            evi.rename('EVI'),ndwi.rename('NDWI'),ndii.rename('NDII')])
        vals=img.reduceRegion(ee.Reducer.mean(),point,10,maxPixels=1e6).getInfo()
        for m in S2_METRICS:
            features[f'y{year}_d{i:02d}_{m}'] = vals.get(m)
    return features

def extract_s1(point, year):
    features = {}
    for i,(s,e) in enumerate(get_dekads(year)):
        s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
            .filterDate(s,e) \
            .filter(ee.Filter.eq('instrumentMode','IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')) \
            .select(['VV','VH']).median()
        vv=s1.select('VV'); vh=s1.select('VH')
        vv_lin=ee.Image(10).pow(vv.divide(10))
        vh_lin=ee.Image(10).pow(vh.divide(10))
        rvi=vh_lin.multiply(4).divide(vv_lin.add(vh_lin)).rename('RVI')
        vhvv=vh.subtract(vv).rename('VHVV')
        img=vv.rename('VV').addBands([vh.rename('VH'),rvi,vhvv])
        vals=img.reduceRegion(ee.Reducer.mean(),point,10,maxPixels=1e6).getInfo()
        for m in S1_METRICS:
            features[f'y{year}_d{i:02d}_{m}'] = vals.get(m)
    return features

def interpolate(features, year, metrics):
    for metric in metrics:
        cols = sorted([k for k in features
                       if k.startswith(f'y{year}') and k.endswith(metric)])
        vals = [features[c] for c in cols]
        xs = [i for i,v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        if len(xs) >= 2:
            for i,c in enumerate(cols):
                if features[c] is None:
                    features[c] = float(np.interp(i, xs, ys))
        else:
            for c in cols:
                if features[c] is None:
                    features[c] = 0.4
    return features

def predict(lat, lng, verbose=True):
    point = ee.Geometry.Point([lng, lat]).buffer(50)
    if verbose: print(f"Extracting S2+S1 for ({lat:.4f}, {lng:.4f})...")

    features = {}
    for year in YEARS:
        features.update(extract_s2(point, year))
        features.update(extract_s1(point, year))
        features = interpolate(features, year, S2_METRICS)
        features = interpolate(features, year, S1_METRICS)
        if verbose: print(f"  {year} ✅")

    X = np.array([[features.get(c, 0.4) for c in feat_cols]])
    probs = clf.predict_proba(X)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    pred_idx = top3_idx[0]
    crop = le.inverse_transform([pred_idx])[0]
    confidence = round(float(probs[pred_idx])*100, 1)

    if confidence >= 70:   conf_cat = 'High'
    elif confidence >= 45: conf_cat = 'Medium'
    else:                  conf_cat = 'Low'

    return {
        'crop': crop,
        'confidence': confidence,
        'confidence_category': conf_cat,
        'top3': [(le.classes_[i], round(float(probs[i])*100,1))
                 for i in top3_idx],
        'lat': lat, 'lng': lng,
        'all_probs': {le.classes_[i]: round(float(p)*100,1)
                      for i,p in enumerate(probs)}
    }

if __name__ == '__main__':
    import json as _json
    with open('pipeline_v2/data/labels/lpis_2024_unique.json') as f:
        parcels = _json.load(f)

    # Pick one parcel per target class
    targets = {
        'Spring Barley':None,'Winter Wheat':None,'Winter Barley':None,
        'OSR':None,'Maize':None,'Potatoes':None,'Spring Oats':None
    }
    for p in parcels:
        if p['crop'] in targets and targets[p['crop']] is None:
            targets[p['crop']] = p
        if all(v is not None for v in targets.values()): break

    init_gee()
    print(f"\n{'='*65}")
    print(f"{'Crop':<20} {'True Label':<20} {'Conf':>6} {'Cat':<8} Top 3")
    print(f"{'='*65}")

    correct = 0
    for true_crop, p in targets.items():
        r = predict(p['lat'], p['lng'], verbose=False)
        match = '✅' if r['crop']==true_crop else '❌'
        if r['crop']==true_crop: correct+=1
        top3_str = ', '.join(f"{c}({v}%)" for c,v in r['top3'])
        print(f"{match} {r['crop']:<20} {true_crop:<20} "
              f"{r['confidence']:>5}% {r['confidence_category']:<8} {top3_str}")

    print(f"\nAccuracy: {correct}/{len(targets)} = {correct/len(targets)*100:.0f}%")
