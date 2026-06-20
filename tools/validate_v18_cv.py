"""
v18 benchmark validation — 5-fold stratified CV.
Produces authoritative accuracy, balanced accuracy, macro F1
and per-class report. Do not modify v18 model files.
"""
import numpy as np, json, warnings, joblib
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                              f1_score, confusion_matrix)
from collections import Counter

# ── Data ──────────────────────────────────────────────────────────────
with open('data/irish_s2_timeseries.json') as f:
    s2_data={d['par_lab']:d for d in json.load(f)}

CM={'Grassland','Barley','Wheat','Oats','Oilseed Rape','Maize','Beans'}
TARGET_SEQ=32; B04,B08,B11=3,7,11

X=[]; labels=[]
for pl,d in s2_data.items():
    label=d.get('label')
    if label not in CM: continue
    ts=np.array(d['timeseries'],dtype=np.float32)
    while ts.shape[0]<TARGET_SEQ: ts=np.vstack([ts,ts[-1:]])
    ts=ts[:TARGET_SEQ,:]
    b04=ts[:,B04]; b08=ts[:,B08]; b11=ts[:,B11]
    ndvi=(b08-b04)/(b08+b04+1e-8)
    peak_idx=int(np.argmax(ndvi))
    pheno=[
        float(np.max(ndvi)),float(np.min(ndvi)),
        float(np.max(ndvi)-np.min(ndvi)),
        float(peak_idx/TARGET_SEQ),
        float(int(np.argmin(ndvi))/TARGET_SEQ),
        float(np.mean(ndvi[:8])),float(np.mean(ndvi[8:16])),
        float(np.mean(ndvi[16:24])),float(np.mean(ndvi[24:])),
        float(np.std(ndvi)),
    ]
    X.append(ts.flatten().tolist()+pheno)
    labels.append(label)

X=np.array(X,dtype=np.float32)
le=LabelEncoder(); y=le.fit_transform(labels)
freq=Counter(labels); total=len(labels)
sw=np.array([total/(len(freq)*freq[labels[i]]) for i in range(len(labels))])

print(f"Dataset: {len(X)} parcels | {X.shape[1]} features")
print(f"Classes: {Counter(labels)}\n")

# ── 5-fold CV ─────────────────────────────────────────────────────────
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
all_y=[]; all_yp=[]

for fold,(tr,te) in enumerate(cv.split(X,y)):
    clf=CatBoostClassifier(iterations=300,depth=5,learning_rate=0.05,
                            loss_function='MultiClass',verbose=0,
                            random_seed=42,thread_count=1)
    clf.fit(X[tr],y[tr],sample_weight=sw[tr])
    yp=clf.predict(X[te]).flatten()
    all_y.extend(y[te]); all_yp.extend(yp)
    fold_acc=(yp==y[te]).mean()*100
    print(f"Fold {fold+1}: {round(fold_acc,1)}%")

all_y=np.array(all_y); all_yp=np.array(all_yp)

acc=(all_yp==all_y).mean()*100
bal=balanced_accuracy_score(all_y,all_yp)*100
mac=f1_score(all_y,all_yp,average='macro')*100

print(f"\n{'='*50}")
print(f"v18 5-Fold CV Results")
print(f"{'='*50}")
print(f"Overall Accuracy:  {round(acc,1)}%")
print(f"Balanced Accuracy: {round(bal,1)}%")
print(f"Macro F1:          {round(mac,1)}%")
print(f"\n{classification_report(all_y,all_yp,target_names=le.classes_)}")

cm=confusion_matrix(all_y,all_yp)
print("Confusion matrix:")
print(f"{'':15}", [c[:5] for c in le.classes_])
for i,cls in enumerate(le.classes_):
    print(f"  {cls:<15}", list(cm[i]))

print(f"\nv15 monthly baseline: 63.5%")
print(f"v18 benchmark target: ~76%")
