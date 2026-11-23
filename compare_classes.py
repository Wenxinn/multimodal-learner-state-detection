import sys, warnings, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
d = pd.read_csv(sys.argv[1]); g = d.participant
META=["participant","qid","passage","subq","english_level","label","strategy","confidence","difficulty","pen_pointer_type","pressure_available"]
X = d[[c for c in d.columns if c not in META]].astype(float).values
SCHEMES = {
 "4 classes": {"locating_information":"locating","understanding":"understanding","reasoning_integrating":"reasoning","answer_construction":"construction"},
 "3 classes A (construction -> reasoning)": {"locating_information":"locating","understanding":"understanding","reasoning_integrating":"reasoning","answer_construction":"reasoning"},
 "3 classes B (understanding + reasoning)": {"locating_information":"locating","understanding":"understand+reason","reasoning_integrating":"understand+reason","answer_construction":"construction"},
}
def models():
    return {"Random Forest":RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight="balanced",random_state=0,n_jobs=-1),
            "XGBoost":XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=0,n_jobs=-1,eval_metric="mlogloss"),
            "SVM (RBF)":make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),SVC(class_weight="balanced"))}
logo=LeaveOneGroupOut(); rows=[]
for sname, mp in SCHEMES.items():
    yt = d.label.map(mp); classes = sorted(yt.unique()); y = yt.map({c:i for i,c in enumerate(classes)}).values
    k=len(classes)
    # baseline: majority per sub-question from training participants
    pb=np.empty(len(y),int)
    for tr,te in logo.split(X,y,g):
        r=pd.Series(y[tr]).groupby(d.subq.values[tr]).agg(lambda s:s.value_counts().idxmax())
        pb[te]=d.subq.iloc[te].map(r).fillna(pd.Series(y[tr]).mode()[0]).astype(int).values
    base=f1_score(y,pb,average="macro")
    rows.append(dict(scheme=sname,model="Baseline (question number)",accuracy=accuracy_score(y,pb),macro_F1=base,chance_F1=1/k,per_class=dict(zip(classes,f1_score(y,pb,average=None).round(2)))))
    for mn in models():
        p=np.empty(len(y),int)
        for tr,te in logo.split(X,y,g):
            m=models()[mn]
            if mn=="XGBoost": m.fit(X[tr],y[tr],sample_weight=compute_sample_weight("balanced",y[tr]))
            else: m.fit(X[tr],y[tr])
            p[te]=m.predict(X[te])
        rows.append(dict(scheme=sname,model=mn,accuracy=accuracy_score(y,p),macro_F1=f1_score(y,p,average="macro"),chance_F1=1/k,per_class=dict(zip(classes,f1_score(y,p,average=None).round(2)))))
    print("done",sname,flush=True)
r=pd.DataFrame(rows)
base=r[r.model.str.startswith("Baseline")].set_index("scheme").macro_F1
r["vs_baseline"]=r.macro_F1-r.scheme.map(base)
r["vs_chance"]=r.macro_F1-r.chance_F1
pd.set_option("display.width",250); pd.set_option("display.max_colwidth",120)
print(r.round(3).to_string(index=False))
r.round(3).to_csv("results_class_schemes.csv",index=False)
