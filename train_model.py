import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

WIN, STEP, SKIP, GAP, CAL = 50, 25, 100, 50, 450
LABELS = ["empty", "standing", "moving"]
NAMES = ["cv_median", "cv_mean", "cv_max", "shift_mean", "shift_median", "diff_mean"]
cols = [f"a{i}" for i in range(64)]

# ---- load every place that has all 3 labels ----
data = {}
for place in sorted(os.listdir("data")):
    p = os.path.join("data", place)
    if not os.path.isdir(p):
        continue
    d = {}
    for lab in LABELS:
        files = sorted(glob.glob(os.path.join(p, f"{lab}_*.csv")))
        if files:
            d[lab] = pd.read_csv(files[-1])[cols].values[SKIP:].astype(float)
    if len(d) == 3:
        data[place] = d
print("places:", list(data))
if not data:
    raise SystemExit("No complete place folders found inside data/")

mask = np.ones(64, bool)
mask[:2] = False
for d in data.values():
    mask &= np.vstack(list(d.values())).mean(axis=0) > 3
print("usable subcarriers:", int(mask.sum()))


def norm(w):
    A = w[:, mask]
    return A / np.maximum(A.mean(axis=1, keepdims=True), 1e-6)


def raw_feats(w, ref):
    A = norm(w)
    m = np.maximum(A.mean(axis=0), 1e-6)
    cv = A.std(axis=0) / m
    shift = np.abs(m - ref) / ref
    d = np.abs(np.diff(A, axis=0)).mean()
    return np.array([np.median(cv), cv.mean(), cv.max(),
                     shift.mean(), np.median(shift), d])


def calibrate(E):
    C = E[:CAL]
    idx = range(0, len(C) - WIN, STEP)
    ref = np.median([norm(C[i:i + WIN]).mean(axis=0) for i in idx], axis=0)
    base = np.median([raw_feats(C[i:i + WIN], ref) for i in idx], axis=0)
    return ref, np.maximum(base, 1e-4)


prep = {pl: (d,) + calibrate(d["empty"]) for pl, d in data.items()}


def make(place, part):
    d, ref, base = prep[place]
    X, y = [], []
    for lab in LABELS:
        arr = d[lab]
        cut = int(len(arr) * 0.7)
        seg = arr[:cut] if part == "train" else arr[cut + GAP:] if part == "test" else arr[CAL:]
        for i in range(0, len(seg) - WIN, STEP):
            X.append(raw_feats(seg[i:i + WIN], ref) / base)
            y.append(lab)
    return np.array(X), np.array(y)


def collect(places, part):
    Xs, ys = zip(*[make(p, part) for p in places])
    return np.vstack(Xs), np.concatenate(ys)


def show(yt, yp, title):
    print(f"\n=== {title} ===")
    print("Accuracy:", round(accuracy_score(yt, yp), 4))
    print("order:", LABELS)
    print(confusion_matrix(yt, yp, labels=LABELS))


def new_rf():
    return RandomForestClassifier(n_estimators=200, random_state=42)


# ---- 1) leave-one-place-out (honest test on an unseen place) ----
if len(data) >= 2:
    for hold in data:
        others = [p for p in data if p != hold]
        Xtr, ytr = collect(others, "train")
        Xte, yte = make(hold, "all")
        clf = new_rf().fit(Xtr, ytr)
        show(yte, clf.predict(Xte), f"UNSEEN PLACE: trained on {others}, tested on {hold}")

# ---- 2) same places, time split ----
Xtr, ytr = collect(list(data), "train")
Xte, yte = collect(list(data), "test")
clf = new_rf().fit(Xtr, ytr)
show(yte, clf.predict(Xte), "SAME PLACES (time split)")
print(classification_report(yte, clf.predict(Xte), labels=LABELS, digits=3))

# ---- final model on all data ----
Xall, yall = collect(list(data), "all")
final = new_rf().fit(Xall, yall)
joblib.dump({"model": final, "mask": mask, "labels": LABELS,
             "win": WIN, "names": NAMES, "cal": CAL}, "model.pkl")
print("\nSaved model.pkl (new calibrated format)")