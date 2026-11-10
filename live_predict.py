import sys
import collections
import time
import joblib
import numpy as np
import serial

PORT, BAUD, N_SUB = "COM7", 500000, 64
STEP, SKIP = 25, 100

m = joblib.load("model.pkl")
clf, mask, WIN, CAL = m["model"], m["mask"], m["win"], m["cal"]


def norm(w):
    A = w[:, mask]
    return A / np.maximum(A.mean(axis=1, keepdims=True), 1e-6)


def raw_feats(w, ref):
    A = norm(w)
    mm = np.maximum(A.mean(axis=0), 1e-6)
    cv = A.std(axis=0) / mm
    shift = np.abs(mm - ref) / ref
    d = np.abs(np.diff(A, axis=0)).mean()
    return np.array([np.median(cv), cv.mean(), cv.max(),
                     shift.mean(), np.median(shift), d])


def read_row():
    line = ser.readline().decode(errors="ignore").strip()
    if not line or line[0] == "#" or line.startswith("timestamp"):
        return None
    parts = line.split(",")
    if len(parts) != 3 + N_SUB:
        return None
    try:
        return np.array([int(p) for p in parts][3:], dtype=float)
    except ValueError:
        return None


try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Cannot open {PORT}: {e}")
    print("-> Close the dashboard / Serial Monitor first.")
    sys.exit(1)

print("CALIBRATING (~11 s) - leave the area and stay still...")
rows = []
while len(rows) < SKIP + CAL:
    r = read_row()
    if r is not None:
        rows.append(r)
C = np.array(rows[SKIP:])
idx = range(0, len(C) - WIN, STEP)
ref = np.median([norm(C[i:i + WIN]).mean(axis=0) for i in idx], axis=0)
base = np.maximum(np.median([raw_feats(C[i:i + WIN], ref) for i in idx],
                            axis=0), 1e-4)
print("Calibration done. Prediction starts.")

buf = collections.deque(maxlen=WIN)
recent = collections.deque(maxlen=3)
n = 0
print("Press Ctrl+C to stop.")
try:
    while True:
        r = read_row()
        if r is None:
            continue
        buf.append(r)
        n += 1
        if len(buf) == WIN and n % STEP == 0:
            x = raw_feats(np.array(buf), ref) / base
            pred = clf.predict([x])[0]
            recent.append(pred)
            vote = collections.Counter(recent).most_common(1)[0][0]
            print(f"{time.strftime('%H:%M:%S')}  {vote.upper():9s} (raw: {pred})")
except KeyboardInterrupt:
    pass
ser.close()
print("Stopped.")