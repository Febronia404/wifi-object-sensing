import sys, os, csv, time, serial
from datetime import datetime

PORT = "COM7"
BAUD = 500000
N_SUB = 64

if len(sys.argv) < 2:
    print("Usage: python collect.py LABEL [SECONDS]")
    sys.exit(1)

label = sys.argv[1]
secs = int(sys.argv[2]) if len(sys.argv) > 2 else 120
place = sys.argv[3] if len(sys.argv) > 3 else "bedroom"

os.makedirs(os.path.join("data", place), exist_ok=True)
fname = os.path.join("data", place, f"{label}_{datetime.now():%Y%m%d_%H%M%S}.csv")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Cannot open {PORT}: {e}")
    print("-> Close the dashboard / Serial Monitor first.")
    sys.exit(1)

f = open(fname, "w", newline="")
w = csv.writer(f)
w.writerow(["timestamp_ms", "rssi", "channel"]
           + [f"a{i}" for i in range(N_SUB)] + ["label"])

for i in range(5, 0, -1):
    print(f"Get in position... starting in {i}")
    time.sleep(1)
ser.reset_input_buffer()
print(f"RECORDING '{label}' for {secs}s")

n = 0
t0 = time.time()
last = t0
while time.time() - t0 < secs:
    line = ser.readline().decode(errors="ignore").strip()
    if not line or line[0] == "#" or line.startswith("timestamp"):
        continue
    parts = line.split(",")
    if len(parts) != 3 + N_SUB:
        continue
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        continue
    w.writerow(vals + [label])
    n += 1
    if time.time() - last >= 10:
        last = time.time()
        print(f"  {int(last - t0)}s  packets={n}")

f.close()
ser.close()
print(f"DONE. packets={n} (expected about {secs * 50})")
print("Saved:", fname)