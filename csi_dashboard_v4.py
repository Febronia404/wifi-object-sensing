"""
WiFi CSI Room Sensing Monitor v3 (ESP32 + ESP-NOW)

3D room scene with a walking figure and WiFi waves, CSI heatmap, live amplitude,
motion plot, two gauges, event log. Siren sound while MOVEMENT is detected.
Every packet is saved to a CSV file (csi_YYYYmmdd_HHMMSS.csv).

Keys:  + / -  = sensitivity up / down      c = recalibrate (empty room)
       b = sound on/off                    f = fullscreen
"""
import collections
import csv
import sys
import threading
import time
from datetime import datetime

import numpy as np
import serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

try:
    import winsound
except ImportError:
    winsound = None

# ============ Settings ============
PORT = "COM7"          # the RX board port
BAUD = 500000          # must match Serial.begin() in the RX sketch
N_SUB = 64
WIN = 50               # packets per window (~1 second at 50 packets/sec)
CALIB_SECONDS = 12     # empty-room calibration time (model needs ~550 packets)
PRESENT_X = 1.4        # motion needed for "present" = baseline x this
MOVE_X = 2.2           # motion needed for "movement" = baseline x this
HOLD_S = 1.5           # seconds before going to a lower state (less flicker)
HEAT_LEN = 150         # packets shown in the heatmap
SOUND_ON = True        # siren while movement (Windows only)
LABEL = "unlabeled"    # used later when collecting labeled data
USE_MODEL = True       # decide with model.pkl (rules still drive the gauges)
# ==================================

MODEL = None
if USE_MODEL:
    try:
        import joblib
        MODEL = joblib.load("model.pkl")
        if MODEL.get("win") != WIN or "cal" not in MODEL:
            raise ValueError("model.pkl is not the new calibrated format")
        print("Model loaded: model.pkl")
    except Exception as e:
        MODEL = None
        print("Model NOT loaded, using rules only:", e)

lock = threading.Lock()
window = collections.deque(maxlen=WIN)
hist = collections.deque(maxlen=HEAT_LEN)
rows = collections.deque(maxlen=4000)
latest = np.zeros(N_SUB)
latest_rssi = 0
pkt_total = 0
running = True

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
except serial.SerialException as e:
    print(f"Cannot open {PORT}: {e}")
    print("-> Close the Arduino Serial Monitor and check the COM port.")
    sys.exit(1)

fname = datetime.now().strftime("csi_%Y%m%d_%H%M%S.csv")
csv_file = open(fname, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp_ms", "rssi", "channel"]
                + [f"a{i}" for i in range(N_SUB)] + ["label"])
print("Saving data to:", fname)


def reader():
    global latest, latest_rssi, pkt_total
    n = 0
    while running:
        try:
            line = ser.readline().decode(errors="ignore").strip()
        except Exception:
            if not running:
                break
            continue
        if not line or line[0] == "#" or line.startswith("timestamp"):
            continue
        parts = line.split(",")
        if len(parts) != 3 + N_SUB:
            continue
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            continue
        writer.writerow(vals + [LABEL])
        n += 1
        if n % 50 == 0:
            csv_file.flush()
        amp = np.array(vals[3:], dtype=float)
        with lock:
            window.append(amp)
            hist.append(amp)
            rows.append(amp)
            latest = amp
            latest_rssi = vals[1]
            pkt_total += 1


threading.Thread(target=reader, daemon=True).start()

# ---------- look ----------
BG = "#120b1f"
PANEL = "#1b1230"
BORDER = "#3a2a5c"
VIOLET = "#b66dff"
PINK = "#ff5fa2"
MUTED = "#a99cc9"
TEXT = "#efe9ff"
COLORS = {"empty": "#2ee6a6", "present": "#ffb020", "moving": "#ff3b3b"}
LABELS = {"empty": "ROOM EMPTY", "present": "PERSON PRESENT",
          "moving": "MOVEMENT DETECTED"}
RANK = {"empty": 0, "present": 1, "moving": 2}

# ---------- detection state ----------
state = {"calib_start": None, "scores": [], "specs": [], "mask": None,
         "base": None, "sd": 0.0, "base_spec": None,
         "shift_med": 0.0, "shift_sd": 0.0, "sens": 1.0,
         "last_pkts": 0, "last_t": time.time(), "pps": 0.0,
         "last_frame": time.time(), "kind": "empty", "low_since": None,
         "present_s": 0.0, "moves": 0, "sound": SOUND_ON, "alarm": False,
         "calib_pkt0": 0, "mref": None, "mbase": None, "mpred": None,
         "mvote": collections.deque(maxlen=3), "mlast": 0.0}
motion_hist = collections.deque(maxlen=400)
shift_hist = collections.deque(maxlen=400)
events = collections.deque(maxlen=5)


def log(msg):
    events.append(f"{datetime.now():%H:%M:%S}  {msg}")


def recalibrate():
    state.update(calib_start=None, scores=[], specs=[], mask=None, base=None,
                 base_spec=None, kind="empty", low_since=None, alarm=False,
                 mref=None, mbase=None, mpred=None, mlast=0.0)
    state["mvote"].clear()
    motion_hist.clear()
    shift_hist.clear()
    log("recalibrating...")


def features(arr, mask):
    """Motion = median over subcarriers of std/mean in the window (after
    removing per-packet gain changes). Spec = mean normalized spectrum."""
    A = arr[:, mask]
    A = A / np.maximum(A.mean(axis=1, keepdims=True), 1e-6)
    m = np.maximum(A.mean(axis=0), 1e-6)
    cv = A.std(axis=0) / m
    return float(np.median(cv)), m


def thresholds():
    b, sd, s = state["base"], state["sd"], state["sens"]
    tp = b + max(b * (PRESENT_X - 1), 3 * sd) / s
    tm = b + max(b * (MOVE_X - 1), 6 * sd) / s
    ts = state["shift_med"] + max(4 * state["shift_sd"],
                                  0.6 * state["shift_med"], 0.01) / s
    return tp, tm, ts


def m_norm(w):
    A = w[:, MODEL["mask"]]
    return A / np.maximum(A.mean(axis=1, keepdims=True), 1e-6)


def m_feats(w, ref):
    A = m_norm(w)
    mm = np.maximum(A.mean(axis=0), 1e-6)
    cv = A.std(axis=0) / mm
    shift = np.abs(mm - ref) / ref
    d = np.abs(np.diff(A, axis=0)).mean()
    return np.array([np.median(cv), cv.mean(), cv.max(),
                     shift.mean(), np.median(shift), d])


def m_calibrate(R):
    # same calibration as live_predict.py (skip the first 100 packets)
    if len(R) < 100 + WIN + 25:
        raise ValueError("not enough packets")
    C = R[100:100 + MODEL["cal"]]
    idx = range(0, len(C) - WIN, 25)
    ref = np.median([m_norm(C[i:i + WIN]).mean(axis=0) for i in idx], axis=0)
    base = np.median([m_feats(C[i:i + WIN], ref) for i in idx], axis=0)
    return ref, np.maximum(base, 1e-4)


def siren():
    while running:
        if state["alarm"] and state["sound"] and winsound:
            try:
                winsound.Beep(1000, 150)
                if state["alarm"]:
                    winsound.Beep(750, 150)
            except Exception:
                time.sleep(0.3)
        else:
            time.sleep(0.05)


threading.Thread(target=siren, daemon=True).start()

# ---------- figure ----------
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 8.5), facecolor=BG)
fig.canvas.manager.set_window_title("WiFi CSI Room Sensing Monitor")


def panel(rect):
    a = fig.add_axes(rect)
    a.set_facecolor(PANEL)
    for sp in a.spines.values():
        sp.set_color(BORDER)
    return a


def blank(a):
    a.set_xticks([])
    a.set_yticks([])
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)


# --- header ---
hdr = panel([0.02, 0.90, 0.96, 0.085])
blank(hdr)
hdr_bar = Rectangle((0, 0), 0.010, 1, fc=COLORS["empty"], ec="none")
hdr.add_patch(hdr_bar)
hdr.text(0.025, 0.80, "WIFI CSI  ROOM SENSING", fontsize=9, color=MUTED,
         va="center")
txt_status = hdr.text(0.025, 0.36, "", fontsize=24, weight="bold",
                      va="center", color=TEXT)
txt_sub = hdr.text(0.44, 0.36, "", fontsize=11, color=MUTED, va="center")
txt_stats = hdr.text(0.985, 0.5, "", fontsize=11, color=TEXT, va="center",
                     ha="right", linespacing=1.5)

# --- 3D scene ---
fig.add_artist(Rectangle((0.02, 0.29), 0.64, 0.60, transform=fig.transFigure,
                         fc=PANEL, ec=BORDER, zorder=-5))
ax3 = fig.add_axes([0.02, 0.29, 0.64, 0.60], projection="3d")
ax3.set_axis_off()
ax3.patch.set_alpha(0)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 6)
ax3.set_zlim(0, 3)
ax3.set_box_aspect((10, 6, 3), zoom=1.25)
ax3.view_init(elev=27, azim=-62)

# floor grid (one line object, segments separated by NaN)
gx, gy, gz = [], [], []
for xv in range(0, 11):
    gx += [xv, xv, np.nan]
    gy += [0, 6, np.nan]
for yv in range(0, 7):
    gx += [0, 10, np.nan]
    gy += [yv, yv, np.nan]
gz = [0] * len(gx)
floor = ax3.plot(gx, gy, gz, lw=0.9, alpha=0.4, color=COLORS["empty"])[0]


def L3(**kw):
    return ax3.plot([0, 0], [0, 0], [0, 0], **kw)[0]


TX = (0.7, 3.0)
RX = (9.3, 3.0)
for (x, y), name in ((TX, "TX"), (RX, "RX")):
    ax3.plot([x, x], [y, y], [0, 1.3], lw=7, color=VIOLET,
             solid_capstyle="round")
    ax3.plot([x], [y], [1.4], marker="o", ms=13, color=PINK, ls="")
    ax3.text(x, y, 1.85, name, color=TEXT, fontsize=12, weight="bold",
             ha="center")
link = ax3.plot([TX[0], RX[0]], [TX[1], RX[1]], [1.4, 1.4], lw=1.5, ls="--",
                color=COLORS["empty"], alpha=0.5)[0]

N_RIP = 4
tx_rip = [L3(lw=2) for _ in range(N_RIP)]
sc_rip = [L3(lw=2) for _ in range(3)]
shadow = L3(lw=2, color="black", alpha=0.5)
LIMBS = ("body", "sh", "hp", "al", "ar", "ll", "lr")
glow = {k: L3(lw=15, alpha=0.1, solid_capstyle="round") for k in LIMBS}
limbs = {k: L3(lw=5, solid_capstyle="round") for k in LIMBS}
head3 = ax3.plot([5], [3], [1.8], marker="o", ms=24, ls="", mew=1.5,
                 mec="white")[0]
TH = np.linspace(0, 2 * np.pi, 72)


def ring(cx, cy, r):
    x = cx + r * np.cos(TH)
    y = cy + r * np.sin(TH)
    out = (x < 0) | (x > 10) | (y < 0) | (y > 6)
    x = np.where(out, np.nan, x)
    y = np.where(out, np.nan, y)
    return x, y, np.full_like(x, 0.02)


def draw_scene(kind, t):
    fg = COLORS[kind]
    px, py, s, dz = 5.0, 3.0, 0.0, 0.0
    if kind == "moving":
        px = 5 + 3.4 * np.sin(t * 0.55)
        py = 3 + 1.3 * np.sin(t * 0.37 + 1)
        s = 0.7 * np.sin(t * 7)
        dz = 0.03 * abs(np.sin(t * 7))
    elif kind == "present":
        dz = 0.012 * np.sin(t * 2.5)
    alpha = 0.18 if kind == "empty" else 1.0
    ls = "--" if kind == "empty" else "-"
    g_alpha = 0.0 if kind == "empty" else 0.12

    hz, sz = 0.95 + dz, 1.5 + dz

    def end(x0, y0, z0, length, ang, dy):
        return (x0 + length * np.sin(ang), y0 + dy, z0 - length * np.cos(ang))

    segs = {
        "body": ((px, py, sz + 0.08), (px, py, hz)),
        "sh": ((px, py - 0.2, sz), (px, py + 0.2, sz)),
        "hp": ((px, py - 0.1, hz), (px, py + 0.1, hz)),
        "al": ((px, py - 0.2, sz), end(px, py - 0.2, sz, 0.6, -0.2 + s, -0.03)),
        "ar": ((px, py + 0.2, sz), end(px, py + 0.2, sz, 0.6, 0.2 + s, 0.03)),
        "ll": ((px, py - 0.1, hz), end(px, py - 0.1, hz, 0.95, -0.08 - s, 0)),
        "lr": ((px, py + 0.1, hz), end(px, py + 0.1, hz, 0.95, 0.08 - s, 0)),
    }
    for k, (p, q) in segs.items():
        xs, ys, zs = [p[0], q[0]], [p[1], q[1]], [p[2], q[2]]
        for line, a in ((limbs[k], alpha), (glow[k], g_alpha)):
            line.set_data_3d(xs, ys, zs)
            line.set_color(fg)
            line.set_alpha(a)
        limbs[k].set_linestyle(ls)
    head3.set_data_3d([px], [py], [1.78 + dz])
    head3.set_markerfacecolor(fg)
    head3.set_alpha(alpha)

    if kind == "empty":
        shadow.set_visible(False)
    else:
        shadow.set_visible(True)
        shadow.set_data_3d(px + 0.4 * np.cos(TH), py + 0.22 * np.sin(TH),
                           np.full_like(TH, 0.02))

    sp = {"empty": 0.2, "present": 0.3, "moving": 0.6}[kind]
    for i, c in enumerate(tx_rip):
        ph = (t * sp + i / N_RIP) % 1.0
        c.set_data_3d(*ring(TX[0], TX[1], 0.2 + ph * 9.5))
        c.set_color(VIOLET if kind == "empty" else fg)
        c.set_alpha(0.55 * (1 - ph))
    for i, c in enumerate(sc_rip):
        if kind == "empty":
            c.set_visible(False)
            continue
        ph = (t * sp * 1.6 + i / 3) % 1.0
        c.set_visible(True)
        c.set_data_3d(*ring(px, py, 0.3 + ph * 2.4))
        c.set_color(fg)
        c.set_alpha(0.8 * (1 - ph))

    floor.set_color(fg)
    link.set_color(fg)
    ax3.view_init(elev=27, azim=-62 + 9 * np.sin(t * 0.3))


# --- gauges ---
gax = panel([0.68, 0.57, 0.30, 0.32])
blank(gax)
gax.text(0.05, 0.91, "MOTION", fontsize=9, color=MUTED)
gax.add_patch(Rectangle((0.05, 0.72), 0.90, 0.13, fc=BG, ec=BORDER))
g_motion = Rectangle((0.05, 0.72), 0.0, 0.13, fc=COLORS["empty"])
gax.add_patch(g_motion)
tick_p = gax.plot([0, 0], [0.69, 0.88], color=COLORS["present"], lw=2,
                  visible=False)[0]
tick_m = gax.plot([0, 0], [0.69, 0.88], color=COLORS["moving"], lw=2,
                  visible=False)[0]
gax.text(0.05, 0.60, "STATIC PRESENCE", fontsize=9, color=MUTED)
gax.add_patch(Rectangle((0.05, 0.41), 0.90, 0.13, fc=BG, ec=BORDER))
g_shift = Rectangle((0.05, 0.41), 0.0, 0.13, fc=COLORS["empty"])
gax.add_patch(g_shift)
tick_s = gax.plot([0, 0], [0.38, 0.57], color=COLORS["present"], lw=2,
                  visible=False)[0]
txt_sens = gax.text(0.05, 0.21, "", fontsize=12, color=VIOLET, weight="bold")
gax.text(0.05, 0.07, "[+/-] sensitivity   [c] recalibrate   [b] sound",
         fontsize=8, color=MUTED)

# --- events ---
eax = panel([0.68, 0.29, 0.30, 0.25])
blank(eax)
eax.text(0.05, 0.90, "EVENT LOG", fontsize=9, color=MUTED)
txt_events = eax.text(0.05, 0.78, "", fontsize=10, color=TEXT, va="top",
                      family="monospace", linespacing=1.6)

# --- bottom row ---
ax_heat = panel([0.05, 0.09, 0.34, 0.16])
heat = ax_heat.imshow(np.zeros((10, HEAT_LEN)), aspect="auto", cmap="plasma",
                      origin="lower", vmin=0, vmax=0.2, interpolation="nearest")
ax_heat.set_title("CSI heatmap (change per subcarrier)", fontsize=10,
                  color=MUTED)
ax_heat.tick_params(labelsize=8)

ax_amp = panel([0.46, 0.09, 0.22, 0.16])
(line_amp,) = ax_amp.plot(np.arange(2, N_SUB), np.zeros(N_SUB - 2), lw=1.6,
                          color=VIOLET)
ax_amp.set_title("Live amplitude", fontsize=10, color=MUTED)
ax_amp.set_xlim(2, N_SUB - 1)
ax_amp.set_ylim(0, 40)
ax_amp.grid(color=BORDER, alpha=0.6)
ax_amp.tick_params(labelsize=8)

ax_mot = panel([0.76, 0.09, 0.22, 0.16])
(line_mot,) = ax_mot.plot([], [], lw=1.6, color=PINK)
h_pres = ax_mot.axhline(0, ls="--", color=COLORS["present"], visible=False)
h_move = ax_mot.axhline(0, ls="--", color=COLORS["moving"], visible=False)
ax_mot.set_title("Motion score", fontsize=10, color=MUTED)
ax_mot.set_xlim(0, 400)
ax_mot.set_ylim(0, 0.05)
ax_mot.grid(color=BORDER, alpha=0.6)
ax_mot.tick_params(labelsize=8)


def update(_):
    now = time.time()
    with lock:
        arr = np.array(window) if len(window) == WIN else None
        lat = latest.copy()
        H = np.array(hist) if len(hist) > 5 else None
        total = pkt_total
        rssi = latest_rssi

    dt = now - state["last_t"]
    if dt >= 1.0:
        state["pps"] = (total - state["last_pkts"]) / dt
        state["last_pkts"] = total
        state["last_t"] = now
    dt_frame = min(now - state["last_frame"], 0.5)
    state["last_frame"] = now

    mask = state["mask"]
    if arr is not None and mask is None:
        m = arr.mean(axis=0) > 3
        m[:2] = False
        if m.sum() >= 10:
            state["mask"] = mask = m
    feat = features(arr, mask) if (arr is not None and mask is not None) else None

    show = "empty"
    title, title_color, sub = "", COLORS["empty"], ""
    motion_s = shift_s = None

    if feat is None or state["pps"] < 5:
        title, title_color = "NO DATA", MUTED
        sub = "Check that TX is powered and RX is connected"
    else:
        motion, spec = feat
        motion_hist.append(motion)
        if state["base"] is None:
            if state["calib_start"] is None:
                state["calib_start"] = now
                state["calib_pkt0"] = total
            state["scores"].append(motion)
            state["specs"].append(spec)
            left = CALIB_SECONDS - (now - state["calib_start"])
            if left <= 0:
                bs = np.median(np.array(state["specs"]), axis=0)
                d = [float(np.mean(np.abs(s - bs) / bs)) for s in state["specs"]]
                state.update(base=max(float(np.median(state["scores"])), 0.003),
                             sd=float(np.std(state["scores"])), base_spec=bs,
                             shift_med=float(np.median(d)),
                             shift_sd=float(np.std(d)))
                log("calibration done")
                if MODEL is not None:
                    n_new = max(1, min(total - state["calib_pkt0"], len(rows)))
                    with lock:
                        R = np.array(list(rows)[-n_new:])
                    try:
                        state["mref"], state["mbase"] = m_calibrate(R)
                        log("model calibrated")
                    except Exception as e:
                        log("model calib FAILED")
                        print("Model calibration failed:", e)
            else:
                title, title_color = f"CALIBRATING... {left:0.0f}s", VIOLET
                sub = "Keep the room EMPTY"
        if state["base"] is not None:
            bs = state["base_spec"]
            shift_hist.append(float(np.mean(np.abs(spec - bs) / bs)))
            motion_s = float(np.mean(list(motion_hist)[-10:]))
            shift_s = float(np.mean(list(shift_hist)[-10:]))
            tp, tm, ts = thresholds()
            cand = ("moving" if motion_s > tm
                    else "present" if (motion_s > tp or shift_s > ts)
                    else "empty")
            if MODEL is not None and state["mref"] is not None:
                if now - state["mlast"] >= 0.5:
                    state["mlast"] = now
                    x = m_feats(arr, state["mref"]) / state["mbase"]
                    p = MODEL["model"].predict([x])[0]
                    state["mvote"].append(p)
                    state["mpred"] = collections.Counter(
                        state["mvote"]).most_common(1)[0][0]
                if state["mpred"] is not None:
                    cand = {"empty": "empty", "standing": "present",
                            "moving": "moving"}[state["mpred"]]
            cur = state["kind"]
            new = cur
            if RANK[cand] >= RANK[cur]:
                new = cand
                state["low_since"] = None
            elif state["low_since"] is None:
                state["low_since"] = now
            elif now - state["low_since"] >= HOLD_S:
                new = cand
                state["low_since"] = None
            if new != cur:
                log(LABELS[new].lower())
                if new == "moving":
                    state["moves"] += 1
            state["kind"] = new
            show = new
            title, title_color = LABELS[new], COLORS[new]
            sub = f"motion {motion_s:.3f}   |   static shift {shift_s:.3f}"
            if state["mpred"] is not None:
                sub += f"   |   model: {state['mpred']}"

    state["alarm"] = (show == "moving")
    if show != "empty":
        state["present_s"] += dt_frame

    draw_scene(show, now)
    fg = COLORS[show]
    hdr_bar.set_facecolor(title_color if title else fg)
    txt_status.set_text(title)
    txt_status.set_color(title_color)
    txt_sub.set_text(sub)
    ps = int(state["present_s"])
    txt_stats.set_text(f"Presence time {ps // 60:02d}:{ps % 60:02d}   "
                       f"Movement events {state['moves']}\n"
                       f"{state['pps']:.0f} pkt/s   RSSI {rssi} dBm")
    txt_sens.set_text(f"SENSITIVITY  {state['sens']:.1f}x")
    txt_events.set_text("\n".join(events) if events else "waiting...")

    # gauges
    if motion_s is not None:
        tp, tm, ts = thresholds()
        full_m, full_s = tm * 1.6, ts * 2.0
        g_motion.set_width(0.90 * min(1.0, motion_s / full_m))
        g_motion.set_facecolor(fg)
        tick_p.set_xdata([0.05 + 0.90 * tp / full_m] * 2)
        tick_m.set_xdata([0.05 + 0.90 * tm / full_m] * 2)
        g_shift.set_width(0.90 * min(1.0, shift_s / full_s))
        g_shift.set_facecolor(COLORS["present"] if shift_s > ts
                              else COLORS["empty"])
        tick_s.set_xdata([0.05 + 0.90 * ts / full_s] * 2)
        for tk in (tick_p, tick_m, tick_s):
            tk.set_visible(True)
    else:
        g_motion.set_width(0.0)
        g_shift.set_width(0.0)
        for tk in (tick_p, tick_m, tick_s):
            tk.set_visible(False)

    # heatmap
    if H is not None and mask is not None:
        V = H[:, mask]
        V = V / np.maximum(V.mean(axis=1, keepdims=True), 1e-6)
        D = np.abs(V - np.median(V, axis=0))
        P = np.zeros((D.shape[1], HEAT_LEN))
        P[:, -D.shape[0]:] = D.T
        heat.set_data(P)
        heat.set_extent((0, HEAT_LEN, 0, D.shape[1]))
        heat.set_clim(0, max(0.15, float(np.percentile(D, 99))))
        ax_heat.set_xlim(0, HEAT_LEN)
        ax_heat.set_ylim(0, D.shape[1])

    # amplitude + motion plot
    line_amp.set_ydata(lat[2:])
    ax_amp.set_ylim(0, max(30, float(lat[2:].max()) * 1.2))
    if motion_hist:
        ys = list(motion_hist)
        line_mot.set_data(np.arange(len(ys)), ys)
        ymax = max(ys)
        if state["base"] is not None:
            tp, tm, ts = thresholds()
            h_pres.set_ydata([tp] * 2)
            h_move.set_ydata([tm] * 2)
            h_pres.set_visible(True)
            h_move.set_visible(True)
            ymax = max(ymax, tm * 1.3)
        else:
            h_pres.set_visible(False)
            h_move.set_visible(False)
        ax_mot.set_ylim(0, ymax * 1.2)


def on_key(event):
    if event.key == "c":
        recalibrate()
    elif event.key == "b":
        state["sound"] = not state["sound"]
        log(f"sound {'ON' if state['sound'] else 'OFF'}")
    elif event.key in ("+", "=", "up"):
        state["sens"] = min(4.0, state["sens"] * 1.2)
        log(f"sensitivity {state['sens']:.1f}x")
    elif event.key in ("-", "down"):
        state["sens"] = max(0.3, state["sens"] / 1.2)
        log(f"sensitivity {state['sens']:.1f}x")


fig.canvas.mpl_connect("key_press_event", on_key)
ani = FuncAnimation(fig, update, interval=60, cache_frame_data=False)

try:
    plt.show()
finally:
    running = False
    try:
        ser.close()
    except Exception:
        pass
    csv_file.close()
    print("Closed. Data saved in", fname)
