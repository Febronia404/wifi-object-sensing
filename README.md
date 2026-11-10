# WiFi Object Sensing: Human Presence & Motion Detection with ESP32 CSI

Detect whether a room is **empty**, has a person **standing still**, or has a person **moving**, using Wi-Fi Channel State Information (CSI) from two ESP32 boards. No router is needed.

> Demo video: *(add your YouTube link here)*

## How it works

- One ESP32 (**TX**) broadcasts ESP-NOW packets at 50 packets/sec on channel 11.
- A second ESP32 (**RX**) extracts the CSI (64 subcarriers, LLTF) from each packet and prints it as CSV over Serial (500000 baud).
- A Python script on the laptop reads the stream, computes features over a 50-packet window (about 1 second), and classifies the state with a Random Forest.

### Features (per window)

Each packet is first divided by its mean amplitude to remove AGC fluctuation. Then six features are computed: `cv_median`, `cv_mean`, `cv_max` (variation of each subcarrier over time), `shift_mean`, `shift_median` (difference from the empty-room spectrum), and `diff_mean` (change between consecutive packets).

### Calibration

For every new location, the first ~11 seconds (room empty) are used to build a reference spectrum and baseline feature levels. All features are divided by this baseline, so the model learns general patterns instead of absolute values tied to one place.

The classifier is a Random Forest (200 trees) with majority voting over the last 3 predictions (about 1.5 seconds) to reduce flicker.

## Hardware and software

- 2 x ESP32 DevKit (classic, CP2102, micro-USB)
- Laptop running Windows
- Arduino IDE 1.8.19 with the ESP32 core (Espressif) 3.3.7
- Python 3.14.3 with: pyserial 3.5, numpy 2.4.2, pandas 3.0.2, scikit-learn 1.9.1, matplotlib 3.10.8

Install the Python packages:

```
pip install -r requirements.txt
```

## Repository contents

| Path | Purpose |
|---|---|
| `firmware/tx/tx.ino` | TX sketch (sends ESP-NOW packets) |
| `firmware/rx/rx.ino` | RX sketch (reads CSI, prints CSV over Serial) |
| `collect.py` | Record labeled CSI data |
| `train_model.py` | Train and evaluate the Random Forest, save `model.pkl` |
| `live_predict.py` | Live prediction in the terminal (auto-calibrates) |
| `csi_dashboard_v3.py` | Dashboard with rule-based decisions |
| `csi_dashboard_v4.py` | Dashboard with decisions from the trained model |
| `data/` | Training recordings (`data/<place>/<label>_<timestamp>.csv`) |
| `model.pkl` | Trained model |

## Usage

### 1. Flash the boards (once)

Open `firmware/tx/tx.ino` and `firmware/rx/rx.ino` in Arduino IDE, select the board (ESP32 Dev Module), and upload one sketch to each board.

### 2. Place the boards

Keep the same setup every time: about 1 m height, 1.5 to 2 m apart, with a clear line of sight between them. Do not move them during a recording session.

### 3. Collect data (per new place)

Edit `PORT` at the top of the Python scripts to match your COM port (the code uses `COM7` by default; check Device Manager). Close the Arduino Serial Monitor first.

```
python collect.py empty 120 PLACE
python collect.py standing 120 PLACE
python collect.py moving 120 PLACE
```

Usage: `python collect.py LABEL SECONDS PLACE`

### 4. Train

```
python train_model.py
```

Each folder inside `data/` must contain all three labels. The script reports a leave-one-place-out test and a time-split test, then saves `model.pkl`.

### 5. Run live

Power the TX board first, then RX, then:

```
python live_predict.py
```

During the first ~11 seconds, stay away from the boards and keep the room still. After calibration, the state is printed every half second.

## Results

| Evaluation | Accuracy |
|---|---|
| Train on room2, test on bedroom (unseen place) | 95.8% |
| Train on bedroom, test on room2 (unseen place) | 87.4% |
| Same places, time split (70% train / 30% test with a gap) | 98.5% |

Data: 3 sessions (empty / standing / moving) of 2 minutes each, in 2 places. The most common error is EMPTY being predicted as MOVING in the place with the weaker signal.

Motion energy alone cannot separate STANDING from EMPTY. The spectrum shift can (median shift: about 0.0044 for EMPTY vs 0.0645 for STANDING).

## Limitations

- Detects presence and motion of a single person only. It does not count people.
- Two boards give one link, so there is no real pose estimation. The human figure in the dashboard is a visual representation of the state (empty / present / moving).
- Needs a calibration (11-12 s with an empty room) in every new place.
- Accuracy on an unseen place is about 87% to 96%, not 100%.
- Sensitive to board placement (see "Place the boards").

## Troubleshooting

- **Cannot open COM7**: close the Serial Monitor or dashboard, or check the COM number in Device Manager.
- **No data**: unplug both boards, plug in TX, then RX.
- **Packet rate far below 50/sec**: boards were moved or are too far apart, or the USB connection is poor.
- **Everything shows MOVING or states look wrong**: calibration was done while someone was near the boards. Step away and recalibrate (press `c` in the dashboard).
- **Model not loaded**: `model.pkl` is missing or in another folder. Run `python train_model.py` again.

## Future work

- More places and more people for better generalization.
- More boards (multiple links) for real location/motion estimation.
- Less reliance on manual calibration.

## License

*(add a license, e.g. MIT, from GitHub's "Add file" menu)*
