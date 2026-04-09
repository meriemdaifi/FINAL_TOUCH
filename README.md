# Driver Monitoring System (DMS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Industrial-grade Driver Monitoring System** with hybrid detection (Haar + MediaPipe + CNN), real-time behavioral analysis, and ECU-like decision engine.
>
> **Author:** Daifi Meriem, Intern at Expleo Group Maroc for Stellantis

---

## Architecture

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Camera     │───>│  Face/Eye       │───>│  Behavioral      │
│   Module     │    │  Detection      │    │  Analysis        │
│ (video_acq.) │    │ (Haar/MediaPipe)│    │ (EAR/PERCLOS/MAR)│
└─────────────┘    └────────┬────────┘    └────────┬─────────┘
                            │                       │
                   ┌────────v────────┐    ┌────────v─────────┐
                   │  Head Pose      │    │  CNN Inference    │
                   │  Estimation     │    │  (OPEN/CLOSED/   │
                   │ (solvePnP)      │    │   YAWNING)       │
                   └────────┬────────┘    └────────┬─────────┘
                            │                       │
                   ┌────────v───────────────────────v─────────┐
                   │           ECU Decision Engine             │
                   │         (Finite State Machine)            │
                   │  ATTENTIVE → WARNING → FATIGUE → EMERGENCY│
                   └────────────────────┬──────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────┐
          │                             │                         │
  ┌───────v──────┐  ┌─────────v─────────┐  ┌──────v──────────────┐
  │ Visual/Audio │  │ Seat/Steering     │  │ Emergency Braking   │
  │ Alert        │  │ Vibration         │  │ + Pull to Side      │
  │ (Level 1)    │  │ (Level 2)         │  │ + Hazard Lights     │
  └──────────────┘  └───────────────────┘  │ (Level 3)           │
                                           └──────────────────────┘
```

## Hybrid Approach

The DMS uses **three complementary detection methods**:

| Method | Strengths | Use Case |
|--------|-----------|----------|
| **Haar Cascade** | Fast, no GPU needed, low compute | Embedded/resource-limited |
| **MediaPipe FaceMesh** | 468 landmarks, accurate, robust | Primary detection mode |
| **CNN (PyTorch)** | Learned features, classifies eye/mouth state | Drowsiness classification |

Switch between Haar and MediaPipe at runtime via `--mode` flag.

## System Requirements

- Python 3.9+
- Webcam (USB or integrated)
- MATLAB R2020a+ (for simulations, optional)
- GPU (optional, for CNN training)

## Installation

```bash
# Clone repository
git clone https://github.com/meriemdaifi/FINAL_TOUCH.git
cd FINAL_TOUCH

# Install dependencies
pip install -r requirements.txt
```

## Training the CNN

### 1. Prepare Dataset

Organize images in this structure:
```
data/
  train/
    open/       # Images of open eyes / normal faces
    closed/     # Images of closed eyes
    yawning/    # Images of yawning faces
  val/
    open/
    closed/
    yawning/
```

### 2. Run Training

```bash
python -m models.train --data_path data/ --epochs 50 --batch_size 32 --lr 0.001
```

Options:
- `--data_path`: Root data directory (default: `data`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 7)

Best model weights are saved to `models/saved_model/best_model.pth`.

## Running the Real-Time System

```bash
# Default: MediaPipe mode
python main.py

# Haar Cascade mode
python main.py --mode haar

# With CNN inference enabled
python main.py --use_cnn --cnn_model_path models/saved_model/best_model.pth

# Custom camera and resolution
python main.py --camera 1 --width 1280 --height 720

# Custom thresholds
python main.py --perclos_threshold 0.30 --ear_threshold 0.22
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | mediapipe | Detection mode: `mediapipe` or `haar` |
| `--camera` | 0 | Camera index |
| `--width` | 640 | Frame width |
| `--height` | 480 | Frame height |
| `--use_cnn` | False | Enable CNN inference |
| `--cnn_model_path` | models/saved_model/best_model.pth | CNN weights path |
| `--perclos_threshold` | 0.35 | PERCLOS fatigue threshold |
| `--ear_threshold` | 0.25 | EAR warning threshold |
| `--no_face_timeout` | 3.0 | Seconds without face → distraction |
| `--head_turn_timeout` | 6.0 | Seconds head turned → distraction |

Press **q** to quit the real-time display.

## Detection Modes

### Mode 1: Haar Cascade
- Uses OpenCV pre-trained cascades (frontal + profile face, eyes)
- Fast but less accurate
- No landmark-based EAR computation
- Head pose via proxy (profile detection ratio)

### Mode 2: MediaPipe FaceMesh
- 468 facial landmarks with sub-pixel accuracy
- Direct EAR/MAR computation from landmarks
- Accurate head pose via solvePnP
- Recommended for production use

## ECU Decision Logic

### Finite State Machine

```
ATTENTIVE  ←→  WARNING_DROWSINESS  →  FATIGUE  →  EMERGENCY
     ↕              ↕                     ↕
WARNING_DISTRACTION → DISTRACTED_HEAD → EMERGENCY
                      DISTRACTED_NO_FACE → EMERGENCY
```

### Priority Order
1. **No face detected** > timeout → `DISTRACTED_NO_FACE`
2. **Head turned** > 6 seconds → `DISTRACTED_HEAD`
3. **PERCLOS** > threshold → `FATIGUE`
4. **EAR low / MAR high / CNN** → `WARNING_DROWSINESS`
5. **Head distracted** (within timeout) → `WARNING_DISTRACTION`
6. Default → `ATTENTIVE`

### Warning Escalation System

| Level | Trigger | Actions |
|-------|---------|---------|
| **Level 1** | Warning state entered | Visual alert + Audio beep |
| **Level 2** | Warning persists / Fatigue | Seat vibration + Steering vibration |
| **Level 3** | Emergency | Emergency braking + Pull to side + Hazard lights (4 signals) |

## Prototype Architecture

### Components

| Component | Description |
|-----------|-------------|
| **Camera Module** | USB/CSI camera, 640×480 @ 30 FPS |
| **Processing Unit** | Raspberry Pi 4 / Jetson Nano / PC |
| **CNN Inference Engine** | PyTorch model on CPU/GPU |
| **ECU Decision Module** | Software FSM with hysteresis |
| **Audio Actuator** | Buzzer/speaker for Level 1 alerts |
| **Seat Vibration Motor** | Eccentric rotating mass (ERM) motor |
| **Steering Vibration** | Linear resonant actuator (LRA) |
| **Braking Interface** | CAN bus signal to brake ECU |
| **Hazard Light Relay** | GPIO-controlled relay for hazard lights |

### Emergency Scenario Flow

```
1. Camera detects driver face → MediaPipe extracts landmarks
2. EAR drops below threshold → PERCLOS rises
3. CNN confirms CLOSED eyes → WARNING_DROWSINESS (Level 1: beep)
4. Driver does not respond → Escalation to Level 2 (vibrations)
5. PERCLOS exceeds fatigue threshold → FATIGUE state
6. Face lost (driver slumps) → DISTRACTED_NO_FACE
7. Timeout exceeded → EMERGENCY (Level 3)
   → Emergency braking activated
   → Vehicle pulls to side of road
   → Hazard lights (4 signals) activated
   → Emergency services notified
```

## Testing

```bash
# Run all tests
pytest tests/test_dms.py -v

# Run specific test class
pytest tests/test_dms.py::TestComputeEAR -v

# Run with coverage
pytest tests/test_dms.py --cov=src --cov=models -v
```

## MATLAB Simulation

```matlab
% Run PERCLOS / EAR / State simulation
run('matlab/dms_simulation.m')

% View Simulink model description
run('matlab/dms_simulink_model.m')

% Run Stateflow simulation
ecu = ecu_stateflow(1/30);
ecu.simulate(60);
```

## Project Structure

```
.
├── src/
│   ├── __init__.py              # Package exports
│   ├── video_acquisition.py     # Webcam capture & preprocessing
│   ├── face_eye_detection.py    # Haar + MediaPipe dual-mode detection
│   ├── head_pose.py             # Head pose estimation (PnP + proxy)
│   ├── behavioral_analysis.py   # EAR, PERCLOS, MAR, blink rate
│   ├── ecu_decision.py          # ECU FSM with warning escalation
│   ├── scenarios.py             # Test scenario generators
│   └── visualization.py         # Frame overlay & dashboard
├── tests/
│   └── test_dms.py              # 30+ pytest unit & integration tests
├── matlab/
│   ├── dms_simulation.m         # PERCLOS/EAR/State simulation
│   ├── dms_simulink_model.m     # Simulink model description
│   └── simulink_model/
│       └── ecu_stateflow.m      # Stateflow state machine
├── models/
│   ├── cnn_model.py             # DrowsinessCNN architecture
│   ├── train.py                 # Training pipeline
│   ├── dataset_loader.py        # Dataset & augmentation
│   └── saved_model/             # Trained weights directory
│       └── .gitkeep
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── rapport_pfe.tex              # PFE report (LaTeX, French)
```

## Author

**Daifi Meriem**
Intern at **Expleo Group Maroc** for **Stellantis** clients

---

*This project was developed as part of a Projet de Fin d'Etudes (PFE) internship focused on Advanced Driver Assistance Systems (ADAS) and driver safety monitoring.*
