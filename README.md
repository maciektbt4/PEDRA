# PEDRA Object Detection (Master’s Thesis Fork)

A fork of **PEDRA** (Programmable Engine for Drone Reinforcement Learning Applications) extended for a master’s thesis with:

- **Searching and capturing an object** (e.g., a ball) in a simulated environment.
- Integration of **object detection (YOLOv8)** into the reinforcement learning loop.
- Improvements to the **DQN/DDQN implementation (+PER depending on configuration)** and experiment logging.
- Configuration handling and automatic generation of `settings.json` for AirSim.

> **Note (important for reproducibility / reviewer comments):** this repository contains research code. Running it requires properly prepared Unreal Engine environments (compiled `.exe`) and compatible Python package versions.

---

## Repository

- Code: `https://github.com/maciektbt4/pedra`

---

## Requirements

### System

- **Windows 10/11** (startup/cleanup scripts use `taskkill`).
- **Python 3.8–3.10** (commonly used/tested with TF 2.10).

### Simulation

Unreal Engine environments built as `.exe` with the following structure:

```
unreal_envs/
  indoor_cloud/
    indoor_cloud.exe
    config.cfg
    ...
  indoor_twist/
    indoor_twist.exe
    config.cfg
    ...
```

- **AirSim (plugin/client)** – the project uses the `airsim` Python package.

### ML / CV

- TensorFlow + Keras (required for RL models).
- YOLOv8 (`ultralytics`) for object detection.

---

## Installation

### 1) Clone the repo

```bash
git clone https://github.com/maciektbt4/PEDRA.git
cd pedra
```

### 2) Python environment (recommended: conda)

```bash
conda create -n pedra_tf2 python=3.8 -y
conda activate pedra_tf2
```

### 3) Install dependencies

The repo includes `requirements_new.txt` (TensorFlow 2.10):

```bash
pip install -r requirements_new.txt
```

**Additionally required for YOLOv8:**

```bash
pip install ultralytics
```

> If you use GPU: make sure your drivers/CUDA setup matches your TensorFlow version.

---

## Configuration

### `configs/config.cfg` (main runtime configuration)

By default `main.py` reads `configs/config.cfg`. Typical fields:

- `env_name`: environment name (folder under `unreal_envs/`)
- `ip_address`: AirSim IP (must match the generated `settings.json`)
- `algorithm`: e.g., `DeepQLearning`
- `mode`: `train` / `infer` / `move_around`

### `configs/DeepQLearning.cfg` (algorithm configuration)

Contains hyperparameters (e.g., epsilon schedule, replay buffer, target update, etc.).

---

## Running

### Training

1. Make sure the UE executable exists at `unreal_envs/<env_name>/<env_name>.exe`.
2. Set `mode: train` in `configs/config.cfg`.
3. Run:

```bash
python main.py
```

The script:
- generates `~\Documents\Airsim\settings.json` based on the config,
- starts the Unreal Engine environment,
- launches the algorithm from `algorithms/<algorithm>.py`.

### Inference (test / demo)

1. Set `mode: infer` in `configs/config.cfg`.
2. Run:

```bash
python main.py
```

### Manual UE environment start (optional)

If you only want to launch the `.exe` without RL:

```bash
python run_env_manual.py
```

---

## Object detection (YOLOv8)

The detector lives in `util/object_detector.py`:
- loads weights `yolov8n.pt` (included in the repo),
- by default filters the COCO class **"sports ball"** (class id = 32),
- exposes:
  - `detect_object(...)`
  - `object_is_captured(...)`
  - `visualize_object_detection(...)`

### Switching to your own model/class

- replace the weights file (e.g., your `.pt`),
- adjust class-id filtering in `util/object_detector.py`.

---

## Artifacts and results

During training the project may produce:
- logs and metrics (e.g., TensorBoard, depending on configuration),
- saved model weights under `models/trained/...`,
- per-agent text logs.

Exact paths depend on `env_type`, `env_name`, `model_name`, `train_type`, and the `.cfg` settings.

---

## Common issues

- **Missing UE environment / wrong path**: verify `unreal_envs/<env>/<env>.exe` exists.
- **AirSim connection problems**: IP in `configs/config.cfg` must match `Documents\Airsim\settings.json`.
- **Missing `ultralytics`**: install with `pip install ultralytics`.
- **Incompatible package versions**: start from a clean conda env (Python 3.8 is the safest baseline).

---

## Citation

If you use this code or results:
- cite the author’s master’s thesis,
- and cite the original PEDRA project (Aqeel Anwar et al.).

---

## License

MIT (as stated in the repository).
