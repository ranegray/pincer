# Pincer Dashboard

Real-time observability dashboard for the Pincer robot. Runs as part of the Pincer runtime and is accessible from any device on the local network.

## Features

- Live joint positions and loads (5-DOF arm + 2-DOF head + gripper)
- End-effector position in base frame
- IK solver state (target, error, convergence)
- MJPEG camera feed from the RealSense D435i
- Behavior control (start/stop detect-and-grasp from the UI)
- Torque enable/disable toggle

## Quick Start

### Install dependencies

```bash
pip install fastapi "uvicorn[standard]"
```

### Build the frontend (one-time)

```bash
cd src/pincer/dashboard/frontend
npm install
npm run build
```

### Run the runtime

```bash
# On the Jetson (with hardware connected)
PYTHONPATH=src python -m pincer.runtime

# Mock mode (no hardware, synthetic data)
PYTHONPATH=src python -m pincer.runtime --mock
```

The dashboard is served at **http://localhost:8080**. From another device on the network, use `http://<jetson-ip>:8080`.

### CLI options

```
--port /dev/ttyACM0         Serial port for motor bus
--camera-serial 310222077874 RealSense camera serial number
--dashboard-port 8080        Port for the dashboard
--host 0.0.0.0               Bind address
--mock                       Run with synthetic telemetry (no hardware)
-v                           Verbose logging
```

## Development

For frontend development with hot reload, run two terminals:

```bash
# Terminal 1: runtime backend
PYTHONPATH=src python -m pincer.runtime --mock

# Terminal 2: Vite dev server
cd src/pincer/dashboard/frontend
npm run dev
```

The Vite dev server runs at `http://localhost:5173` and proxies `/ws`, `/stream`, and `/api` to the runtime on port 8080.

## Architecture

```
Pincer Runtime (single process)
├── Hardware layer        Owns motor bus + camera, 10Hz read loop
├── RobotState            Thread-safe shared state store
├── Behavior engine       Runs behaviors in worker threads
└── Dashboard server      FastAPI in a background daemon thread
    ├── WS /ws            10Hz JSON telemetry
    ├── GET /stream       MJPEG camera feed (15fps)
    ├── POST /api/behavior/start   Trigger a behavior
    ├── POST /api/behavior/stop    Cancel active behavior
    ├── GET /api/state    REST snapshot
    └── GET /             Static React app
```

## API

### WebSocket `/ws`

Pushes JSON telemetry at ~10Hz:

```json
{
  "stamp": 12345.678,
  "joints": {
    "arm": [-12.3, 45.1, -30.0, 15.5, 0.2],
    "head": [5.2, -3.1],
    "gripper": 80.0,
    "loads": [142, 389, 210, 95, 40]
  },
  "ee": [0.123, -0.045, 0.210],
  "ik": {
    "target": [0.150, -0.050, 0.200],
    "error": 0.0032,
    "iterations": 47,
    "converged": true
  },
  "detection": {
    "bboxes": [[120, 200, 50, 60]],
    "centroids": [[145, 230]],
    "label": "red"
  },
  "loop_hz": 9.8,
  "behavior": { "name": "detect_and_grasp", "status": "running" },
  "torque_enabled": true
}
```

### `GET /stream`

MJPEG camera stream. Embed with `<img src="/stream">`.

### `POST /api/behavior/start`

```json
{ "name": "detect_and_grasp", "params": {} }
```

### `POST /api/behavior/stop`

Cancels the active behavior.

### `GET /api/state`

Returns the same JSON as the WebSocket (single snapshot).
