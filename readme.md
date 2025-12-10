# Dynamic_Loc

Dynamic_Loc is a framework for cleaning/retrieving distance measures from UWB hardwar finding the positions and giving real time indoor sub-meter localisation.
The main idea is to recreate and predict the trajectory of an agent given only the anchors position and the distance between the current position and those anchors.

---

## Project Structure

```text
Dynamic_Loc/
├── data_preparing/
│   ├── clean_positions.py       - Clean and filter trajectory files
│   ├── physical_atribute.py     - Compute kinematic properties (speed, acceleration)
│   ├── get_distance.py          - Extract distances from trajectory data
│   ├── predict_path.py          - Reconstruct path from distance matrix with Kalman filter
├── generating_bloc/
│   ├── convert_data.py          - Convert trajectory data
│   ├── anchor_positions.npy     - Anchor positions for distance calculations
│   ├── woltDynDev.json          - Configuration data
├── test_site/
│   ├── trajectories/            - Generated trajectory files (.npy)
│   └── test_*.csv               - Test data files
├── logs/                        - Log files
├── acc_spped.py                 - Analyze speed and acceleration
├── analyse.py                   - Compare cleaned paths with reference paths
├── cercles.py                   - Analyze circle intersections
├── mr_robot.py                  - Data collection script
├── visualisation.py             - Real-time trajectory visualization
├── Dynamic_loc.sh               - Main pipeline script (MQTT + collection)
├── generator_config.ini         - Configuration file
├── requirements.txt             - Python dependencies
└── readme.md
```
## Configuration File: generator config.ini
    Controls trajectory generation, distance matrix computation, and degradation:
    • [GENERATE] – Trajectory parameters: anchor positions, number of paths, mean speed, sampling frequency, plotting options, and random seed.
    • [MATRIX] – Distance matrix settings: input trajectory, anchor positions, output matrix file, and
      trajectory index.
    • [CANAL MODEL] – Degradation settings: input matrix, output file, maximum anchor masking
      ratio, and noise level.
    Allows easy adjustment of generation and simulation parameters without modifying the code.

## 1) `data_preparing`

This module contains scripts to retrieve trajectory data: get positions (trajectory), compute physical attributes, and apply noise reduction and kalman filters.

### Files

- `clean_positions.py` — Script to clean and filter `.npy` path files.

  **Usage:**
  ```bash
  python3 data_preparing/clean_positions.py --input <trajectory_file_to_clean> --output <where_to_save_cleaned_trajectory>
  ```
  Example:
  ```bash
  python3 data_preparing/clean_positions.py --input trajectory.npy --output clean_trajectory.npy

  ```


- `physical_atribute.py` — Computes kinematic properties (speed, acceleration, and velocity vectors) from 2D trajectories.  
Used for analyzing motion data or supporting higher-level filters (e.g., Kalman filter and outlier_filter).

- `predict_path.py` — This script reconstructs a path (trajectory) from a distance matrix using least-squares optimization.  
  It reads measured distances between a moving target and fixed anchors, estimates the target’s position step by step, and saves the reconstructed path in `.npy` format.

  **Algorithm steps:**
  - Handles missing or zero distance values.
  - Computes weighted initial position guesses to improve convergence.
  - Uses Levenberg–Marquardt (least-squares) optimization to estimate positions.
  - Applies a Kalman-like filter to smooth the trajectory and handle cases where the optimisation fail.
  - Saves the final path (`x`, `y`, `timestamp`) in a NumPy array.

  **Usage:**
  ```bash
  python3 -m data_preparing.predict_path --input <path_to_input_csv> --output <path_to_output_npy>

  ```
  ```bash
  python3 -m data_preparing.predict_path --input degradation_Filtre/degraded_dist_matrix.csv --output trajectory.npy
  
  ```

---



## 2) `generating_bloc`

Handles data convergence. And stores anchors topologies.

### Files

- `anchor.json` — Anchor positions used to scale paths.
- `convert_data.py` Converte data from Geo-localisation coordinates to local coordinates (handles the anchors and there palcement)


---

## 3) `analyse.py`

Compare cleaned paths to reference paths and compute metrics like RMSE, max pointwise distance, and trajectory analysis.

**Usage:**
```bash
python3 analyse.py <cleaned_file.npy> <reference_file.npy> --path_line 0
```

---

## Metrics & Evaluation

- RMSE (Root Mean Square Error)  
- MAE (Mean Absolute Error)  
- Max pointwise Euclidean distance  
- velocity and acceleration

---

## Quick Start
the input information is in generator_config.ini

**Get live positions and trajectories**
```bash
bash Dynamic_loc.sh
```

**live Visualisation**
```bash
python3 visualisation.py
```

**Analysis**
```bash
python3 analyse.py
```

---
## License

Wizzilab
