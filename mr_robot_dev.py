import pandas as pd
import numpy as np
import json
from time import sleep
import argparse
from data_preparing.predict_path import PositionEstimator
from data_preparing.clean_positions import noise_cleaning
import traceback

# ---------------------------
# Argument parsing
# ---------------------------
"""
Parse command-line arguments defining:
- the movement scenario to test
- the anchor positions file
"""
parser = argparse.ArgumentParser()
parser.add_argument(
            "--movement"
            , type=str, 
            help="The movement you want to test",
            default="test"
            )

parser.add_argument(
            "--anchor_npy",
            type=str, 
            help =" The anchors' positions in a np.ndarray ",
            default="generating_bloc/anchor_positions.npy"
            )

args = parser.parse_args()
movement = args.movement
anchors_array= np.load(args.anchor_npy)

# ---------------------------
# Tags
# ---------------------------
"""
Hard-coded list of tag identifiers.
Each tag represents a tracked device.
"""
tags = [
    "001BC50C70027E04",
    "001BC50C70027E08",
    "001BC50C70027E47",
    "001BC50C70027E5F",
    "001BC50C7002A6B1"
]

# ---------------------------
# Initialize data structures
# ---------------------------
"""
Dictionaries indexed by tag UID to store:
- raw distance measurements
- estimated trajectories
- GPS positions
- uncertainty radii
- initialization state
"""

data = {tag: [] for tag in tags}
trajectories = {tag: np.empty((1,1,3)) for tag in tags}
positions = {tag: [] for tag in tags}
incertitude = {tag: [] for tag in tags}
initialisation_traj={tag: True for tag in tags}
# ---------------------------
# Functions
# ---------------------------

def get_json(line):
    """
    Extract JSON payload from a log line.

    Args:
        line (str): Raw log line

    Returns:
        str: JSON substring
    """   
    return line.split(' ')[1]


def transform(df: pd.DataFrame, freq: float):
    """
    Normalize timestamps based on sampling frequency.

    Args:
        df (pd.DataFrame): DataFrame containing measurements
        freq (float): Measurement frequency in Hz
    """

    df['time stamp'] = [i / freq for i in range(len(df))]
    print("Time conversion and normalization are done!")


def save_positions_cartesian(positions, freq, anchors_path='generating_bloc/woltDynDev.json',
                             out_cm='positions_cartesian_cm.npy'):
    """
    Convert geographic coordinates (lat, lon) into local Cartesian coordinates.

    The conversion uses a reference defined in the anchors configuration file.
    Output is saved in centimeters.

    Args:
        positions (list): List of [latitude, longitude]
        freq (float): Measurement frequency in Hz
        anchors_path (str): Path to anchor configuration JSON
        out_cm (str): Output file path (.npy)
    """
    try:
        with open(anchors_path, 'r') as f:
            anchors_cfg = json.load(f)
    except Exception:
        anchors_cfg = {}

    t = anchors_cfg.get("transform", {})
    ref_lat = float(t.get("latitude", positions[0][0]))
    ref_lon = float(t.get("longitude", positions[0][1]))
    lon_factor = float(t.get("longitude_factor", 1.0))

    R = 6371000.0  # Earth radius (m)
    ref_lat_rad = np.radians(ref_lat)

    lats = np.radians(np.array([p[0] for p in positions], dtype=float))
    lons = np.radians(np.array([p[1] for p in positions], dtype=float))

    dY = R * (lats - np.radians(ref_lat))
    dX = R * np.cos(ref_lat_rad) * (lons - np.radians(ref_lon)) * lon_factor
    time = np.array([i / freq for i in range(len(positions))])
    arr_m = np.column_stack((dX, dY, time / 100))
    np.save(out_cm,np.array( [arr_m * 100.0]))  # cm
    print(f"Saved {arr_m.shape[0]} positions -> {out_cm} (cm)")


def process_metadata(list_meta_data, tags, UID, data, positions,incertitude):
    """
    Process metadata logs and populate distance measurements and positions.

    Args:
        list_meta_data (list): Raw metadata JSON strings
        tags (list): Known tag identifiers
        UID (dict): Anchor UID mapping
        data (dict): Distance measurements per tag
        positions (dict): GPS positions per tag
        incertitude (dict): Measurement uncertainty per tag
    """
    
    for line in list_meta_data:
        line = json.loads(line)
        tag = line["meta"]["uid"]
        
        
        incertitude[tag].append(line["accuracy_xy"]*100)
        if tag not in tags:
            continue  # skip unknown tags

        row = {"time stamp": line["timestamp"]}
        for anchor_uid in UID.keys():
            distance = line["poi"]["anchors"].get(anchor_uid, {}).get("dst", 0)
    
            
            if distance!=0:
                row[anchor_uid] = distance * 100
            else:
                row[anchor_uid] = 0

        # Append position
        positions[tag].append([float(line["latitude"]), float(line["longitude"])])
        save_positions_cartesian(positions[tag][(-70):], 2,out_cm=f"test_site/trajectories/{tag}positions_cartesian_cm{movement}.npy")
        # Append row
        data[tag].append(row)
        print (f"Carried data for {tag} is :{len(data[tag])}")
        

def intersection_cercles(x0, y0, r0, x1, y1, r1):
    """
    Compute intersection points of two circles.

    Returns None if there is no valid intersection.

    Args:
        (x0, y0), r0: Center and radius of first circle
        (x1, y1), r1: Center and radius of second circle

    Returns:
        list | None: Two intersection points or None
    """    
    
    d = np.hypot(x1 - x0, y1 - y0)

    # Aucun ou infini de points
    if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
        return None

    # Calcul intermédiaire
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(r0**2 - a**2)

    # Point P2 sur la ligne entre les centres
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d

    # Points d’intersection
    rx = -(y1 - y0) * (h / d)
    ry =  (x1 - x0) * (h / d)

    p1 = (x2 + rx, y2 + ry)
    p2 = (x2 - rx, y2 - ry)
    
    return [p1, p2]


def kalman_filter(positions, time_window = 4, coeff=0.3, freq=2):

    """
    Apply a simplified Kalman-like smoothing filter to a trajectory.

    Args:
        positions (np.ndarray): Trajectory array (n, d)
        time_window (int): Number of past steps for velocity estimation
        coeff (float): Blending coefficient (0..1)
        freq (float): Measurement frequency in Hz

    Returns:
        np.ndarray: Smoothed trajectory
    """    
    if coeff < 0 or coeff > 1:
        coeff = 0.3
        print("Coefficient must be between 0 and 1")
        
    n, d = positions.shape
    d=d-1
    dt = 1.0 / freq

    # Copy positions to initialize filtered trajectory
    filtered = np.copy(positions)
    old_velocity = np.zeros(d)
    for i in range(2, n):
        # Determine the start of the window
        start_idx = max(0, i - time_window)
        pos_start = filtered[start_idx,:2]
        pos_n_1 = filtered[i - 1,:2]

        # Compute velocity from start of window to previous point
        old_dt = (i - 1 - start_idx) * dt
        if old_dt == 0:
            old_velocity = np.zeros(d)
        else:
            old_velocity = (pos_n_1 - pos_start) / old_dt
        # Extrapolate to current step
        new_dt = dt
        estimated_pos = pos_n_1 + old_velocity * new_dt

        # Blend with the current measurement
        filtered[i,:2] = coeff * estimated_pos + (1 - coeff) * positions[i,:2]


    return np.array(filtered)


def post_filtre(trajectory:np.ndarray ,raw_trajectory,incertitude_radius: list):
    """Return a filtered trajectory where each element is a consistent [x, y, time] triple.

    - `trajectory` is expected to be an (n, 2) or (n, 3) array with at least x,y.
    - `raw_trajectory` is expected to be an (n, 3) array containing original x,y,time.
    - `incertitude_polygon` is a list/array of polygons aligned with `trajectory`.

    This function ensures the returned array has homogeneous rows (same length) so numpy
    can build a regular ndarray instead of a ragged one.
    """
    filtred_trajecory = []
    tlr=5 #5cm 

    for i in range(len(trajectory)):
        position = trajectory[i]
        radius = incertitude_radius[i] 
        is_inside = False
        
        try:
            pt = position[:2]
            centre = raw_trajectory[i,:2]
            vect = [pt[0]-centre[0],pt[1]-centre[1]]
            dist = np.sqrt(vect[0]**2+ vect[1]**2)
            is_inside = (dist<=radius+tlr)
        except Exception:
            # Any error-> treat as outside
            is_inside = False

        if is_inside:
            # ensure consistent [x,y,time]
            filtred_trajecory.append([float(position[0]), float(position[1]), i/2])
        else:
            # fallback to the original raw point (force floats)
            scale = radius / dist
            new_position = [centre[0]+scale*vect[0],centre[1]+scale*vect[1]]
            filtred_trajecory.append([new_position[0], new_position[1],  i/2 ])

    # keep the previous outer-wrap behavior (1, n, 3) for compatibility
    return np.array([filtred_trajecory])
        

def prediction(df, trajectory, initialisation_traj,incertitude_tag):
    """
    Main prediction pipeline:
    - Estimate positions from distances
    - Handle degenerate anchor cases
    - Apply smoothing and post-filtering

    Args:
        df (pd.DataFrame): Distance matrix
        trajectory (np.ndarray): Previous trajectory
        initialisation_traj (bool): Initialization flag
        incertitude_tag (list): Uncertainty per step

    Returns:
        tuple: (filtered_path, raw_path)
    """
    estimator = PositionEstimator()

    distance = df
    distance = distance.fillna(0)

    anchors_path = "generating_bloc/woltDynDev.json"  # Hardcoded for now

    if not initialisation_traj:
        step = len(trajectory) 
        new_path = list(trajectory[0])
    else:
        step = 0
        new_path = []

    while step < len(distance):

        try:
            estimated_position = estimator.estimate_position(
                distance, anchors_path, step
            )

            new_path.append([
                estimated_position[0],
                estimated_position[1],
                step * (1 / estimator.freq)
            ])

            if estimator.STATUS[0] == "1_DATA":
                kalman_pos = kalman_filter(
                    np.array(new_path),
                    coeff=0.7,
                    time_window=5
                )[step, :2]

            if estimator.STATUS[0] == "2_DATA":
                anchors = estimator.STATUS[1]

                dots = intersection_cercles(
                    anchors[0][0][0], anchors[0][0][1], anchors[0][1],
                    anchors[1][0][0], anchors[1][0][1], anchors[1][1]
                )

                kalman_pos = kalman_filter(
                    np.array(new_path),
                    coeff=1
                )[step, :2]

                # Safe handling when circles do not intersect (intersection_cercles returns None)
                if dots is None:
                    # fallback to Kalman prediction
                    new_path[step][0] = kalman_pos[0]
                    new_path[step][1] = kalman_pos[1]

                    print("Warning: circles do not intersect — using Kalman prediction")
                else:
                    p1, p2 = dots

                    # choose the intersection point closest to Kalman prediction
                    if np.linalg.norm(kalman_pos - np.array(p1)) <= np.linalg.norm(
                        kalman_pos - np.array(p2)
                    ):
                        chosen = p1
                    else:
                        chosen = p2

                    new_path[step][0] = chosen[0]
                    new_path[step][1] = chosen[1]

                    print(
                        "Warning : Only two anchors are detected. "
                        "Taking the closest position to Kalman prediction "
                    )

            if estimator.STATUS[0] == "0_DATA":
                kalman_pos = kalman_filter(
                    np.array(new_path),
                    coeff=0.9,
                    time_window=7
                )[step, :2]

                new_path[step][0] = kalman_pos[0]
                new_path[step][1] = kalman_pos[1]

                print(
                    "Warning : No anchor is detected. "
                    "Taking position as the Kalman prediction "
                )

            if estimator.STATUS[0] == "OPTIMIZATION_FAIL":
                new_position = estimator.estimate_position(
                    distance, anchors_path, step
                )
                new_path[step] = [
                    new_position[0],
                    new_position[1],
                    step * (1 / estimator.freq)
                ]
                pass

            step += 1

        except ValueError as e:
            print("ValueError :", e)
            df.to_csv(f"error_distance_{step}.csv", index=False)
            print("step =", step)
            break

    new_path = np.array(new_path)
    raw_path = new_path.copy()

    new_path = noise_cleaning(new_path, window_size=3)

    try:
        inc_coeff = 0.3 
    except Exception:
        inc_coeff = 0.3 
   
    filtered_path = kalman_filter(
        new_path[0],
        coeff=inc_coeff
    )

    filtered_path =post_filtre(new_path[0],raw_path,incertitude_tag)
    raw_path = np.array([raw_path])
   
    return filtered_path, raw_path


def run_predictor(tag,rows,trajectories,initialisation_traj=initialisation_traj,movement=movement):
    """
    Run the position prediction pipeline for a single tag.

    This function:
    - Converts raw distance rows into a DataFrame
    - Normalizes timestamps
    - Estimates and filters the trajectory
    - Updates internal state for the given tag
    - Saves both trajectory and distance data to disk

    Args:
        tag (str): Unique identifier of the tracked tag.
        rows (list): List of distance measurement dictionaries.
        trajectories (dict): Dictionary storing trajectories per tag.
        initialisation_traj (dict, optional): Tracks whether a tag is being
            initialized for the first time.
        movement (str, optional): Movement scenario name used for output files.
    """   
    print("Running run_predictor")
    
    df = pd.DataFrame(rows)
    transform(df,freq=1)
    filtered_trajectory,trajectories[tag]= prediction(df,trajectories[tag],initialisation_traj[tag],incertitude[tag])
    initialisation_traj[tag]= False
    
    np.save(f"test_site/trajectories/{movement}_{tag}.npy", filtered_trajectory[:,(-70):])
    print(f"saved positions -> test_site/trajectories/{movement}_{tag}.npy (cm)")
    df.to_csv(f"test_site/{movement}_{tag}_distance.csv",index=False)
 
  
    
# ---------------------------
# Load anchor UID mapping
# ---------------------------
"""
Build a dictionary mapping anchor UIDs to their altitude.

The altitude value is currently not used directly in the positioning algorithm
but is preserved for future 3D extensions or compatibility.
"""
UID = {}

with open('generating_bloc/woltDynDev.json', 'r') as f:
    anchors = json.load(f)

    for device in anchors.get("devices", []):
        UID[device["uid"]] = device.get("altitude", 0)

# ---------------------------
# Real-time log processing loop
# ---------------------------
"""
Continuously monitor a log file for new positioning messages.

The loop:
- Resumes reading from the last file position
- Extracts JSON payloads from new log entries
- Updates distance and position buffers
- Triggers trajectory estimation for each active tag
"""
file_position = 0
last_hist_len = 0

while True:
    try:
        with open('logs/test.log', 'r') as file:

            # Resume reading from the last known file position
            file.seek(file_position)

            new_lines = []

            # Read newly appended log entries
            for line in file:
                if line.startswith('/applink/'):
                    new_lines.append(get_json(line))

            # Store the current file pointer for the next iteration
            file_position = file.tell()

            # If no new data is available, wait before retrying
            if len(new_lines) == 0:
                sleep(2)
                continue

            # Parse and store metadata from new log lines
            process_metadata(
                new_lines,
                tags,
                UID,
                data,
                positions,
                incertitude
            )

            # Run prediction for each tag with available data
            for tag, rows in data.items():
                if rows:
                    run_predictor(
                        tag,
                        rows,
                        trajectories
                    )

                    # Reset data buffer after processing
                    data = {tag: [] for tag in tags}

    except Exception as e:
        # Catch-all safety net to prevent loop termination
        print(f"Error: {e}")
        traceback.print_exc()

        # Pause briefly before retrying to avoid rapid failure loops
        sleep(2)


