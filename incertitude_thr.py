import numpy as np 

from numba import njit
import pandas as pd 
from scipy.optimize import least_squares, minimize
import json
from generating_bloc.convert_data import get_anchors_position_and_id
import matplotlib.pyplot as plt 
import argparse
from tqdm import tqdm




#======================================================
#                    Functions 
#======================================================
   
# Normalize anchors into a lookup dict: keys are strings of indices

def load_anchors(anchors_path: str) -> dict:
    with open(anchors_path, "r") as f:
        data = json.load(f)
    ref_data = data["transform"]
    anchor_data = data["devices"]
    anchors_positions = get_anchors_position_and_id(ref_data, anchor_data)
    return anchors_positions


def residuals(x: np.ndarray, distance_series: pd.Series, anchors_lookup: dict) -> np.ndarray:
    """
    Compute residuals for least_squares.
    - distance_series: pd.Series indexed by anchor id (can be strings or ints)
    - anchors_lookup: dict mapping string id -> (x,y,...) array
    Returns vector of residuals.
    """
    outside_scale = 4
    MEW = 2
    eps = 1e-8
    res = []

    for anchor_id in distance_series.index:
        d_mes = distance_series[anchor_id]
        if pd.isna(d_mes) or d_mes == 0:
            continue

        # robust anchor lookup
        key = str(anchor_id)
        anchor_pos = None
        if key in anchors_lookup:
            anchor_pos = anchors_lookup[key]
        else:
            # try converting to int index
            try:
                anchor_pos = anchors_lookup[str(int(anchor_id))]
            except Exception:
            # skip if anchor not found
                continue

        anchor_pos = np.asarray(anchor_pos)
            # use first two dims only
        anchor_xy = anchor_pos[:2].astype(float)

        d = np.linalg.norm(x - anchor_xy)
        e = d - float(d_mes)

        # safe division by measured distance
        base = e / (float(d_mes) + eps)
        if d > d_mes:
            cost = base + (e * MEW) ** outside_scale
        else:
            cost = base

        res.append(cost)

    return np.asarray(res)


def initial_guess(distance: pd.Series, anchors_lookup: dict) -> np.ndarray:
        """
        Weighted average initial guess to reduce outliers & optimize convergence
        """


        weights = [
            1 / (distance[e] ** 1) for e in distance.index if distance[e] != 0 and e!='time stamp']
  
        X0 = sum(
            [(1 / (distance[e] ** 1)) * np.array(anchors_lookup[e][:2])
             for e in distance.index if distance[e] != 0 and e!='time stamp']
        )
        if sum(weights) == 0:
            X0 = np.mean([np.array(anchors_lookup[e][:2]) for e in distance.index if e!='time stamp'], axis=0)
        else:
            X0 = X0 / sum(weights)
        return X0[:2]


def get_circle_params(distance_series: pd.Series, anchors_lookup: dict):
    circle_params = []
    for anchor_id in distance_series.index:
        d_mes = distance_series[anchor_id]
        if pd.isna(d_mes) or d_mes == 0:
            continue
        key = str(anchor_id)
        try:
            anchor_pos = np.asarray(anchors_lookup[key])[:2].astype(float)
        except Exception:
            try:
                anchor_pos = np.asarray(anchors_lookup[str(int(anchor_id))])[:2].astype(float)
            except Exception:
                continue
        circle_params.append([anchor_pos[0], anchor_pos[1], float(d_mes)])
    return circle_params



def intersection_polygon(circles, num_points_per_circle=200):
    """
    circles: list of [x, y, r]
    num_points_per_circle: resolution along circumference
    Returns: np.array of points [[x1,y1],[x2,y2],...] forming the intersection polygon
    """
    if len(circles) < 2:
        return np.array([])  # Not enough circles for intersection

    # Collect candidate points along all circles
    candidate_points = []

    for cx, cy, r in circles:
        angles = np.linspace(0, 2*np.pi, num_points_per_circle, endpoint=False)
        x_pts = cx + r * np.cos(angles)
        y_pts = cy + r * np.sin(angles)
        circle_pts = np.column_stack((x_pts, y_pts))
        candidate_points.append(circle_pts)

    candidate_points = np.vstack(candidate_points)

    # Filter points inside all circles
    def inside_all(pt):
        x, y = pt
        return all((x - cx)**2 + (y - cy)**2 <= r**2 + 1e-9 for cx, cy, r in circles)

    mask = np.array([inside_all(pt) for pt in candidate_points])
    intersection_pts = candidate_points[mask]

    if intersection_pts.shape[0] == 0:
        return np.array([])

    # Sort points counterclockwise around centroid
    centroid = np.mean(intersection_pts, axis=0)
    angles = np.arctan2(intersection_pts[:,1] - centroid[1], intersection_pts[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    polygon = intersection_pts[sorted_indices]

    return polygon


def polygon_bounding_box(polygon,result):
    """
    polygon: np.array of points [[x1,y1],[x2,y2],...]
    Returns: min_x, max_x, min_y, max_y
    """
    if polygon.size == 0:
        return result.x[0]-10,result.x[0]+10,result.x[1]-10,result.x[1]+10 , True # No intersection

    min_x = np.min(polygon[:,0])
    max_x = np.max(polygon[:,0])
    min_y = np.min(polygon[:,1])
    max_y = np.max(polygon[:,1])

    return min_x, max_x, min_y, max_y, False



def Binary_search(thr_log_min,thr_log_max,target,mask,cell_area,z,real_volum,real_z,iter_max=80): 
    thr_log = thr_log_max
    iter=0
        
    while iter <= iter_max :
        mask_inf = (z[mask] <= thr_log)  
        V_i = np.abs(z[mask][mask_inf]).sum() * cell_area
        fraction = V_i/real_volum
            
        if np.abs(fraction - target)<= 0.01:
            break
                
        elif fraction >= target :
            thr_log_max = thr_log
        else :
            thr_log_min = thr_log
        thr_log =(thr_log_max+thr_log_min)/2
        iter+=1
    #The metrics 
    is_target_In=(real_z < thr_log)
    intersection_area = mask.sum()*cell_area
    area=mask_inf.sum()*cell_area           

    return thr_log, V_i, fraction, is_target_In, intersection_area, area, mask_inf


def distance_from_volume(plan_X,plan_Y,mask1,mask_inf,real_position,result):
            
   
    X = plan_X[mask1][mask_inf]
    Y = plan_Y[mask1][mask_inf]
            
    dist = np.sqrt(np.sqrt((X - real_position[0])**2 + (Y - real_position[1])**2))
    try:
        error = np.min(dist)
    except:
        
        print(f"No distsance was found dist = {dist}")
        error = np.linalg.norm( result.x -real_position )
        real_z =np.log( np.linalg.norm(residuals(real_position, distance, anchors)))
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(plan_X,plan_Y,z,cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        ax.scatter(real_position[0], real_position[1], real_z, color='red', s=50)
        plt.show()
    return error



def create_grid(circles,result, grid_size=100, grid_offset=10):
    """
    Prepare the plan_X, plan_Y meshgrid and mask for the circles.
    """
    polygon= intersection_polygon(circles, num_points_per_circle=200)
    min_x, max_x, min_y, max_y, _ = polygon_bounding_box(polygon,result)

    plan_X, plan_Y = np.meshgrid(
        np.linspace(min_x - grid_offset, max_x + grid_offset, grid_size),
        np.linspace(min_y - grid_offset, max_y + grid_offset, grid_size)
    )

    mask = np.ones_like(plan_X, dtype=bool)
    for (x, y, r) in circles:
        mask &= (np.hypot(plan_X - x, plan_Y - y) <= r)

    dx = np.mean(np.abs(np.diff(plan_X[0, :])))
    dy = np.mean(np.abs(np.diff(plan_Y[:, 0])))
    cell_area = dx * dy

    return plan_X, plan_Y, mask, cell_area

def compute_z(plan_X, plan_Y, mask, distance, anchors):
    """
    Compute z values for each point in the grid, respecting the mask.
    """
    points = np.column_stack((plan_X.ravel(), plan_Y.ravel()))
    z = np.full(plan_X.size, np.nan, dtype=float)

    for i, p in enumerate(points):
        r = residuals(p, distance, anchors)
        if r is not None and len(r) > 0:
            z[i] = np.log(np.linalg.norm(r))

    z = z.reshape(plan_X.shape)
    z[~mask] = np.nanmax(z[mask])  # optional: zero outside mask
    return z

def compute_target_uncertainty(z, mask, cell_area, real_position, distance, anchors, target):
    """
    Compute real volume and Binary_search threshold/fraction at a target.
    """
    real_volum = np.nansum(np.abs(z[mask])) * cell_area
    if real_volum == 0 or np.isnan(real_volum):
        raise ValueError("real_volum is zero or NaN. Check grid or mask!")

    r = residuals(real_position, distance, anchors)
    if r is None or len(r) == 0:
        real_z = np.nan
    else:
        real_z = np.log(np.linalg.norm(r))

    thr_log_max, thr_log_min = (np.nanmax(z[mask]), np.nanmin(z[mask])) if z[mask].size else (0, -20)

    thr_log, V_i, fraction, is_target_In, intersection_area, area, mask_inf = Binary_search(
        thr_log_min, thr_log_max, target, mask, cell_area, z, real_volum, real_z
    )

    return thr_log, V_i, fraction, is_target_In, intersection_area, area, mask_inf, real_volum

#======================================================
#                     Inputs
#======================================================
parser = argparse.ArgumentParser(description="Run incertitude with supplied data paths")
parser.add_argument("--anchors",       default="generating_bloc/test_anchor.json", help="anchors json file")
parser.add_argument("--distances",     default="degradation_Filtre/degraded_dist_matrix.csv", help="distances CSV")
parser.add_argument("--anchors_raw",   default="generating_bloc/anchor_positions.npy", help="raw anchors .npy")
parser.add_argument("--real_positions",default="generating_bloc/paths/walk_paths.npy", help="real positions .npy")
parser.add_argument("--noise_level", default= 0)
parser.add_argument("--target", default= 0.8, help= "give the amount of the volume you want to keep (from 0 to 1)")
args = parser.parse_args()

noise_level=args.noise_level


anchors = load_anchors(args.anchors)
distances = pd.read_csv(args.distances, sep=",")
anchors_raw = np.load(args.anchors_raw, allow_pickle=True)
real_positions = np.load(args.real_positions, allow_pickle=True)

success_rate=[]
mean_error=[]


pbar = tqdm(total=1000, desc="Targets progress")

for target in np.arange(0, 1, 0.1):


    intersection_area_list=[]
    incertitude_area_list=[]
    success= []
    error = []
    for pos_index in range(100):
    #======================================================================
                                #prediction
    #======================================================================
        # Load distance (one row -> Series) and anchors
        distance = distances.iloc[pos_index]
        if np.all(distance[1:]==0) :
            
            print("Not enough data : not included in computation !")
            continue
        real_position=real_positions[0][pos_index,:2]
        area = 0           
        # prepare x0 and run solver
        x0 = initial_guess(distance,anchors)
        result = least_squares(residuals,x0,args=(distance, anchors),loss="huber",tr_solver="exact")
    #================================================================================
                                        #Incertitude bloc
    #================================================================================
        circles = get_circle_params(distance,anchors)
        
        plan_X, plan_Y, mask, cell_area=create_grid(circles,result, grid_size=100, grid_offset=10)
        
        z= compute_z(plan_X, plan_Y, mask, distance, anchors)

        thr_log,V_i,fraction,is_target_In,intersection_area, area,mask_inf,real_volum =compute_target_uncertainty(z, mask, cell_area, real_position, distance, anchors, target)




        #print(f"noise ={noise_level}|Intersection area ={intersection_area:.2f} cm2|Incertitude area={area:.2f} cm2|target in {is_target_In}|mask={len(mask)}|mask_inf= {len(mask_inf)}")
        intersection_area_list.append(intersection_area)
        incertitude_area_list.append(area)
        success.append(int(is_target_In))   
        error.append(distance_from_volume(plan_X,plan_Y,mask,mask_inf,real_position,result))
        
        
        #print(f"Confidance={fraction*100:.2f}%| V_i={V_i:.2f} cm3| total volum={real_volum:.2f} cm3|" f"Incertitude area={area:.2f} cm2|Intersection area ={intersection_area:.2f} cm2|target in {is_target_In}")

        
        z_thr = np.full(z.shape , thr_log,dtype=float)
        
        
        # if only two circles, keep previous behaviour (divide by 2)
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(plan_X,plan_Y,z,cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
        #ax.plot_surface(plan_X, plan_Y, z_thr, color='red', alpha=0.5, linewidth=0, antialiased=False)
        plt.show()
        """
        pbar.update(1)
        
        
    success_rate.append(np.mean(success))
    mean_error.append(np.mean(error))
    
x=np.arange(0, 1, 0.1)
test_data_line={'noise': noise_level}


for i in range (len(x)):
    test_data_line[f'success_rate_{i}'] = success_rate[i]
for i in range (len(x)):
    test_data_line[f'Mean-error_at{i}'] = mean_error[i]

test_data=pd.read_csv("test/volume_thr1.csv", sep = ',')
test_data = pd.concat([test_data, pd.DataFrame([test_data_line])], ignore_index = True)
test_data.to_csv("test/volume_thr1.csv",index=False)    
    