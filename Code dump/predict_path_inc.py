import numpy as np
import pandas as pd
import argparse
import json
import configparser
from scipy.optimize import least_squares
from generating_bloc.convert_data import get_anchors_position_and_id
from data_preparing.clean_positions import noise_cleaning
import math
import time
from scipy.spatial import ConvexHull





class PositionEstimator:
    
    
    def __init__(self, config_path: str = "generator_config.ini"):
        
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.freq = self.config.getfloat("GENERATE", "freq")
        self.intercetion_history=[]
        self.raw_trajectory=[]
        self.polygons_reduit=[]
        
        # Parsed args (default None until parse_args is called)
        self.input = self.config.get("CANAL_MODEL","output")
        self.output = None


    def parse_arguments(self):
        
        """Parse CLI arguments and update object attributes."""
        parser = argparse.ArgumentParser(
            description="Estimate path from distance matrix with handling missing data")
        
        parser.add_argument("--input", 
                            help="Path to the noisy position (csv file)"
                            , default= self.input)
        
        parser.add_argument("--output",
                            help="Path for the output file (.npy)")
        
        parser.add_argument(
            "--freq", type=float
            , help="Measurement frequency in Hz", default=self.freq)
        
        args = parser.parse_args()
        self.input = args.input
        self.output = args.output
        self.freq = args.freq
        
    
    # ----------------------
    # Utility static methods
    # ----------------------
    @staticmethod
    def dist(p1: np.ndarray, p2: np.ndarray) -> float:
        
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def error(x, dist):
        
        return x - dist

    @staticmethod
    def load_matrix(matrix_path: str) -> pd.DataFrame:
        
        return pd.read_csv(matrix_path, sep=",")

    @staticmethod
    def load_anchors(anchors_path: str) -> dict:
        
        with open(anchors_path, "r") as f:
            data = json.load(f)
            
        ref_data = data["transform"]
        anchor_data = data["devices"]
        anchors_positions = get_anchors_position_and_id(ref_data, anchor_data)
        return anchors_positions

    @staticmethod
    def measur_at_step(dist_matrix: pd.DataFrame, step: int) -> pd.Series:
        
        dist_matrix = dist_matrix.drop(columns=["time stamp"])
        return dist_matrix.iloc[step]
    
    
    
    @staticmethod
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
    
    
    def intersection_polygon(self,circles, num_points_per_circle=100):
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
            return all((x - cx)**2 + (y - cy)**2 <= r**2 + 5 for cx, cy, r in circles)

        mask = np.array([inside_all(pt) for pt in candidate_points])
        intersection_pts = candidate_points[mask]

        if intersection_pts.shape[0] == 0:
            # No intersection points — return empty polygon early to avoid mean on empty slice
            self.polygon = np.array([])
            return np.array([])

        # Sort points counterclockwise around centroid
        centroid = np.mean(intersection_pts, axis=0)
        angles = np.arctan2(intersection_pts[:,1] - centroid[1], intersection_pts[:,0] - centroid[0])
        sorted_indices = np.argsort(angles)
        polygon = intersection_pts[sorted_indices]
        self.polygon = polygon
        
        return polygon


    def polygon_bounding_box(self,result):
        """
        polygon: np.array of points [[x1,y1],[x2,y2],...]
        Returns: min_x, max_x, min_y, max_y
        """
        if self.polygon.size == 0:
            print( "Attention : could not find a polygon !")
            return result.x[0]-10,result.x[0]+10,result.x[1]-10,result.x[1]+10 , True # No intersection

        min_x = np.min(self.polygon[:,0])
        max_x = np.max(self.polygon[:,0])
        min_y = np.min(self.polygon[:,1])
        max_y = np.max(self.polygon[:,1])

        return min_x, max_x, min_y, max_y, False

     
    def Binary_search(self,thr_log_min,thr_log_max,target,mask,cell_area,z,real_volum,iter_max=40): 
        thr_log = thr_log_max
        iter=0
            
        while iter <= iter_max :
            mask_inf = (z[mask] <= thr_log)  
            V_i = np.abs(z[mask][mask_inf]).sum() * cell_area
            fraction = V_i/real_volum
                
            if np.abs(fraction - target)<= 0.05:
                break
                    
            elif fraction >= target :
                thr_log_max = thr_log
            else :
                thr_log_min = thr_log
            thr_log =(thr_log_max+thr_log_min)/2
            iter+=1
        #The metrics 
        
        
        area=mask_inf.sum()*cell_area           

        return thr_log, V_i, fraction, area, mask_inf

    
    def create_grid(self,circles,result, grid_size=100, grid_offset=10):
        """
        Prepare the plan_X, plan_Y meshgrid and mask for the circles.
        """
        self.polygon= self.intersection_polygon(circles, num_points_per_circle=200)
        min_x, max_x, min_y, max_y, _ = self.polygon_bounding_box(result)

        plan_X, plan_Y = np.meshgrid(
            np.linspace(min_x - grid_offset, max_x + grid_offset, grid_size),
            np.linspace(min_y - grid_offset, max_y + grid_offset, grid_size)
        )

        mask = np.ones_like(plan_X, dtype=bool)
        
        for (x, y, r) in circles:
            mask &= (np.hypot(plan_X - x, plan_Y - y) <= r)
            
        if mask.sum() == 0:
            mask = np.ones_like(plan_X, dtype=bool)
            print("Couldn't find an intersection area ! try a bigger grid_size")
            
        dx = np.mean(np.abs(np.diff(plan_X[0, :])))
        dy = np.mean(np.abs(np.diff(plan_Y[:, 0])))
        cell_area = dx * dy

        return plan_X, plan_Y, mask, cell_area


    def compute_z(self,plan_X, plan_Y, mask, distance, anchors):
        """
        Compute z values for each point in the grid, respecting the mask.
        """
        points = np.column_stack((plan_X.ravel(), plan_Y.ravel()))

        z = np.full(plan_X.size, np.nan, dtype=float)
        start = time.time()
        
        for i, p in enumerate(points):
            r = self.residuals(p, distance, anchors)
            if r is not None and len(r) > 0:
                z[i] = np.log(np.linalg.norm(r))
                
        end = time.time()
        print("Elapsed:", end - start, "seconds")
        
        try :
            z = z.reshape(plan_X.shape)
            z[~mask] = np.nanmax(z[mask]) # optional: zero outside mask
            
        except : 
            if mask.sum()==0 :
                print("the cicrles do not intersect , Redo the fitting ! ")

        return z

    
    def compute_target_uncertainty(self,z, mask, cell_area, distance, anchors, target):
        """
        Compute real volume and Binary_search threshold/fraction at a target.
        """
        real_volum = np.nansum(np.abs(z[mask])) * cell_area
        if real_volum == 0 or np.isnan(real_volum):
            raise ValueError("real_volum is zero or NaN. Check grid or mask!")



        thr_log_max, thr_log_min = (np.nanmax(z[mask]), np.nanmin(z[mask])) if z[mask].size else (0, -20)

        thr_log, V_i, fraction, area, mask_inf = self.Binary_search(
            thr_log_min, thr_log_max, target, mask, cell_area, z, real_volum)

        return thr_log, V_i, fraction, area, mask_inf, real_volum


    def incertitude(self,distance,anchors,result,target):
        circles = self.get_circle_params(distance,anchors)

        plan_X, plan_Y, mask, cell_area=self.create_grid(circles,result, grid_size=100, grid_offset=10)
        
 
        z= self.compute_z(plan_X, plan_Y, mask, distance, anchors)
        
        thr_log,V_i,fraction, area,mask_inf,real_volum =self.compute_target_uncertainty(z, mask, cell_area, distance, anchors, target)
        contour_mask = z < thr_log
        points_polygon = np.column_stack((plan_X[contour_mask], plan_Y[contour_mask]))

        if points_polygon.shape[0] > 2:
            hull = ConvexHull(points_polygon)
            points_polygon_sorted = points_polygon[hull.vertices]
        else:
            points_polygon_sorted = points_polygon

        self.polygons_reduit.append(points_polygon_sorted)
            
        

    def initial_guess(self, distance: pd.DataFrame, step: int, anchors_path: str) -> np.ndarray:
        """
        Weighted average initial guess to reduce outliers & optimize convergence
        """
        distance_step = self.measur_at_step(distance, step)
        anchors_position = self.load_anchors(anchors_path)

        weights = [
            1 / (distance_step[e] ** 1) for e in distance_step.index if distance_step[e] != 0
        ]
  
        X0 = sum(
            [(1 / (distance_step[e] ** 1)) * np.array(anchors_position[e])
             for e in distance_step.index if distance_step[e] != 0]
        )
        if sum(weights) == 0:
            X0 = np.mean([np.array(anchors_position[e]) for e in distance_step.index], axis=0)
        else:
            X0 = X0 / sum(weights)
        return X0[:2]



    def incertitude_residuls(self,x,window=5):
        n =len(self.intercetion_history)-1
        window = min(window , n)
        if n == 0:
            return []
        
        outside_scale = 4
        MEW = 2 
        incertitude_res=[]
        
        for i in range(n-window,n):
            prev = np.array(self.raw_trajectory[i])
            fictif_anchor = (prev- np.array(self.raw_trajectory[n])/n-i)*(n-i+1)
            d_mes=np.sqrt(self.intercetion_history[i])
            if d_mes == 0:  # missing data → skip
                continue
            if self.intercetion_history[i]!=0:
                normalisation = np.sqrt( self.intercetion_history[i])
                d = self.dist(x,fictif_anchor)
                e =  np.abs( d - d_mes) # signed error
                if d > d_mes+normalisation:
                    cost = e /(d_mes*normalisation) +  (((e/normalisation) * MEW)**2)
                elif d > normalisation:
                    cost = e /(d_mes*normalisation) +  (((e/normalisation) * MEW)**2)
                else:
                    cost = e /(d_mes*normalisation)
                incertitude_res.append(cost)
        
        
        return incertitude_res
                    
        
    def residuals(self, x: np.ndarray, distance: pd.Series, anchors_position: dict) -> list:
        outside_scale = 4
        MEW = 2
        residuals = []
        self.STATUS = ["standard"]
        anchor_tarce = []
        for i in range(len(distance.index)):
            d_mes = distance[distance.index[i]]  # measured distance
            if d_mes == 0:  # missing data → skip
                continue

            anchor_pos = np.array(anchors_position[distance.index[i]])[:2]  # only (x,y)
            d = self.dist(x, anchor_pos)  # estimated distance
            e =   d - d_mes # signed error
            anchor_tarce.append([anchor_pos,d_mes])
            if d > d_mes:
                cost = e / d_mes + (e * MEW) ** outside_scale
            else:
                cost = e / d_mes

            residuals.append(cost)

        # Status checks
        if len(residuals) < 1:
            self.STATUS = ["0_DATA", anchor_tarce]
        elif len(residuals) == 1:
            self.STATUS = ["1_DATA",anchor_tarce]
        elif len(residuals) == 2:
            self.STATUS = ["2_DATA", anchor_tarce] 

        return residuals #+ self.incertitude_residuls(x)

    
    def estimate_position(self,distance: pd.DataFrame, anchors_path: str, step: int,First_iter=True)-> np.ndarray:
        anchors_position = self.load_anchors(anchors_path)
        dist_step = self.measur_at_step(distance, step)
        x0 = self.initial_guess(distance, step, anchors_path)
        bounds = (-np.inf * np.ones_like(x0), np.inf * np.ones_like(x0))

        result = least_squares(self.residuals,x0=x0,loss='huber',  bounds=bounds,tr_solver='exact',args=(dist_step,anchors_position))
        if not result.success and First_iter :
            #print(f"Optimization failed at step {step}: {result.message}")
            self.STATUS=['OPTIMIZATION_FAIL']
            print('OPTIMIZATION_FAIL',step)


        estimated_positions= result.x
        self.raw_trajectory.append(estimated_positions)
        self.incertitude(dist_step,anchors_position,result,0.8)
        
        return np.array(estimated_positions)


def intersection_cercles(x0, y0, r0, x1, y1, r1):
    d = math.hypot(x1 - x0, y1 - y0)

    # Aucun ou infini de points
    if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
        return None

    # Calcul intermédiaire
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = math.sqrt(r0**2 - a**2)

    # Point P2 sur la ligne entre les centres
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d

    # Points d’intersection
    rx = -(y1 - y0) * (h / d)
    ry =  (x1 - x0) * (h / d)

    p1 = (x2 + rx, y2 + ry)
    p2 = (x2 - rx, y2 - ry)
    
    return [p1, p2]


def kalman_filter(positions, time_window = 3, coeff=0.3, freq=1):
    """
    Apply a simplified Kalman-like filter on the entire trajectory.

    Args:
        positions (np.ndarray): array of positions, shape (n, d)
        time_window (int): number of steps to look back for velocity estimation
        coeff (float): blending coefficient (0..1)
        freq (float): measurement frequency in Hz

    Returns:
        np.ndarray: filtered trajectory, same shape as positions
    """
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
            v = (pos_n_1 - pos_start) / old_dt
            if np.linalg.norm(v) >300:
                old_velocity = v #cap velocity 

        # Extrapolate to current step
        new_dt = dt
        estimated_pos = pos_n_1 + old_velocity * new_dt

        # Blend with the current measurement
        filtered[i,:2] = coeff * estimated_pos + (1 - coeff) * positions[i,:2]


    return np.array(filtered)


def main():
    estimator=PositionEstimator()
    estimator.parse_arguments()
    
    distance = estimator.load_matrix(estimator.input)
    anchors_path = "generating_bloc/woltDynDev.json" #Hardcoded for now
    step = 0
    new_path = []
    while step < len(distance):

        try:
    
            estimated_position = estimator.estimate_position(distance, anchors_path, step)
   
            new_path.append([estimated_position[0],estimated_position[1],step*(1/estimator.freq)])
            if estimator.STATUS[0]=="1_DATA" :
                kalman_pos=kalman_filter(np.array(new_path) ,coeff=0.6,time_window=5)[step,:2]
            
            if estimator.STATUS[0] == "2_DATA" :
                anchors=estimator.STATUS[1]

                
                dots = intersection_cercles(anchors[0][0][0],anchors[0][0][1],anchors[0][1],
                                            anchors[1][0][0],anchors[1][0][1],anchors[1][1])
                print(step)
                kalman_pos=kalman_filter(np.array(new_path) ,coeff=1)[step,:2]
                    # Safe handling when circles do not intersect (intersection_cercles returns None)
                if dots is None:
                    # fallback to Kalman prediction
                    new_path[step][0] = kalman_pos[0]
                    new_path[step][1] = kalman_pos[1]
                    
                    print("Warning: circles do not intersect — using Kalman prediction")
                else:
                    p1, p2 = dots
                    # choose the intersection point closest to Kalman prediction
                    if np.linalg.norm(kalman_pos - np.array(p1)) <= np.linalg.norm(kalman_pos - np.array(p2)):
                        chosen = p1
                    else:
                        chosen = p2
                    new_path[step][0] = chosen[0]
                    new_path[step][1] = chosen[1]

                    print("Warning : Only two anchors are detected. Taking the closest position to Kalman prediction ")
               
            if estimator.STATUS[0]== "0_DATA" :
                kalman_pos=kalman_filter(np.array(new_path) ,coeff=0.7,time_window=7)[step,:2]
                
            
                
                new_path[step][0] = kalman_pos[0]
                new_path[step][1] = kalman_pos[1]
                print("Warning : No anchor is detected. Taking position as the Kalman prediction ")
            if estimator.STATUS[0]== "OPTIMIZATION_FAIL" :
                
                new_position = estimator.estimate_position(distance, anchors_path, step,First_iter=False)
                new_path[step]=[new_position[0],new_position[1],step*(1/estimator.freq)]
                
    
            step+=1
        except ValueError:
            print("Value error",step)
            break
        
    new_path = np.array(new_path) 
    new_path=noise_cleaning(new_path, window_size=3)
    
    filtered_path= kalman_filter(new_path[0])

    filtered_path=np.array([filtered_path])
    np.save(estimator.output, filtered_path)       


"""
        
if __name__ == "__main__":
    main()
    
    
    print("path is saved")
    
"""    