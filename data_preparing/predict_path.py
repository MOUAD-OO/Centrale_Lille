import numpy as np
import pandas as pd
import argparse
import json
import configparser
from scipy.optimize import least_squares
from generating_bloc.convert_data import get_anchors_position_and_id
from data_preparing.clean_positions import noise_cleaning
import matplotlib.pyplot as plt
import math
import time



class PositionEstimator:
    """
    PositionEstimator handles trajectory estimation from noisy distance measurements.

    It loads configuration parameters, parses command-line arguments, 
    provides utility functions for distance calculations and data loading, 
    and estimates positions using a weighted initial guess and least-squares optimization.

    Attributes:
        config (ConfigParser): Configuration loaded from a .ini file.
        freq (float): Measurement frequency in Hz.
        input (str): Path to input CSV containing noisy positions.
        output (str | None): Path to output file (.npy), set after CLI parsing.
        STATUS (list): Optional list capturing estimation status for each step.
    """
    
    def __init__(self, config_path: str = "generator_config.ini"):
        """
        Initialize the estimator by loading parameters from a configuration file.

        Args:
            config_path (str): Path to the configuration .ini file. Defaults to "generator_config.ini".
        """
        # Load configuration
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.freq = self.config.getfloat("GENERATE", "freq")

        # Parsed args (default None until parse_args is called)
        self.input = self.config.get("CANAL_MODEL","output")
        self.output = None

    def parse_arguments(self):
        """
        Parse command-line arguments to update input, output, and measurement frequency.

        CLI arguments take priority over configuration file values.
        """
        parser = argparse.ArgumentParser(
            description="Estimate path from distance matrix with handling missing data"
        )
        parser.add_argument("--input", help="Path to the noisy position (csv file)", default= self.input)
        parser.add_argument("--output", help="Path for the output file (.npy)")
        parser.add_argument(
            "--freq", type=float, help="Measurement frequency in Hz", default=self.freq
        )
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
        """
        Load anchor positions from a JSON file.

        Args:
            anchors_path (str): Path to the JSON file containing anchors.

        Returns:
            dict: Mapping of anchor IDs to their positions.
        """        
        with open(anchors_path, "r") as f:
            data = json.load(f)
        ref_data = data["transform"]
        anchor_data = data["devices"]
        anchors_positions = get_anchors_position_and_id(ref_data, anchor_data)
        return anchors_positions

    @staticmethod
    def measur_at_step(dist_matrix: pd.DataFrame, step: int) -> pd.Series:
        """
        Retrieve the distance measurements at a specific time step.

        Args:
            dist_matrix (pd.DataFrame): Full distance matrix.
            step (int): Index of the step to extract.

        Returns:
            pd.Series: Distances for all anchors at the specified step.
        """
        
        dist_matrix = dist_matrix.drop(columns=["time stamp"])
        return dist_matrix.iloc[step]

    # ----------------------
    # Core estimation logic
    # ----------------------
    def initial_guess(self, distance: pd.DataFrame, step: int, anchors_path: str) -> np.ndarray :
        """
        Compute a weighted initial guess for position based on inverse distances.

        Weighted average reduces the influence of outliers and improves convergence.

        Args:
            distance (pd.DataFrame): Distance matrix.
            step (int): Step index to compute initial guess.
            anchors_path (str): Path to anchor JSON file.

        Returns:
            np.ndarray: Initial guess coordinates (x, y).
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

    def residuals(self, x: np.ndarray, distance: pd.Series, anchors_position: dict) -> list:
        """
        Compute residuals for least-squares optimization.

        The residuals are weighted and include an outlier-penalizing term.
        Handles missing measurements and tracks estimation status.

        Args:
            x (np.ndarray): Estimated position (x, y).
            distance (pd.Series): Measured distances to anchors.
            anchors_position (dict): Dictionary of anchor positions.

        Returns:
            list: Residuals for optimization.
        """        
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
            e =  d - d_mes # signed error
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

        return residuals
    def new_estimate(self,distance: pd.DataFrame,last_dot, anchors_path: str, step: int):
        """
        Compute a new position estimate based on the previous position.

        Args:
            distance (pd.DataFrame): Distance matrix.
            last_dot (np.ndarray): Previous estimated position.
            anchors_path (str): Path to anchors JSON file.
            step (int): Step index for the estimation.

        Returns:
            np.ndarray: Estimated position (x, y).
        """        
        anchors_position = self.load_anchors(anchors_path)
        dist_step = self.measur_at_step(distance, step)
        result = least_squares(self.residuals,x0=last_dot,loss='huber', tr_solver='exact',args=(dist_step,anchors_position))
        estimated_positions= result.x
        if not result.success:
            print(f"Optimization failed at step {step}: {result.message}")

        return np.array(estimated_positions)
        
    def estimate_position(self,distance: pd.DataFrame, anchors_path: str, step: int)-> np.ndarray:
        """
        Estimate the position at a given step using least-squares optimization.

        Starts from a weighted initial guess to improve convergence.

        Args:
            distance (pd.DataFrame): Distance matrix.
            anchors_path (str): Path to anchors JSON file.
            step (int): Step index to estimate.

        Returns:
            np.ndarray: Estimated position (x, y).
        """       
        anchors_position = self.load_anchors(anchors_path)
        dist_step = self.measur_at_step(distance, step)
        x0 = self.initial_guess(distance, step, anchors_path)
        result = least_squares(self.residuals,x0=x0,loss='huber', tr_solver='exact',args=(dist_step,anchors_position))
        if not result.success:
            #print(f"Optimization failed at step {step}: {result.message}")
            self.STATUS=['OPTIMIZATION_FAIL']


        estimated_positions= result.x

        return np.array(estimated_positions)







class KalmanFilter:
    
    def __init__(self,config_path="generator_config.ini",):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        self.freq = self.config.getfloat("GENERATE", "freq")
        self.dt = 1.0 / self.freq
        

    def evolution_matrix(self):
        """This fuction gives the evolution matrix for the kalman filter 

        Args:
            dt ( float ): The time between two steps in seconds
        """
        return np.array([[1 ,0 ,self.dt,0 ],
                        [0 ,1 ,0 ,self.dt],
                        [0 ,0 ,1 ,0 ],
                        [0 ,0 ,0 ,1 ]])
        
    
    
    

    def Velocity_vector(self , point1, point2):
        """This function computes the velocity vector between two points 

        Args:
            point1 (np.ndarray): the first point 
            point2 (np.ndarray): the second point 
            dt (float): the time between two points in seconds

        Returns:
            np.ndarray: the velocity vector 
        """
        return (point2 - point1) / self.dt







    def kalman_filter_trajectory(self,positions, time_window = 5, coeff=0.4):
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
        freq= self.freq
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


        return np.array([filtered])



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
        placed = False

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
                    print(new_path)
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
                
                new_position = estimator.new_estimate(distance,new_path[step][:2], anchors_path, step)
                new_path[step]=[new_position[0],new_position[1],step*(1/estimator.freq)]
                
    
            step+=1
        except ValueError:
            print("Value error",step)
            break
        
    new_path = np.array(new_path) 
    new_path=noise_cleaning(new_path, window_size=3)
    
    filtered_path= kalman_filter(new_path[0])

    filtered_path=np.array([filtered_path])
    #np.save(estimator.output, filtered_path)      
    return  filtered_path
   
  
"""  
 
if __name__ == "__main__":

    main()
    
    print("path is saved")
"""