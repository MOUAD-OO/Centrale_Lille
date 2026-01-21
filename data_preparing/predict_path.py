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







