#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_preparing.physical_atribute import speed_time_series, acceleration_time_series
import folium
import json
import webbrowser


class PathAnalyzer:
    def __init__(self, real_path_file, est_file_1=None, est_file_2=None, path_index=0):
        self.real_path_file = real_path_file
        self.est_file_1 = est_file_1
        self.est_file_2 = est_file_2
        self.path_index = path_index
        self.real_path = None
        self.est_path_1 = None
        self.est_path_2 = None
        self.load_paths()

    def load_paths(self):
        self.real_path = np.load(self.real_path_file)[self.path_index]
        print("Loaded real_path:", self.real_path.shape)
        if self.est_file_1:
            self.est_path_1 = np.load(self.est_file_1)[self.path_index]
            print("Loaded est_path_1:", self.est_path_1.shape)
        if self.est_file_2:
            self.est_path_2 = np.load(self.est_file_2)[self.path_index]
            print("Loaded est_path_2:", self.est_path_2.shape)

    def compute_distances(self, est_path):
        """Compute distances between corresponding points in the estimated and real paths."""
        min_len = min(len(self.real_path), len(est_path))
        distances = np.sqrt(
            (self.real_path[:min_len, 0] - est_path[:min_len, 0])**2 +
            (self.real_path[:min_len, 1] - est_path[:min_len, 1])**2
        )
        return distances
    def compute_distances(self, est_path,a=-0.22,b=140):
        """
        Compute perpendicular distances from trajectory points to a line y = a*x + b.

        Parameters
        ----------
        traj : np.ndarray, shape (N, 2)
            Trajectory points (x, y)
        a : float
            Line slope
        b : float
            Line intercept

        Returns
        -------
        distances : np.ndarray, shape (N,)
            Perpendicular distances to the line
        """
        min_len = min(len(self.real_path), len(est_path))
        x = est_path[:min_len, 0]
        y = est_path[:min_len, 1]

        distances = np.abs(a * x - y + b) / np.sqrt(a**2 + 1)
        return distances
        

    def compute_rmse(self, distances):
        """Compute cumulative RMSE over time."""
        return [np.sqrt(np.mean(distances[:i+1]**2)) for i in range(len(distances))]

    def compute_mean_error(self, distances):
        """Compute the mean distance error."""
        return np.mean(distances)

    def coordinate_with_max_error(self, est_path):
        """Determine which coordinate (X or Y) contributes the most to the error."""
        min_len = min(len(self.real_path), len(est_path))
        x_error = np.abs(self.real_path[:min_len, 0] - est_path[:min_len, 0])
        y_error = np.abs(self.real_path[:min_len, 1] - est_path[:min_len, 1])
        x_mean_error = np.mean(x_error)
        y_mean_error = np.mean(y_error)
        return "X" if x_mean_error > y_mean_error else "Y", x_mean_error, y_mean_error

    def compute_speed_acceleration(self, path, frequency):
        """Compute speed and acceleration time series for a path."""
        speed = speed_time_series(path, frequency)
        acc = acceleration_time_series(path, frequency)
        return speed, acc
    
    def compute_delay(self):
        """
        Compute the average delay between the real path and each estimated path.
        Delay is based on the difference in movement vectors.
        Returns a dict with delays for est_path_1 and est_path_2.
        """
        def vecteur(p1, p2):
            return np.array(p2[:2]) - np.array(p1[:2])

        delays = {}
        for idx, est_path in enumerate([self.est_path_1, self.est_path_2], start=1):
            if est_path is not None:
                path_delays = []
                min_len = min(len(self.real_path), len(est_path))
                for i in range(1, min_len):
                    real_v = vecteur(self.real_path[i - 1], self.real_path[i])
                    est_v = vecteur(est_path[i - 1], est_path[i])
                    norm_real_v = np.linalg.norm(real_v)
                    if norm_real_v != 0:
                        delay = np.dot(est_v, real_v) / norm_real_v - norm_real_v
                        path_delays.append(delay)
                if path_delays:
                    delays[f"Estimation {idx}"] = np.mean(path_delays)
                else:
                    delays[f"Estimation {idx}"] = None
        return delays

    def plot_speed_acceleration(self, frequency):
        """Plot speed and acceleration for all paths."""
        time_real = self.real_path[:, 2]
        plt.figure(figsize=(12, 6))
        # Speed plot
        plt.subplot(2, 1, 1)
        speed_real, acc_real = self.compute_speed_acceleration(self.real_path, frequency)
        plt.plot(time_real, speed_real, label="Real Speed", color="green")
        if self.est_path_1 is not None:
            speed_1, _ = self.compute_speed_acceleration(self.est_path_1, frequency)
            plt.plot(self.est_path_1[:, 2], speed_1, label="Est. 1 Speed", color="blue")
        if self.est_path_2 is not None:
            speed_2, _ = self.compute_speed_acceleration(self.est_path_2, frequency)
            plt.plot(self.est_path_2[:, 2], speed_2, label="Est. 2 Speed", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (m/s)")
        plt.title("Speed Over Time")
        plt.legend()
        plt.grid(True)
        # Acceleration plot
        plt.subplot(2, 1, 2)
        plt.plot(time_real, acc_real, label="Real Acceleration", color="green")
        if self.est_path_1 is not None:
            _, acc_1 = self.compute_speed_acceleration(self.est_path_1, frequency)
            plt.plot(self.est_path_1[:, 2], acc_1, label="Est. 1 Acceleration", color="blue")
        if self.est_path_2 is not None:
            _, acc_2 = self.compute_speed_acceleration(self.est_path_2, frequency)
            plt.plot(self.est_path_2[:, 2], acc_2, label="Est. 2 Acceleration", color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s²)")
        plt.title("Acceleration Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_metrics(self, distances, rmses, label="Estimation"):
        """Plot RMSE and distances over time."""
        time = self.real_path[:len(distances), 2]
        plt.figure(figsize=(10, 6))
        plt.plot(time, rmses, label=f"{label} RMSE", color="red")
        plt.plot(time, distances, label=f"{label} Distance", alpha=0.5, color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (cm)")
        plt.title(f"Error Over Time ({label})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_static_paths(self):
        """Plot all paths statically."""
        plt.figure(figsize=(10, 8))
        anchor_positions = np.load("generating_bloc/anchor_positions.npy")
        plt.scatter(anchor_positions[:, 0], anchor_positions[:, 1], c='red', label='Anchors')
        plt.plot(self.real_path[:, 0], self.real_path[:, 1], marker='o', markersize=4,
                 linestyle='-', alpha=0.7, color="green", label="Real Path")
        if self.est_path_1 is not None:
            plt.plot(self.est_path_1[:, 0], self.est_path_1[:, 1], marker='s', markersize=4,
                     linestyle='-', alpha=0.5, color="blue", label="Estimation 1")
        if self.est_path_2 is not None:
            plt.plot(self.est_path_2[:, 0], self.est_path_2[:, 1], marker='x', markersize=4,
                     linestyle='-.', alpha=0.5, color="orange", label="Estimation 2")
        for i in range (min(len(self.est_path_1),len(self.est_path_2))):
            X=[self.est_path_1[i,0],self.est_path_2[i,0]]
            Y =[self.est_path_1[i,1],self.est_path_2[i,1]]
            #plt.plot(X,Y,c="red")
        plt.xlabel("X (cm)")
        plt.ylabel("Y (cm)")
        plt.title("Static Path Comparison")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()
        

                
    def plot_X_evolution(self):
        plt.figure(figsize=(10,8))
        plt.plot(self.real_path[:, 0], self.real_path[:, 2], marker='o', markersize=4,
                 linestyle='-', alpha=0.7, color="green", label="Real Path")
        if self.est_path_1 is not None:
            plt.plot(self.est_path_1[:, 0], self.est_path_1[:, 2], marker='s', markersize=4,
                     linestyle='-', alpha=0.5, color="blue", label="Estimation 1")
        if self.est_path_2 is not None:
            plt.plot(self.est_path_2[:, 0], self.est_path_2[:, 2], marker='x', markersize=4,
                     linestyle='-.', alpha=0.5, color="orange", label="Estimation 2")
        plt.xlabel("X (cm)")
        plt.ylabel("T (s)")
        plt.title("X_evolution  Comparison")
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_Y_evolution(self):
        plt.figure(figsize=(10,8))
        plt.plot(self.real_path[:, 1], self.real_path[:, 2], marker='o', markersize=4,
                 linestyle='-', alpha=0.7, color="green", label="Real Path")
        if self.est_path_1 is not None:
            plt.plot(self.est_path_1[:, 1], self.est_path_1[:, 2], marker='s', markersize=4,
                     linestyle='-', alpha=0.5, color="blue", label="Estimation 1")
        if self.est_path_2 is not None:
            plt.plot(self.est_path_2[:, 1], self.est_path_2[:, 2], marker='x', markersize=4,
                     linestyle='-.', alpha=0.5, color="orange", label="Estimation 2")
        plt.xlabel("Y (cm)")
        plt.ylabel("T (s)")
        plt.title("Y_evolution  Comparison")
        plt.grid(True)
        plt.legend()
        plt.show()


    def plot_paths_on_real_map(self, anchors_json_path="generating_bloc/woltDynDev.json"):
        """
        Create a REAL interactive OpenStreetMap and open it automatically.
        """
        # Load transform
        with open(anchors_json_path, "r") as f:
            cfg = json.load(f)

        t = cfg["transform"]
        ref_lat = float(t["latitude"])
        ref_lon = float(t["longitude"])
        lon_factor = float(t.get("longitude_factor", 1.0))

        R = 6371000.0
        ref_lat_rad = np.radians(ref_lat)

        def xy_to_latlon(x, y):
            lat = np.degrees(y / R + ref_lat_rad)
            lon = np.degrees(x / (R * np.cos(ref_lat_rad) * lon_factor) + np.radians(ref_lon))
            return lat, lon

        # Create map centered on transform
        m = folium.Map(
        location=[ref_lat, ref_lon],
        zoom_start=19,
        tiles="CartoDB Positron"
        )

        # ---- ANCHORS ----
        anchor_positions = np.load("generating_bloc/anchor_positions.npy")
        for position in anchor_positions:
            ax=position[0]
            ay= position[1]
            lat, lon = xy_to_latlon(ax/100, ay/100)
            folium.CircleMarker(
                location=[lat, lon],
                radius=1, color="red", fill=True, popup="Anchor"
            ).add_to(m)

        # ---- REAL PATH ----
        real_coords = [xy_to_latlon(x/100, y/100) for x,y,_ in self.real_path]
        folium.PolyLine(real_coords, color="green", weight=0.2, popup="Real Path").add_to(m)

        # ---- ESTIMATED PATHS ----
        if hasattr(self, "est_path_1") and self.est_path_1 is not None:
            est_coords = [xy_to_latlon(x/100, y/100) for x,y,_ in self.est_path_1]
            folium.PolyLine(est_coords, color="blue", weight=0.2, popup="Estimation 1").add_to(m)

        if hasattr(self, "est_path_2") and self.est_path_2 is not None:
            est_coords = [xy_to_latlon(x/100, y/100) for x,y,_ in self.est_path_2]
            folium.PolyLine(est_coords, color="orange", weight=0.2, popup="Estimation 2").add_to(m)

        # ---- SAVE AND AUTO OPEN ----
        out_file = "paths_map.html"
        m.save(out_file)
        webbrowser.open("file://" + os.path.realpath(out_file))
        print("Real OpenStreetMap opened automatically.")
      
      
    
    def plot_moving_paths(self, interval=100):
        """Animate paths moving point by point."""
        anchor_positions = np.load("generating_bloc/anchor_positions.npy")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(anchor_positions[:, 0], anchor_positions[:, 1], c='red', label='Anchors')
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_title("Animated Path Comparison")
        ax.grid(True)
        ax.axis('equal')

        # Initialize lines
        real_line, = ax.plot([], [], marker='o', markersize=4, linestyle='-', color="green", alpha=0.7, label="Real Path")
        est1_line, = ax.plot([], [], marker='s', markersize=4, linestyle='-', color="blue", alpha=0.5, label="Estimation 1") if self.est_path_1 is not None else (None,)
        est2_line, = ax.plot([], [], marker='x', markersize=4, linestyle='-.', color="orange", alpha=0.5, label="Estimation 2") if self.est_path_2 is not None else (None,)

        ax.legend()

        # Determine the max length to animate
        max_len = max(len(self.real_path),
                    len(self.est_path_1) if self.est_path_1 is not None else 0,
                    len(self.est_path_2) if self.est_path_2 is not None else 0)

        def update(frame):
            real_line.set_data(self.real_path[:frame, 0], self.real_path[:frame, 1])
            if self.est_path_1 is not None:
                est1_line.set_data(self.est_path_1[:frame, 0], self.est_path_1[:frame, 1])
            if self.est_path_2 is not None:
                est2_line.set_data(self.est_path_2[:frame, 0], self.est_path_2[:frame, 1])
            return real_line, est1_line, est2_line

        ani = FuncAnimation(fig, update, frames=max_len, interval=interval, blit=True)
        plt.show()    
        
    
      
def main():
    config = configparser.ConfigParser()
    config.read('generator_config.ini')
    frequency = config.getint('GENERATE','freq',fallback=2)
    parser = argparse.ArgumentParser(description="Compare up to two estimated trajectories to a real trajectory")
    parser.add_argument("--est1", type=str, help="Path to the first estimated trajectory file")
    parser.add_argument("--est2", type=str, help="Path to the second estimated trajectory file")
    parser.add_argument("--real", type=str, required=True, help="Path to the real trajectory file")
    parser.add_argument("--path_line", type=int, default=0, help="Index of the path to be plotted")
    args = parser.parse_args()

    analyzer = PathAnalyzer(args.real, args.est1, args.est2, path_index=args.path_line)

    # Pour chaque estimation, calculer et afficher les métriques
    for idx, est_path in enumerate([analyzer.est_path_1, analyzer.est_path_2], start=1):
        if est_path is not None:
            distances = analyzer.compute_distances(est_path)
            rmses = analyzer.compute_rmse(distances)
            mean_error = analyzer.compute_mean_error(distances)
            coord, x_error, y_error = analyzer.coordinate_with_max_error(est_path)
            print(f"[Estimation {idx}] Mean distance error: {mean_error:.2f} cm")
            #print(f"[Estimation {idx}] Coordinate with the most error: {coord} (X error: {x_error:.2f} cm, Y error: {y_error:.2f} cm)")
            print(f"[Estimation {idx}] Final RMSE: {rmses[-1]:.2f} cm")
            print(f"[Estimation {idx}] Max distance: {np.max(distances):.2f} cm")

            analyzer.plot_metrics(distances, rmses, label=f"Estimation {idx}")
        print('--------------------------------------------------------------------')
    delays = analyzer.compute_delay()
    for label, value in delays.items():
        print(f"{label} average delay: {value:.4f} (units of path)")

    analyzer.plot_speed_acceleration(frequency)
    analyzer.plot_static_paths()
    analyzer.plot_moving_paths()
    analyzer.plot_paths_on_real_map()
    
    
    #analyzer.plot_X_evolution()
    #analyzer.plot_Y_evolution()
    
    
if __name__ == "__main__":
    main()