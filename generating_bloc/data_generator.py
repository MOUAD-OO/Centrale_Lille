#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import json
from convert_data import convert_gps_to_cartesian_cm
import configparser

dir_path = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read("generator_config.ini")

# Get defaults from config file, with fallback values
default_anchor = config.get("GENERATE", "Anchor_position", fallback=os.path.join(dir_path, "woltDynDev.json"))
default_num_paths = config.getint("GENERATE", "number_of_paths", fallback=5)
default_mean_speed = config.getfloat("GENERATE", "mean_speed", fallback=1.3)
default_freq = config.getfloat("GENERATE", "freq", fallback=1.0)
default_with_plots = config.getboolean("GENERATE", "with_plots", fallback=False)
default_plot_all = config.getboolean("GENERATE", "plot_all_in_one", fallback=False)
seed=config.getint("GENERATE","seed")

parser = argparse.ArgumentParser(
    description=(
        "This script generates random paths (graphs) and saves them as .npy files.\n"
        "By default, it will generate 5 paths of each type (linear, polynomial, circular, sinusoidal).\n"
        "Default parameters:\n"
        "  --Anchor_position: ./test_anchor.json\n"
        "  --number_of_paths: 5\n"
        "  --with_plots: False\n"
        "  --plot_all_in_one: False\n"
        "  --mean_speed: 1.3 (m/s)\n"
        "  --freq: 1.0 (Hz)\n"
    ),
    usage="python data_generator.py [--Anchor_position JSON FILE] [--number_of_paths N] [--with_plots] [--plot_all_in_one] [--mean_speed SPEED] [--freq FREQ]"
)

parser.add_argument("--Anchor_position", help="Path to the anchor position JSON file", type=str, default=default_anchor)
parser.add_argument("-n","--number_of_paths", help="Number of paths to generate", type=int, default=default_num_paths)
parser.add_argument("--with_plots", help="Plot the generated paths", action='store_true', default=default_with_plots)
parser.add_argument("--plot_all_in_one", help="Plot all generated paths in one figure", action='store_true', default=default_plot_all)
parser.add_argument("--mean_speed", help="Mean speed of the moving object (m/s)", type=float, default=default_mean_speed)
parser.add_argument("--freq", help="Frequency of measurements (Hz)", type=float, default=default_freq)





args = parser.parse_args()


anchor_position_file=args.Anchor_position #path to the anchor position file (json format)
number_of_paths=args.number_of_paths #number of paths (graphs)
with_plots=args.with_plots #if you want to plot the generated paths
plot_all_in_one=args.plot_all_in_one
mean_speed=args.mean_speed #mean speed of the moving object (m/s)
freq=args.freq #frequency of measurements (Hz)





    


def anchor_pos(anchor_position_file):
    """ Get the anchor positions in cartesian coordinates (cm) from a json file.
    Args:
        anchor_position_file : str
            The file path to the anchor position (json format).

    Returns:
    anchor_position: np.ndarray
        The anchor positions in cartesian coordinates (cm).  
    """ 
    print("------------------------------------------------------------------")
    print("extracting anchors' position...")
    print("------------------------------------------------------------------")
    with open(anchor_position_file, "r") as f:
        anchor_position = json.load(f)
        
    devices=anchor_position["devices"]
    anchor_position= convert_gps_to_cartesian_cm(anchor_position["transform"], devices)
    print("extracting is done !")
    return anchor_position



#getr the anchor position in cartesian coordinates 
anchor_position = anchor_pos(anchor_position_file)



x_min, x_max = anchor_position[:, 0].min(), anchor_position[:, 0].max()
y_min, y_max = anchor_position[:, 1].min(), anchor_position[:, 1].max()
y_mean = anchor_position[:, 1].mean()
x_mean = anchor_position[:, 0].mean()





def number_of_points(mean_speed, freq, x_min=x_min, x_max=x_max,y_min=y_min,y_max=y_max):
    """ Calculate the number of points based on mean speed, frequency, and desired number of dots.
    """
    distance = x_max - x_min  # in cm
    if mean_speed ==0:
        time_seconds=30
        num_points=1
    else :
        time_seconds = distance / (mean_speed * 100)  # convert speed to cm 
        num_points = int(time_seconds * freq) # ensure at least 2 points
    return num_points



num_dots=number_of_points(mean_speed, freq)



# Generate time vector
t = [0]  # time vector
for n in range(num_dots-1): # time vector based on frequency
    t.append(t[n]+(1/ freq))
t = np.array(t)





def generate_linear_paths(
    filename: str,
    num_paths: int=number_of_paths,
    num_points: int=num_dots,
    t:np.ndarray=t,
    x_min=x_min, x_max=x_max,y_min=y_min,y_max=y_max, y_mean=y_mean):
    
    """
    Generate random linear paths scaled to anchor positions and store as a numpy array.
    Each path is an array of shape (num_points, 2) for (X, Y).
    All paths are stored in an array of shape (num_paths, num_points, 2).
    """
    np.random.seed(seed)

    
    linear_coeff = 2*np.random.rand(num_paths, 2) - 1  # random linear coefficients for Y
    paths = []
    
 
    
    print("------------------------------------------------------------------")
    print("                      Generating paths...")
    print("------------------------------------------------------------------")
    
    for i in range(num_paths):
        X = np.random.uniform(1.2*x_min,1.2* x_max, num_points) # to generate X with random speed variations 
        #X= np.linspace(x_min, x_max, num_points) # to generate X with constant speed
        X.sort()  # Sort X values to create a more natural path
        Y = linear_coeff[i, 0] * X + linear_coeff[i, 1] * y_max#+ np.random.uniform(y_min, y_max)
        path = np.stack((X, Y,t), axis=1)  # shape (num_points, 3)
        paths.append(path)
    paths = np.array(paths)  # shape (num_paths, num_points, 3)
    np.save(filename, paths)

    print(f"Saved {num_paths} paths with {num_points} points each to {filename}")






def generate_polynomial_paths(
    filename: str,
    num_paths: int=number_of_paths,
    num_points: int=num_dots,
    t:np.ndarray=t,
    x_min=x_min, x_max=x_max,y_min=y_min,y_max=y_max):
    """
    Generate random polynomial paths scaled to anchor positions and store as a numpy array.
    """
    np.random.seed(seed)

    if y_max!=0:
        scale=np.abs(y_max)
    elif y_min!=0:
        scale=np.abs(y_min)
    else:
        scale=0.001
        
    coeff = 2*np.random.rand(num_paths, 3) - 1  # random coefficients for Y
    paths = []
    
    
    print("------------------------------------------------------------------")
    print("                      Generating paths...")
    print("------------------------------------------------------------------")
    for i in range(num_paths):
        X=np.random.uniform(1.1*x_min, 1.1*x_max, num_points)
        # Sort X values to create a more natural path
        #X= np.linspace(x_min, x_max, num_points) # to generate X with constant speed
        X.sort()  
        Y= (coeff[i,0]*(X-x_mean)**2 + coeff[i,1]*(X-x_mean))/scale +y_max*coeff[i,2]
        path = np.stack((X, Y,t), axis=1)  # shape (num_points, 3)
        paths.append(path)
    paths = np.array(paths)  # shape (num_paths, num_points, 3)
    np.save(filename, paths)    
    print(f"Saved {num_paths} paths with {num_points} points each to {filename}")    





def generate_circular_paths(
    filename: str,
    num_paths: int=number_of_paths,
    num_points: int=num_dots,
    t:np.ndarray=t,
    x_min=x_min, x_max=x_max,y_min=y_min,y_max=y_max, y_mean=y_mean):
    """
    Generate random circular paths scaled to anchor positions and store as a numpy array.
    """
    np.random.seed(seed)
    paths = []
    
    print("------------------------------------------------------------------")
    print("                      Generating paths...                         ")
    print("------------------------------------------------------------------")
    for i in range(num_paths):
        low = min(x_min, y_min)
        high = max(x_max, y_max)
        epsilon = 5.0  # distance autour des bornes

        if np.random.rand() < 0.5:
            radius = np.random.uniform(low, low + epsilon)
        else:
            radius = np.random.uniform(high - epsilon, high)
        center_x = np.random.uniform(x_min ,x_max)
        center_y = np.random.uniform(y_min, y_max )
        theta = np.linspace(0, 2 * np.pi, num_points)
        X = center_x + radius * np.cos(theta) + np.random.normal(0, 1, num_points)
        Y = center_y + radius * np.sin(theta) #+ np.random.normal(0, np.sqrt(np.abs(y_mean)), num_points)
        path = np.stack((X, Y,t), axis=1)  # shape (num_points, 3)
        paths.append(path)
    paths = np.array(paths)  # shape (num_paths, num_points, 3)
    np.save(filename, paths)    
    
    print(f"Saved {num_paths} circular paths with {num_points} points each to {filename}")





def generate_sinus_paths(    
    filename: str,
    num_paths: int=number_of_paths,
    num_points: int=num_dots,
    t:np.ndarray=t,
    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, x_mean=x_mean, y_mean=y_mean):


    np.random.seed(seed)
    periode = (x_max - x_min) / 3 +np.random.normal(0,np.abs(x_mean)) # period of the sine wave in cm
    paths = []
    print("------------------------------------------------------------------")
    print("                      Generating paths...")
    print("------------------------------------------------------------------")
    for i in range(num_paths):
        
        X = np.sort(np.random.uniform(1.2 * x_min, 1.2 * x_max, num_points))
        #X= np.linspace(x_min, x_max, num_points) # to generate X with constant speed
        
        amplitude = np.random.uniform(0.3, 1.0) * (y_max - y_min) / 2
        phase = np.random.uniform(0, 2 * np.pi)
        vertical_shift = np.random.uniform(0.9*y_min, 0.9*y_max - amplitude)
        noise = np.random.normal(0, (y_max - y_min) * 0.03, num_points)
        
        Y = amplitude * np.sin(((X - X[0]) )/ periode + phase) + vertical_shift #+ noise # if we want to add noise to the sinusoidal path
        # Ensure Y stays within [y_min, y_max]
        
        path = np.stack((X, Y,t), axis=1)  # shape (num_points, 3)
        paths.append(path)
    paths = np.array(paths)  # shape (num_paths, num_points, 3)
    np.save(filename, paths)
    print(f"Saved {num_paths} wave paths with {num_points} points each to {filename}")




def random_walk(    
    filename: str,
    num_paths: int=number_of_paths,
    num_points: int=num_dots,
    t:np.ndarray=t,
    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, x_mean=x_mean, y_mean=y_mean,
    mean_speed: float = mean_speed,
    freq= freq,
):
    """
    Generate multiple kinematic-style paths (simulating human or vehicle motion)
    and save them to a .npy file.

    Parameters
    ----------
    filename : str
        Path to the output .npy file.
    num_paths : int
        Number of paths to generate.
    num_points : int
        Number of points per path.
    t : np.ndarray
        Time array (same length as num_points).
    x_min, x_max, y_min, y_max : float
        Spatial boundaries.
    x_mean, y_mean : float
        Mean starting position (center of the area).
    mean_speed : float, optional
        Mean movement speed (default = 1.0).

    seed : int, optional
        Random seed for reproducibility (default = 42).
    """

    np.random.seed(seed)

    print("------------------------------------------------------------------")
    print("                   Generating kinematic paths...")
    print("------------------------------------------------------------------")

    paths = []

    for i in range(num_paths):
        # Initialize starting point near center
        x, y = [np.random.uniform(x_min*0.6, 0.6*x_max)], \
               [np.random.uniform( 0.6*y_min, 0.6*y_max)]
        
        v = mean_speed
        heading = np.random.uniform(0, 2 * np.pi)

        for j in range(num_points - 1):
            # Random acceleration and turn
            accel = np.random.normal(0, 3.5)
            v = np.clip(v + accel * (1/freq), -mean_speed*3, mean_speed * 3) 
            turn = np.random.uniform(-np.pi /8, np.pi /8)
            heading += turn
            dt = 1/freq
            dx = v * np.cos(heading) * dt
            dy = v * np.sin(heading) * dt

            new_x = x[-1] + dx*100 #multiply by 100 to go from m/s to cm/s
            new_y = y[-1] + dy*100#multiply by 100 to go from m/s to cm/s
            
            if  np.abs(new_y) > max(np.abs(y_max*1.1),np.abs(y_min*1.1)) :
                heading = 0
            if np.abs(new_x) > max(np.abs(x_max*1.1),np.abs(x_min*1.1)) :
                heading = np.pi/2
            if new_y > (y_max*1.1) and  new_x < x_min*1.1 :
                heading= np.pi*(7/4)
            if new_y > (y_max*1.1) and new_x > (x_max*1.1) : 
                heading = np.pi*(5/4)
            if new_y < y_min*1.1 and  new_x < x_min*1.1 :
                heading= np.pi*(1/4)
            if new_y < y_min*1.1 and  new_x > x_max*1.1 :
                heading= np.pi*(3/4)
                
                

            x.append(new_x)
            y.append(new_y)

        path = np.stack((x, y, t[:num_points]), axis=1)  # shape (num_points, 3)
        paths.append(path)

    paths = np.array(paths)  # shape (num_paths, num_points, 3)
    np.save(filename, paths)

    print(f"Saved {num_paths} kinematic paths with {num_points} points each to {filename}")

        
        

    

def plot_paths(paths_file:str, anchor_cartesian_data: np.ndarray):
    """Plot generated paths and anchor positions.
    Args:
        paths_file (str): Path to the .npy file containing generated paths.
        anchor_cartesian_data (np.ndarray): Anchor positions in cartesian coordinates (cm)."""
    print("------------------------------------------------------------------")
    print(f"Plotting paths from {paths_file} and anchor positions from provided data.")
    print("------------------------------------------------------------------")
    
    
    paths=np.load(paths_file)
    plt.figure(figsize=(10, 8))
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], marker='x', markersize=4, linestyle='-', alpha=0.5)
    plt.scatter(anchor_cartesian_data[:, 0], anchor_cartesian_data[:, 1], color='red', label='Anchor Positions', alpha=0.7, s=100, edgecolor='k')
    plt.title('Generated Paths and Anchor Positions')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.legend()
    plt.grid(True)
    #plt.axis('equal')
    
    
    
    
    
def plot_all(anchor_cartesian_data: np.ndarray, dir_path, path_folder: str = 'paths'):
    """Plot all generated paths from .npy files in a specified directory along with anchor positions.
    Args:
        anchor_cartesian_data (np.ndarray): _description_
        dir_path (_type_): _description_
        path_folder (str, optional): _description_. Defaults to 'paths'.
    """
    
    plt.figure(figsize=(10, 8))
    paths_dir = os.path.join(dir_path, path_folder)
    for filename in os.listdir(paths_dir):
        if filename.endswith('.npy'):
            paths_file = os.path.join(paths_dir, filename)
            paths = np.load(paths_file)
            
            for path in paths:
                plt.plot(path[:, 0], path[:, 1], marker='o', markersize=5, linestyle='-', alpha=0.9)
    plt.scatter(anchor_cartesian_data[:, 0], anchor_cartesian_data[:, 1], color='red', label='Anchor Positions', alpha=0.7, s=100, edgecolor='k')
    plt.title('Generated Paths and Anchor Positions')
    plt.xlabel('X (cm)')
    plt.ylabel('Y (cm)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    
    

def main():
    #get the directory of the current script
    dir_path = os.path.dirname(os.path.realpath(__file__))
    

    if with_plots:
        #plot_paths(os.path.join(dir_path,"paths/linear_paths.npy"), anchor_position)
        #plot_paths(os.path.join(dir_path,"paths/polynomial_paths.npy"), anchor_position)
        #plot_paths(os.path.join(dir_path,"paths/circular_paths.npy"), anchor_position)
        #plot_paths(os.path.join(dir_path,"paths/wave_paths.npy"), anchor_position)
        
        
        plt.show()
        
    if plot_all_in_one:
        plot_all(anchor_position,dir_path)
    
if __name__ == "__main__":
    main()