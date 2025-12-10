import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os 
import itertools
from data_preparing.physical_atribute import speed_time_series, acceleration_time_series


anchors = np.load("generating_bloc/anchor_positions.npy")

def func(metric):

    fig,ax = plt.subplots(figsize=(16,12))
    
    
    trajectories = []
    lines = []
    filenames = []
    file_info = {}

    def load_trajectory(path):
        try:
            arr = np.load(path)
            traj = arr[0, :, :2]
            return traj
        except Exception:
            return None

    def refresh_files():
        """Scan directory for .npy trajectory files and update internal lists.

        - Adds new files as new lines on the plot
        - Updates cached trajectories when file mtime changes
        """
        files = [f for f in os.listdir("test_site/trajectories") if f.endswith('.npy')]
        for f in files:
            p = os.path.join("test_site/trajectories", f)
            try:
                mtime = os.path.getmtime(p)
            
            except FileNotFoundError:
                continue

            info = file_info.get(f)

            if info is None:
                traj = load_trajectory(p)
                if metric == "speed":
                    traj = speed_time_series(traj, frequency=1)
                if metric == "acc":
                    traj = acceleration_time_series(traj, frequency=1)
                if traj is None:
                    continue
                filenames.append(f)
                
                trajectories.append(traj)
                line, = ax.plot([], [], marker='o', markersize=4, linestyle='-', alpha=0.7, label=f)
                lines.append(line)
                file_info[f] = {"mtime": mtime, "index": len(trajectories) - 1}
                ax.legend()
            else:
                if mtime > info["mtime"]:
                    traj = load_trajectory(p)
                    if traj is None:
                        continue
                    idx = info["index"]
                    trajectories[idx] = traj
                    file_info[f]["mtime"] = mtime



    def update(frame):
        refresh_files()
        
        
        lenght=[]
        y_min =[]
        y_max =[]
        for line, traj in zip(lines, trajectories):
            n = min(frame + 1, len(traj))
            lenght.append(n)
            if n > 0:
                # For 1D time series (speed/acc), use frame index as x-axis
                x_data = np.arange(n)
                y_data = np.atleast_1d(traj)[:n]
                line.set_data(x_data, y_data)
                y_min.append(min(y_data))
                y_max.append(max(y_data))
            
        ax.set_ylim(min(y_min), max(y_max))
        ax.set_xlim(0,np.max(lenght))
        return lines

    # Run the animation indefinitely; update() will pick up new files as they appear.
    ani = FuncAnimation(fig, update, frames=itertools.count(), interval=10, blit=False)

    plt.legend()
    plt.grid()    
    plt.show()
    
    
func("speed")



