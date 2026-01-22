import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import os 
import itertools


anchors = np.load("generating_bloc/anchor_positions.npy")

fig, ax = plt.subplots(figsize=(16,12))
x=[-1140,1238]
y=[290.2,-279.74]
#ax.plot(x,y, linestyle='--', color='gray', label='Chemin idéal')
ax.scatter(anchors[:,0],anchors[:,1],marker='s',label= "Anchors",c='black')
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
            if traj is None:
                continue
            filenames.append(f)
            trajectories.append(traj)
            line, = ax.plot([], [], marker='o', markersize=3, linestyle='-', alpha=0.7, label=f, linewidth=2)
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
    all_x = []
    all_y = []
    for traj in trajectories:
        if len(traj) > 0:
            all_x.extend(traj[:, 0].tolist())
            all_y.extend(traj[:, 1].tolist())
    if all_x and all_y:
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        # expand if outside current limits
        pad = 200
        new_xmin = min(cur_xlim[0], min_x - pad)
        new_xmax = max(cur_xlim[1], max_x + pad)
        new_ymin = min(cur_ylim[0], min_y - pad)
        new_ymax = max(cur_ylim[1], max_y + pad)
        ax.set_xlim(new_xmin, new_xmax)
        ax.set_ylim(new_ymin, new_ymax)

    for line, traj in zip(lines, trajectories):
        n = min(frame + 1, len(traj))
        if n > 0:
            line.set_data(traj[:n, 0], traj[:n, 1])
    return lines


# Run the animation indefinitely; update() will pick up new files as they appear.
ani = FuncAnimation(fig, update, frames=itertools.count(), interval=1, blit=False)

plt.axis('equal')
plt.legend()
plt.grid()
#ani.save("animation.mp4", writer="ffmpeg", fps=3)
plt.show()
