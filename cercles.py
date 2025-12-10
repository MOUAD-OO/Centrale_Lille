import pandas as pd
import numpy as np 
from generating_bloc.convert_data import get_anchors_position_and_id
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from scipy.optimize import least_squares




idx =1




anchor_path= "generating_bloc/woltDynDev.json"
def anchors(anchors_path:str)->dict:
    with open(anchors_path, 'r') as f:
        data = json.load(f)
    ref_data = data['transform']
    anchor_data = data['devices']
    anchors_positions = get_anchors_position_and_id(ref_data, anchor_data)
    return anchors_positions

anchor_position =anchors(anchor_path)
data=pd.read_csv('test_site/test_001BC50C70027E47.csv', sep= ',')

anchors_cercles_list = []

for i in range(len(data)):
    # create a list of [position, distance] for all anchors at step i
    step_circles = [[anchor_position[key], data.at[i, key]] for key in anchor_position]
    anchors_cercles_list.append(step_circles)

fig, ax = plt.subplots()

# Generate distinct colors for each group
num_groups = len(anchors_cercles_list)
colors = cm.get_cmap('tab10', num_groups)  # or 'viridis', 'rainbow', etc.

for i, anchors_cercles in enumerate(anchors_cercles_list):
    color = colors(i)  # pick a color from the colormap
    if i ==idx:
        for pos, r in anchors_cercles:
            if r==0:
                ax.plot(pos[0], pos[1], 'o', color='yellow', label= "Unused Anchors")
            if r!=0:
                circle = plt.Circle(
                    (pos[0], pos[1]), r,
                    color='red',
                    alpha=0.1,
                    linewidth=1)
                ax.add_patch(circle)
                # Optional: mark the anchor
                
                ax.plot(pos[0], pos[1], 'o', color='green', label = "Used Anchors")






input = 'test_site/test_001BC50C70027E47.csv'


def matrix_traitment(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix = matrix.copy()
    
    # Replace zeros with NaN so we can interpolate
    matrix.replace(0, np.nan, inplace=True)
    
    # Apply interpolation column by column
    for column in matrix.columns:
        col_values = matrix[column].values  # get as numpy array
        n = len(col_values)
        i = 0
        
        while i < n:
            if np.isnan(col_values[i]):
                start = i - 1 if i > 0 else None
                
                # find end of NaN run
                j = i
                while j < n and np.isnan(col_values[j]):
                    j += 1
                
                end = j if j < n else None
                
                if start is None:  # NaNs at the beginning
                    col_values[0:j] = col_values[j]
                elif end is None:  # NaNs at the end
                    col_values[i:n] = col_values[start]
                else:  # interpolate between start and end
                    num_missing = j - start + 1
                    interp = np.linspace(col_values[start], col_values[end], num_missing)
                    col_values[start+1:end] = interp[1:-1]
                
                i = j  # skip to end of NaN run
            else:
                i += 1
        
        matrix[column] = col_values
    
    return matrix


def dist(p1:np.ndarray, p2:np.ndarray)->float:
    return np.linalg.norm(p1 - p2)

def error(x,dist):
    return np.abs(x-dist)

def matrix(matrix:str)->pd.DataFrame:
    distance_matrix=pd.read_csv(matrix,sep=",")
    return distance_matrix


def measur_at_step(dist_matrix:pd.DataFrame, step:int)->pd.DataFrame:
    dist_matrix=dist_matrix.drop(columns=["time stamp"])
    return dist_matrix.iloc[step]

def initial_guess(distance,step,anchors_path):# afunction to give the best initial guess to minimize outliers and optimize the computation time 
    distance= measur_at_step(distance,step)
    anchors_position=anchors(anchors_path)

    weights = [
        1 / (distance[e] ** 1) for e in distance.index if distance[e] != 0
    ]
  
    X0= sum([(1/(distance[e]**1))*np.array(anchors_position[e]) for e in distance.index if distance[e]!=0 ])
    if sum(weights) == 0:
        X0 = np.mean([np.array(anchors_position[e][:2]) for e in distance.index], axis=0)
    else:
        X0 = X0 / sum(weights)
    
    return X0[:2]

def residual_function(x, distance: pd.DataFrame, anchors_positions, step: int) -> list:# there is a list to save the hhistory of x if you want to take the function delet the first line 
    x_history.append(x.copy())
    outside_scale = 4
    MEW = 2

    residuals = []

    for i in range(len(distance.index)):   
        d_mes = distance[distance.index[i]]  # measured distance

        # Skip if measurement is zero or invalid
        if d_mes == 0:
            continue

        anchor_pos = np.array(anchors_positions[distance.index[i]])[:2]  # only (x,y)
        d = dist(x, anchor_pos)  # estimated distance
            
        e = d - d_mes  # signed error
        if d > d_mes:
            cost = e/ d_mes + (e * MEW) ** outside_scale 
        else:
            cost = e/ d_mes

        residuals.append(cost)

    return residuals


def jac(x, distance: pd.DataFrame, anchors_positions, step: int):

    outside_scale = 4
    MEW = 2

    jac = []

    for i in range(len(distance.index)):
        d_mes = distance[distance.index[i]]
        
        if d_mes == 0:
            continue  
        anchor_pos=np.array(anchors_positions[distance.index[i]])[:2]
        d = dist(x, anchor_pos)  # estimated distance
            
        e = d - d_mes  # signed error
        if d> d_mes:
            w_i =   1/d_mes +((MEW**outside_scale)*outside_scale*(e**(outside_scale-1)))/d     
        else:
            w_i = 1/(d_mes*d)
            
        jac.append(w_i*(x-anchor_pos))
    
    return jac





# localise the Fst position with the least square method
distance=matrix(input)
#distance=matrix("/home/nathan/Dynamic_Loc/generating_bloc/dist_matrix.csv")
anchors_path="generating_bloc/woltDynDev.json"
anchors_positions = anchors(anchors_path)   # dict {UID: position}
anchor_array = np.array(list(anchors_positions.values()))[:, :2]  # only x,y

# Bounds for x and y
lower_bounds = anchor_array.min(axis=0)  # [min_x, min_y]
upper_bounds = anchor_array.max(axis=0)  # [max_x, max_y] 

x_history_list=[]
new_path=[]
for step in range(len(distance['time stamp'])):
    X0=initial_guess(distance,step,anchors_path)
    x_history= []
    anchors_positions = anchors(anchors_path)   # dict {UID: position}
    distance_ = measur_at_step(distance, step) 
    solution=least_squares(residual_function,x0=X0,loss='huber', tr_solver='exact',args=(distance_,anchors_positions,step))
   
    pos_time=list(solution.x)
    new_path.append(pos_time)

    x_history_list.append(np.array(x_history))



new_path = np.array(new_path)  # convert list of lists to NumPy array


for i in range(len(x_history_list)) :
    if i==200:
        arr=x_history_list[i]
        plt.plot(arr[:,0], arr[:,1], 'x',linestyle='-',color='red',alpha=0.2)



plt.axis('equal')
plt.plot(new_path[idx,0], new_path[idx,1], 's',linestyle='dashed',label='The predicted Position')
#plt.plot(real_path[idx,0],real_path[idx,1], 'x',linestyle='-')
plt.legend()
plt.xlabel("X (cm)")
plt.ylabel("Y (cm)")
plt.grid(True)
plt.show()
