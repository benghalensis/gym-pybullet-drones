import matplotlib.pyplot as plt
import numpy as np
import os
from track import AlphaPilot, Single, Straight, HalfCircle
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

GATE_WIDTH = 1.25
GATE_HEIGHT = 0.2

if __name__=="__main__":
    track = AlphaPilot()
    initial_xyzs = track.initial_xyzs
    center = np.array(track.obstaclesCenterPosition)
    orientation = np.array(track.obstaclesOrientation)
    num_gates = center.shape[0]

    fig, ax = plt.subplots()
    # Drone Start location
    ax.scatter(initial_xyzs[:,0], initial_xyzs[:,1], color='red', marker="^")
    ax.plot(center[:,0], center[:,1], color='blue', linewidth=1)
    # ax.scatter(center[:,0], center[:,1], color='blue')
    gate_rectangles = [mpatches.Rectangle([center[i,0] - GATE_WIDTH/2, center[i,1]-GATE_HEIGHT/2], 
                       GATE_WIDTH, GATE_HEIGHT, angle=orientation[i,2]*180/np.pi, rotation_point='center', ec="none") for i in range(num_gates)]
    collection = PatchCollection(gate_rectangles, color='green', alpha=0.3)
    ax.add_collection(collection)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal', 'datalim')
    ax.margins(0.1)

    if False:
        drone_path_list = []
        dir_path = '/media/enigma/Amplitude/Study/UW/CSE579/Project/gym-pybullet-drones/results/single/evaluations/drone_paths/'
        for file in os.listdir(dir_path):
            drone_path = np.load(dir_path + file)
            drone_path_list.append(drone_path)
        
        for drone_path in drone_path_list:
            ax.plot(drone_path[:,0], drone_path[:,1], color='red', alpha=0.3)

    plt.show()




    

    
