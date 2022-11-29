import pybullet as p
import pybullet_data
import pkg_resources
import time
import random

CLIENT = p.connect(p.GUI)
obstaclesCenterPosition = [[0,1,0.625], [0,2,0.625], [0,3,0.625], [0,4,0.625], [0,5,0.625]]
obstaclesOrientation = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
obstaclesStd = 0.01
obstacleIDs = []
gate_width = 1.25

if __name__=='__main__':
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setPhysicsEngineParameter(fixedTimeStep=1/100)
    p.setGravity(0, 0, -9.81)

    assert len(obstaclesCenterPosition) == len(obstaclesOrientation)
    n = len(obstaclesCenterPosition)

    for i in range(n):
        center = [random.gauss(obstaclesCenterPosition[i][0], obstaclesStd), 
                  random.gauss(obstaclesCenterPosition[i][1], obstaclesStd), 
                  random.gauss(obstaclesCenterPosition[i][2], obstaclesStd)]

        if center[2] < gate_width/2: center[2] = gate_width/2

        obstacleID = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', "assets/gates.urdf"),
                                    center,
                                    p.getQuaternionFromEuler(obstaclesOrientation[i]),
                                    useFixedBase=1,
                                    physicsClientId=CLIENT
                    )
        obstacleIDs.append(obstacleID)

    print(obstacleIDs)

    while True:
        p.stepSimulation()
        time.sleep(0.01)
    