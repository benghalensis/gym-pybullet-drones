import pybullet as p
import pybullet_data
import pkg_resources
import time
import random
import math

from scipy.spatial.transform import Rotation as R
import numpy as np

# GATE_DIMENSION = [Inside_Square, Outside_Square, Gate_thickness]
GATE_DIMENSION = [0.75, 1.25, 0.2]


def _build_gate(center, orientation, client):
    sphereRadius = 0.05
    center = np.array(center)
    rotation = R.from_quat(orientation).as_matrix()
    x_axis = rotation[:,0]
    z_axis = rotation[:,2]
    delta = (GATE_DIMENSION[1] + GATE_DIMENSION[0])/4
    mass = 1

    # Piller dimensions
    piller_x = (GATE_DIMENSION[1] - GATE_DIMENSION[0])/2 - 0.01
    piller_y = GATE_DIMENSION[2]- 0.01
    piller_z = GATE_DIMENSION[1]- 0.01

    # Slab dimensions
    slab_x = GATE_DIMENSION[0]- 0.01
    slab_y = GATE_DIMENSION[2]- 0.01
    slab_z = (GATE_DIMENSION[1] - GATE_DIMENSION[0])/2 - 0.01
    
    # Move in the +x axis by (GATE_DIMENSION[1] + GATE_DIMENSION[0])/2
    piller_1_center = center + delta * x_axis
    piller_2_center = center - delta * x_axis
    roof_center = center + delta * z_axis
    floor_center = center - delta * z_axis

    piller_collision_ID = p.createCollisionShape(p.GEOM_BOX, halfExtents=[piller_x/2, piller_y/2, piller_z/2], physicsClientId=client)
    piller_visual_ID = p.createVisualShape(p.GEOM_BOX, halfExtents=[piller_x/2, piller_y/2, piller_z/2], physicsClientId=client)

    slab_collision_ID = p.createCollisionShape(p.GEOM_BOX, halfExtents=[slab_x/2, slab_y/2, slab_z/2], physicsClientId=client)
    slab_visual_ID = p.createCollisionShape(p.GEOM_BOX, halfExtents=[slab_x/2, slab_y/2, slab_z/2], physicsClientId=client)

    piller_1_ID = p.createMultiBody(mass, 
                                  piller_collision_ID, 
                                  piller_visual_ID, 
                                  piller_1_center, 
                                  orientation,
                                  physicsClientId=client)

    piller_2_ID = p.createMultiBody(mass, 
                                  piller_collision_ID, 
                                  piller_visual_ID, 
                                  piller_2_center, 
                                  orientation,
                                  physicsClientId=client)

    roof_ID = p.createMultiBody(mass, 
                                  slab_collision_ID, 
                                  slab_visual_ID, 
                                  roof_center, 
                                  orientation,
                                  physicsClientId=client)

    floor_ID = p.createMultiBody(mass, 
                                  slab_collision_ID, 
                                  slab_visual_ID, 
                                  floor_center, 
                                  orientation,
                                  physicsClientId=client)

    piller_1_constraint = p.createConstraint(piller_1_ID, -1, -1, -1, 
                                             p.JOINT_FIXED, 
                                             [0, 0, 0], 
                                             [0, 0, 0], 
                                             piller_1_center, 
                                             parentFrameOrientation=[0, 0, 0, 1],
                                             childFrameOrientation=orientation,
                                             physicsClientId=client)

    piller_2_constraint = p.createConstraint(piller_2_ID, -1, -1, -1, 
                                             p.JOINT_FIXED,
                                             [0, 0, 0], 
                                             [0, 0, 0], 
                                             piller_2_center,
                                             parentFrameOrientation=[0, 0, 0, 1],
                                             childFrameOrientation=orientation,
                                             physicsClientId=client)

    roof_constraint = p.createConstraint(roof_ID, -1, -1, -1, p.JOINT_FIXED, 
                                             [0, 0, 0], 
                                             [0, 0, 0], 
                                             roof_center,
                                             parentFrameOrientation=[0, 0, 0, 1],
                                             childFrameOrientation=orientation,
                                             physicsClientId=client)

    floor_constraint = p.createConstraint(floor_ID, -1, -1, -1, 
                                             p.JOINT_FIXED, 
                                             [0, 0, 0], 
                                             [0, 0, 0], 
                                             floor_center,
                                             parentFrameOrientation=[0, 0, 0, 1],
                                             childFrameOrientation=orientation,
                                             physicsClientId=client)

    return [piller_1_constraint, piller_2_constraint, roof_constraint, floor_constraint]

if __name__=='__main__':
    client = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.setPhysicsEngineParameter(fixedTimeStep=1/100)
    p.setGravity(0, 0, -9.81)

    obstaclesCenterPosition = [[0,1,0.625], [0,2,0.625], [0,3,0.625], [0,4,0.625], [0,5,0.625]]
    obstaclesOrientation = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
    obstaclesStd = 0.01
    obstacleIDs = []
    gate_width = GATE_DIMENSION[1]

    assert len(obstaclesCenterPosition) == len(obstaclesOrientation)
    n = len(obstaclesCenterPosition)

    # Test
    test_collision_ID = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
    test_visual_ID = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)

    test_ID = p.createMultiBody(baseMass=1,
                                baseCollisionShapeIndex=test_collision_ID, 
                                baseVisualShapeIndex=test_visual_ID, 
                                basePosition=[0,0,1], 
                                baseOrientation=[0,0,0,1])
    # End of test

    for i in range(n):
        center = [random.gauss(obstaclesCenterPosition[i][0], obstaclesStd), 
                  random.gauss(obstaclesCenterPosition[i][1], obstaclesStd), 
                  random.gauss(obstaclesCenterPosition[i][2], obstaclesStd)]

        if center[2] < gate_width/2: center[2] = gate_width/2

        obstacle_ID_list =_build_gate(obstaclesCenterPosition[i], p.getQuaternionFromEuler(obstaclesOrientation[i]), client)

        obstacleIDs += obstacle_ID_list

    print(obstacleIDs)

    contact_information = p.getContactPoints()
    print(contact_information)

    while True:
        p.stepSimulation()
        time.sleep(0.01)
    