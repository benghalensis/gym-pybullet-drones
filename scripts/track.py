import numpy as np

class Track:
    def __init__(self, obstaclesCenterPosition, 
                       obstaclesOrientation, 
                       obstaclesStd, 
                       obstacleOrientationStd,
                       initial_xyzs,
                       initial_xyzs_std,
                       gate_width=1.25):
        self.obstaclesCenterPosition = obstaclesCenterPosition
        self.obstaclesOrientation = obstaclesOrientation
        self.obstaclesStd = obstaclesStd
        self.obstacleOrientationStd = obstacleOrientationStd

        # The outer width of the gate
        self.gate_width = gate_width

        # The initial position of the drone
        self.initial_xyzs = initial_xyzs
        self.initial_xyzs_std = initial_xyzs_std

class Straight(Track):
    def __init__(self,):
        obstaclesCenterPosition = [[0,1.5,1], [0,3,1], [0,4.5,1], [0,6,1], [0,7.5,1]]
        obstaclesOrientation = [[0,0,0], [0,0,0], [0,0,0], [0,0,0], [0,0,0]]
        obstaclesStd = 0.05
        obstacleOrientationStd = 0.05
        initial_xyzs = np.array([[0.0,0.0,1.0]])
        initial_xyzs_std = 0.2

        super().__init__(obstaclesCenterPosition, 
                       obstaclesOrientation, 
                       obstaclesStd, 
                       obstacleOrientationStd, 
                       initial_xyzs, 
                       initial_xyzs_std)

class HalfCircle(Track):
    def __init__(self,):
        obstaclesCenterPosition = [[0,2,1], [0.202,4,1], [0.835,6,1], [2,8,1], [4, 10, 1], [10, 12, 1],
                                   [16, 10, 1], [18, 8, 1], [19.165, 6, 1], [19.798, 4, 1], [20, 2, 1]]
        obstaclesOrientation = [[0, 0, 0], [0, 0, -0.201], [0, 0, -0.411], [0, 0, -0.643], [0, 0, -0.927], [0, 0, -1.571],
                                [0, 0, -2.215], [0, 0, -2.499], [0, 0, -2.731], [0, 0, -2.941], [0, 0, -3.142]]
        obstaclesStd = 0.05
        obstacleOrientationStd = 0.05
        initial_xyzs = np.array([[0.0,0.0,1.0]])
        initial_xyzs_std = 0.2

        super().__init__(obstaclesCenterPosition, 
                       obstaclesOrientation, 
                       obstaclesStd, 
                       obstacleOrientationStd, 
                       initial_xyzs, 
                       initial_xyzs_std)

class Single(Track):
    def __init__(self,):
        obstaclesCenterPosition = [[0,1.5,1]]
        obstaclesOrientation = [[0,0,0]]
        obstaclesStd = 0.05
        obstacleOrientationStd = 0.05
        initial_xyzs = np.array([[0.0,0.0,1.0]])
        initial_xyzs_std = 0.2

        super().__init__(obstaclesCenterPosition, 
                       obstaclesOrientation, 
                       obstaclesStd, 
                       obstacleOrientationStd,
                       initial_xyzs,
                       initial_xyzs_std)

class AlphaPilot(Track):
    def __init__(self,):
        obstaclesCenterPosition = [[20.0,-10.0, 3.0], # Obs-0.5
                                    [20.0, 5.0, 3.0], # Obs-1
                                    [15.0, 40.0, 3.0], # Obs-2
                                    [ 5.0, 30.0, 3.0], # Obs-3
                                    [ 0.0, 5.0, 3.0], # Obs-4
                                    [-5.0,-10.0, 3.0], # Obs-5
                                    [-5.0,-25.0, 3.0], # Obs-6
                                    [ 0.0,-30.0, 3.0], # Obs-7
                                    [ 5.0,-25.0, 3.0], # Obs-8
                                    [ 5.0,-10.0, 3.0], # Obs-9
                                    [-10.0, 5.0, 3.0], # Obs-10
                                    [-10.0, 35.0, 3.0], # Obs-11
                                    ]

        obstaclesOrientation = [[0.0, 0.0, 0.0], # Obs-0.5
                                [0.0, 0.0, 0.0], # Obs-1
                                [0.0, 0.0, np.pi/4], # Obs-2
                                [0.0, 0.0, np.pi], # Obs-3
                                [0.0, 0.0, np.pi], # Obs-4
                                [0.0, 0.0, np.pi], # Obs-5
                                [0.0, 0.0, 5/4*np.pi], # Obs-6
                                [0.0, 0.0, 3/2*np.pi], # Obs-7
                                [0.0, 0.0, 7/4*np.pi], # Obs-8
                                [0.0, 0.0, 0.0], # Obs-9
                                [0.0, 0.0, 0.0], # Obs-10
                                [0.0, 0.0, 0.0], # Obs-11
                                ]

        obstaclesStd = 0.25
        obstacleOrientationStd = 0.1
        initial_xyzs = np.array([[20.0, -25.0, 3.0]])
        initial_xyzs_std = 0.2


        super().__init__(obstaclesCenterPosition, 
                       obstaclesOrientation, 
                       obstaclesStd, 
                       obstacleOrientationStd,
                       initial_xyzs,
                       initial_xyzs_std)
