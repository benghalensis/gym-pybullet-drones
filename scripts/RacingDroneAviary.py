import os
import numpy as np
import pybullet as p
import pkg_resources
import random
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from reward_function import final_reward
from add_track import _build_gate
from gym import spaces

class RacingDroneAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly through a gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_xyzs_std: float = 0.0,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 obstaclesCenterPosition=None,
                 obstaclesOrientation=None,
                 obstaclesStd: float = 0.001,
                 obstacleOrientationStd: float = 0.001,
                 gate_width: float = 1.25,
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        # Note that the length of obstaclesCenterPosition and obstaclesOrientation should be equal
        assert len(obstaclesCenterPosition) == len(obstaclesOrientation)

        self.obstaclesCenterPosition = obstaclesCenterPosition
        self.obstaclesOrientation = obstaclesOrientation
        self.obstaclesStd = obstaclesStd
        self.obstacleOrientationStd = obstacleOrientationStd
        self.gate_width = gate_width
        # [WARNING] - spawn_position and orientation
        self.initial_xyzs = initial_xyzs
        self.initial_xyzs_std = initial_xyzs_std
        # use these value to find mean and std deviation from 1000 rollouts
        self.all_vel=[]
        self.all_rpy=[]
        self.all_acc=[]
        self.all_ang_v=[]
        self.all_gate_obs=[]
        self.computeZScore=False

        self.saved_drone_path = np.empty(shape=(0,3), dtype=np.float64)

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        self.obstacleIDs = []
        self.current_obstable_index = 0
        self.circuit_complete = False
        self.crash_location = np.zeros(3)
        self.crashed = False
        self.crashed_into_gate = False
        self.INIT_XYZS = np.random.normal(self.initial_xyzs, self.initial_xyzs_std)
        self.drone_path = np.empty(shape=(0,3), dtype=np.float64)
        super()._housekeeping()

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        n = len(self.obstaclesCenterPosition)

        for i in range(n):
            center = [random.gauss(self.obstaclesCenterPosition[i][0], self.obstaclesStd),
                      random.gauss(self.obstaclesCenterPosition[i][1], self.obstaclesStd),
                      random.gauss(self.obstaclesCenterPosition[i][2], self.obstaclesStd)]

            orientation = [random.gauss(self.obstaclesOrientation[i][0], self.obstacleOrientationStd),
                           random.gauss(self.obstaclesOrientation[i][1], self.obstacleOrientationStd),
                           random.gauss(self.obstaclesOrientation[i][2], self.obstacleOrientationStd)]
            
            if center[2] < self.gate_width / 2:
                center[2] = self.gate_width / 2

            obstacle_ID_list =_build_gate(center, p.getQuaternionFromEuler(orientation), client=self.CLIENT)

            self.obstacleIDs += obstacle_ID_list

    def _getGatesStateVector(self, nth_drone):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
            [o_1, alpha_1, o_2, alpha_2]
            sperical coordinate o_1, o_2 = (p_r, p_theta, p_phi),
            alpha_1, alpha_2 = angle between gate normal and the vector pointing from the quadrotor to the center of gate i
        """
        # Get the drone pose
        drone_state = self._getDroneStateVector(nth_drone)
        drone_center = drone_state[0:3]
        drone_rotation_matrix = R.from_quat(drone_state[3:7]).as_matrix()

        current_gate_index = self.current_obstable_index
        next_gate_index = self.current_obstable_index + 1
        currentGatePos = self.obstaclesCenterPosition[current_gate_index]
        currentGateOri = self.obstaclesOrientation[current_gate_index]
        current_gate_rotation_matrix = R.from_euler('xyz', currentGateOri).as_matrix()

        if next_gate_index >= len(self.obstaclesCenterPosition):
            next_gate_rotation_matrix = current_gate_rotation_matrix
            nextGatePos = np.array(currentGatePos) + (current_gate_rotation_matrix @ np.array([[0], [1], [0]])).flatten()
        else:
            nextGatePos = self.obstaclesCenterPosition[next_gate_index]
            nextGateOri = self.obstaclesOrientation[next_gate_index]
            next_gate_rotation_matrix = R.from_euler('xyz', nextGateOri).as_matrix()

        # consider there is a vector p from plane coordinate system to gate center,
        vec_p = np.array(currentGatePos) - np.array(drone_center)

        # get p_r: The distance between drone center and gate center
        p_r = np.linalg.norm(vec_p)

        # get p_theta:
        p_along_z_axis = np.dot(drone_rotation_matrix[:, 2], vec_p) * drone_rotation_matrix[:, 2]
        p_along_xy_plane = vec_p - p_along_z_axis
        p_theta = np.arccos(np.dot(p_along_xy_plane, vec_p) / np.linalg.norm(vec_p) / np.linalg.norm(p_along_xy_plane))

        # get p_phi:
        p_phi = np.arccos(np.dot(drone_rotation_matrix[:, 2], vec_p) / np.linalg.norm(vec_p))

        # get alpha_1: Note that the y axis of gate is the normal of the gate
        alpha_1 = np.arccos(np.dot(current_gate_rotation_matrix[:, 1], vec_p) / np.linalg.norm(vec_p))

        # consider there is a vector q from first gate coordinate system to second gate center,
        q = np.array(nextGatePos) - np.array(currentGatePos)

        # get p_r: The distance between drone center and gate center
        q_r = np.linalg.norm(q)

        # get q_theta:
        q_along_z_axis = np.dot(current_gate_rotation_matrix[:, 2], q) * current_gate_rotation_matrix[:, 2]
        q_along_xy_plane = q - q_along_z_axis
        q_theta = np.arccos(np.dot(q_along_xy_plane, q) / np.linalg.norm(q) / np.linalg.norm(q_along_xy_plane))

        # get q_phi:
        q_phi = np.arccos(np.dot(current_gate_rotation_matrix[:, 2], q) / np.linalg.norm(q))

        # get alpha_2: Note that the y axis of gate is the normal of the gate
        alpha_2 = np.arccos(np.dot(next_gate_rotation_matrix[:, 1], q) / np.linalg.norm(q))

        return np.array([p_r, p_theta, p_phi, alpha_1, q_r, q_theta, q_phi, alpha_2])

    ################################################################################

    def _observationSpace(self):
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # Observation vector ###   v_x   v_y   v_z   a_x   a_y   a_z   r11   r12   r13   r21   r22   r23  r31   r32   r33   w_1   w_2   w_3   p_r   p_th  p_phi a_1   q_r   q_th  q_phi  a_2
        obs_lower_bound = np.array([-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   0,    0,    0,    0,    0,    0])
        obs_upper_bound = np.array([ 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   1,    1,    1,    1,    1,    1])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################

    def _computeObs(self):
        """Computes the current observation.

        Returns
        -------
        float
            quadrotor state: [quadrotor linear velocity, linear acceleration, rotation matrix, and angular velocity]
            gate state: [quadrotor linear velocity, linear acceleration, rotation matrix, and angular velocity]

        """
        nth_drone = 0
        rotation_matrix = R.from_quat(self.quat[nth_drone, :]).as_matrix()
        quad_state = np.hstack((self.vel[nth_drone, :], self.acc[nth_drone, :], rotation_matrix.flatten(), self.ang_v[nth_drone, :]))

        gate_state = self._getGatesStateVector(nth_drone)

        full_state = np.hstack((quad_state, gate_state))

        ############################################################
        #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        #### Observation vector ###   v_x   v_y   v_z   a_x   a_y   a_z   r11   r12   r13   r21   r22   r23  r31   r32   r33   w_1   w_2   w_3   p_r   p_th  p_phi a_1   q_r   q_th  q_phi a_2
        # obs_lower_bound = np.array([-1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,   -1,    0,    0,   0,    0,    0,    0,    0,    0])
        # obs_upper_bound = np.array([ 1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,   1,    1,    1,    1,    1,    1])
        ############################################################
        if self.computeZScore:
            return full_state
        return self.zNormalizeState(full_state)

    ################################################################################

    def _cleared_obstacle(self):
        current_gate_center = self.obstaclesCenterPosition[self.current_obstable_index]
        current_gate_rotation = R.from_euler('xyz', self.obstaclesOrientation[self.current_obstable_index]).as_matrix()

        plane_normal = current_gate_rotation[:,1]
        d = -np.dot(plane_normal, current_gate_center)

        pos = self.pos[0]

        if np.dot(plane_normal, pos) + d > 0:
            self.current_obstable_index += 1

        if self.current_obstable_index == len(self.obstaclesCenterPosition):
            self.circuit_complete = True

        contact_information = p.getContactPoints(physicsClientId=self.CLIENT)
        if contact_information:
            body_ids = contact_information[0][1:3]
            if any(item in self.obstacleIDs for item in body_ids):
                self.crashed_into_gate = True
            self.crashed = True
            self.crash_location = (np.array(contact_information[0][5]) + np.array(contact_information[0][6]))/2

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        # Check if the obstable has been cleared and change the current obstacle.
        self._cleared_obstacle()

        if self.circuit_complete:
            return 0.0

        if self.current_obstable_index-1 < 0:
            prev_gate_center = self.INIT_XYZS[0]
            prev_gate_rotation = self.INIT_RPYS[0]
        else:
            prev_gate_center = np.array(self.obstaclesCenterPosition[self.current_obstable_index-1])  # prev_gate_center centre
            prev_gate_rotation = R.from_euler('xyz', self.obstaclesOrientation[self.current_obstable_index-1]).as_matrix()

        # print("self.current_obstable_index:",self.current_obstable_index)
        current_gate_center = np.array(self.obstaclesCenterPosition[self.current_obstable_index])  # current_gate_center centre
        # print("current_gate_center:", current_gate_center)
        current_gate_rotation = R.from_euler('xyz', self.obstaclesOrientation[self.current_obstable_index]).as_matrix()
        # print("current_gate_rotation:", current_gate_rotation)

        if self.computeZScore:
            self.all_vel.append(self.vel[0])
            self.all_acc.append(self.acc[0])
            self.all_rpy.append(self.rpy[0])
            self.all_ang_v.append(self.ang_v[0])
            self.all_gate_obs.append(self._getGatesStateVector(0))
            print("vel\t",np.mean(self.all_vel,0),"\n",np.std(self.all_vel,0))
            print("acc\t",np.mean(self.all_acc,0),"\n",np.std(self.all_acc,0))   
            print("orientation\t",np.mean(self.all_rpy,0),"\n",np.std(self.all_rpy,0)) 
            print("angl_vel\t",np.mean(self.all_ang_v,0),"\n",np.std(self.all_ang_v,0))
            print("gate obs\t",np.mean(self.all_gate_obs,0),"\n",np.std(self.all_gate_obs,0))
        
    
        wt = self.ang_v[0]  # body rates

        p = self.pos[0]  # current position
        p_prev = self.prev_pos[0]  # previous position
        wg = self.gate_width  # side length of the rectangular gate

        dmax = 2.5  # specifies a threshold on the distance to the gate center in order to activate the safety reward
        a = 0.15  # hyperparameter that trades off between progress maximization and risk minimization
        b = -0.0  # weight for penalty body rate

        reward = final_reward(p, p_prev, prev_gate_center, current_gate_center, prev_gate_rotation, current_gate_rotation, a, b, dmax, wt, wg, crashed=self.crashed, crash_location=self.crash_location)
        # print("reward:", reward)
        return reward

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        elif self.circuit_complete:
            return True
        elif self.crashed:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  # Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################

    def zNormalizeState(self, state):
        """Z-Normalizes a drone's state.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        STD_LIN_ACC_XY = 40
        STD_LIN_ACC_Z = 20

        STD_ANG_VEL_XY = 5
        STD_ANG_VEL_Z = 2

        STD_ANGLE = np.pi

        vel_mean =np.array([0.0, 0.0, 0.0])
        vel_std=np.array([2.29446675, 2.11655072, 1.51620822])
        acc_mean=np.array([0.0, 0.0, 0.0])
        acc_std=np.array([STD_LIN_ACC_XY, STD_LIN_ACC_XY, STD_LIN_ACC_Z])
        # rpy_mean=[0.18491563, -0.11516139, -0.37774102] 
        # rpy_std=[2.50923123, 0.51063424, 1.36510486]
        ang_vel_mean=np.array([0.0, 0.0, 0.0])
        ang_vel_std=np.array([STD_ANG_VEL_XY, STD_ANG_VEL_XY, STD_ANG_VEL_Z])

        gate_obs_mean= np.array([2.0, 0.49788855, 1.33366795, 0.36415743]) 
        gate_obs_std=np.array([4.37988081e-01, 3.95491805e-01, 5.89980432e-01, 3.34656134e-01])

        pr_mean = 2.0
        qr_mean = 2.0
        pr_std = 2.0
        qr_std = 2.0

        # gate_center_mean=np.array([0. ,        1.50585938, 1.        ]) 
        # gate_center_std=np.array([0.,         0.09356672, 0.,        ])
        # gate_rotation_mean=np.ndarray([[1., 0., 0.],[0., 1., 0.],[0., 0.,1.]]) 
        # gate_rotation_std=np.ndarray([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]])

        normalized_vel = (state[0:3]-vel_mean)/vel_std
        normalized_acc = (state[3:6]-acc_mean)/acc_std
        # rpy_mean = R.from_euler('xyz', rpy_mean).as_matrix().flatten()
        # rpy_std=R.from_euler('xyz', rpy_std).as_matrix().flatten()
        # normalized_rpy=(state[6:15]-rpy_mean)/rpy_std
        normalized_ang_vel=(state[15:18]-ang_vel_mean)/ang_vel_std
        # normalized_gate_obs=(state[18:22]-gate_obs_mean)/gate_obs_std

        normalized_p_r = (state[18] - pr_mean) / pr_std
        normalized_p_theta = state[19] / STD_ANGLE
        normalized_p_phi = state[20] / STD_ANGLE
        normalized_a_1 = state[21] / STD_ANGLE
        normalized_q_r = (state[22] - qr_mean) / qr_std
        normalized_q_theta = state[23] / STD_ANGLE
        normalized_q_phi = state[24] / STD_ANGLE
        normalized_a_2 = state[25] / STD_ANGLE

        normalized_state = np.hstack([normalized_vel,
                                      normalized_acc,
                                      state[6:15],
                                      normalized_ang_vel,
                                      normalized_p_r,
                                      normalized_p_theta,
                                      normalized_p_phi,
                                      normalized_a_1,
                                      normalized_q_r,
                                      normalized_q_theta,
                                      normalized_q_phi,
                                      normalized_a_2,
                                      ]).reshape(26,)

        return normalized_state

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z * self.EPISODE_LEN_SEC

        MAX_LIN_ACC_XY = 40
        MAX_LIN_ACC_Z = 20

        MAX_PITCH_ROLL = np.pi  # Full range

        MAX_DIST_TO_GATE = 5

        # v_x   v_y
        clipped_vel_xy = np.clip(state[0:2], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        # v_z
        clipped_vel_z = np.clip(state[2], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        # a_x   a_y
        clipped_acc_xy = np.clip(state[3:5], -MAX_LIN_ACC_XY, MAX_LIN_ACC_XY)
        # a_z
        clipped_acc_z = np.clip(state[5], -MAX_LIN_ACC_Z, MAX_LIN_ACC_Z)
        # w_1   w_2   w_3

        # p_r  p_th  p_phi a_1   q_r   q_th  q_phi a_2
        clipped_p_r = np.clip(state[18], 0, MAX_DIST_TO_GATE)
        clipped_q_r = np.clip(state[22], 0, MAX_DIST_TO_GATE)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_vel_xy,
                                               clipped_vel_z,
                                               clipped_acc_xy,
                                               clipped_acc_z,
                                               clipped_p_r,
                                               clipped_q_r
                                               )

        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_acc_xy = clipped_acc_xy / MAX_LIN_ACC_XY
        normalized_acc_z = clipped_acc_z / MAX_LIN_ACC_Z
        # I don't know why this is done
        normalized_ang_vel = state[15:18] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_p_r = clipped_p_r / MAX_DIST_TO_GATE
        normalized_p_theta = state[19] / MAX_PITCH_ROLL
        normalized_p_phi = state[20] / MAX_PITCH_ROLL
        normalized_a_1 = state[21] / MAX_PITCH_ROLL
        normalized_q_r = clipped_q_r / MAX_DIST_TO_GATE
        normalized_q_theta = state[23] / MAX_PITCH_ROLL
        normalized_q_phi = state[24] / MAX_PITCH_ROLL
        normalized_a_2 = state[25] / MAX_PITCH_ROLL

        norm_and_clipped = np.hstack([normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_acc_xy,
                                      normalized_acc_z,
                                      state[7:16],
                                      normalized_ang_vel,
                                      normalized_p_r,
                                      normalized_p_theta,
                                      normalized_p_phi,
                                      normalized_a_1,
                                      normalized_q_r,
                                      normalized_q_theta,
                                      normalized_q_phi,
                                      normalized_a_2,
                                      ]).reshape(26,)

        return norm_and_clipped

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      clipped_acc_xy,
                                      clipped_acc_z,
                                      clipped_p_r,
                                      clipped_q_r,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_vel_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_vel_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[2]))
        if not(clipped_acc_xy == np.array(state[3:5])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy accleration [{:.2f} {:.2f}]".format(state[3], state[4]))
        if not(clipped_acc_z == np.array(state[5])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z accleration [{:.2f}]".format(state[5]))
        if not(clipped_p_r == np.array(state[18])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped p_r [{:.2f}]".format(state[18]))
        if not(clipped_q_r == np.array(state[22])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped q_r [{:.2f}]".format(state[22]))