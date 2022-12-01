import os
import numpy as np
import pybullet as p
import pkg_resources
import random
from scipy.spatial.transform import Rotation as R

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from reward_function import final_reward


class RacingDroneAviary(BaseSingleAgentAviary):
    """Single agent RL problem: fly through a gate."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
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
        # Note that the length of obstaclesCenterPosition and obstaclesOrientation should be equal
        assert len(obstaclesCenterPosition) == len(obstaclesOrientation)
        self.obstaclesCenterPosition = obstaclesCenterPosition
        self.obstaclesOrientation = obstaclesOrientation
        self.obstaclesStd = obstaclesStd
        self.obstacleIDs = []
        self.gate_width = gate_width
        self.current_obstable_index = 0

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Extends the superclass method and add the gate build of cubes and an architrave.

        """
        # super()._addObstacles() # Has functions relate to Camera on the drone
        n = len(self.obstaclesCenterPosition)

        for i in range(n):
            center = [random.gauss(self.obstaclesCenterPosition[i][0], self.obstaclesStd),
                      random.gauss(self.obstaclesCenterPosition[i][1], self.obstaclesStd),
                      random.gauss(self.obstaclesCenterPosition[i][2], self.obstaclesStd)]

            if center[2] < self.gate_width / 2:
                center[2] = self.gate_width / 2

            obstacleID = p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', "assets/gates.urdf"),
                                    center,
                                    p.getQuaternionFromEuler(self.obstaclesOrientation[i]),
                                    useFixedBase=1,
                                    physicsClientId=self.CLIENT
                                    )
            self.obstacleIDs.append(obstacleID)

    def _getGatesStateVector(self, nth_drone, gateIDs):
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
        drone_center = drone_state[0, 3]
        drone_rotation_matrix = R.from_quat(drone_state[3, 7]).as_matrix()

        # Get the gate pose
        currentGateID = gateIDs[0]
        nextGateID = gateIDs[1]
        currentGatePos, currentGateOri = p.getBasePositionAndOrientation(objectUniqueId=currentGateID)
        current_gate_rotation_matrix = R.from_quat(currentGateOri).as_matrix()
        if nextGateID == -999:
            next_gate_rotation_matrix = current_gate_rotation_matrix
            nextGatePos = currentGatePos + current_gate_rotation_matrix @ np.array([[0], [1], [0]])
        else:
            nextGatePos, nextGateOri = p.getBasePositionAndOrientation(objectUniqueId=nextGateID)
            next_gate_rotation_matrix = R.from_quat(nextGateOri).as_matrix()

        # consider there is a vector p from plane coordinate system to gate center,
        p = np.array(currentGatePos) - np.array(drone_center)

        # get p_r: The distance between drone center and gate center
        p_r = np.linalg.norm(p)

        # get p_theta:
        p_along_z_axis = np.dot(drone_rotation_matrix[:, 2], p) * drone_rotation_matrix[:, 2]
        p_along_xy_plane = p - p_along_z_axis
        p_theta = np.arccos(np.dot(p_along_xy_plane, p) / np.linalg.norm(p) / np.linalg.norm(p_along_xy_plane))

        # get p_phi:
        p_phi = np.arccos(np.dot(drone_rotation_matrix[:, 2], p) / np.linalg.norm(p))

        # get alpha_1: Note that the y axis of gate is the normal of the gate
        alpha_1 = np.arccos(current_gate_rotation_matrix[:, 1], p) / np.linalg.norm(p)

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
        alpha_2 = np.arccos(next_gate_rotation_matrix[:, 1], q) / np.linalg.norm(q)

        return np.array([p_r, p_theta, p_phi, alpha_1, q_r, q_theta, q_phi, alpha_2])

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

        # Find the current gate and next gate
        if self.current_obstable_index + 1 < len(self.obstacleIDs):
            gateIDs = [self.obstacleIDs[self.current_obstable_index], self.obstacleIDs[self.current_obstable_index + 1]]
        else:
            gateIDs = [self.obstacleIDs[self.current_obstable_index], -999]

        gate_state = self._getGatesStateVector(nth_drone, gateIDs)

        return np.hstack((quad_state, gate_state))

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        #TODO: get the previous position
        g1 = self.obstaclesCenterPosition[0]  # gate1 centre
        g2 = self.obstaclesCenterPosition[1]  # gate2 centre

        gm1 = self.obstaclesOrientation[0]
        gm2 = self.obstaclesOrientation[1]

        wt = self.ang_v[0]  # body rates

        p = self.pos[0]  # current position
        p_prev = self.prev_pos[0]  # previous position
        wg = self.gate_width  # side length of the rectangular gate

        dmax = 2  # specifies a threshold on the distance to the gate center in order to activate the safety reward
        a = 2  # hyperparameter that trades off between progress maximization and risk minimization
        b = -0.5  # weight for penalty body rate
        
        return final_reward(p, p_prev, g1, g2, gm1, gm2, a, b, dmax, wt, wg)

    ################################################################################

    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        # update current_obstable_index
        if self.step_counter / self.SIM_FREQ > self.EPISODE_LEN_SEC:
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

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped

    def _clipAndNormalizeTrackState(self, track_state):
        # TODO
        # Clip the track states
        pass

    ################################################################################

    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
