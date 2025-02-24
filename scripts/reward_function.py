import numpy as np
from scipy.spatial.transform import Rotation as R

def dp(p, g, gm):
    '''
    distance to the gate plane
    '''
    vector_pg = g - p
    gate_y_axis = gm[:, 1] # Should be a unit vector.

    return np.dot(vector_pg, gate_y_axis) / np.linalg.norm(gate_y_axis)


def dn(p, g, gm):
    '''
    distance of the quadrotor to the gate normal
    '''
    vector_pg = g - p
    gate_y_axis = gm[:, 1] # Should be a unit vector.
    vector_pg_along_gate_y_axis = dp(p, g, gm) * gate_y_axis

    return np.linalg.norm(vector_pg - vector_pg_along_gate_y_axis)

def s(p, g1, g2):
    '''
    Defines the progress along the path segment that connects the previous gate center g1 with the next gate center g2.
    p:position
    '''
    # TODO: find previous gate and next gate
    return np.dot(p - g1, g2 - g1) / np.linalg.norm(g1 - g2)


def progress_reward(p, p_prev, g1, g2):
    '''
    Progress Reward
    Increases when the drone move in the direction joining gate1 and gate2
    '''
    return s(p, g1, g2) - s(p_prev, g1, g2)


def safety_reward(p, g, gm, dmax, wg):
    '''
    Safety Reward (only activated when drone is within threshold dmax) 
    Value lies between 0 and -1.
    increases when position is away from normal
    '''
    f = max(1 - dp(p, g, gm) / dmax, 0.0)
    v = max((1 - f) * wg / 6, 0.05)
    return -f**2 * (1 - np.exp(-0.5 * dn(p, g, gm)**2 / v))


def final_reward(p, p_prev, g1, g2, gm1, gm2, a, b, dmax, wt, wg, crashed, crash_location, debug=False):
    '''
    Final Reward
    p_crash position of crash
    g = position of center of gate where crash occured
    wt=body rates
    '''

    wt_mag = np.linalg.norm(wt)
    if debug:
        print("progress_reward:", progress_reward(p, p_prev, g1, g2))
        print("safety_reward:", safety_reward(p, g2, gm2, dmax, wg))
        print("wt_mag", wt_mag)
    reward = progress_reward(p, p_prev, g1, g2) + a * safety_reward(p, g2, gm2, dmax, wg) + b * wt_mag
    if crashed:
        dg = np.linalg.norm(p - g2)
        terminal_reward = -min((dg / wg)**2, 20.0)
        print("terminal_reward", terminal_reward)
        return reward + terminal_reward
    else:
        return reward

if __name__ == "__main__":
    prev_gate_center = np.array([20.0, 5.0, 3.0])  # gate1 centre (The gate it has cleared)
    current_gate_center = np.array([15.0, 40.0, 3.0])  # gate2 centre (The gate in front of it)
    prev_gate_rotation = R.from_euler('xyz', np.array([0.0, 0.0, 0.0])).as_matrix()
    current_gate_rotation = R.from_euler('xyz', np.array([0.0, 0.0, np.pi/4])).as_matrix()
    ang_vel = np.array([0.0, 0.0, 0.0])  # body rates
    debug = True
    
    p = np.array([25.0, 7.0, 3.0])  # current position
    p_prev = np.array([20.0, 5.0, 3.0])  # previous position
    wg = 0.75  # side length of the rectangular gate
    crash_location = np.array([0, 4, 1])

    dmax = 2  # specifies a threshold on the distance to the gate center in order to activate the safety reward
    a = 0.5  # hyperparameter that trades off between progress maximization and risk minimization
    b = -0.0  # weight for penalty body rate

    if debug:
        print('previous gate: dp(p, g, gm):', dp(p, prev_gate_center, prev_gate_rotation))
        print('previous gate: dn(p, g, gm):', dn(p, prev_gate_center, prev_gate_rotation))
        print('current gate: dp(p, g, gm):', dp(p, current_gate_center, current_gate_rotation))
        print('current gate: dn(p, g, gm):', dn(p, current_gate_center, current_gate_rotation))
        print(final_reward(p=p, 
                           p_prev=p_prev, 
                           g1=prev_gate_center, 
                           g2=current_gate_center, 
                           gm1=prev_gate_rotation, 
                           gm2=current_gate_rotation, 
                           a=a, 
                           b=b, 
                           dmax=dmax, 
                           wt=ang_vel, 
                           wg=wg, 
                           crashed=False, 
                           crash_location=crash_location, 
                           debug=debug))
