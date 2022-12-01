import numpy as np

def dp(p, g, gm):
    '''
    distance to the gate plane
    '''
    vector_pg = g - p
    gate_y_axis = gm[:, 1] # Should be a unit vector.

    return np.linalg.norm(np.dot(vector_pg, gate_y_axis)) / np.linalg.norm(gate_y_axis)


def dn(p, g, gm):
    '''
    distance of the quadrotor to the gate normal
    '''
    vector_pg = g - p
    gate_y_axis = gm[:, 1] # Should be a unit vector.
    vector_pg_along_gate_y_axis = dp(p, g, gm) * gate_y_axis

    return np.linalg.norm(vector_pg - vector_pg_along_gate_y_axis)


def nearest_gate(p, g1, g2, gm1, gm2):
    '''
    Caluculates the nearest gate
    '''
    if np.linalg.norm(p - g1) <= np.linalg.norm(p - g2):
        return g1, gm1
    else:
        return g2, gm2


def collision_check(p):
    return False


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


def final_reward(p, p_prev, g1, g2, gm1, gm2, a, b, dmax, wt, wg, debug=False):
    '''
    Final Reward
    p_crash position of crash
    g = position of center of gate where crash occured
    wt=body rates
    '''

    wt_mag = np.linalg.norm(wt)
    g, gm = nearest_gate(p, g1, g2, gm1, gm2)
    if debug:
        print("progress_reward:", progress_reward(p, p_prev, g1, g2))
        print("safety_reward:", safety_reward(p, g, gm, dmax, wg))
        print("wt_mag", wt_mag)
    reward = progress_reward(p, p_prev, g1, g2) + a * safety_reward(p, g, gm, dmax, wg) + b * wt_mag
    if collision_check(p):
        dg = np.linalg.norm(p - g)
        rt = -min((dg / wg)**2, 20.0)
        return reward + rt
    else:
        return reward

if __name__ == "__main__":
    gate1_center = np.array([0, 2, 0.625])  # gate1 centre (The gate it has cleared)
    gate2_center = np.array([0, 4, 0.625])  # gate2 centre (The gate in front of it)
    gate1_rotation = np.identity(3)
    gate2_rotation = np.identity(3)
    ang_vel = np.array([0.0, 0.0, 0.0])  # body rates
    debug = True

    p = np.array([0.1, 1.0, 0.625])  # current position
    p_prev = np.array([0, 0, 0.625])  # previous position
    wg = 0.75  # side length of the rectangular gate

    dmax = 2  # specifies a threshold on the distance to the gate center in order to activate the safety reward
    a = 2  # hyperparameter that trades off between progress maximization and risk minimization
    b = -0.5  # weight for penalty body rate

    if debug:
        print('dp(p, g, gm):', dp(p, gate1_center, gate1_rotation))
        print('dn(p, g, gm):', dn(p, gate1_center, gate1_rotation))
        print(final_reward(p, p_prev, gate1_center, gate2_center, gate1_rotation, gate2_rotation, a, b, dmax, ang_vel, wg, debug=debug))
