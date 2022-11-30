import numpy as np

# g1 = np.array([0, 1, 5])  # gate1 centre
# g2 = np.array([0, 4, 5])  # gate2 centre
# # gm1 = np.array([[1, 2, 0], [1, 2, 0], [1, 2, ]])
# gm1 = np.identity(3)
# gm2 = np.identity(3)
# # gm2 = np.array([[1, 2, 5], [1, 6, 3], [8, 2, 3]])
# wt = np.array([1, 0, 0])  # body rates

# p = np.array([0, 2, 5])  # current position
# p_prev = np.array([0, 0, 0])  # previous position
# wg = 0.75  # side length of the rectangular gate


# dmax = 2  # specifies a threshold on the distance to the gate center in order to activate the safety reward
# a = 2  # hyperparameter that trades off between progress maximization and risk minimization
# b = -0.5  # weight for penalty body rate


def dp(p, g, gm):
    '''
    distance to the gate plane
    '''
    return np.linalg.norm(np.dot(g - p, gm[:, 1]) / (np.linalg.norm(p - g)) * gm[:, 1])


def dn(p, g, gm):
    '''
    distance of the quadrotor to the gate normal
    '''
    return np.linalg.norm(g - p - np.dot(g - p, gm[:, 1]) / (np.linalg.norm(p - g)) * gm[:, 1])


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


def rp(p, p_prev, g1, g2):
    '''
    Progress Reward
    Increases when the 
    '''
    return s(p, g1, g2) - s(p_prev, g1, g2)


def rs(p, g, gm, dmax, wg):
    '''
    Safety Reward (only activated when drone is within threshold dmax) 
    Value lies between 0 and -1.
    increases when position is away from normal
    '''
    f = max(1 - dp(p, g, gm) / dmax, 0.0)
    v = max((1 - f) * wg / 6, 0.05)
    return -f**2 * (1 - np.exp(-0.5 * dn(p, g, gm)**2 / v))


def final_reward(p, p_prev, g1, g2, gm1, gm2, a, b, dmax, wt, wg):
    '''
    Final Reward
    p_crash position of crash
    g = position of center of gate where crash occured
    wt=body rates
    '''

    wt_mag = np.linalg.norm(wt)
    g, gm = nearest_gate(p, g1, g2, gm1, gm2)
    print("rp", rp(p, p_prev, g1, g2))
    print("rs", rs(p, g, gm, dmax, wg))
    reward = rp(p, p_prev, g1, g2) + a * rs(p, g, gm, dmax, wg) + b * wt_mag
    if collision_check(p):
        dg = np.linalg.norm(p - g)
        rt = -min((dg / wg)**2, 20.0)
        return reward + rt
    else:
        return reward


print(final_reward(p, p_prev, g1, g2, gm1, gm2, a, b, dmax, wt, wg))
