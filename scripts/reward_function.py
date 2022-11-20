import numpy as np
import math

g1 = [0.0, 0.0, 0.0]  # gate1 centre
g2 = [0.0, 0.0, 0.0]  # gate2 centre
dp = 0.0   # distance to the gate plane
dn = 0.0  # distance of the quadrotor to the gate normal
dmax = 0.0  # specifies a threshold on the distance to the gate center in order to activate the safety reward
wg = 0.0  # side length of the rectangular gate
a = 0.0  # hyperparameter that trades off between progress maximization and risk minimization
b = 0.0  # weight for penalty w
p = [0.0, 0.0, 0.0]  # current position
wt = [0.0, 0.0, 0.0]  # body rates


def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2 + (point1[2] - point2[2])**2)


def s(p):
    '''
    Defines the progress along the path segment that connects the previous gate center g1 with the next gate center g2.
    '''
    return (p - g1) * (g2 - g1) / euclidean_distance(g1, g2)


def rp():
    '''
    Progress Reward
    '''
    return s(p[t]) - s(p[t - 1])


def rs():
    '''
    Safety Reward
    '''
    f = max(1 - dp / dmax, 0.0)
    v = max((1 - f) * wg / 6, 0.05)
    return -f**2 * (1 - math.exp(-0.5 * dn**2 / v))


def collision_check(p,gate):

    return [False,p_crash,gate]


def wb():
    '''
    Body rate
    '''
    return [0.0, 0.0, 0.0]


def r():
    '''
    Final Reward
    p_crash position of crash
    g = position of center of gate where crash occured
    wt=body rates
    '''

    wt_mag = euclidean_distance(wt, 0)
    reward = rp() + a * rs() - b * wt_mag
    if collision_check():
        dg = euclidean_distance(p_crash, g)
        rt = -min((dg / wg)**2, 20.0)
        return reward + rt
    else:
        return reward
