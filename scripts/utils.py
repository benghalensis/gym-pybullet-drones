import numpy as np

class Spherical:
    def __init__(self,r,theta,z):
        self.r=r
        self.theta=theta
        self.z=z
    def magnitude(self):
        return np.sqrt(self.rr**2+self.z**2)
    