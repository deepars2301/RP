import numpy as np
import matplotlib.pyplot as plt
from mavion_model import Mavion

rad2deg = 180/np.pi


if __name__ == "__main__":

    mavion = Mavion()

    vit = np.arange(0, 30, 1)
    rot = np.arange(0, 1, 0.1)

    for v in vit:
        dx, de, theta = mavion.trim([v, 0])
        theta = (theta + np.pi) % (2*np.pi) - np.pi
        print(dx, de, theta*rad2deg)
        print("------------------------------------")

        for om in rot :
            phi = np.arctan(v*om/9.81)
            print(phi*rad2deg)

            dx1, dx2, de1, de2, theta, phi = mavion.trim_turn([v, om])
            print(dx1, dx2, de1, de2)
            theta = (theta + np.pi) % (2*np.pi) - np.pi
            phi = (phi + np.pi/2) % (np.pi) - np.pi/2
            print(theta*rad2deg, phi*rad2deg)



        