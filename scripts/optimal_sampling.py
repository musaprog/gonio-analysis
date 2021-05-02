
from math import sin, cos, tan, sqrt, radians, pi

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from gonioanalysis.coordinates import camera2Fly, optimal_sampling


def test():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    verticals = [1, 30, 45, 60, 90]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    for vertical, color in zip(verticals, colors):
        horizontals = np.linspace(-90, 90)

        for horizontal in horizontals:
            x,y,z = camera2Fly(horizontal, vertical)
            ax.scatter(x,y,z, color=color)
        
    plt.show()

def plot_optimal(points):


    print(len(points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*list(zip(*points)))
    
    plt.show()



if __name__ == "__main__":

    steps = (10, 10)


    # For Gabor
    #horizontals = np.arange(-90,40+0.01, steps[0])
    #verticals = np.arange(-20,180+0.01, steps[1])
    
    # For vector map
    horizontals = np.arange(-50,40+0.01, steps[0])
    verticals = np.arange(-20,180+0.01, steps[1])
    
    # Full sphere just for illustration
    #horizontals = np.arange(-90,90+0.01, steps[0])
    #verticals = np.arange(0,360+0.01, steps[1])
 
    points = optimal_sampling(horizontals, verticals, steps)
    plot_optimal(points)
