
from math import sin, cos, tan, sqrt, radians, pi

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from pupilanalysis.coordinates import camera2Fly


def distance(a, b):
    '''
    Calculates distance between two points in 3D cartesian space.
    a,b      (x,y,z)
    '''

    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)



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


def optimal(horizontals, verticals):
    '''
    Determine optimal way to sample using two orthogonal goniometers.
    '''
    
    steps = ((horizontals[1]-horizontals[0]), (verticals[1]-verticals[0]))

    min_distance = 0.75 * distance(camera2Fly(steps[0], steps[1]), camera2Fly(0,0))

  
    goniometer_vals = {}
    
    points = []

    for vertical in verticals:
        goniometer_vals[vertical] = []
        for horizontal in horizontals:
            point = camera2Fly(horizontal, vertical)
            
            append = True
            
            for previous_point in points:
                if distance(previous_point, point) < min_distance:
                    append = False
                    break

            if append:
                points.append(point)
                goniometer_vals[vertical].append(horizontal)

    #for hor, vers in sorted(goniometer_vals.items(), key=lambda x: int(x[0])):
    #    print('{}: {}'.format(hor, vers))
    
    return np.array(points)

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
 
    points = optimal(horizontals, verticals, steps)
    plot_optimal(points)
