'''
Everything here is garbage to throw away.


Transferring between lab coordinates and fly (rotational stages)
coordinates.
'''

from math import pi
import numpy as np
from numpy import sin, cos, sqrt, tan
from scipy.linalg import solve

import matplotlib.pyplot as plt

def A_matrix(fi, th):
    A = np.array(
            [[cos(fi), - sin(fi)*cos(th), cos(fi)*sin(th)],
            [sin(fi), cos(fi)*cos(th), - sin(fi)*sin(th)],
            [0, -sin(th), cos(th)]])
    return A



def B_matrix(fi, th): 
    B = np.array(
            [[cos(fi), sin(fi)*cos(th), sin(fi)*sin(th)],
            [-sin(fi), cos(fi)*cos(th), cos(fi)*sin(th)],
            [0, -sin(th), cos(th)]])
    return B



def fly2LabCoordinates(xf, yf, zf, fi, th):
    '''
    xf,yf,zf        Cartesian in fly coordinate system
    fi              Horizontal angle (in radians)
    th              Fly pitch angle (in radians)

    fi and th rotate fly coordinate system in 3d space as in
    the pupil imaging setup (fly bench).
    
    Return x,y,z in lab coordinates where the camera is facing
    towards the positive y-axis.
    '''
    
    #x = cos(fi)*xf - sin(fi)*cos(th)*yf + sin(fi)*sin(th)*zf
    #y = sin(fi)*xf + cos(fi)*cos(th)*yf - sin(fi)*sin(th)*zf
    #z = sin(th)*(sin(th)*sqrt(1-sin(fi)**2*cos(th)**2))*yf + cos(fi)*(cos(th)*sqrt(1-sin(fi)**2*sin(th)**2))*zf

    #WRONG ONES TO GIVE CRAZY PATTERN
    #z = sin(th)*(sin(th)+sin(fi)*cos(th))*yf + cos(fi)*(sin(th)+sin(fi)*cos(th))*zf
    #z = sin(th)*cos(fi)*yf + cos(fi)*cos(th)*zf
    
    # Matrix formulation
    return np.matmul(A_matrix(-fi, -th), np.array([xf, yf, zf]))

def lab2FlyCoordinates(xl, yl, zl, fi, th):
    return np.matmul(B_matrix(-fi, -th), np.array([xl, yl, zl]))


def lin(x):
    '''
    like sine but triangle waveform
    x -> -1...+1
    '''
    period = 2*pi
    
    sign = np.sign(x)
    x = abs(x)

    # Force x between interval 0...period
    if x > period:
        x -= int(x/period) * period

    if 0 <= x < period/4:
        mag = x/(period/4)

    if period/4 <= x < 3*period/4:
        mag = 1 - (x-(period/4))/(period/4)
    
    if 3*period/4 <= x:
        mag = -1 + (x-(3*period/4))/(period/4)
    
    return sign * mag

def cameraInFlyCoordinates(xi, yi, horizontal, pitch):
    '''
    Transform camera coordinates

    INPUT ARGUMENTS
    xi, yi                  Vector endpoints in x and y in camera coordinates
                            
                            Image coordinate system
                             - - - - >
                            |       X-axis
                            |
                            |
                            v  Y-axis


    horizontal, pitch       Conjumeter positions, in radians

    Returns vector x,y,z in fly's coordinate system.
    '''
    
    
    cam_horizontal = -horizontal + pi/2  #* cos(pitch) + sin(pitch) * pi/2
    cam_pitch = -pitch * lin(horizontal) + (pi/2)
    rotation = sin(horizontal) * pitch
    
    # Neutralize rotation
    xi_rotated = cos(rotation) * xi - sin(rotation) * yi
    yi_rotated = sin(rotation) * xi + cos(rotation) * yi
    
    r = 10
    x0 = r * sin(cam_pitch) * cos(cam_horizontal)
    y0 = r * sin(cam_pitch) * sin(cam_horizontal)
    z0 = r * cos(cam_pitch)
    
    #x0, y0, z0 = fly2LabCoordinates(0, 0, 1, cam_horizontal, cam_pitch)
    
    #x1 =
    #y1 = r * sin(cam_pitch) * sin(cam_horizontal)
    #z1 = r * cos(cam_pitch)
    
    x1, y1, z1 = [0,0,0]

    #x = cos(cam_horizontal) * -xi
    #y = sin(cam_horizontal) * xi
    #z = yi_rotated * sin(cam_pitch)


    return [x0, y0, z0], [x1, y1, z1]


def cameraInFlyTest():
    
    horizontal = np.radians(np.linspace(-180,180,100))
    vertical = np.radians(np.linspace(80,80, 100))


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
 
    XF,YF,ZF = [[],[],[]]
    for h,v in zip(horizontal, vertical):
        print("{},{}".format(h,v))
        start, end = cameraInFlyCoordinates(0,0, h, v)
        xf, yf, zf = start
        XF.append(xf)
        YF.append(yf)
        ZF.append(zf)
    
    
    ax.scatter(XF,YF,ZF)
    plt.show()

def lintest():
    xx = []
    X = np.linspace(-2*pi, 2*pi, 300)
    for x in X:
        xx.append(lin(x))

    plt.plot(xx)
    plt.show()

def unit_test1():
    th = np.radians(1)
    fi = np.radians(1)

    f = [5,3,2]
    print(f)
    
    l = fly2LabCoordinates(*f, fi, th)
    print(l)



def unit_test2():


    from mpl_toolkits.mplot3d import Axes3D

    th = np.radians(np.linspace(0,360))
    fi = np.radians(np.linspace(0,180))

    angles = []

    for t in th:
        for f in fi:
            angles.append((t,f))

    XF,YF,ZF = [[],[],[]]
    for t,f in angles:
        xf,yf,zf = fly2LabCoordinates(1, 1, 1, f,t)
        XF.append(xf)
        YF.append(yf)
        ZF.append(zf)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(XF,YF,ZF)
    plt.show()

def unit_test3():
    th = np.radians(90)
    fi = np.radians(0)


    f = [1,1,1]
    print(f)
    
    bf = lab2FlyCoordinates(*f, fi, th)
    print(bf)


if __name__ =='__main__':
    cameraInFlyTest()
    #lintest()

