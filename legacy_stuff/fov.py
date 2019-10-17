'''
Estimating binocular overlap

TODO
- angle correction if fly wasn't straight in the beginning


'''

import itertools
import csv

import matplotlib.pyplot as plt

from preprocessing import RowAligner
import plotter.default

def arrayToAngles(center_points, N_images_list, pitch_step, horizontal_step):
    '''
    center_points       
    N_images_list       [N_images_stack1, N_mages_stack2, ...]
    '''
    
    angle_pairs = []
    
    for j, cp in enumerate(center_points):
        pitch = pitch_step * j
        
        for i in range(N_images_list[j]):
            horizontal = horizontal_step * (i-center_points[j])
            angle_pairs.append((pitch, horizontal))

    return angle_pairs





def binocularOverlap(booleans, angles_pairs):
    '''
    Pick only angles that have marked binocular overlap by boolean True/False value.

    booleans        [True, False, ...]
    angles_pairs     [(pitch_angle, horizontal_angle), ...]

    '''
    
    points = []

    for boolean, angles_pair in zip(booleans, angles_pairs):

        if boolean == True:
            points.append(angles_pair)
    
    return points


def __findContourPoints(overlapping_angle_pairs):
    '''
    Returns a list of
    [[leftside_points],[rightside_points]]
    '''
    contour_points = [[], []] 
    previous_pitch = False

    for i, angles in enumerate(overlapping_angle_pairs):
        
        if angles[0] == previous_pitch:
            pass
        else:
            contour_points[1].append(overlapping_angle_pairs[i-1])
            contour_points[0].append(overlapping_angle_pairs[i])

        previous_pitch = angles[0]
    
    contour_points[1].pop(0)
    return contour_points


def plotOverlap2D(overlapping_angle_pairs):
    '''
    '''
    
    contour_points = __findContourPoints(overlapping_angle_pairs) 
    contour_points[1].reverse()
    
    pitches = [x[0] for x in contour_points[0]+contour_points[1]]
    horizontals = [x[1] for x in contour_points[0]+contour_points[1]]
    
    fig, ax = plt.subplots()
    ax.plot(horizontals, pitches)
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 180)




def plotOverlap1D(overlapping_angle_pairs):
    contour_points = __findContourPoints(overlapping_angle_pairs) 

    X, Y = [[], []]

    for left, right in zip(*contour_points):
        pitch = left[0]
        overlap_width = right[1] - left[1]
        X.append(pitch)
        Y.append(overlap_width)

    #fig, ax = plt.subplots()
    #plt.plot(X, Y, marker='x')
    #plt.show()
    
    plotter.default.simpleplot(X, Y, xlabel='Pitch angle', ylabel='Horizontal angle')
    
    with open('overlap1D.csv', 'w') as fp:
        writer = csv.writer(fp)
        for x, y in zip(X, Y):
            writer.writerow([x, y])



def main():
    
    aligner = RowAligner(None)
    aligner.loadAlign()
    
    centers = aligner.centers
    N_images_list = aligner.N_images_list
    booleans = [i for j in aligner.both_visible for i in j]


    angle_pairs = arrayToAngles(centers, N_images_list, 5, 1)
    

    angle_pairs = binocularOverlap(booleans, angle_pairs)

    
    plotOverlap1D(angle_pairs)
    

if __name__ == "__main__":
    main()


