#!/usr/bin/env python3
'''
Analysing DrosoM flies that contain pseudopupil movement.


TODO
- the file is getting too long, split to submodules
- start up time is getting too long?

'''
import sys
import cv2
import os
import json
import shutil
import ast
import math
import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import scipy.interpolate
from mayavi import mlab

# Plotting 3D in matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

from droso import DrosoSelect
from marker import Marker
from pupil_imsoft.anglepairs import toDegrees
from movie import Encoder
from image_adjusting import ROIAdjuster
from coordinates import findDistance, findClosest
from coordinates import camera2Fly, camvec2Fly, rotate_about_x, force_to_tplane
from directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR, PROCESSING_TEMPDIR_BIGFILES, DROSO_DATADIR
from optimal_sampling import optimal
from optic_flow import flow_direction, field_error


import plotter
from imalyser.movement import Movemeter

def angleFromFn(fn):
    '''
    Returns the horizontal and vertical angles from a given filename,
    that is imsoft formatted as
        im_pos(-30, 170)_rep0_0.tiff
    '''
    hor, ver = fn.split('(')[1].split(')')[0].split(',')
    hor = int(hor)
    ver = int(ver)
    
    angles = [[hor,ver]]
    toDegrees(angles)
    return angles[0]

def get_data(drosom_folder):
    '''
    Imports DrosoM imaging data from the following save structure

    DrosoM2
        pos(0, 0)
            .tif files
        pos(20, 20)
            .tif files
        pos(0, 10)
            .tif files
        ...

    in a dictionary where the keys are str((horizontal, pitch)) and the items are
    a list of image stacks:
        
        stacks_dictionary = {"(hor1, pitch1): [[stack_rep1], [stack_rep2], ...]"},
        
        where stack_rep1 = [image1_fn, image2_fn, ...].

    '''
    repetition_indicator = 'rep'
    position_indicator = 'pos' 
        
    
    stacks_dictionary = {}

    pos_folders = os.listdir(drosom_folder)

    # Import all tif images
    for folder in pos_folders:
        str_angles = folder[len(position_indicator):]     # Should go from "pos(0, 0)" to "(0, 0)"
     
        files = os.listdir(os.path.join(drosom_folder, folder))
        tiff_files = [f for f in files if f.endswith('.tiff') or f.endswith('.tif')]
        
        #tiff_files.sort() 
        
        
        # FIXED sorting does not work becauce imsfot lasyness in indexing, no zero padding!!! :DDDDD
        tiff_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
        
        stacks_dictionary[str_angles] = []

        # Subdivide into repetitions
        for tiff in tiff_files:
            i_repetition = int(tiff[tiff.index(repetition_indicator)+len(repetition_indicator):].split('_')[0])
            
            while i_repetition >= len(stacks_dictionary[str_angles]):
                stacks_dictionary[str_angles].append([])
            
            
            stacks_dictionary[str_angles][i_repetition].append(os.path.join(drosom_folder, folder, tiff))
    
    return stacks_dictionary


class MLoader:
    '''
    Loads DrosoM imaging data.
    Mainly used directly through MAnalyser.
    '''
    
    def __init__(self):
        
        # For decoding parameters from filenames
        self.repetition_indicator = 'rep'
        self.position_indicator = 'pos' 
        
    def getData(self, drosom_folder):
        '''
        Imports DrosoM imaging data from the following save structure

        DrosoM2
            pos(0, 0)
                .tif files
            pos(20, 20)
                .tif files
            pos(0, 10)
                .tif files
            ...

        in a dictionary where the keys are str((horizontal, pitch)) and the items are
        a list of image stacks:
            
            stacks_dictionary = {"(hor1, pitch1): [[stack_rep1], [stack_rep2], ...]"},
            
            where stack_rep1 = [image1_fn, image2_fn, ...].

        '''
        stacks_dictionary = {}

        pos_folders = os.listdir(drosom_folder)

        # Import all tif images
        for folder in pos_folders:
            str_angles = folder[len(self.position_indicator):]     # Should go from "pos(0, 0)" to "(0, 0)"
   
         
            files = os.listdir(os.path.join(drosom_folder, folder))
            tiff_files = [f for f in files if f.endswith('.tiff') or f.endswith('.tif')]
            
            #tiff_files.sort() 
            
            
            # FIXED sorting does not work becauce imsfot lasyness in indexing, no zero padding!!! :DDDDD
            tiff_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                
            
            stacks_dictionary[str_angles] = []

            # Subdivide into repetitions
            for tiff in tiff_files:
                i_repetition = int(tiff[tiff.index(self.repetition_indicator)+len(self.repetition_indicator):].split('_')[0])
                
                while i_repetition >= len(stacks_dictionary[str_angles]):
                    stacks_dictionary[str_angles].append([])
                
                
                stacks_dictionary[str_angles][i_repetition].append(os.path.join(drosom_folder, folder, tiff))
        
        return stacks_dictionary


        
class MAnalyser():
    '''
    Cross-correlation analysis of DrosoM data.
    '''

    def __init__(self, data_path, folder, clean_tmp=False):
        '''
        INPUT ARGUMENTS     DESCRIPTION 
        data_path           directory where DrosoM folder lies
        folder              Name of the DrosoM folder, for example "DrosoM1"
        
        '''
        self.data_path = data_path
        self.folder = folder
        
        self.CROPS_SAVEFN = os.path.join(PROCESSING_TEMPDIR, "MAnalyser/ROIs/dynamic_{}_crops.json".format(folder))
        self.MOVEMENTS_SAVEFN = os.path.join(PROCESSING_TEMPDIR, "MAnalyser/movements/dynamic_{}_{}_movements.json".format(folder, '{}'))
        
        # Ensure the directories where the crops and movements are saved exist
        os.makedirs(os.path.dirname(self.CROPS_SAVEFN), exist_ok=True)
        os.makedirs(os.path.dirname(self.MOVEMENTS_SAVEFN), exist_ok=True)
        
        #self.CORRECTIONS_SAVEFN = os.path.join(PROCESSING_TEMPDIR, "MAnalyser/motion_corrections/dynamic_{}_{}_motion_corrections.json".format(folder, '{}'))
        
        # The images with which the cropping is done are copied here
        #self.CROPPING_IMAGES_COPYDIR = os.path.join(PROCESSING_TEMPDIR, "MAnalyser/cropping_images/")
        #os.makedirs(self.CROPPING_IMAGES_COPYDIR, exist_ok=True)
        

        loader = MLoader()
        self.stacks = loader.getData(os.path.join(self.data_path, self.folder))

        
        self.antenna_level_correction = self._getAntennaLevelCorrection(folder)
        if self.antenna_level_correction == False:
            print('No antenna level correction value for fly {}'.format(folder))


    def __fileOpen(self, fn):
        with open(fn, 'r') as fp:
            data = json.load(fp)
        return data

    
    def __fileSave(self, fn, data):
        with open(fn, 'w') as fp:
            json.dump(data, fp)
    
    
    def getFolderName(self):
        '''
        Return the name of the data (droso) folder, such as DrosoM42
        '''
        return self.folder

    
    @staticmethod
    def getPosFolder(image_fn):
        '''
        Gets the name of the folder where an image lies, for example
        /a/b/c/image -> c
        '''
        return os.path.split(os.path.dirname(image_fn))[1]
    
    
    @staticmethod
    def _getAntennaLevelCorrection(fly_name): 
        fn = os.path.join(ANALYSES_SAVEDIR, 'antenna_levels', fly_name+'.txt')

        if os.path.exists(fn):
            with open(fn, 'r') as fp:
                antenna_level_offset = float(fp.read())
        else:
            antenna_level_offset = False
        
        return antenna_level_offset

    
    def _correctAntennaLevel(self, angles):
        '''
        angles  In degrees, tuples, (horizontal, pitch)
        '''
        if self.antenna_level_correction != False:
            for i in range(len(angles)):
                #print('Before correcting {}'.format(angles[i]))
                angles[i][1] -= self.antenna_level_correction
                #print('Before correcting {}'.format(angles[i]))

        return angles

    def loadROIs(self):
        '''
        Load ROIs (pseudopupils selected before) for the left/right eye.
        
        INPUT ARGUMENTS     DESCRIPTION
        eye                 'left' or 'right'
        
        DETAILS
        While selecting ROIs, pseudopupils of both eyes are selcted simultaneously. There's
        no explicit information about from which eye each selected ROI/pseudopupil is from.
        Here we reconstruct the distinction to left/right using following way:
            1 ROI:      horizontal angle determines
            2 ROIs:     being left/right in the image determines
        
        Notice that this means that when the horizontal angle is zero (fly is facing towards the camera),
        image rotation has to be so that the eyes are on image's left and right halves.
        '''

        self.ROIs = {'left': {}, 'right': {}}

        with open(self.CROPS_SAVEFN, 'r') as fp:
            marker_markings = json.load(fp)
        
        for image_fn, ROIs in marker_markings.items():
            
            pos = self.getPosFolder(image_fn)[3:]
            horizontal, pitch = ast.literal_eval(pos)

            # ROI belonging to the eft/right eye is determined solely by
            # the horizontal angle when only 1 ROI exists for the position
            if len(ROIs) == 1:
                
                if horizontal > 0:
                    self.ROIs['left'][pos] = ROIs[0]
                else:
                    self.ROIs['right'][pos] = ROIs[0]

            # If there's two ROIs
            if len(ROIs) == 2:
                
                if ROIs[0][0] > ROIs[1][0]:
                    self.ROIs['left'][pos] = ROIs[0]
                    self.ROIs['right'][pos] = ROIs[1]
                else:
                    self.ROIs['left'][pos]= ROIs[1]
                    self.ROIs['right'][pos] = ROIs[0]
                    

    def selectROIs(self):
        '''
        Selecting the ROIs from the loaded images.
        Currently, only the first frame of each recording is shown.
        '''
        
        to_cropping = [stacks[0][0] for str_angles, stacks in self.stacks.items()]

        fig, ax = plt.subplots()
        marker = Marker(fig, ax, to_cropping, self.CROPS_SAVEFN)
        marker.run()
        
    def isROIsSelected(self):
        '''
        Returns True if a file for crops/ROIs is found.
        '''
        return os.path.exists(self.CROPS_SAVEFN)

    def isMovementsAnalysed(self):
        '''
        Returns (True, True) if analyseMovement results can be found for the fly and bot eyes.
        '''
        return (os.path.exists(self.MOVEMENTS_SAVEFN.format('left')), os.path.exists(self.MOVEMENTS_SAVEFN.format('right')))

    def loadAnalysedMovements(self):
        self.movements = {}
        with open(self.MOVEMENTS_SAVEFN.format('right'), 'r') as fp:
            self.movements['right'] = json.load(fp)
        with open(self.MOVEMENTS_SAVEFN.format('left'), 'r') as fp:
            self.movements['left'] = json.load(fp)
        
    def analyseMovement(self, eye):
        '''
        Performs cross-correlation analysis for the selected pseudopupils (ROIs, regions of interest)
        using Movemeter package.

        If pseudopupils/ROIs haven't been selected, calls method self.selectROIs.
        Movements are saved into a tmp directory.

        INPUT ARGUMENTS         DESCRIPTION
        eye                     'left' or 'right'

        Cross-correlation analysis is the slowest part of the DrosoM pipeline.
        '''
        
        self.movements = {}
        
        if not os.path.exists(self.CROPS_SAVEFN):
            self.selectROIs() 
        self.loadROIs()
        

        angles = []
        stacks = []
        ROIs = []
        
        for angle in self.stacks:
            if angle in str(self.ROIs[eye].keys()):
                for i_repetition in range(len(self.stacks[angle])):
                    angles.append(angle)
                    stacks.append( self.stacks[angle][i_repetition] )
                    ROIs.append( [self.ROIs[eye][angle]] )
        
        print('angles len {}'.format(len(angles)))

        meter = Movemeter(upscale=4)
        meter.setData(stacks, ROIs)
       

        for stack_i, angle in enumerate(angles):
            print('Analysing {} eye pseudopupil motion from position {}, done {}/{} for this eye'.format(eye.upper(), angle, stack_i+1, len(ROIs)))

            print("Calculating ROI's movement...")
            x, y = meter.measureMovement(stack_i, max_movement=15)[0]
            
            print('Done.')
            
            # Failsafe for crazy values
            if not max(np.max(np.abs(x)), np.max(np.abs(y))) > 100:
                try:
                    self.movements[angle]
                except KeyError:
                    self.movements[angle] = []
                
                tags = meter.getMetadata(stack_i)['Image ImageDescription'].values.split('"')
                time = tags[tags.index('start_time') + 2]
                self.movements[angle].append({'x': x, 'y':y, 'time': time})
 
        
        # Save momevemtns
        with open(self.MOVEMENTS_SAVEFN.format(eye), 'w') as fp:
            json.dump(self.movements, fp)
        
        
        #for key, data in self.movements.items():
        #    plt.plot(data['x'])
        #    plt.plot(data['y'])
        #    plt.show()


    def timePlot(self):
        '''
        Not sure anymore what this is but probably movement magnitude over time.
        '''

        data = []
        
        for eye in ['left', 'right']:
            for angle in self.movements[eye]:
                data.extend(self.movements[eye][angle])

        data.sort(key=lambda x: x['time'])
        
        for d in data:
            print(d['time'])

        X = [0]
        X.extend( [(datetime.datetime.strptime(data[i]['time'], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(data[i-1]['time'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
            for i in range(1, len(data))] )
        
        xx = [x['x'][-1]-x['x'][0] for x in data]
        yy = [x['y'][-1]-x['y'][0] for x in data]
        Z = [math.sqrt(x**2+y**2) for (x,y) in zip(xx, yy)]
        
        print(X)
        print(Z)

        plt.scatter(X, Z)
        plt.show()


    def getTimeOrdered(self):
        '''
        Returns a list of all taken images and ROIs, ordered in time from the first to the last.
        '''
        self.loadROIs()

        times_and_images = []
        seen_angles = []

        for eye in self.movements:
            for angle in self.movements[eye]:
                
                if not angle in seen_angles: 
                    time = self.movements[eye][angle][0]['time']
                    fn = self.stacks[angle][0]
                    #ROI = self.ROIs[eye][angle]
                    ROI = self.getMovingROIs(eye, angle)
                    times_and_images.append([time, fn, ROI])
         
                    seen_angles.append(angle)

        times_and_images.sort(key=lambda x: x[0])
        
        images = []
        ROIs = []

        for time, fns, ROI in times_and_images:
            images.extend(fns)
            #ROIs.extend([ROI]*len(fns))
            ROIs.extend(ROI)
        return images, ROIs


    def get2DVectors(self, eye, mirror_horizontal=True, mirror_pitch=True, correct_level=True):
        '''
        Creates 2D vectors from the movements analysis data.
            Vector start point: Pupils position at the firts frame
            Vector end point: Pupil's position at the last frame

        mirror_pitch    Should make so that the negative values are towards dorsal and positive towards frontal
                            (this is how things on DrosoX were)
        '''
       
        angles = [list(ast.literal_eval(angle)) for angle in self.movements[eye]]
        values = [self.movements[eye][angle] for angle in self.movements[eye]]
        
     
        toDegrees(angles)
        
        if correct_level:
            angles = self._correctAntennaLevel(angles)


        if mirror_horizontal:
            for i in range(len(angles)):
                angles[i][0] *= -1

        if mirror_pitch:
            for i in range(len(angles)):
                angles[i][1] *= -1

        

        # Vector X and Y components
        # Fix here if repetitions are needed to be averaged
        # (don't take only x[0] but average)
        X = [x[0]['x'][-1]-x[0]['x'][0] for x in values]
        Y = [x[0]['y'][-1]-x[0]['y'][0] for x in values]

        return angles, X, Y
    
    def getMagnitudeTraces(self, eye):
        '''
        Return a dictionary of movement magnitudes over time.
        Keys are the angle pairs.  
        '''
        magnitude_traces = {}

        for angle in self.movements[eye]:
            x = self.movements[eye][angle][0]['x']
            y = self.movements[eye][angle][0]['y']
            

            mag = np.sqrt(np.asarray(x)**2 + np.asarray(y)**2)
            magnitude_traces[angle] = mag

        return magnitude_traces

    
    def getMovingROIs(self, eye, angle):
        '''
        Returns a list of ROIs how they move over time.
        Useful for visualizing.
        '''

        moving_ROI = []

        self.loadROIs()

        movements = self.movements[eye][angle][0]
        rx,ry,rw,rh = self.ROIs[eye][angle]
        
        for i in range(len(movements['x'])):
            x = -movements['x'][i]
            y = -movements['y'][i]
            
            moving_ROI.append([rx+x,ry+y,rw,rh])
        return moving_ROI
        
    
    def get_3d_vectors(self, eye, correct_level=True, normalize_length=0.1):
        '''
        Return 3D vectors
    
        correct_level           Use estimated antenna levels
        '''
        vectors = []

        angles, X, Y = self.get2DVectors(eye, mirror_pitch=False, mirror_horizontal=True, correct_level=False)
        
        for angle, x, y in zip(angles, X, Y):
            horizontal, pitch = angle

            point0 = camera2Fly(horizontal, pitch)
            point1 = camvec2Fly(x, y, horizontal, pitch, normalize=normalize_length)
            
            if correct_level:
                rotation = -self.antenna_level_correction
                point0 = rotate_about_x(point0, rotation)
                point1 = rotate_about_x(point1, rotation)
            
            x0,y0,z0 = point0
            x1,y1,z1 = point1

            vectors.append( (tuple(angle), (x0,x1), (y0,y1), (z0, z1)) )
        return vectors
    


class MAverager:
    '''
    Combining and averaging results from many MAnalyser objects.
    
    MAverager acts like MAnalyser object for getting data (like get2DVectors)
    but lacks the movement analysis (cross-correlation) related parts.
    '''
    def __init__(self, manalysers):
        
        self.manalysers = manalysers


    def getFolderName(self):
        return 'averaged_'+'_'.join([manalyser.getFolderName() for manalyser in self.manalysers])

    def setInterpolationSteps(self, horizontal_step, vertical_step):
        '''
        Set the resolution of the N-nearest neighbour interpolation in Maverager.get2DVectors.

        INPUT ARGUMENTS
        horizontal_step
        vertical_step

        Arguments horizontal_step and vertical_step refer to the rotation stages.

        '''

        self.intp_step = (horizontal_step, vertical_step)

    @staticmethod
    def findDistance(point1, point2):
        '''
        Returns PSEUDO-distance between two points in the rotation stages angles coordinates (horizontal_angle, vertical_angle).
        
        It's called pseudo-distance because its not likely the real 3D cartesian distance + the real distance would depend
        on the radius that is in our angles coordinate system omitted.

        In the horizontal/vertical angle system, two points may seem to be far away but reality (3D cartesian coordinates)
        the points are closed to each other. For example, consider points
            (90, 10) and (90, 70)
        These points are separated by 60 degrees in the vertical (pitch) angle, but because the horizontal angle is 90 degrees
        in both cases, they are actually the same point in reality (with different camera rotation)

        INPUT ARGUMENTS     DESCRIPTION
        point1              (horizontal, vertical)


        TODO:   - Implement precise distance calculation in 3D coordinates
        '''
        # Scaler: When the horizontal angle of both points is close to 90 or -90 degrees, distance
        # should be very small
        #scaler = abs(math.sin((point1[0] + point2[0])/ 2))
        # All this is probably wrong, right way to do this is calculate distances on a sphere
        #return scaler * math.sqrt( (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )
    
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    


    # TODO REMOVEME
    def get2DVectors(self, eye):
        '''
        N-nearest neighbour interpolation of 2D vectors, for more see MAnalyser.get2DVectors method.
        
        N-NEAREST NEIGHBOUR
        
        Here N-nearest neighbour means, that for each point to be interpolated, we cruerly select N nearest neighbours
        whose average sets value to the interpolated point. N = N_manalysers, meaning the number of recorded flies.
        
        his interpolation method was selected because the way how data was sampled; For each fly, the sampling grid
        was similar (every 10 degrees in horizontal and vertical angles, every 5 degrees in horizontal near frontal area),
        so weighting the values offers no extra.


        Temporal solution, better would be to make vectors in 3D cartesian coordinates
        and then interpolate these and then average.

        TODO:   - there's optimization work, see fixme markings in the code
                - see self.getDistance, this would need work
        '''
        
        N = len(self.manalysers)

        # Getting the 2D vectors
        ANGLES, X, Y = ([], [], [])
        for analyser in self.manalysers:
            angles, x, y = analyser.get2DVectors(eye)
            ANGLES.extend(angles)
            X.extend(x)
            Y.extend(y)
        
        # Interpolation borders or "limits" by taking extreme values of horizontal
        # and vertical angles.
        # FIXME: What is there's a single outliner far away from others, we have not enough data
        # to interpolate in between?
        HORIZONTALS = [hor for (hor,pit) in ANGLES]
        VERTICALS = [pit for (hor,pit) in ANGLES]
        horlim = [np.min(HORIZONTALS), np.max(HORIZONTALS)]
        verlim = [np.min(VERTICALS), np.max(VERTICALS)]
        
        # Performing the interpolation
        INTP_ANGLES, INTP_X, INTP_Y = ([], [], [])
        
        for hor in np.arange(*horlim, self.intp_step[0]):
            for ver in np.arange(*verlim, self.intp_step[1]):
                
                # Here we find N closest point to the point being interpolated
                # This is SLOW (FIXME): We wouldn't have to calculate all points if the AGNLES
                # would be somehow better sorted
                distances_and_indices = [[self.findDistance((hor,ver), angle), i] for (i,angle) in enumerate(ANGLES)]
                distances_and_indices.sort(key=lambda x: x[0])
                
                N_closest = [d_and_i[1] for d_and_i in distances_and_indices[0:N]]
                
                INTP_X.append( np.mean([X[i] for i in N_closest]) )
                INTP_Y.append( np.mean([Y[i] for i in N_closest]) )
                INTP_ANGLES.append((hor,ver))
                
        return INTP_ANGLES, INTP_X, INTP_Y

    def nearest_neighbour_3d(self, point_A, vectors, max_distance=None):
        '''
        Return the nearest point to the point_A

        vectors     [((),(),()),]
        '''
        mindist = np.inf
        argmin = None
    
        ax, ay, az = point_A

        for vector in vectors:
            angle, (x0, x1), (y0, y1), (z0, z1) = vector
            
            distance = np.sqrt( (ax-x0)**2 + (ay-y0)**2 + (az-z0)**2 )
            
            if distance < mindist:
                mindist = distance
                argmin = vector

            #print(mindist)

        if mindist > max_distance:
            return False

        return argmin
    
    @staticmethod
    def vector_length(vector):
        if type(vector) == type([]):
            angle, x,y,z = vector
            length = np.sqrt( (x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2 )
        elif type(vector) == type(np.array([0])):
            x,y,z = vector
            length = np.sqrt( (x)**2 + (y)**2 + (z)**2 )
        return length

    def average_vector_3d(self, angle_tag, point, vectors):
        '''
        Average vectors and return a vector at point point.

        DIDNT WORK BECAUSE VECTORS NOT ON THE SAME POINT and WE
        WANT THE VECTOR TO BE TANGENTIAL TO THE SPEHRE
        '''
        
        real_vectors = []

        for vector in vectors:
            angle, x, y, z = vector


            X = x[1] - x[0]
            Y = y[1] - y[0]
            Z = z[1] - z[0]
            real_vectors.append(np.array([X,Y,Z]))
        

        av = np.mean(real_vectors, axis=0)
        if self.vector_length(av) != 0:

            av += np.array(point)
            av = force_to_tplane(point, av)


            
            for i in range(0,len(vectors)):
                wanted_len = self.vector_length(real_vectors[0])
                
                if wanted_len != 0:
                    break

            av -= np.array(point)
            av = (av / self.vector_length(av) * wanted_len)
            av += np.array(point)
        else:
            av = point

        x,y,z = point
        
        return (angle_tag, (x, av[0]), (y, av[1]), (z, av[2]) )
        

    def get_3d_vectors(self, eye, correct_level=True, normalize_length=0.1):
        '''
        
        '''
        interpolated = []
        
        R = 1
        intp_dist = (2 * R * np.sin(math.radians(self.intp_step[0])))
        
        vectors_3d = []

        for analyser in self.manalysers:
            vectors_3d.append( analyser.get_3d_vectors(eye, correct_level=True, normalize_length=normalize_length) )
            
        
        intp_points = optimal(np.arange(-90, 90.01, self.intp_step[0]), np.arange(0, 360.01, self.intp_step[1]))
        
        angle_tag = (0,0)

        for intp_point in intp_points:
            
            nearest_vectors = []
            for vectors in vectors_3d:
                nearest_vector = self.nearest_neighbour_3d(intp_point, vectors, max_distance=intp_dist)
                #print(nearest_vector) 
                if nearest_vector != False: 
                    nearest_vectors.append(nearest_vector)

            if len(nearest_vectors) > len(vectors_3d)/2:
                interpolated.append( self.average_vector_3d(angle_tag, intp_point, nearest_vectors) )

        return interpolated
        

class MPlotter:

 
    def __init__(self):
        
        self.plots = {}
        self.savedir = os.path.join(ANALYSES_SAVEDIR, 'mplots')
        os.makedirs(self.savedir, exist_ok=True)
    
    def save(self):
        '''
        Saves all the plots made to the analysis savedirectory.
        '''

        for plot_name in self.plots:
            fn = os.path.join(self.savedir, plot_name)
            self.plots[plot_name]['fig'].savefig(fn+'.svg', format='svg')

    def setLimits(self, limit):
        '''
        Sets all the plots to the same limits

        limit       'common', 'individual' or to set fixed for all: (min_hor, max_hor, min_pit, max_pit)
        '''

        if len(limit) == 4:
            limit = tuple(limit)
        
        elif limit == 'common':
            
            manalysers = [self.plots[plot_name]['manalyser'] for plot_name in self.plots]

            limits = [1000, -1000, 1000, -1000]
            
            for manalyser in manalysers:
                for eye in ['left', 'right']:
                    angles, X, Y = manalyser.get2DVectors(eye)
                    horizontals, pitches = zip(*angles)
                    
                    #print(horizontals)
                    limits[0] = min(limits[0], np.min(horizontals))
                    limits[1] = max(limits[1], np.max(horizontals))
                    limits[2] = min(limits[2], np.min(pitches))
                    limits[3] = max(limits[3], np.max(pitches))
        
        for ax in [self.plots[plot_name]['ax'] for plot_name in self.plots]:
            ax.set_xlim(limits[0]-10, limits[1]+10)
            ax.set_ylim(limits[2]-10, limits[3]+10)


    def plotDirection2D(self, manalyser):
        '''
        manalyser       Instance of MAnalyser class or MAverager class (having get2DVectors method)
        
        limits          Limits for angles, [min_hor, max_hor, min_pitch, max_pitch]
        '''
        

        fig, ax = plt.subplots()
        
   
        for color, eye in zip(['red', 'blue'], ['left', 'right']):
            angles, X, Y = manalyser.get2DVectors(eye)
            for angle, x, y in zip(angles, X, Y):
               
                horizontal, pitch = angle
                
                # If magnitude too small, its too unreliable to judge the orientation so skip over
                movement_magnitude = math.sqrt(x**2 + y**2)
                #if movement_magnitude < 2:
                #    continue 
  
                # Vector orientation correction due to sample orientation dependent camera rotation
                #xc = x * np.cos(np.radians(pitch)) + y * np.sin(np.radians(pitch))
                #yc = x * np.sin(np.radians(pitch)) + y * np.cos(np.radians(pitch))

                # Scale all vectors to the same length
                scaler = math.sqrt(x**2 + y**2) / 5 #/ 3
                #scaler = 0
                if scaler != 0:
                    x /= scaler
                    y /= scaler /2.4    # FIXME


                #ar = matplotlib.patches.Arrow(horizontal, pitch, xc, yc)
                ar = matplotlib.patches.FancyArrowPatch((horizontal, pitch), (horizontal-x, pitch+y), mutation_scale=10, color=color, picker=True)
                #fig.canvas.mpl_connect('pick_event', self.on_pick)
                ax.add_patch(ar)
        
        ax.set_xlabel('Horizontal (degrees)')
        ax.set_ylabel('Pitch (degrees)')
        
        for key in ['top', 'right']:
            ax.spines[key].set_visible(False)
       
        plot_name = 'direction2D_{}'.format(manalyser.getFolderName())
        self.plots[plot_name] = {'fig': fig, 'ax': ax, 'manalyser': manalyser}

    def plotMagnitude2D(self, manalyser):
        '''

        TODO
        - combine eyes to yield better picture
        - axes from pixel values to actual

        '''
        
        distancef = lambda p1,p2: math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        
        fig, ax = plt.subplots(ncols=2)
     
        for eye_i, (color, eye) in enumerate(zip(['red', 'blue'], ['left', 'right'])):
            angles, X, Y = manalyser.get2DVectors(eye)
            
                
            HOR = []
            PIT = []
            for angle, x, y in zip(angles, X, Y):
                horizontal, pitch = angle
                HOR.append(horizontal)
                PIT.append(pitch) 

            # TRY NEAREST NEIGHBOUR INTERPOLATION
            res = (50, 50)
            xi = np.linspace(np.min(HOR), np.max(HOR), res[0]) 
            yi = np.linspace(np.min(PIT), np.max(PIT), res[1]) 
            zi = np.zeros(res)
            for j in range(len(yi)):
                for i in range(len(xi)):
                    point = findClosest((xi[i], yi[j]), angles, distance_function=distancef)
                    
                    index = angles.index(point)
                    
                    zi[j][i] = (math.sqrt(X[index]**2 + Y[index]**2))


            print('{} to {}'.format(xi[0], xi[-1]))
            print('{} to {}'.format(yi[0], yi[-1]))

            im = ax[eye_i].imshow(zi, interpolation='none', extent=[xi[0], xi[-1], yi[0], yi[-1]])
            #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax[eye_i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            
            fig.colorbar(im, cax=cax)

            ax[eye_i].title.set_text('{} eye'.format(eye.capitalize()))          

            #XYZ.append([xi,yi,zi])
    
        #fig = plotter.contourplot(XYZ, 1, 2, colorbar=True) 
        #X,Y = np.meshgrid(X, Y)
        #plt.pcolor(X,Y,Z)

        #ax.set_xlim(-np.max(HOR)-10, -np.min(HOR)+10)
        #ax.set_ylim(-np.max(PIT)-10, -np.min(PIT)+10)
        #ax.set_xlabel('Horizontal angle (degrees)')
        #ax.set_ylabel('Pitch angle (degrees)')
    

    def plotTimeCourses(self, manalyser, exposure=0.010):
        '''
        Plotting time courses

        FIXME This became dirty
        '''
        avg = []

        for eye in ['left', 'right']:
            traces = manalyser.getMagnitudeTraces(eye)
            
            for angle in traces:
                print(np.max(traces[angle]))
                trace = traces[angle]
                #trace /= np.max(traces[angle])
                if np.isnan(trace).any():
                    print('Nan')
                    continue
                avg.append(trace)
        
        for trace in avg:
            time = np.linspace(0, exposure*len(trace)*1000, len(trace))
            plt.plot(time, trace)
            #plt.show(block=False)
            #plt.pause(0.1)
            #plt.cla()
        print(len(avg))

        avg = np.mean(np.asarray(avg), axis=0)
        print(avg)
        plt.plot(time, avg, color='black')

        plt.show()



    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    

    def plot_3d_vectormap_mayavi(self, manalyser):
        '''
        Use mayavi to make the 3D image that then can be saved in obj file format.
        '''

        for color, eye in zip([(1.,0,0), (0,0,1.)], [('left'), 'right']):
            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True)


            N = len(vectors_3d)
            arrays = [np.zeros(N) for i in range(6)]

            for i in range(N):
                arrays[0][i] = vectors_3d[i][1][0]
                arrays[1][i] = vectors_3d[i][2][0]
                arrays[2][i] = vectors_3d[i][3][0]
                
                arrays[3][i] = vectors_3d[i][1][1] - arrays[0][i]
                arrays[4][i] = vectors_3d[i][2][1] - arrays[1][i]
                arrays[5][i] = vectors_3d[i][3][1] - arrays[2][i]

            mlab.quiver3d(*arrays, color=color)
        
        mlab.show()
    
    def when_moved(self, event):
        '''
        Callback to make two axes to have synced rotation when rotating
        self.axes[0].
        '''
        if event.inaxes == self.axes[0]:
            self.axes[1].view_init(elev = self.axes[0].elev, azim = self.axes[0].azim)
        self.fig.canvas.draw_idle()

    def plot_3d_vectormap(self, manalyser, with_optic_flow=False, animation=False):
        '''
        relp0   Relative zero point

        with_optic_flow         Angle in degrees. If non-false, plot also estimated optic
                                flow with this parameter
        animation           Sequence of (elevation, azimuth) points to create an
                                animation of object rotation
        '''

        fig = plt.figure(figsize=(15,15))
        
        if with_optic_flow:
            axes = []
            axes.append( fig.add_subplot(121, projection='3d') )
            axes.append( fig.add_subplot(122, projection='3d') )

        else:
            axes = [fig.add_subplot(111, projection='3d')]
        

        
        points = []
        pitches = []

    
        for color, eye in zip(['red', 'blue'], ['left', 'right']):
            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True)

            # Direction
            for (angle, x, y, z) in vectors_3d:
                horizontal, pitch = angle
                
                #ax.text(xs0, ys0, zs0, '{},{}'.format(horizontal,pitch), fontsize=8)
                #ax.text(x[0], y[0], z[0], '{},{}'.format(int(horizontal),int(pitch)), fontsize=7)
                
                points.append([x[0], y[0], z[0]])
                pitches.append(pitch)

                ar = self.Arrow3D(x, y, z, arrowstyle="-|>", lw=2, mutation_scale=10, color=color)
                axes[0].add_artist(ar)
                
                if with_optic_flow:
                    flowP = flow_direction(points[-1], xrot=with_optic_flow) + np.array(points[-1])
                    ar = self.Arrow3D(*zip(points[-1], flowP), arrowstyle="-|>",
                            lw=2, mutation_scale=10, color=color)
                    axes[1].add_artist(ar)
                
        for ax in axes:
            if not with_optic_flow:
                ax.scatter([p[0] for p in points],[p[1] for p in points],[p[2] for p in points], c=pitches)
            

            ax.set_xlim(-1, 1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1, 1)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)



        if with_optic_flow and not animation:
            connection = fig.canvas.mpl_connect('motion_notify_event', self.when_moved)
            self.axes = axes
            self.fig = fig

        if animation:
            savedir = os.path.join(self.savedir, 'vectormap_3d_anim')
            os.makedirs(savedir, exist_ok=True)

            plt.show(block=False)

            for i, (elevation, azimuth) in enumerate(animation):
                print('{} {}'.format(elevation, azimuth)) 
                for ax in axes:
                    ax.view_init(elev=elevation, azim=azimuth)
                fig.canvas.draw_idle()

                fn = 'image_{:0>8}.png'.format(i)
                fig.savefig(os.path.join(savedir, fn))
                #plt.pause(0.1)

        # make the panes transparent
        #ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        #ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        plt.savefig('vectormap.svg', transparent=True)
        plt.show()



class TerminalDrosoM:
    '''
    Using drosom.py from terminal (command line).
    '''


    def __init__(self): 
        
        # These determine what is considered proper 
        #self.allowed_shorts = 'a'
        #self.allowed_longs = ['timeplot', '2dplot', '3dplot', 'saveonly', 'averaged', 'movie', 'magnitude', 'averaged-magnitude', 'show', 'save', 'magtrace', 'recalculate']
        self.argv = sys.argv
        #self.parseInputArguments()

    def help(self):
        '''
        Prints the help when the script is ran from terminal.
        '''
        print('Usage (drosom.py):')
        print('TODO')
    

    def parseInputArguments(self):
        
        self.args = []
        
        for arg in sys.argv[1:]:
            if not arg in self.allowed_longs:
                print('Invalid option {}'.format(arg))
                self.help()
                raise ValueError

    
    def main(self):
        
        selector = DrosoSelect()
        directories = selector.askUser(startswith='DrosoM')

        analysers = []

        # Set up analysers at the selected DrosoM folders
        for directory in directories: 
            path, folder_name = os.path.split(directory)
            analyser = MAnalyser(path, folder_name) 
            analysers.append(analyser)
            

        # Ask ROIs if not selected
        for analyser in analysers:
            if not analyser.isROIsSelected():
                analyser.selectROIs()

        # Analyse movements if not analysed, othewise load these
        for analyser in analysers:

            if not analyser.isMovementsAnalysed() == (True, True) or 'recalculate' in self.argv:
                analyser.analyseMovement(eye='left')
                analyser.analyseMovement(eye='right')
            analyser.loadAnalysedMovements()
        
        
        plotter = MPlotter()

        # Plot results if asked so
        for analyser in analysers:
            if 'timeplot' in self.argv:
                analyser.timePlot()
            if 'magtrace' in self.argv:
                plotter.plotTimeCourses(analyser)
            if '2dplot' in self.argv:
                plotter.plotDirection2D(analyser)
            if '3dplot' in self.argv:
                plotter.plot_3d_vectormap(analyser)

            if 'magnitude' in self.argv:
                plotter.plotMagnitude2D(analyser)

            if 'movie' in self.argv:
                print(analyser.getFolderName())
                images, ROIs = analyser.getTimeOrdered()
                
                workdir = os.path.join(PROCESSING_TEMPDIR_BIGFILES, 'movie_{}'.format(str(datetime.datetime.now())))
                os.makedirs(workdir, exist_ok=True)

                newnames = [os.path.join(workdir, '{:>0}.jpg'.format(i)) for i in range(len(images))]
                

                adj = ROIAdjuster()
                newnames = adj.writeAdjusted(images, ROIs, newnames, extend_factor=3, binning=1)

                enc = Encoder()
                fps = 25
                enc.encode(newnames, os.path.join(ANALYSES_SAVEDIR, 'movies','{}_{}fps.mp4'.format(analyser.getFolderName(), fps)), fps)
                


                for image in newnames:
                    os.remove(image)
                try:
                    os.rmdir(workdir)
                except OSError:
                    print("Temporal directory {} left behind because it's not empty".format(workdir))

        if 'averaged' in self.argv:
            avg_analyser = MAverager(analysers)
            avg_analyser.setInterpolationSteps(5,5)
            #plotter.plotDirection2D(avg_analyser)
           
            if 'animation' in self.argv:
                animation = []
                step = 0.5
                sidego = 30
                # go up, to dorsal
                for i in np.arange(-30,60,step):
                    animation.append((i,90))
                #rotate azim
                for i in np.arange(90,90+sidego,step*2):
                    animation.append((60,i))
                # go back super down, to ventral
                for i in np.arange(0,120,step):
                    animation.append((60-i,90+sidego))
                # rotate -azim
                for i in np.arange(0,2*sidego,step*2): 
                    animation.append((-60,90+sidego-i))
                # go up back to dorsal
                for i in np.arange(0,120, step):
                    animation.append((-60+i,90-sidego))



            else:
                animation = False

            if 'optimal_optic_flow' in self.argv:
                
                vectors_3d = [vector[1:] for vector in avg_analyser.get_3d_vectors('left')]
                #measured_vecs = [np.array(v[1])-np.array(v[0]) for v in vectors_3d]

                measured_vecs = []
                for vec in vectors_3d:
                    x,y,z = vec
                    measured_vecs.append( np.array([x[1]-x[0], y[1]-y[0], z[1]-z[0]]) )

                rotations = np.linspace(-180, 180, 100)
                
                errors = []
                for rot in rotations: 
                    flow_vecs = [flow_direction(P0, xrot=rot) for P0 in measured_vecs]
                    
                    

                    er = field_error(measured_vecs, flow_vecs) 
                    print('Error of {} for rotation {}deg'.format(er, rot))
                    errors.append(er)

                plt.plot(rotations, errors)
                plt.show()
                
                plotter.plot_3d_vectormap(avg_analyser,
                        with_optic_flow=rotations[np.argmin(errors)], animation=animation)

                

            else:
                if 'mayavi' in self.argv:
                    plotter.plot_3d_vectormap_mayavi(avg_analyser)
                else:
                    plotter.plot_3d_vectormap(avg_analyser, animation=animation)
            


        if 'averaged-magnitude' in self.argv:
            avg_analyser = MAverager(analysers)
            avg_analyser.setInterpolationSteps(10,10)
            plotter.plotMagnitude2D(avg_analyser)

        plotter.setLimits('common')

        if 'save' in self.argv:
            plotter.save()
        
        if 'show' in self.argv:
            plt.show()

        

def main():
    terminal = TerminalDrosoM()
    terminal.main()

if __name__ == "__main__":
    main()
