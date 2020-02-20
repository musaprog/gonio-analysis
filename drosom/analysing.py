'''
Measuring deep pseudopupil movement values forom DrosoM data.

-------
Classes
-------
  MAnalyser
    Main programmatic interfacce to process and interact with imaging data
    produced by pupil_imsoft        

  MAverager         
    Takes in many MAnalysers to generate a mean specimen.
    Only implements some of MAnalyser methods 

  VectorGettable
    Internal, caches results for better performance


'''

import os
import json
import ast
import math
import datetime

import numpy as np
import matplotlib.pyplot as plt

from .loading import load_data, angles_from_fn
from pupil.coordinates import camera2Fly, camvec2Fly, rotate_about_x, nearest_neighbour, mean_vector
from pupil.directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR
from pupil.optimal_sampling import optimal


from pupil_imsoft.anglepairs import toDegrees
from marker import Marker

from imalyser.movement import Movemeter


class VectorGettable:
    '''
    Inheriting this class grants abilities to get vectors and caches results
    for future use, minimizing computational time penalty when called many times.
    '''

    def __init__(self):
        self.cached = {}

        # Define class methods dynamically
        #for key in self.cached:
        #    exec('self.get_{} = ')


    def _get(self, key, *args, **kwargs):
        '''

        '''
        dkey = ''
        for arg in args:
            dkey += arg
        for key, val in kwargs.items():
            dkey += '{}{}'.format(key, val)
        
        try:
            self.cached[dkey]
        except KeyError:
            self.cached[dkey] = self._get_3d_vectors(*args, **kwargs)
        return self.cached[dkey]


    def get_3d_vectors(self, *args, **kwargs):
        '''
        Returns the sampled points and cartesian 3D-vectors at these points.
        '''
        return self._get('3d_vectors', *args, **kwargs)
        

        
class MAnalyser(VectorGettable):
    '''
    Cross-correlation analysis of DrosoM data, saving and loading, and getting
    the analysed data out.

    ------------------
    Input argument naming convetions
    ------------------
    - specimen_name
    - recording_name

    -----------
    Attributes
    -----------
    - self.movements      self.movements[eye][angle][i_repeat][x/y/time]
                    where eye = "left" or "right"
                    angle = recording_name.lstrip('pos'), so for example angle="(0, 0)_uv"
        

    


    '''

    def __init__(self, data_path, folder, clean_tmp=False, no_data_load=False):
        '''
        INPUT ARGUMENTS     DESCRIPTION 
        data_path           directory where DrosoM folder lies
        folder              Name of the DrosoM folder, for example "DrosoM1"
        no_data_load        Skip loading data in the constructor
        '''
        super().__init__()
        
        self.data_path = data_path
        self.folder = folder
        
        self.CROPS_SAVEFN = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser', 'ROIs', 'dynamic_{}_crops.json'.format(folder))
        self.MOVEMENTS_SAVEFN = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser', 'movements', 'dynamic_{}_{}_movements.json'.format(folder, '{}'))
        
        # Ensure the directories where the crops and movements are saved exist
        os.makedirs(os.path.dirname(self.CROPS_SAVEFN), exist_ok=True)
        os.makedirs(os.path.dirname(self.MOVEMENTS_SAVEFN), exist_ok=True)
        
        if no_data_load:
            # no_data_load was speciefied, skip all data loading
            pass
        else:
            self.stacks = load_data(os.path.join(self.data_path, self.folder))
            
            # Load movements and ROIs if they exists
            if self.is_rois_selected():
                self.loadROIs()
            
            if self.is_measured():
                self.loadAnalysedMovements()

            self.antenna_level_correction = self._getAntennaLevelCorrection(folder)
            if self.antenna_level_correction == False:
                print('No antenna level correction value for fly {}'.format(folder))
        
        # For cahcing frequently used data
        self.cahced = {'3d_vectors': None}
    
        self.stop_now = False
        
        


    def __fileOpen(self, fn):
        with open(fn, 'r') as fp:
            data = json.load(fp)
        return data

    
    def __fileSave(self, fn, data):
        with open(fn, 'w') as fp:
            json.dump(data, fp)

    
    def list_imagefolders(self, list_special=True):
        '''
        Returns a list of the images containing folders (subfolders).
        
        list_special        Sets wheter to list also image folders with suffixes
        '''
        image_folders = []
        special_image_folders = []

        for key in self.stacks.keys():
            try:
                horizontal, vertical = ast.literal_eval(key)
            except (SyntaxError, ValueError):
                special_image_folders.append('pos'+key)   
                continue

            image_folders.append('pos'+key)

        #image_folders = [fn for fn in os.listdir(os.path.join(self.data_path, self.folder)) if os.path.isdir(os.path.join(self.data_path, self.folder, fn))]
        
        return sorted(image_folders) + sorted(special_image_folders)

    def get_specimen_directory(self):
        return os.path.join(self.data_path, self.folder)

    
    def list_images(self, image_folder):
        '''
        List all image filenames in an image folder
        
        FIXME: Alphabetical order not right because no zero padding

        '''
        return sorted([fn for fn in os.listdir(os.path.join(self.data_path, self.folder, image_folder)) if fn.endswith('.tiff') or fn.endswith('.tif')])
    
    def getFolderName(self):
        '''
        Return the name of the data (droso) folder, such as DrosoM42
        '''
        return self.folder
   
    def get_specimen_name(self):
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
    
    def get_antenna_level_correction(self):
        '''
        Return the antenna level correction or if no correction exists, False.
        '''
        return self._getAntennaLevelCorrection(self.folder)


    def _load_descriptions_file(self):
        '''
        Finds and loads the descriptions file that contains imaging parameters,
        fly sex, age, name, and (in current versions) image_folders for each setting
        '''

        descriptions = []

        # The descriptions file can be in 2 locations depending version of the
        # pupil_imsoft.
        # The old version contains only one set of imaging_parameters, common
        # to all of the image_folders (and thus inaccurate if settings changed)
        current_version_loc = os.path.join(self.data_path, self.folder, self.folder+'.txt')
        old_version_loc = os.path.join(self.data_path, self.folder+'.txt')
        
        #print(current_version_loc)
        #print(old_version_loc)
        
        
        if os.path.exists(current_version_loc):
            
            with open(current_version_loc, 'r') as fp:
                for line in fp:
                    descriptions.append(line)
            return descriptions

        if os.path.exists(old_version_loc):

            with open(old_version_loc, 'r') as fp:
                for line in fp:
                    descriptions.append(line)
            return descriptions

        return descriptions

    def get_imaging_parameters(self, image_folder):
        '''
        Returns the imaging parameters for the image_folder byt reading the newest
        matching entry in the destriptions file (pupil_imsoft).

        The descriptions file made by pupil_imsoft contains imaging_parameters
        and imaged image_folders beneath. New entries are appended to this file.
        '''

        parameters = []
        
        try:
            self.descriptions_file
        except AttributeError:
            self.descriptions_file = self._load_descriptions_file()
        
        # Find from the bottom, where the image_folder shows up the first time
        location = 0
        for location, line in enumerate(reversed(self.descriptions_file)):
            #print(line)
            if image_folder in line.strip('\n'):
                break
        

        # If not found
        if location == len(self.descriptions_file)-1:
            return ''
        
        # If found, get the parameters
        parameters = []
        at_parameters = False
        for line in reversed(self.descriptions_file[0:len(self.descriptions_file)-location-2]):
            
            line = line.strip('\n')
            if not line:
                continue
            
            #print(line)
            
            if self.folder+'\\' in line:
                # If foldername entry    
                if at_parameters == True:
                    break

            else:
                # If parameter name+value entry
                parameters.append(line)
                at_parameters = True
        
        return reversed(parameters)


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
                        
            # ROIs smaller than 7 pixels a side are not loaded
            good_rois = []
            for i_roi in range(len(ROIs)):
                if not (ROIs[i_roi][2] < 7 and ROIs[i_roi][3] < 7):
                    good_rois.append(ROIs[i_roi])
            ROIs = good_rois

            pos = self.getPosFolder(image_fn)
            horizontal, pitch = angles_from_fn(pos)
            pos = pos[3:]

            # ROI belonging to the eft/right eye is determined solely by
            # the horizontal angle when only 1 ROI exists for the position
            if len(ROIs) == 1:
                
                if horizontal > 0:
                    self.ROIs['left'][pos] = ROIs[0]
                else:
                    self.ROIs['right'][pos] = ROIs[0]

            # If there's two ROIs
            elif len(ROIs) == 2:
                
                if ROIs[0][0] > ROIs[1][0]:
                    self.ROIs['left'][pos] = ROIs[0]
                    self.ROIs['right'][pos] = ROIs[1]
                else:
                    self.ROIs['left'][pos]= ROIs[1]
                    self.ROIs['right'][pos] = ROIs[0]
            
            else:
                print('Warning. len(ROIs) == {} for {}'.format(len(ROIs), image_fn))

        self.N_folders_having_rois = len(marker_markings)
        
        print('ROIs left: {}'.format(len(self.ROIs['left'])))
        print('ROIs right: {}'.format(len(self.ROIs['right'])))
        
    def selectROIs(self, **kwargs):
        '''
        Selecting the ROIs from the loaded images.
        Currently, only the first frame of each recording is shown.

        kwargs      Passed to the marker constructor
        '''
        
        to_cropping = [stacks[0][0] for str_angles, stacks in self.stacks.items()]

        fig, ax = plt.subplots()
        marker = Marker(fig, ax, to_cropping, self.CROPS_SAVEFN, **kwargs)
        marker.run()


    def select_ROIs(self):
        return self.selectROIs()


    def isROIsSelected(self):
        '''
        Returns True if a file for crops/ROIs is found.
        '''
        return os.path.exists(self.CROPS_SAVEFN)
    

    def is_rois_selected(self):
        return self.isROIsSelected()
    

    def count_roi_selected_folders(self):
        '''
        Returns the number of imagefolders that have ROIs selected
        '''
        if self.is_rois_selected():
            return self.N_folders_having_rois
        else:
            return 0

    def folder_has_rois(self, image_folder):
        '''
        Returns True if for specified image_folder at least one
        ROI exsits. Otherwise False.
        ''' 
        try:
            self.ROIs
        except AttributeError:
            return False

        if self.get_rois(image_folder) != []:
            return True

        return False


    def get_rois(self, image_folder):
        rois = []
        for eye in ['left', 'right']:
            try:
                roi = self.ROIs[eye][image_folder[3:]]
                rois.append(roi)
            except KeyError:
                continue
        return rois


    def isMovementsAnalysed(self):
        '''
        Returns (True, True) if analyseMovement results can be found for the fly and bot eyes.
        '''
        print(self.MOVEMENTS_SAVEFN)
        return (os.path.exists(self.MOVEMENTS_SAVEFN.format('left')), os.path.exists(self.MOVEMENTS_SAVEFN.format('right')))


    def is_measured(self):
        return all(self.isMovementsAnalysed())


    def folder_has_movements(self, image_folder):
        '''
        Returns True if for specified image_folder has movements
        measured. Otherwise False.
        '''
        try:
            self.movements
        except AttributeError:
            return False

        if any([image_folder[3:] in self.movements[eye].keys()] for eye in ['left', 'right']):
            return True
        return False


    def loadAnalysedMovements(self):
        self.movements = {}
        with open(self.MOVEMENTS_SAVEFN.format('right'), 'r') as fp:
            self.movements['right'] = json.load(fp)
        with open(self.MOVEMENTS_SAVEFN.format('left'), 'r') as fp:
            self.movements['left'] = json.load(fp)
        

    def load_analysed_movements(self):
        return self.loadAnalysedMovements()


    def measure_both_eyes(self, **kwargs):
        for eye in ['left', 'right']:
            self.analyseMovement(eye, **kwargs)


    def analyseMovement(self, eye, only_folders=None):
        '''
        Performs cross-correlation analysis for the selected pseudopupils (ROIs, regions of interest)
        using Movemeter package.

        If pseudopupils/ROIs haven't been selected, calls method self.selectROIs.
        Movements are saved into a tmp directory.

        INPUT ARGUMENTS         DESCRIPTION
        eye                     'left' or 'right'
        only_folders            Analyse only image folders in the given list (that is only_folders).

        Cross-correlation analysis is the slowest part of the DrosoM pipeline.
        '''
        
        self.movements = {}
        
        if not os.path.exists(self.CROPS_SAVEFN):
            self.selectROIs() 
        self.loadROIs()
        

        angles = []
        stacks = []
        ROIs = []

        if not self.ROIs[eye] == {}:

            #print(self.ROIs)
            for angle in self.stacks:
                #if angle in str(self.ROIs[eye].keys()):
                
                # Continue if no ROI for this eye exists
                try :
                    self.ROIs[eye][angle]
                except KeyError:
                    continue

                # Continue if only_folders set and the angle is not in
                # the only folders
                if only_folders and not 'pos'+angle in only_folders:
                    continue

                # Fuse if only one frame per repetition
                if len(self.stacks[angle][0]) == 1:
                    fuse = True
                else:
                    fuse = False
                if fuse:
                    fused = []
                    for i_repetition in range(len(self.stacks[angle])):
                        fused += self.stacks[angle][i_repetition]
                
                    self.stacks[angle] = [fused]

                for i_repetition in range(len(self.stacks[angle])):
                    angles.append(angle)
                    stacks.append( self.stacks[angle][i_repetition] )
                    ROIs.append( [self.ROIs[eye][angle]] )

            
            if ROIs == []:
                return None


            # Old upscale was 4
            meter = Movemeter(upscale=10)
            meter.setData(stacks, ROIs)
            
            for stack_i, angle in enumerate(angles):
                
                if self.stop_now:
                    self.stop_now = False
                    self.movements = {}
                    print('{} EYE CANCELLED'.format(eye.upper()))
                    return None

                
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
        else:
            self.movements = {}
            
        # If only_folders set ie. only some angles were (re)measured,
        # load previous movements also for saving
        if only_folders:
            with open(self.MOVEMENTS_SAVEFN.format(eye), 'r') as fp:
                 previous_movements = json.load(fp)
            
            # Update previous movements with the new movements and set
            # the updated previous movements to be the current movements
            previous_movements.update(self.movements)
            self.movements = previous_movements


        # Save movements
        with open(self.MOVEMENTS_SAVEFN.format(eye), 'w') as fp:
            json.dump(self.movements, fp)
        
        
        #for key, data in self.movements.items():
        #    plt.plot(data['x'])
        #    plt.plot(data['y'])
        #    plt.show()


    def measure_movement(self, *args, **kwargs):
        return self.analyseMovement(*args, **kwargs)


    def timePlot(self):
        '''
        Not sure anymore what this is but probably movement magnitude over time.

        UPDATE
        probably the magnitude of the movement as a function of ISI
        '''

        data = []
        
        for eye in ['left', 'right']:
            for angle in self.movements[eye]:
                data.extend(self.movements[eye][angle])

        data.sort(key=lambda x: x['time'])
        
        for d in data:
            print(d['time'])

        X = [0]
        #X.extend( [(datetime.datetime.strptime(data[i]['time'], '%Y-%m-%d %H:%M:%S.%f') - datetime.datetime.strptime(data[i-1]['time'], '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
        #    for i in range(1, len(data))] )
        
        for i in range(1, len(data)):
            # We need these try blocks because sometimmes seconds are integer and we have no %f
            try:
                this_time = datetime.datetime.strptime(data[i]['time'], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                this_time = datetime.datetime.strptime(data[i]['time'], '%Y-%m-%d %H:%M:%S') 
            try:
                previous_time = datetime.datetime.strptime(data[i-1]['time'], '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                previous_time = datetime.datetime.strptime(data[i-1]['time'], '%Y-%m-%d %H:%M:%S') 
            
            X.append((this_time-previous_time).total_seconds())

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


    def get_movements_from_folder(self, image_folder):
        '''
        
        '''
        data = {}
        for eye in ['left', 'right']:
            try:
                data[eye] = self.movements[eye][image_folder[3:]]
            except KeyError:
                pass
        
        return data


    def get_raw_xy_traces(self, eye):
        '''
        Return angles, values
        angles      Each recorded fly orientation in steps
        values      X and Y
        '''
        angles = [list(ast.literal_eval(angle)) for angle in self.movements[eye]]
        movement_dict = [self.movements[eye][str(angle)] for angle in angles]
        
        return angles, movement_dict


    def get2DVectors(self, eye, mirror_horizontal=True, mirror_pitch=True, correct_level=True):
        '''
        Creates 2D vectors from the movements analysis data.
            Vector start point: Pupils position at the firts frame
            Vector end point: Pupil's position at the last frame

        mirror_pitch    Should make so that the negative values are towards dorsal and positive towards frontal
                            (this is how things on DrosoX were)
        '''

        # Make the order of angles deterministic
        sorted_angle_keys = sorted(self.movements[eye])

        angles = [list(ast.literal_eval(angle.split(')')[0]+')' )) for angle in sorted_angle_keys]
        values = [self.movements[eye][angle] for angle in sorted_angle_keys]

     
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
        
    
    def _get_3d_vectors(self, eye, angle_tagged=False, correct_level=True, normalize_length=0.1):
        '''
        Returns 3D vectors and their starting points.
    
        correct_level           Use estimated antenna levels
        '''
        angles, X, Y = self.get2DVectors(eye, mirror_pitch=False, mirror_horizontal=True,
                correct_level=False)
        
        
        N = len(angles)

        points = np.zeros((N,3))
        vectors = np.zeros((N,3))

       
        for i, (angle, x, y) in enumerate(zip(angles, X, Y)):
            horizontal, pitch = angle

            point0 = camera2Fly(horizontal, pitch)
            point1 = camvec2Fly(x, y, horizontal, pitch, normalize=normalize_length)
            
            if correct_level:
                rotation = -self.antenna_level_correction
                point0 = rotate_about_x(point0, rotation)
                point1 = rotate_about_x(point1, rotation)
            
            x0,y0,z0 = point0
            x1,y1,z1 = point1

            #vectors.append( (tuple(angle), (x0,x1), (y0,y1), (z0, z1)) )
            points[i] = np.array(point0)            
            vectors[i] = np.array(point1) - points[i] 

        if angle_tagged:
            return points, vectors, angles
        else:
            return points, vectors


    def get_recording_time(self, recording_name, i_rep=0):
        '''
        Returns the timestamp of a recording.

        angle       Recording name, such as
        '''

        angle = recording_name.lstrip('pos')

        for eye in ['left', 'right']:
            try:
                return self.movements[eye][angle][i_rep]['time']
            except KeyError:
                pass

        raise ValueError(('self.movements[left/right] has no (angle) key {}'
                'List of available angles: {}').format(angle, list(self.movements['left'].keys())+list(self.movements['right'].keys())))

    def stop(self):
        '''
        Stop long running activities (now measurement).
        '''
        self.stop_now = True


class ShortNameable:
    '''
    Inheriting this class adds getting and setting
    short_name attribute, and style for matplotlib text.
    '''

    def get_short_name(self):
        '''
        Returns the short_name of object or an emptry string if the short_name
        has not been set.
        '''
        try:
            return self.short_name
        except AttributeError:
            return ''

    def set_short_name(self, short_name):
        self.short_name = short_name

    
    #def get_short_name_style(self):
    #    try:
    #        return self.short_name_style
    #    except AttributeError:
    #        return 'normal'

    #def set_short_name_style(self, style):
    #    self.short_name_style = syle


class MAverager(VectorGettable, ShortNameable):
    '''
    Combining and averaging results from many MAnalyser objects.
    
    MAverager acts like MAnalyser object for getting data (like get2DVectors)
    but lacks the movement analysis (cross-correlation) related parts.
    '''
    def __init__(self, manalysers, short_name=''):
        
        self.manalysers = manalysers

    def get_N_specimens(self):
        return len(self.manalysers)

    def get_specimen_name(self):
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

# COMMENTING OUT 2D AVERAGING BECAUSE IT'S JUST A BAD APPROXIMATE VERSION OF THE 3D VERSION
#
#    @staticmethod
#    def findDistance(point1, point2):
#        '''
#        Returns PSEUDO-distance between two points in the rotation stages angles coordinates (horizontal_angle, vertical_angle).
#        
#        It's called pseudo-distance because its not likely the real 3D cartesian distance + the real distance would depend
#        on the radius that is in our angles coordinate system omitted.
#
#        In the horizontal/vertical angle system, two points may seem to be far away but reality (3D cartesian coordinates)
#        the points are closed to each other. For example, consider points
#            (90, 10) and (90, 70)
#        These points are separated by 60 degrees in the vertical (pitch) angle, but because the horizontal angle is 90 degrees
#        in both cases, they are actually the same point in reality (with different camera rotation)
#
#        INPUT ARGUMENTS     DESCRIPTION
#        point1              (horizontal, vertical)
#
#
#        TODO:   - Implement precise distance calculation in 3D coordinates
#        '''
#        # Scaler: When the horizontal angle of both points is close to 90 or -90 degrees, distance
#        # should be very small
#        #scaler = abs(math.sin((point1[0] + point2[0])/ 2))
#        # All this is probably wrong, right way to do this is calculate distances on a sphere
#        #return scaler * math.sqrt( (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2 )
#    
#        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
#    
#
#
#    # TODO REMOVEME
#    def get2DVectors(self, eye):
#        '''
#        N-nearest neighbour interpolation of 2D vectors, for more see MAnalyser.get2DVectors method.
#        
#        N-NEAREST NEIGHBOUR
#        
#        Here N-nearest neighbour means, that for each point to be interpolated, we cruerly select N nearest neighbours
#        whose average sets value to the interpolated point. N = N_manalysers, meaning the number of recorded flies.
#        
#        his interpolation method was selected because the way how data was sampled; For each fly, the sampling grid
#        was similar (every 10 degrees in horizontal and vertical angles, every 5 degrees in horizontal near frontal area),
#        so weighting the values offers no extra.
#
#
#        Temporal solution, better would be to make vectors in 3D cartesian coordinates
#        and then interpolate these and then average.
#
#        TODO:   - there's optimization work, see fixme markings in the code
#                - see self.getDistance, this would need work
#        '''
#        
#        N = len(self.manalysers)
#
#        # Getting the 2D vectors
#        ANGLES, X, Y = ([], [], [])
#        for analyser in self.manalysers:
#            angles, x, y = analyser.get2DVectors(eye)
#            ANGLES.extend(angles)
#            X.extend(x)
#            Y.extend(y)
#        
#        # Interpolation borders or "limits" by taking extreme values of horizontal
#        # and vertical angles.
#        # FIXME: What is there's a single outliner far away from others, we have not enough data
#        # to interpolate in between?
#        HORIZONTALS = [hor for (hor,pit) in ANGLES]
#        VERTICALS = [pit for (hor,pit) in ANGLES]
#        horlim = [np.min(HORIZONTALS), np.max(HORIZONTALS)]
#        verlim = [np.min(VERTICALS), np.max(VERTICALS)]
#        
#        # Performing the interpolation
#        INTP_ANGLES, INTP_X, INTP_Y = ([], [], [])
#        
#        for hor in np.arange(*horlim, self.intp_step[0]):
#            for ver in np.arange(*verlim, self.intp_step[1]):
#                
#                # Here we find N closest point to the point being interpolated
#                # This is SLOW (FIXME): We wouldn't have to calculate all points if the AGNLES
#                # would be somehow better sorted
#                distances_and_indices = [[self.findDistance((hor,ver), angle), i] for (i,angle) in enumerate(ANGLES)]
#                distances_and_indices.sort(key=lambda x: x[0])
#                
#                N_closest = [d_and_i[1] for d_and_i in distances_and_indices[0:N]]
#                
#                INTP_X.append( np.mean([X[i] for i in N_closest]) )
#                INTP_Y.append( np.mean([Y[i] for i in N_closest]) )
#                INTP_ANGLES.append((hor,ver))
#                
#        return INTP_ANGLES, INTP_X, INTP_Y
#
#    def nearest_neighbour_3d(self, point_A, points_B, max_distance=None):
#        '''
#        Return the nearest point to the point_A from points_B.
#
#        point_A         1D np.array [x0, y0, z0]
#        points_B        2D np.array [ [x1,y1,z1], [z2,y2,z2], ... ]
#        '''
#
#        distances = np.linalg.norm(points_B - point_A, axis=1)
#        
#        i_shortest = np.argmin(distances)
#
#        if max_distance:
#            if distances[i_shortest] > max_distance:
#                return False
#            
#        return i_shortest
#
# OLD, BEFORE VECTORIZATION
#        mindist = np.inf
#        argmin = None
#    
#        #ax, ay, az = point_A
#
#        for vector in vectors:
#            #angle, (x0, x1), (y0, y1), (z0, z1) = vector
#            pointB, vectorsB = vector
#
#            #distance = np.sqrt( (ax-x0)**2 + (ay-y0)**2 + (az-z0)**2 )
#
#            distance = np.linalg.norm(pointB-np.asarray(pointA))
#
#            if distance < mindist:
#                mindist = distance
#                argmin = vector
#
#            #print(mindist)
#
#        if mindist > max_distance:
#            return False
#
#        return argmin
#    
#    @staticmethod
#    def vector_length(vector):
#        return np.linalg.norm(vector)
#
##        if type(vector) == type([]):
##            angle, x,y,z = vector
##            length = np.sqrt( (x[0]-x[1])**2 + (y[0]-y[1])**2 + (z[0]-z[1])**2 )
##        elif type(vector) == type(np.array([0])):
##            x,y,z = vector
##            length = np.sqrt( (x)**2 + (y)**2 + (z)**2 )
##        return length
##
#    def average_vector_3d(self, point, vectors):
#        '''
#        Average vectors and return a vector at point point.
#
#        DIDNT WORK BECAUSE VECTORS NOT ON THE SAME POINT and WE
#        WANT THE VECTOR TO BE TANGENTIAL TO THE SPEHRE
#        '''
#        
#        #real_vectors = [x[1] for x in vectors]
#        #real_vectors = vectors[1]
#        #for point, vec in vectors:
#        #    #angle, x, y, z = vector
#        #    
#        #    #X = x[1] - x[0]
#        #    #Y = y[1] - y[0]
#        #    #Z = z[1] - z[0]
#        #    #real_vectors.append(np.array([X,Y,Z]))
#        # 
#        #    #real_vectors.append(vec)
#
#
#        av = np.mean(vectors, axis=0)
#        if self.vector_length(av) != 0:
#
#            av += np.array(point)
#            av = force_to_tplane(point, av)
#
#
#            
#            for i in range(0,len(vectors)):
#                wanted_len = self.vector_length(vectors[i])
#                
#                if wanted_len != 0:
#                    break
#
#            av -= np.array(point)
#            av = (av / self.vector_length(av) * wanted_len)
#        else:
#            av = np.array([0,0,0])
#            pass
#        #x,y,z = point
#        
#        #return (angle_tag, (x, av[0]), (y, av[1]), (z, av[2]) )
#        return av
#        
#
    def get_3d_vectors(self, eye, correct_level=True, normalize_length=0.15):
        '''
        Equivalent to MAnalysers get_3d_vectors but interpolates with N-nearest
        neughbours.
        '''
        interpolated = [[],[]]
        
        R = 1
        intp_dist = (2 * R * np.sin(math.radians(self.intp_step[0])))
        
        vectors_3d = []

        for analyser in self.manalysers:
            vec = analyser.get_3d_vectors(eye, correct_level=True,
                    normalize_length=normalize_length)
            

            vectors_3d.append(vec)
        
        intp_points = optimal(np.arange(-90, 90.01, self.intp_step[0]), np.arange(0, 360.01, self.intp_step[1]))
        

        for intp_point in intp_points:
            
            nearest_vectors = []
            for vectors in vectors_3d:
                i_nearest = nearest_neighbour(intp_point, vectors[0], max_distance=intp_dist)
                if not i_nearest is False:
                    nearest_vectors.append(vectors[1][i_nearest])

            if len(nearest_vectors) > len(vectors_3d)/2:
                avec = mean_vector(intp_point, nearest_vectors)
                interpolated[0].append(np.array(intp_point))
                interpolated[1].append(avec)
        
        return np.array(interpolated[0]), np.array(interpolated[1])
        

