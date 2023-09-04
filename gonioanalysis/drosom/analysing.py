'''Main MAnalyser classes for goniometric motion analysis.
'''

import os
import json
import ast
import math
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree

from gonioanalysis.drosom.loading import load_data, angles_from_fn, arange_fns
from gonioanalysis.coordinates import (
        camera2Fly,
        camvec2Fly,
        rotate_about_x,
        nearest_neighbour,
        mean_vector,
        optimal_sampling,
        where_vertical_between,
        )
from gonioanalysis.directories import ANALYSES_SAVEDIR, PROCESSING_TEMPDIR
from gonioanalysis.rotary_encoders import to_degrees, step2degree, DEFAULT_STEPS_PER_REVOLUTION

from roimarker import Marker
from movemeter import Movemeter



class AnalyserBase:
    '''Common base class for all analysers.
    
    Features
    --------
    - setting vertical and horizontal ange limits
    - exposing selected attributes as UI options

    Attributes
    ----------
    ui_options : dict
        Info about attributes should be exposed in a user interface. Each item
        is another dictionary with keys "help" (help string) and "type" (type
        conversion function such as int).
    va_limits, ha_limits : list
        Vertical angle and horizontal angle limits for get_3d_vectors.
    '''

    def __init__(self):
        self.ui_options = {}
        
        self.va_limits = [None, None]
        self.ha_limits = [None, None]
        self.alimits_reverse = False


    def get_ui_options(self):
        '''Lists UI options for the analyser: Their values, help and convert.
        '''
        dictionary = []
        for key in self.ui_options:
            dictionary[key] = {}
            dictionary[key]['value'] = getattr(self, key, None)
            dictionary[key]['help'] = self.ui_options[key]['help']
            dictionary[key]['type'] = self.ui_options[key]['type']
        
        return dictionary

    def set_ui_options(self, dictionary):
        '''
        '''
        for key, item in dictionary.items():
            if key in self.ui_options:
                convert = self.ui_options[key].get('type', str)
                setattr(self, key, convert(item))
            else:
                valid_keys = list(self.ui_options.keys())
                raise KeyError(f'Key "{key}" not valid. Valid keys are {valid_keys}')
            

    def set_angle_limits(self, va_limits=(None, None), reverse=False):
        '''
        Limit get_3d_vectors

        All units in degrees.
        '''
        self.va_limits = va_limits
        self.alimits_reverse = reverse



class MAnalyser(AnalyserBase):
    '''Motion analysis for GonioImsoft data.

    Attributes
    ----------
    data_path : string
        A full file path to the specimen folder's location.
    folder : string
        The name of the specimen folder.
    active_analysis : string
        Name of the active analysis
    ROIs
    movements : dict
        Nested dictionary of the measured 2D movements from Movemeter.
        
        Nested structure
        -----------------
        self.movements[eye][angle][i_repeat][x/y/time]
            eye = "left" or "right"
            angle = recording_name.lstrip('pos') // for example angle="(0, 0)_uv"
    eyes : tuple of strings
        Default ("left", "right").
    vector_rotation : float or None
        Rotation of 2D vectors (affects 3D)
    imagefolder_skiplist : dict
    '''

    def __init__(self, data_path, folder, clean_tmp=False, no_data_load=False,
                 active_analysis=''):
        '''Initialize the MAnalyser object.

        Arguments
        ---------
        no_data_load : bool
            If True, skips loading data at constructing the object (use if
            needing many short lived objects.
        active_analysis : string
            Name of the activated analysis
        '''
        super().__init__()
        self._no_data_load = no_data_load
        
        self.ROIs = None 
        self.data_path = data_path
        self.folder = folder

        
        # Skip image_folders. i_repeat
        self.imagefolder_skiplist = {}

        # Python dictionary for linked data
        self.linked_data = {}
        
        self.manalysers = [self]
        self.eyes = ("left", "right")
        self.vector_rotation = None
        
        # Different file or folder paths
        self._rois_skelefn = 'rois_{}{}.json' # specimen_name active_analysis
        self._movements_skelefn = 'movements_{}_{}{}.json' # specimen_name, eye, active_analysis
        self._skiplist_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', folder, 'imagefolder_skiplist.json')
        self._crops_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', folder, self._rois_skelefn.format(folder, ''))
        self._movements_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', folder, self._movements_skelefn.format(folder, '{}', ''))
        self._link_savedir= os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', folder, 'linked_data')
        
        
        # Load some things
        if no_data_load == False:
            self.stacks = load_data(os.path.join(self.data_path, self.folder))

            if os.path.isfile(self._skiplist_savefn):
                with open(self._skiplist_savefn, 'r') as fp:
                    self.imagefolder_skiplist = json.load(fp)
            
            self.antenna_level_correction = self._getAntennaLevelCorrection(folder)

            self.load_linked_data()

            # Ensure the directories where the crops and movements are saved exist
            os.makedirs(os.path.dirname(self._crops_savefn), exist_ok=True)
            os.makedirs(os.path.dirname(self._movements_savefn), exist_ok=True)


        # Set the active analysis, and loads ROIs and movements if available
        self.active_analysis = active_analysis

        self.stop_now = False
    
        # No data load to affect only this constructor
        self._no_data_load = False
        
        # Info data about available UI options
        self.ui_options = {
                'vector_rotation': {'help': 'Vector rotation in the 2D imaging plane', 'type': float}
                }
    

    @property
    def name(self):
        return self.folder
    
    @name.setter
    def name(self, name):
        self.folder=name


    @property
    def active_analysis(self):
        if self.__active_analysis == '':
            return 'default'
        else:
            return self.__active_analysis


    @active_analysis.setter
    def active_analysis(self, name):
        '''Sets the active analysis and loads ROIs an movements.

        Arguments
        ---------
        name : string
            The analysis default is "default" or "" (empty string).
        '''
        
        if name == 'default':
            name = ''

        self.__active_analysis = name
        
        if name == '':
            self._crops_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', self.folder, self._rois_skelefn.format(self.folder, ''))
            self._movements_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', self.folder, self._movements_skelefn.format(self.folder, '{}', ''))
        else:
            self._crops_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', self.folder, self._rois_skelefn.format(self.folder, '_'+name))
            self._movements_savefn = os.path.join(PROCESSING_TEMPDIR, 'MAnalyser_data', self.folder, self._movements_skelefn.format(self.folder, '{}', '_'+name))
    
        if self._no_data_load == False:

            if self.are_rois_selected():
                self.load_ROIs()
            else:
                try:
                    del self.ROIs
                except AttributeError:
                    pass
            if self.is_measured():
                self.load_analysed_movements()
            else:
                try:
                    del self.movements
                except AttributeError:
                    pass
            



    def list_analyses(self):
        '''Returns a list of existing analysis names.
        '''
        
        manalyser_dir = os.path.dirname(self._movements_savefn)

        if os.path.isdir(manalyser_dir):
            fns = [fn for fn in os.listdir(manalyser_dir) if
                    self._movements_skelefn.split('{')[0] in fn and
                    self.eyes[0] in fn]
        else:
            fns = []


        names = []

        for fn in fns:
            secondlast, last = fn.split('.')[0].split('_')[-2:]
            
            if fn.split('.')[0].split('_')[-1] in self.eyes:
                names.append('default')
            else:
                analysis = fn.split(self.eyes[0])[1].split('.')[0].removeprefix('_')
                names.append(analysis)

        return names


    def __fileOpen(self, fn):
        with open(fn, 'r') as fp:
            data = json.load(fp)
        return data

    
    def __fileSave(self, fn, data):
        with open(fn, 'w') as fp:
            json.dump(data, fp)


    def mark_bad(self, image_folder, i_repeat):
        '''
        Marks image folder and repeat to be bad and excluded
        when loading movements.

        i_repeat : int or 'all'
        '''
        
        if self.imagefolder_skiplist.get(image_folder, None) is None:
            self.imagefolder_skiplist[image_folder] = []

        self.imagefolder_skiplist[image_folder].append(i_repeat)
        
        with open(self._skiplist_savefn, 'w') as fp:
            json.dump(self.imagefolder_skiplist, fp)



    def list_rotations(self, list_special=True, special_separated=False,
            horizontal_condition=None, vertical_condition=None,
            _return_imagefolders=False):
        '''
        List all the imaged vertical-horizontal pair rotations.
        
        Arguments
        ---------
        list_special : bool
            If false, include only rotations whose folders have no suffix
        special_separated : bool
            If true, return standard and special image folders separetly.
        horizontal_condition : callable or None
            A callable, that when supplied with horizontal (in steps),
            returns either True (includes) or False (excludes).
        vertical_condition : callable or None
            Same as horizontal condition but for vertical rotation

        Returns
        -------
        rotations : list of tuples
            List of rotations.
        '''

        def check_conditions(vertical, horizontal):
            if callable(horizontal_condition):
                if not horizontal_condition(horizontal):
                    return False
            if callable(vertical_condition):
                if not vertical_condition(vertical):
                    return False
            return True
 
        standard = []
        special = []

        for key in self.stacks.keys():
            try:
                horizontal, vertical = ast.literal_eval(key)
                if check_conditions(vertical, horizontal) == False:
                    continue

            except (SyntaxError, ValueError):
                # This is now a special folder, ie. with a suffix or something else
                # Try to get the angle anyhow
                splitted = key.replace('(', ')').split(')')
                if len(splitted) == 3:
                    try:
                        horizontal, vertical = splitted[1].replace(' ', '').split(',')
                        horizontal = int(horizontal)
                        vertical = int(vertical)
                        
                        if check_conditions(vertical, horizontal) == False:
                            continue
                    except:
                        pass
                
                if _return_imagefolders:
                    special.append('pos'+key)
                else:
                    special.append((horizontal, vertical))
                continue

            if _return_imagefolders:
                standard.append('pos'+key)
            else:
                standard.append((horizontal, vertical))
        
        if not list_special:
            special = []

        if special_separated:
            return standard, special
        else:
            return standard + special

    
    def list_imagefolders(self, endswith='', only_measured=False, **kwargs):
        '''
        Returns a list of the image folders (specimen subfolders that contain
        the images).
        
        Arguments
        ---------
        only_measured : bool
            Return only image_folders with completed movement analysis
        
        See list_rotations for other allowed keyword arguments. 
        
        Returns
        -------
        image_folders : list of strings
        '''

        image_folders, special_image_folders = self.list_rotations(
                special_separated=True,
                _return_imagefolders=True,
                **kwargs)
        
        all_folders = [fn for fn in sorted(image_folders) + sorted(special_image_folders) if fn.endswith(endswith)]
        
        if only_measured:
            all_folders = [fn for fn in all_folders if self.folder_has_movements(fn)]

        return all_folders


    def get_horizontal_vertical(self, image_folder, degrees=True):
        '''
        Tries to return the horizontal and vertical for an image folder.

        image_folder
        degrees             If true, return in degrees
        '''
        # Trusting that ( and ) only reserved for the angle
        splitted = key.replace('(', ')').split(')')
        if len(splitted) == 3:
            horizontal, vertical = splitted[1].replace(' ', '').split(',')
            horizontal = int(horizontal)
            vertical = int(vertical)
        
        if degrees:
            return step2degree(horizontal), step2degree(vertical)
        else:
            return horizontal, vertical


    def get_specimen_directory(self):
        return os.path.join(self.data_path, self.folder)

    
    def list_images(self, image_folder, absolute_path=False):
        '''
        List all image filenames in an image folder
        
        FIXME: Alphabetical order not right because no zero padding
        
        image_folder        Name of the image folder
        absolute_path       If true, return filenames with absolute path instead of relative

        '''

        #fns = [fn for fn in os.listdir(os.path.join(self.data_path, self.folder, image_folder)) if fn.endswith('.tiff') or fn.endswith('.tif')]
        #fns = arange_fns(fns)
        #if absolute_path:
        #    fns = [os.path.join(self.data_path, self.folder, image_folder, fn) for fn in fns]
        
        fns = []
        # Flatten out the i_repeat structure
        for repetitions_images in self.stacks[image_folder.removeprefix('pos')]:
            fns.extend(repetitions_images)

        if not absolute_path:
            fns = [os.path.basename(fn) for fn in fns]

        return fns

   
    def get_specimen_name(self):
        '''
        Return the name of the data (droso) folder, such as DrosoM42
        '''              
        return self.folder
  

    def get_imagefolder(self, image_fn):
        '''Gets the image containing folder (for example, /a/b/c/image -> c).

        Arguments
        ---------
        image_fn : string
            Full path to the image.
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
                angles[i][1] -= self.antenna_level_correction

        return angles


    def get_antenna_level_correction(self):
        '''
        Return the antenna level correction or if no correction exists, False.
        '''
        return self._getAntennaLevelCorrection(self.folder)


    def get_imaging_parameters(self, image_folder):
        '''
        Returns a dictionary of the Gonio Imsoft imaging parameters.
        The dictionary is empty if the descriptions file is missing.

        image_folder : string
        '''
        
        parameters = {}


        fn = os.path.join(self.data_path, self.folder, image_folder, 'description.txt')
        
        if not os.path.isfile(fn):
            # Fallback for older Imsoft data where only
            # one descriptions file for each imaging
            old_fn = os.path.join(self.data_path, self.folder, self.folder+'.txt')
            if os.path.isfile(old_fn):
                fn = old_fn
            else:
                return {}


        with open(fn, 'r') as fp:
            for line in fp:
                if line.startswith('#') or line in ['\n', '\r\n']:
                    continue
                split= line.strip('\n\r').split(' ')

                if len(split) >= 1:
                    parameters[split[0]] = split[1]
                else:
                    parameters[split[0]] = ''
        
        return parameters

    def get_specimen_age(self):
        '''
        Returns age of the specimen, or None if unkown.

        If many age entries uses the latest for that specimen.
        '''
        
        try:
            self.descriptions_file
        except AttributeError:
            self.descriptions_file = self._load_descriptions_file()
        
        for line in self.descriptions_file[::-1]:
            if line.startswith('age '):
                return line.lstrip('age ')

        return None


    def get_specimen_sex(self):
        '''
        Returns sex of the specimen, or None if unkown.

        If many sex entries uses the latest for that specimen.
        '''
        
        try:
            self.descriptions_file
        except AttributeError:
            self.descriptions_file = self._load_descriptions_file()
        
        for line in self.descriptions_file[::-1]:
            if line.startswith('sex '):
                return line.lstrip('sex ').strip(' ').strip('\n')

        return None

    
    def get_imaging_frequency(self, image_folder, fallback_value=100.):
        '''
        Return imaging frequency (how many images per second) for an image folder
        by searching for frame_length field in the descriptions file.

        Arguments
        ---------
        image_folder : string or None
            Name. If None (not recommended), returns the imaging frequency from
            any image folder that is available (~random).
        fallback_value : float or None
            What to return if fs cannot be determined.

        Returns
        -------
        fs : float
            The imaging frequency. Default of 100 HzNone if the imaging frequency could not be determined.
        '''
        if image_folder is None:
            folders = self.list_imagefolders()
        else:
            folders = [image_folder]
        
        for folder in folders:
            fs = self.get_imaging_parameters(folder).get('frame_length', None)
            if fs is not None:
                break

        if fs is None:
            # FIXME
            return fallback_value
        else:
            return 1/float(fs)
    
    def get_pixel_size(self, image_folder):
        '''
        Return the pixel size of the imaging.
        Currently always returns the same static value of 1.22375. 
        '''
        # Based on the stage micrometer;
        # 0.8 µm in the images 979 pixels 
        return 1/1.22376
    

    def get_rotstep_size(self):
        '''
        Returns how many degrees one rotation encoder step was
        (the return value * steps == rotation in degrees)
        '''
        return 360/DEFAULT_STEPS_PER_REVOLUTION


    def get_snap_fn(self, i_snap=0, absolute_path=True):
        '''
        Returns the first snap image filename taken (or i_snap'th if specified).

        Many time I took a snap image of the fly at (0,0) horizontal/vertical, so this
        can be used as the "face photo" of the fly. 
        '''

        snapdir = os.path.join(self.data_path, self.folder, 'snaps')
        fns = [fn for fn in os.listdir(snapdir) if fn.endswith('.tiff')]
        fns.sort()

        if absolute_path:
            fns = [os.path.join(snapdir, fn) for fn in fns]

        return fns[i_snap]


    def load_ROIs(self):
        '''
        Load ROIs (selected before) for the left/right eye.
        
        INPUT ARGUMENTS     DESCRIPTION
        eye                 'left' or 'right'
        
        DETAILS
        While selecting ROIs, both eyes are selcted simultaneously. There's
        no explicit information about from which eye each selected ROI is from.
        Here we reconstruct the distinction to left/right using following way:
            1 ROI:      horizontal angle determines
            2 ROIs:     being left/right in the image determines
        
        Notice that this means that when the horizontal angle is zero (fly is facing towards the camera),
        image rotation has to be so that the eyes are on image's left and right halves.
        '''

        self.ROIs = {'left': {}, 'right': {}}

        with open(self._crops_savefn, 'r') as fp:
            marker_markings = json.load(fp)

        for image_fn, ROIs in marker_markings.items():
            
            # Since use to relative filenames in the ROIs savefile
            image_fn = os.path.join(self.data_path, self.folder, image_fn)

            # ROIs smaller than 7 pixels a side are not loaded
            good_rois = []
            for i_roi in range(len(ROIs)):
                if not (ROIs[i_roi][2] < 7 and ROIs[i_roi][3] < 7):
                    good_rois.append(ROIs[i_roi])
            ROIs = good_rois

            pos = self.get_imagefolder(image_fn)
            try:
                horizontal, pitch = angles_from_fn(pos)
            except:
                horizontal, pitch = (0, 0)
            pos = pos[3:]

            if '_cam' in image_fn:
                # Allows cameras 0-9
                try:
                    i_camera = int(image_fn[image_fn.index('_cam') + 4])
                    pos += f'_cam{i_camera}'
                except: pass

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
            
            elif len(ROIs) > 2:

                # With drift_correction, many ROIs are possible
                #if self.active_analysis == 'drift_correction':
                #    for i_roi, roi in enumerate(ROIs):
                #        self.ROIs[f'roi{i_roi}'] = roi

                print('Warning. len(ROIs) == {} for {}'.format(len(ROIs), image_fn))

        self.N_folders_having_rois = len(marker_markings)
        
        

    def select_ROIs(self, **kwargs):
        '''
        Selecting the ROIs from the loaded images.
        Currently, only the first frame of each recording is shown.

        kwargs      Passed to the marker constructor
        '''
        
        to_cropping = [stacks[0][0] for str_angles, stacks in self.stacks.items()]

        fig, ax = plt.subplots()
        marker = Marker(fig, ax, to_cropping, self._crops_savefn,
                relative_fns_from=os.path.join(self.data_path, self.folder), **kwargs)
        marker.run()


    def are_rois_selected(self):
        '''
        Returns True if a file for crops/ROIs is found.
        '''
        return os.path.exists(self._crops_savefn)


    def count_roi_selected_folders(self):
        '''
        Returns the number of imagefolders that have ROIs selected
        '''
        if self.are_rois_selected():
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
            except:
                continue
        return rois


    def is_measured(self):
        '''
        Returns (True, True) if analyseMovement results can be found for the fly and bot eyes.
        '''
        return all((os.path.exists(self._movements_savefn.format('left')), os.path.exists(self._movements_savefn.format('right'))))



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
            if len(self.get_displacements_from_folder(image_folder)) > 0:
                return True
        return False


    def load_analysed_movements(self):
        self.movements = {}
        with open(self._movements_savefn.format('right'), 'r') as fp:
            self.movements['right'] = json.load(fp)
        with open(self._movements_savefn.format('left'), 'r') as fp:
            self.movements['left'] = json.load(fp)
        
        if self.__active_analysis == 'drift_correction':
            for key in self.movements['right']:
                if key not in self.movements['left']:
                    self.movements['left'] = self.movements['right']

        # Special analysis name: "drift_correction". If present, subtracted
        # from all the active analysis
        if 'drift_correction' in self.list_analyses() and self.__active_analysis != 'drift_correction':
            dc_analyser = MAnalyser(
                    self.data_path, self.folder, active_analysis='drift_correction')
            
            for eye in ['left', 'right']:
                for image_folder, data in dc_analyser.movements[eye].items():
                    if image_folder not in self.movements[eye]:
                        continue
                    for i_repeat in range(len(data)):
                        A = self.movements[eye][image_folder][i_repeat]
                        B = data[i_repeat]
                        for key in ['x', 'y']:
                            dcx = np.array(list(range(len(B[key]))))
                            fit = np.polynomial.polynomial.Polynomial.fit(dcx, B[key],1)
                            corrected = (np.array(A[key]) - fit(dcx) ).tolist()
                            #corrected = [a-b for a, b in zip(A[key], B[key])]
                            self.movements[eye][image_folder][i_repeat][key] = corrected


        if self.imagefolder_skiplist:
            
            for image_folder, skip_repeats in self.imagefolder_skiplist.items():
                for eye in self.eyes:

                    if self.movements[eye].get(image_folder[3:], None) is None:
                        continue
                    
                    # Iterate repeats reversed so we can just pop things
                    for i_repeat in sorted(skip_repeats)[::-1]:
                        self.movements[eye][image_folder[3:]].pop(i_repeat)

                    


    def measure_both_eyes(self, **kwargs):
        '''
        Wrapper to self.measure_movement() for both left and right eyes.
        '''
        for eye in self.ROIs:
            self.measure_movement(eye, **kwargs)


    def measure_movement(self, eye, only_folders=None,
            max_movement=30, absolute_coordinates=False, join_repeats=False,
            stop_event=None):
        '''
        Performs cross-correlation analysis for the selected ROIs (regions of interest)
        using Movemeter package.

        If ROIs haven't been selected, calls method self.selectROIs.
        Movements are saved into a tmp directory.

        INPUT ARGUMENTS         DESCRIPTION
        eye                     'left' or 'right'
        only_folders            Analyse only image folders in the given list (that is only_folders).
        max_movement            Maximum total displacement in x or y expected. Lower values faster.
        absolute_coordinates    Return movement values in absolute image coordinates
        join_repeats            Join repeats together as if they were one long recording.
        stop_event              None or threading.Event for stopping the movement measurement
            

        Cross-correlation analysis is the slowest part of the DrosoM pipeline.
        '''
        
        self.movements = {}
        
        if not os.path.exists(self._crops_savefn):
            self.selectROIs() 
        self.load_ROIs()
        

        angles = []
        stacks = []
        ROIs = []

        if not self.ROIs[eye] == {}:

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

                if join_repeats:
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
            meter = Movemeter(upscale=10, absolute_results=absolute_coordinates)
            meter.set_data(stacks, ROIs)
            
            for stack_i, angle in enumerate(angles):
                
                if stop_event and stop_event.is_set():
                    self.stop_now = True

                if self.stop_now:
                    self.stop_now = False
                    self.movements = {}
                    print('{} EYE CANCELLED'.format(eye.upper()))
                    return None

                
                print('Analysing {} eye, motion from position {}, done {}/{} for this eye'.format(eye.upper(), angle, stack_i+1, len(ROIs)))

                print("Calculating ROI's movement...")
                x, y = meter.measure_movement(stack_i, max_movement=max_movement)[0]
                
                print('Done.')
                
                try:
                    self.movements[angle]
                except KeyError:
                    self.movements[angle] = []
                
                tags = meter.get_metadata(stack_i)['Image ImageDescription'].values.split('"')
                
                # GonioImsoft start time tag in the images
                if 'start_time' in tags:
                    time = tags[tags.index('start_time') + 2]
                else:
                    time = None

                self.movements[angle].append({'x': x, 'y':y, 'time': time})

        else:
            self.movements = {}
            
        # If only_folders set ie. only some angles were (re)measured,
        # load previous movements also for saving
        if only_folders:
            with open(self._movements_savefn.format(eye), 'r') as fp:
                 previous_movements = json.load(fp)
            
            # Update previous movements with the new movements and set
            # the updated previous movements to be the current movements
            previous_movements.update(self.movements)
            self.movements = previous_movements


        # Save movements
        with open(self._movements_savefn.format(eye), 'w') as fp:
            json.dump(self.movements, fp)
        
        
        #for key, data in self.movements.items():
        #    plt.plot(data['x'])
        #    plt.plot(data['y'])
        #    plt.show()
    


    def get_time_ordered(self, angles_in_degrees=True, first_frame_only=False,
            exclude_imagefolders=[]):
        '''
        Get images, ROIs and angles, ordered in recording time for movie making.
        
        exclude_imagefolders : list
            Imagefolders to exclude

        Returns 3 lists: image_fns, ROIs, angles
                image_fns
        '''
        self.load_ROIs()

        times_and_data = []
        seen_angles = []

        for eye in self.movements:
            for angle in self.movements[eye]:
                
                if 'pos'+angle in exclude_imagefolders:
                    continue

                if not angle in seen_angles: 
                    time = self.movements[eye][angle][0]['time']
                    
                    fn = self.stacks[angle][0]
                    ROI = self.get_moving_ROIs(eye, angle)
                    deg_angle = [list(ast.literal_eval(angle.split(')')[0]+')' ))]
                    
                    if angles_in_degrees:
                        to_degrees(deg_angle)
                    
                    deg_angle = [deg_angle[0] for i in range(len(fn))]

                    times_and_data.append([time, fn, ROI, deg_angle])
                    seen_angles.append(angle)
        
        # Everything gets sorted according to the time
        times_and_data.sort(key=lambda x: x[0])
        
        image_fns = []
        ROIs = []
        angles = []
        
        if not first_frame_only:
            for time, fns, ROI, angle in times_and_data:
                image_fns.extend(fns)
                ROIs.extend(ROI)
                angles.extend(angle)
        else:
            for time, fns, ROI, angle in times_and_data:
                image_fns.append(fns[0])
                ROIs.append(ROI[0])
                angles.append(angle[0])
        
        return image_fns, ROIs, angles


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


    def get_displacements_from_folder(self, image_folder):
        '''
        Returns a list of 1D numpy arrays, which give displacement
        over time for each repeat.

        If no displacement data, returns an empty list

        Calculated from separete (x,y) data
        '''
        displacements = []
        
        for eye, data in self.get_movements_from_folder(image_folder).items():
            for repetition_data in data:
                x = repetition_data['x']
                y = repetition_data['y']
                mag = np.sqrt(np.asarray(x)**2 + np.asarray(y)**2)
                displacements.append(mag)

        return displacements


    def get_raw_xy_traces(self, eye):
        '''
        Return angles, values
        angles      Each recorded fly orientation in steps
        values      X and Y
        '''
        angles = [list(ast.literal_eval(angle)) for angle in self.movements[eye]]
        movement_dict = [self.movements[eye][str(angle)] for angle in angles]
        
        return angles, movement_dict
    

    def get_2d_vectors(self, eye, mirror_horizontal=True, mirror_pitch=True,
                       correct_level=True, repeats_separately=False,
                       mirror_movements=False):
        '''
        Creates 2D vectors from the movements analysis data.
            Vector start point: ROI's position at the first frame
            Vector end point: ROI's position at the last frame

        mirror_pitch    Should make so that the negative values are towards dorsal and positive towards frontal
                            (this is how things on DrosoX were)
        mirror_movements : bool
            If True, mirrors the movement directions (X=-X and Y=-Y)
        '''

        # Make the order of angles deterministic
        sorted_angle_keys = sorted(self.movements[eye])

        angles = [list(ast.literal_eval(angle.split(')')[0]+')' )) for angle in sorted_angle_keys]
        values = [self.movements[eye][angle] for angle in sorted_angle_keys]

        to_degrees(angles)
        
        if correct_level:
            angles = self._correctAntennaLevel(angles)

        
        if mirror_horizontal:
            for i in range(len(angles)):
                angles[i][0] *= -1
            xdirchange = -1
        else:
            xdirchange = 1
        
        if mirror_pitch:
            for i in range(len(angles)):
                angles[i][1] *= -1

        # Vector X and Y components
        # Fix here if repetitions are needed to be averaged
        # (don't take only x[0] but average)
        if repeats_separately:
            tmp_angles = []            
            X = []
            Y = []
            
            for angle, val in zip(angles, values):
                for repeat in val:
                    tmp_angles.append(angle)
                    X.append( xdirchange*(repeat['x'][-1]-repeat['x'][0]) )
                    Y.append( repeat['y'][-1]-repeat['y'][0] )

            angles = tmp_angles

        else:

            X = [xdirchange*(x[0]['x'][-1]-x[0]['x'][0]) for x in values]
            Y = [x[0]['y'][-1]-x[0]['y'][0] for x in values]
        
        if mirror_movements:
            X = [-x for x in X]
            Y = [-y for y in Y]

        if self.vector_rotation:
            r = math.radians(self.vector_rotation)
            for i in range(len(X)):
                x = X[i]
                y = Y[i]
                
                if angles[i][1] > 0:
                    sr = -1 * r
                else:
                    sr = 1 * r

                if eye == 'left':
                    sr = sr
                elif eye == 'right':
                    sr = -sr

                X[i] = x * math.cos(sr) - y * math.sin(sr)
                Y[i] = x * math.sin(sr) + y * math.cos(sr)

        return angles, X, Y


            
    def get_magnitude_traces(self, eye, image_folder=None,
            mean_repeats=False, mean_imagefolders=False,
            microns=False, _phase=False, _derivative=False):
        '''
        Get all movement magnitudes (sqrt(x**2+y**2)) from the specified eye.
        The results are returned as a dictionary where the keys are the
        angle pairs (self.movements keys)

        eye : string or None
            "left" or "right".
            None leads to taking mean where eyes overlap
        image_folder : string
            If specified, return movements from this image folder.
            Otherwise by default None, movements from all image folders.
        mean_repeats : bool
            Wheter take mean of the mean repeats.
        mean_imagefolders : bool
            Only makes sense when image_folder is None
        mean_eyes : bool
            Only makes sense when eye is None
        microns : bool
            Call self.get_pixel_size(image_folder) to convert from
            pixel units to micrometers.
        _phase: bool
            If true return phase in degrees instead.

        Returns
            if mean_repeats == True
                magnitude_traces = {angle_01: [mag_mean], ...}
            if mean_repeats == False
                magnitude_traces = {angle_01: [mag_rep1, mag_rep2,...], ...}
            
            if mean_imagefolders, there's only one key 'mean'

        '''
        alleye_magnitude_traces = {}
        
        if eye is None:
            eyes = self.eyes
        else:
            eyes = [eye]

        if image_folder is None:
            movement_keys = set().union(*[list(self.movements[eye].keys()) for eye in eyes])
        else:
            movement_keys = [image_folder[3:]]
        
        for eye in eyes:
            magnitude_traces = {}
            for angle in movement_keys:
                
                if self.movements[eye].get(angle, None) is None:
                    # Continue if data for this eye
                    continue

                magnitude_traces[angle] = []
                
                for i_repeat in range(len(self.movements[eye][angle])):
                    x = self.movements[eye][angle][i_repeat]['x']
                    y = self.movements[eye][angle][i_repeat]['y']

                    if _phase:
                        mag = np.degrees(np.arctan2(y, -np.asarray(x)))
                    else:
                        mag = np.sqrt(np.asarray(x)**2 + np.asarray(y)**2)
                    
                    if _derivative:
                        mag = np.diff(mag)

                    magnitude_traces[angle].append( mag )
                
                if mean_repeats:
                    magnitude_traces[angle] = [np.mean(magnitude_traces[angle], axis=0)]

            if magnitude_traces == {}:
                # If nothing for this eye
                continue

            if mean_imagefolders:
                tmp = np.mean([val for val in magnitude_traces.values()], axis=0)
                
                magnitude_traces = {}
                magnitude_traces['imagefoldersmean'] = tmp

            alleye_magnitude_traces[eye] = magnitude_traces

        


        if len(eyes) > 1:
            merge = {}
            # Merge (mean) eyes where one imagefolder hols data from both eyes

            angles = [list(val.keys()) for val in alleye_magnitude_traces.values()]
            angles = set().union(*angles)
            
            for angle in angles:

                data = [alleye_magnitude_traces.get(eye, {}).get(angle, None) for eye in eyes]
                data = [d for d in data if d is not None]

                merge[angle] = np.mean(data, axis=0)


            magnitude_traces = merge

        if microns and not _phase:
            for image_folder in magnitude_traces:
                pixel_size = self.get_pixel_size(image_folder)
                magnitude_traces[image_folder] = [t*pixel_size for t in magnitude_traces[image_folder]]


        return magnitude_traces

    
    def get_moving_ROIs(self, eye, angle, i_repeat=0):
        '''
        Returns a list of ROIs how they move over time.
        Useful for visualizing.
        '''

        moving_ROI = []
        
        if not self.ROIs:
            self.load_ROIs()

        movements = self.movements[eye][angle][i_repeat]
        rx,ry,rw,rh = self.ROIs[eye][angle]
        
        for i in range(len(movements['x'])):
            x = -movements['x'][i]
            y = -movements['y'][i]
            
            moving_ROI.append([rx+x,ry+y,rw,rh])
        return moving_ROI
        
    
    def get_3d_vectors(self, eye, return_angles=False, correct_level=True, repeats_separately=False, normalize_length=0.1, strict=None, vertical_hardborder=None):
        '''
        Returns 3D vectors and their starting points.
    
        correct_level           Use estimated antenna levels

        va_limits       Vertical angle limits in degrees, (None, None) for no limits
        '''
        angles, X, Y = self.get_2d_vectors(eye, mirror_pitch=False, mirror_horizontal=True,
                correct_level=False, repeats_separately=repeats_separately)
        
        
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

        # Vertical/horizontal angle limiting
        booleans = where_vertical_between(points, lower=self.va_limits[0],
                upper=self.va_limits[1], reverse=self.alimits_reverse)
        points = points[booleans]
        vectors = vectors[booleans]

        if return_angles:
            return points, vectors, angles
        else:
            return points, vectors


    def get_recording_time(self, image_folder, i_rep=0):
        '''
        Returns the timestamp of a recording, if measure_movement() method has been
        run for the recording.

        If no time is found,
            returns None

        recording_Name      Recording name
        i_rep               Return the time for recording repeat i_rep
                                By default, i_rep=0
        '''

        angle = image_folder.lstrip('pos')

        for eye in ['left', 'right']:
            try:
                return self.movements[eye][angle][i_rep]['time']
            except (KeyError, IndexError):
                pass
            except AttributeError:
                print('No time for {} because movements not analysed'.format(image_folder))
                return None
        
        return None



    def stop(self):
        '''
        Stop long running activities (now measurement).
        '''
        self.stop_now = True


    # ------------
    # LINKED DATA
    # linking external data such as ERGs to the DPP data (MAnalyser)
    # ------------

    def link_data(self, key, data):
        '''
        Data linked to the MAnalyser
        '''
        self.linked_data[key] = data


    def save_linked_data(self):
        '''
        Attempt saving the linked data on disk in JSON format.
        '''
        os.makedirs(self._link_savedir, exist_ok=True)
        
        for key, data in self.linked_data.items():
            with open(os.path.join(self._link_savedir, "{}.json".format(key)), 'w') as fp:
                json.dump(data, fp)

    
    def load_linked_data(self):
        '''
        Load linked data from specimen datadir.
        '''
        # Initialize linked data to an empty dict
        self.linked_data = {}

        # Check if linked data directory exsists, if not, the no linked data for this specimen
        if os.path.exists(self._link_savedir):

            dfiles = [fn for fn in os.listdir(self._link_savedir) if fn.endswith('.json')]
            
            for dfile in dfiles:
                with open(os.path.join(self._link_savedir, dfile), 'r') as fp:
                    data = json.load(fp)
                    self.linked_data[dfile.replace('.json', '')] = data
        

class MAverager(AnalyserBase):
    '''
    Combining and averaging results from many MAnalyser objects.
    
    MAverager acts like MAnalyser object for getting data (like get_2d_vectors)
    but lacks the movement analysis (cross-correlation) related parts.
    '''
    def __init__(self, manalysers):
        super().__init__()

        self.manalysers = manalysers

        self.interpolation = {}

        self.intp_step = (5, 5)
        
        self.eyes = manalysers[0].eyes
        self.vector_rotation = manalysers[0].vector_rotation

        self.interpolated_raw= {}
        
        # Info data about available UI options
        self.ui_options = manalysers[0].ui_options.copy() 

    def get_N_specimens(self):
        return len(self.manalysers)


    def get_specimen_name(self):
        return 'averaged_'+'_'.join([manalyser.folder for manalyser in self.manalysers])
    
    @property
    def name(self):
        return self.get_specimen_name()
    
    @name.setter
    def name(self, name):
        return


    def setInterpolationSteps(self, horizontal_step, vertical_step):
        '''
        Set the resolution of the N-nearest neighbour interpolation in Maverager.get_2d_vectors.

        INPUT ARGUMENTS
        horizontal_step
        vertical_step

        Arguments horizontal_step and vertical_step refer to the rotation stages.

        '''

        self.intp_step = (horizontal_step, vertical_step)

    
    def get_2d_vectors(self, eye, **kwargs):
        '''
        Get's the 2D movement vectors (in the camera coordinate system)
        using N_nearest neighbour interpolation and averaging.
        '''
        #Modified from get_3d_vectors
        
        interpolated = [[],[],[]]
        
        points_2d = []
        vectors_2d = []

        for analyser in self.manalysers:
            angles, X, Y = analyser.get_2d_vectors(eye, mirror_horizontal=False, mirror_pitch=False)
            vecs = [[x,y] for x,y in zip(X, Y)]
            points_2d.append(np.array(angles))
            vectors_2d.append( np.array(vecs) )
        
        vectors_2d = np.array(vectors_2d)
        
        kdtrees = [KDTree(points) for points in points_2d]

        hmin, hmax = (-90, 90)
        vmax, hmax = (-180, 180)
       
        intp_points = []
        for h in np.arange(hmin, hmax+0.01, 10):
            for v in np.arange(hmin, hmax+0.1, 10):
                intp_points.append((h,v))
        
        for intp_point in intp_points:
            
            nearest_vectors = []

            for kdtree, vectors in zip(kdtrees, vectors_2d):
                distance, index = kdtree.query(intp_point)
                
                if distance < math.sqrt(self.intp_step[0]**2+self.intp_step[1]**2):
                    nearest_vectors.append(vectors[index])

            if len(nearest_vectors) > len(vectors_2d)/2:
                avec = np.mean(nearest_vectors, axis=0)
                avec /= np.linalg.norm(avec)
                interpolated[0].append(np.array(intp_point))
                interpolated[1].append(avec[0])
                interpolated[2].append(avec[1])

        angles, x, y = interpolated
        return angles, x, y


    def get_3d_vectors(self, eye, correct_level=True, normalize_length=0.1,
            recalculate=False, strict=False, vertical_hardborder=False,
            repeats_separately=False, **kwargs):
        '''
        Equivalent to MAnalysers get_3d_vectors but interpolates with N-nearest
        neughbours.

        repeats_separately : bool
            If True, return underlying MAnalyser vectors separetly
            (same points get repeated many times)
        '''

        cachename = ';'.join([str(item) for item in [self.vector_rotation, correct_level, normalize_length, strict, vertical_hardborder]])


        if self.interpolation.get(eye, {}).get(cachename) is None or recalculate:

            interpolated = [[],[]]
            self.interpolated_raw[eye] = [] # key points str, value list of vectors

            R = 1
            intp_dist = (2 * R * np.sin(math.radians(self.intp_step[0])))
            
            vectors_3d = []

            for analyser in self.manalysers:
                analyser.vector_rotation = self.vector_rotation
                vec = analyser.get_3d_vectors(eye, correct_level=True,
                        normalize_length=normalize_length, **kwargs)
                

                vectors_3d.append(vec)
            
            if not strict:
                intp_points = optimal_sampling(np.arange(-90, 90.01, self.intp_step[0]), np.arange(0, 360.01, self.intp_step[1]))
            else:
                if eye == 'left':
                    intp_points = optimal_sampling(np.arange(-90, 0.01, self.intp_step[0]), np.arange(0, 360.01, self.intp_step[1]))
                else:
                    intp_points = optimal_sampling(np.arange(0, 90.01, self.intp_step[0]), np.arange(0, 360.01, self.intp_step[1]))

            for intp_point in intp_points:
                
                nearest_vectors = []
                for vectors in vectors_3d:
                    i_nearest = nearest_neighbour(intp_point, vectors[0], max_distance=intp_dist)
                    
                    if not i_nearest is False:
                        
                        if vertical_hardborder:
                            if np.sign(intp_point[2]) != np.sign(vectors[0][i_nearest][2]):
                                continue

                        nearest_vectors.append(vectors[1][i_nearest])

                if len(nearest_vectors) > len(vectors_3d)/2:
                    avec = mean_vector(intp_point, nearest_vectors)
                    interpolated[0].append(np.array(intp_point))
                    interpolated[1].append(avec)
                    
                    self.interpolated_raw[eye].append(nearest_vectors)

            self.interpolation[eye] = {}
            self.interpolation[eye][cachename] = np.array(interpolated[0]), np.array(interpolated[1])
            
        else:
            pass
        
        
        points, vectors = self.interpolation[eye][cachename]
        
        if repeats_separately:
            newpoints = []
            newvecs = []
            for i_point, point in enumerate(points):
                for vec in self.interpolated_raw[eye][i_point]:
                    newpoints.append(point)
                    newvecs.append(vec)
            points = np.array(newpoints)
            vectors = np.array(newvecs)

        
        # Vertical/horizontal angle limiting
        booleans = where_vertical_between(points, lower=self.va_limits[0],
                upper=self.va_limits[1], reverse=self.alimits_reverse)
        points = points[booleans]
        vectors = vectors[booleans]
        

        return points, vectors

