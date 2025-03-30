

import numpy as np


from .analysing import AnalyserBase


def insert_virt_suffix(folder_name, i_virt):
    parts = folder_name.split('_')
    parts.insert(1, f'virt{i_virt}')
    return '_'.join(parts)

def remove_virt_suffix(folder_name):
    parts = folder_name.split('_')
    if parts[1].startswith('virt'):
        i_virt = int(parts.pop(1).removeprefix('virt'))
    return '_'.join(parts), i_virt

class VirtualAnalyser(AnalyserBase):
    '''Creates a new virtual analyser based on one or more analysers

    Similar to MAverager but does not average and interpolate
    data.

    Each image folder gets a virt suffix in its name to avoid
    name collisions. For example, "pos(0,0)_bigsmoke" becomes
    "pos(0,0)_virt0_bigsmoke" (and "pos(0,0)_virt1_bigsmoke" if
    the second analyser also has this exact same folder)
    '''

    def __init__(self, name, analysers):
        super().__init__()

        self.name = name
        self.folder = name

        self.analysers = analysers
        self.eyes = analysers[0].eyes

        self.vector_rotation = 0
        self.attributes = {'yaw': 0}

    @property
    def active_analysis(self):
        return self.analysers[0].active_analysis

    @active_analysis.setter
    def active_analysis(self, name):
        for an in self.analysers:
            an.active_analysis = name


    def list_analyses(self):
        analyses = []
        for an in self.analysers:
            for analysis in an.list_analyses():
                if analysis in analyses:
                    continue
                analyses.append(analysis)
        return analyses
    
    def save_attributes(self):
        pass

    def mark_bad(self, image_folder, i_repeat, i_is_relative=True):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].mark_bad(
                folder, i_repeat, i_is_relative=i_is_relative)

    # List rotations is missing, not sure how to handle this
    # and if it would be better to remove list_rotations alltogether
    #def list_rotations(self)
    
    def list_imagefolders(self, *args, **kwargs):
        folders = []
        for i_an, an in enumerate(self.analysers):
            fols = an.list_imagefolders(*args, **kwargs)

            # Add index of the analyser to the folder suffix beginning
            fols = [insert_virt_suffix(fol, i_an) for fol in fols]

            folders.extend(fols)
        
        return folders

    def get_horizontal_vertical(self, image_folder, *args, **kwargs):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_horizontal_vertical(
                folder, *args, **kwargs)

    def get_specimen_directory(self):
        '''Returns a list of the directories
        '''
        return [an.get_specimen_directory() for an in self.analysers]

    def get_N_repeats(self, image_folder, *args, **kwargs):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_N_repeats(
                folder, *args, **kwargs)

    def list_images(self, image_folder, *args, **kwargs):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].list_images(
                folder, *args, **kwargs)

    def get_specimen_name(self):
        return self.name
   
    def get_imaging_parameters(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_imaging_parameters(folder)

    def get_specimen_age(self):
        '''Assumes all specimens have the same age
        '''
        ages = [an.get_specimen_age() for an in self.analysers]
        # FIXME: add warning if ages do not match
        return ages[0]

    def get_specimen_sex(self):
        '''Assumes all specimens have the same sex
        '''
        sexes = [an.get_specimen_sex() for an in self.analysers]
        # FIXME: add warning if sexes do not match
        return sexes[0]
    
    def get_imaging_frequency(self, image_folder, fallback_value=100):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_imaging_frequency(
                folder, fallback_value)

    def get_pixel_size(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_pixel_size(folder)

    def get_snap_fn(self, i_snap=0, absolute_path=True):
        # FIXME: Only gives snaps from the first specimen
        return self.analysers[0].get_snap_fn(
                i_snap, absolute_path=absolute_path)
    
    def load_ROIs(self):
        for an in self.analysers:
            an.load_ROIs()

    def select_ROIs(self, **kwargs):
        for an in self.analysers:
            an.select_ROIs(**kwargs)

    def are_rois_selected(self, *args, **kwargs):
        selected = [an.are_rois_selected() for an in self.analysers]
        return all(selected)
    
    def count_roi_selected_folders(self):
        count = 0
        for an in self.analysers:
            count = an.count_roi_selected_folders()
        return count

    def folder_has_rois(self, image_folder):
        '''
        image_folder : string
            Name of the image folder with the virt suffix
        '''
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].folder_has_rois(folder)

    def get_rois(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_rois(folder)

    def is_measured(self, *args, **kwargs):
        measured = [an.is_measured() for an in self.analysers]
        return all(measured)
    
    def folder_has_movements(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].folder_has_movements(folder)

    def load_analysed_movements(self, *args, **kwargs):
        for an in self.analysers:
            an.load_analysed_movements(*args, **kwargs)

    def get_recording_time(image_folder, i_rep=0):
        return None

    def measure_both_eyes(self, **kwargs):
        for an in self.analysers:
            an.measure_both_eyes(**kwargs)

    def measure_movement(self, eye, *args, **kwargs):
        for an in self.analysers:
            an.measure_movement(eye, *args, **kwargs)
   
    def get_time_ordered(self, *args, **kwargs):
        # FIXME: Doesn not check which eye was measured first
        # FIXME: May break some logic that uses this method?
        image_fns = []
        ROIs = []
        angles = []
        for an in self.analysers:
            fns, rois, angs = an.get_time_ordered(*args, **kwargs)
            image_fns.extend(fns)
            ROIs.extend(rois)
            angles.extend(angs)
        return image_fns, ROIs, angles

    def get_movements_from_folder(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_movements_from_folder(folder)

    def get_displacements_from_folder(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_displacements_from_folder(
                folder)
   
    def get_2d_vectors(self, eye, *args, **kwargs):
        # FIXME: Probably doesn't give sense making results
        # with yaw combining.
        angles = []
        X = []
        Y = []
        for an in self.analysers:
            ang, x, y = an.get_2d_vectors(eye, *args, **kwargs)
            angles.extend(ang)
            X.extend(x)
            Y.extend(y)
        return angles, X, Y

    def get_magnitude_traces(self, eye, image_folder=None,
                             *args, **kwargs):
        
        # Case A: image_folder is given
        if image_folder is not None:
            folder, i_virt = remove_virt_suffix(image_folder)
            return self.analysers[i_virt].get_magnitude_traces(
                    eye, folder, *args, **kwargs)
        
        # Case B: image_folder is None
        # -> append suffix virt_ian to all keys
        traces = {}
        for i_an, an in enumerate(self.analysers):
            data = an.get_magnitude_traces(eye, None, *args, **kwargs)
            for key, value in data.items():
                traces[f'{key}_virt{i_an}'] = value

        return traces

    def get_moving_ROIs(self, eye, image_folder, i_repeat=0):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_moving_ROIs(
                eye, folder, i_repeat=i_repeat)
        

    def get_3d_vectors(self, eye, *args, **kwargs):
        
        points = []
        vectors = []

        for analyser in self.analysers:
            
            yaw = analyser.attributes['yaw']
            
            # Make sure the coloring comes up correct
            # yaw == 0: default case, manalyser code splits L-R
            # yaw==90 or yaw==-90: Should be only one eye recording
            if yaw == 0:
                P, V = analyser.get_3d_vectors(eye, *args, **kwargs)
                points.append(P)
                vectors.append(V)
            elif (yaw == 90 and eye == 'left') or (yaw == -90 and eye == 'right'):
                for aeye in analyser.eyes:
                    P, V = analyser.get_3d_vectors(
                            aeye, *args, **kwargs) 
                    if len(P) > 0:
                        points.append(P)
                        vectors.append(V)

        points = np.concatenate((*points,))
        vectors = np.concatenate((*vectors,))

        merge_distance = kwargs.get('merge_distance', 0)
        if merge_distance:
            points, vectors = self._merge_by_distance(
                    points, vectors, merge_distance)

        return points, vectors

    
    def get_recording_time(self, image_folder, *args, **kwargs):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_recording_time(
                folder, *args, **kwargs)
    
    def stop(self):
        for an in self.analysers:
            an.stop()

