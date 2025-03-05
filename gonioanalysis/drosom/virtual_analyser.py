

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
    
    def load_ROIs(self):
        for an in self.analysers:
            an.load_ROIs()

    

    def are_rois_selected(self, *args, **kwargs):
        selected = [an.are_rois_selected() for an in self.analysers]
        return all(selected)

    def is_measured(self, *args, **kwargs):
        measured = [an.is_measured() for an in self.analysers]
        return all(measured)

    def load_analysed_movements(self, *args, **kwargs):
        for an in self.analysers:
            an.load_analysed_movements(*args, **kwargs)

    def get_recording_time(image_folder, i_rep=0):
        return None


    def list_imagefolders(self, *args, **kwargs):
        folders = []
        for i_an, an in enumerate(self.analysers):
            fols = an.list_imagefolders(*args, **kwargs)

            # Add index of the analyser to the folder suffix beginning
            fols = [insert_virt_suffix(fol, i_an) for fol in fols]

            folders.extend(fols)
        
        return folders

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

    def folder_has_movements(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].folder_has_movements(folder)

    def load_analysed_movements(self):
        for an in self.analysers:
            an.load_analysed_movements()

    def list_images(self, image_folder, *args, **kwargs):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].list_images(
                folder, *args, **kwargs)

    def get_specimen_name(self):
        return self.name

    
    def get_imaging_parameters(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_imaging_parameters(folder)

    def get_imaging_frequency(self, image_folder, fallback_value=100):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_imaging_frequency(
                folder, fallback_value)

    def get_pixel_size(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_pixel_size(folder)



    def get_movements_from_folder(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_movements_from_folder(folder)

    def get_displacements_from_folder(self, image_folder):
        folder, i_virt = remove_virt_suffix(image_folder)
        return self.analysers[i_virt].get_displacements_from_folder(
                folder)

    def get_magnitude_traces(self, eye, image_folder=None,
                             *args, **kwargs):
        
        # Case A: image_folder is given
        if image_folder is not None:
            folder, i_virt = remove_virt_suffix(image_folder)
            return self.analysers[i_virt].get_magnitude_traces(
                    eye, folder, *args, **kwargs)
        
        # Case B: image_folder is None
        raise NotImplementedError('VirtualAnalyser not complete')
        

    def get_3d_vectors(self, eye, *args, **kwargs):
        
        points = []
        vectors = []

        for analyser in self.analysers:
            P, V = analyser.get_3d_vectors(eye, *args, **kwargs) 
            points.append(P)
            vectors.append(V)

        points = np.concatenate((*points,))
        vectors = np.concatenate((*vectors,))

        return points, vectors


