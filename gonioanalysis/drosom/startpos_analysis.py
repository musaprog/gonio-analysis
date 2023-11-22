import os
import json

from gonioanalysis.directories import PROCESSING_TEMPDIR
from gonioanalysis.drosom.analysing import MAnalyser
from movemeter.stacks import stackread, stackwrite
from movemeter import Movemeter

class StartposAnalyser(MAnalyser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._movements_skelefn = self._movements_skelefn.replace('movements_', 'startpos_')
        self.active_analysis = ''

        self._startpos_stack = os.path.join(PROCESSING_TEMPDIR, 'startpos_stacks', self.folder)


    def measure_movement(self, eye, *args, max_movement=30, **kwargs):
        '''Measures how the first frame repeat ROI moves across the repeats.
        '''



        self.movements = {}
        
        if not os.path.exists(self._crops_savefn):
            self.selectROIs() 
        self.load_ROIs()
        
        #angles = []
        image_folders = []
        stacks = []
        ROIs = []

        
        # Pull out the first frames out of each repeats
        # (needed especially if a stack saved)
        for image_folder in self.list_imagefolders():
            #images = self.list_images(image_folder, absolute_path=True)
            images = self.stacks[image_folder.removeprefix('pos')]

            print(f'Image folder {image_folder} has following images in it')

            start_frames = []

            for fns in images:
                print(fns[0])
                stack = stackread(fns[0])
                start_frames.append( next(stack) )
            
            os.makedirs(self._startpos_stack, exist_ok=True)
            write_fn = os.path.join(self._startpos_stack, f'{image_folder}.tiff')
            with stackwrite(write_fn) as fp:
                for frame in start_frames:
                    fp.write(frame)
            
            rois = self.get_rois(image_folder)
            for roi in rois:
                ROIs.append([roi])
                stacks.append([write_fn])
                image_folders.append(image_folder)

            print(f'Wrote {write_fn}')


        meter = Movemeter(upscale=10, absolute_results=True)
        meter.set_data(stacks, ROIs)
        

        for stack_i, image_folder in enumerate(image_folders):
            print('Analysing {} eye, done {}/{} for this eye'.format(eye.upper(), stack_i+1, len(stacks)))
            print("Calculating ROI's movement...")
            x, y = meter.measure_movement(stack_i, max_movement=max_movement)[0]
            print('Done.')
            
            angle = image_folder.removeprefix('pos')
            if angle not in self.movements:
                self.movements[angle] = []

            self.movements[angle].append( {'x': x, 'y': y} )


        # Save movements
        with open(self._movements_savefn.format(eye), 'w') as fp:
            json.dump(self.movements, fp)
        

    def is_measured(self):
        fns = [self._movements_savefn.format(eye) for eye in self.eyes]
        return all([os.path.exists(fn) for fn in fns])
