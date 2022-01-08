'''
Simple tool (Encoder) to transform an image recording into a movie clip.
'''

import os
import datetime
import subprocess

PROCESSING_TEMPDIR = '/tmp/videowrapper_tmp'


class VideoWrapper:
    '''
    Simple Python interface for encoding images into a movie clip
    using ffmpeg (required to be installed).

    See Encoder.encode for usage.
    '''

    def __init__(self):
        
        # Ensure that ffmpeg is installed
        try:
            subprocess.check_output('ffmpeg -version', shell=True)
        except subprocess.CalledProcessError:
            raise RuntimeError('ffmpeg is not installed (required by movie.Encoder)')
        
        # Temporal stuff directory
        self.tempdir = os.path.join(PROCESSING_TEMPDIR, 'movie-encoder')
        os.makedirs(self.tempdir, exist_ok=True)


    def _makeImageList(self, images, fps):
        '''
        PRIVATE Used from self.encode.
        
        Creates a textfile that contains all the image filenames and is passed to ffmpeg.
        '''
        imagelist_fn = os.path.join(self.tempdir, 'imagelist_{}.txt'.format(str(datetime.datetime.now()).replace(' ', '_')))
        imagelist_fn = os.path.abspath(imagelist_fn)
        with open(imagelist_fn, 'w') as fp:
            for image in images:
                fp.write("file '"+os.path.abspath(image)+"'\n")
                fp.write('duration {}\n'.format(float(1/fps)))
        
        return imagelist_fn


    def images_to_video(self, images, output_fn, fps, codec='libaom-av1'):
        '''
        Encoding a video clip using ffmpeg.

        INPUT ARGUMENTS         DESCRIPTION
        images                  Ordered list of images; Movie will be encoded in this order.
        output_fn               Output name of the movie.
        fps                     Desrided frames per second.

        Encoding libx264, preset veryslow and crf 28 will be used.
        '''

        # Check that input is correct and that images actually exist (avoids many trouble).
        
        options = {}

        if type(output_fn) != type('string'):
            raise TypeError('output_fn has to be a string (required by movie.Encoder)')

        try:
            float(fps)
        except:
            raise TypeError('fps has to be floatable')
        
        for image in images:
            if not os.path.exists(image):
                raise FileNotFoundError('Precheck in movie.Encoder failed. File {} does not exist'.format(image))
        


        # Check that directory for ouput_fn exsits, if not, create 
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        
        imagelist_fn = self._makeImageList(images, fps)
        print(imagelist_fn)
        command = 'ffmpeg -r {} -f concat -safe 0 -i "{}" -c:v {} -pix_fmt gray12 -crf 7 -b:v 0 "{}"'.format(fps,imagelist_fn, codec, output_fn)
        subprocess.check_output(command, shell=True)
        
        os.remove(imagelist_fn)
