
import subprocess
import tifffile
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

class FramePuller:
    '''
    Read image frames from a video file.
    Uses ffmpeg's piping and Pillow to read.
    '''
    def __init__(self, fn):
        self.fn = fn
    
    def _get_cmd(self, vframes):
        return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", self.fn, "-r", "1/1", "-c:v", "tiff", "-vframes", str(vframes), '-pixel_format', 'gray16le', "-f", "image2pipe", "-"]
    
    def get_frames(self):
        process = subprocess.Popen(self._get_cmd(20), stdout=subprocess.PIPE)
        output = process.communicate()[0]
 
        barr = bytearray(output)

        b1 = barr[0:4]
        parts = barr.split(b1)
        
        images = []
        
        for part in parts[1:]:
            part = bytes(b1 + part)
            
            f = io.BytesIO()
            f.write(part)
            
            images.append(Image.open(f))
        
        return np.asarray(images)
        
    
def main():
    
    fn = TESTVIDEOFILE
    
    images = FramePuller(fn).get_frames()
    
    print(images.shape)
    
    i = 4000
    for image in images:
        tifffile.imsave(os.path.join(os.path.dirname(fn), os.path.basename(fn)+str(i)+'.tiff'), np.array(image))
        i+=1
        
if __name__ == "__main__":
    main()
