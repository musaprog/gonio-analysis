
import os
import cv2
from multiprocessing import Pool
import tempfile

import tifffile

from videowrapper import VideoWrapper
from gonioanalysis.drosom.loading import arange_fns, split_to_repeats


datadir = input("DATADIR TO COMPRESS (av1 encode): ")
newdir = datadir+'_av1compressed'
print("New data will be saved in {}".format(newdir))
os.makedirs(newdir, exist_ok=True)


def _expand_stack(fn, tmp_path):
    '''
    Expand a tiff stack to separate tiff files using tifffile.
    '''
    stack = tifffile.TiffFile(fn)

    for i_page in range(len(stack.pages)):
        image = stack.asarray(key=i_page)
        
        newfn = os.path.join(tmp_path, fn+str(i_page).zfill(8)+'.tiff'
        tifffile.imsave(newfn, image)

    return tmp_fns


def _clear_tmp(tmp_fns):
    for fn in tmp_fns:
        os.remove(fn)


def process_dir(root):
    directory = root
    
    fns = [fn for fn in os.listdir(directory) if fn.endswith(('.tiff', '.tif'))]
    fns = arange_fns(fns)
    fns = split_to_repeats(fns)
    
    tmpdir = tmpfile.TemporaryDirectory()
    
    for repeat in fns:
        common = os.path.commonprefix(repeat)
        savedir = root.replace(os.path.basename(datadir), os.path.basename(newdir))
        
        if 'stack' in common.lower():
            tmpdir.cleanup()
            if len(repeat) != 1:
                raise NotImplementedError('One repeat, multiple stacks, not implemented')
            _expand_stack(repeat, tmpdir.name)

            location = tmpdir.name
            movie_fn = os.path.join(savedir, common)
        else:
            location = root
            movie_fn = os.path.join(savedir, common+'stack.mp4')
        
        if os.path.exists(movie_fn):
            continue
        
        os.makedirs(savedir, exist_ok=True)
        video = VideoWrapper()
        video.images_to_video([os.path.join(location, fn) for fn in repeat], movie_fn, 1)
        

def main():

    tiffdirs = []
    for root, dirs, fns in os.walk(datadir):
        
        if os.path.basename(root) == "snaps":
            continue
        
        hastiffs = any([fn.endswith('.tiff') for fn in fns])
        
        if hastiffs:
            tiffdirs.append(root)
            
    with Pool() as p:
        p.map(process_dir, tiffdirs)

if __name__ == "__main__":
    main()
