
import os
import tifffile
import cv2
from multiprocessing import Pool

from videowrapper import VideoWrapper
from gonioanalysis.drosom.loading import arange_fns, split_to_repeats

datadir = input("DATADIR TO COMPRESS (av1 encode): ")
newdir = datadir+'_av1compressed'
print("New data will be saved in {}".format(newdir))
os.makedirs(newdir, exist_ok=True)


def process_dir(root):
    directory = root
    
    fns = [fn for fn in os.listdir(directory) if fn.endswith('.tiff')]
    fns = arange_fns(fns)
    fns = split_to_repeats(fns)
    
    for repeat in fns:
        common = os.path.commonprefix(repeat)
        savedir = root.replace(os.path.basename(datadir), os.path.basename(newdir))
        
        movie_fn = os.path.join(savedir, common+'stack.mp4')
        
        if os.path.exists(movie_fn):
            continue
        
        os.makedirs(savedir, exist_ok=True)
        video = VideoWrapper()
        video.images_to_video([os.path.join(root, fn) for fn in repeat], movie_fn, 1)


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
