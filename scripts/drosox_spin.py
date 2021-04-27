'''
A script to make a video where DrosoM format saved fly
is spinned over the horizontal rotation (binocular overlap control).
'''

import sys
import os


from PIL import Image
import numpy as np
import tifffile
import tkinter as tk

from tk_steroids.matplotlib import SequenceImshow

def main():
    '''
    usage
    python3 drosox_spin.py dir1 dir2 dir3 ...
    '''

    order = ['nostop', 'ownstop', 'extstop']

    #ycrop = 300
    #yshift = -50
    ycrop, yshift = (1,0
    data = []
    clip = 95

    folders = []
    for name in order:
        for arg in sys.argv[1:]:
            if name in arg:
                folders.append(arg)
                break

    print(folders)

    # get data
    for arg in folders:
        print(arg) 
        folders = [f for f in os.listdir(arg) if os.path.isdir(os.path.join(arg, f)) and 'snap' not in f]
        hors = [int( f.split('(')[1].split(',')[0] ) for f in folders]

        hors, folders = zip(*sorted(zip(hors, folders), key=lambda x: x[0]))


        imfns = [[os.path.join(arg, f, fn) for fn in os.listdir(os.path.join(arg, f)) if fn.endswith('.tiff')][0] for f in folders]
        
        data.append([np.array(hors), imfns])


    # min max
    hor_start = max([hors[0] for hors, imfns in data])
    hor_stop = min([hors[-1] for hors, imfns in data])

    print('hor_start {}, hor_stop {}'.format(hor_start, hor_stop))

    

    # select center points
    for hors, imfns in data:
        break 
        loaded = [tifffile.imread(fn) for i_fn, fn in enumerate(imfns) if not print('{}/{}'.format(i_fn, len(imfns)))]
        print(len(loaded))
        tkroot = tk.Tk()
        imshow = SequenceImshow(tkroot)
        imshow.imshow(loaded, cmap='gray', slider=True)
        imshow.grid(row=1, column=1, sticky='NSWE')
        tkroot.columnconfigure(1, weight=1)
        tkroot.rowconfigure(1, weight=1)
        tkroot.mainloop()
    
    # normalization values
    normalization = []
    for hors, imfns in data:
        image = tifffile.imread(imfns[0])[ycrop+yshift:-ycrop+yshift:,]
        normalization.append( np.percentile(image, clip) )

    print(normalization)

    remove_fns = []

    for i_image, hor in enumerate(range(hor_start, hor_stop+1, 1)):
        images = []
        for i_data, (hors, imfns) in enumerate(data):
            i_closest = np.argmin(np.abs(hors-hor))
            
            image = np.clip(tifffile.imread(imfns[i_closest])[ycrop+yshift:-ycrop+yshift:,], 0, normalization[i_data])
            image = image - np.min(image)
            image = 255*image.astype(np.float) / np.max(image)
            images.append(image)
             
        image = Image.fromarray( np.concatenate((*images,),axis=0).astype(np.uint8) )
        
        #imfn = 'images/im{0:03d}.jpg'.format(i_image)
        imfn = 'images/im{0:03d}_horXZ.jpg'.format(i_image).replace('XZ', '{}').format(hor)
        image.save(imfn)

        remove_fns.append(imfn)


    # encode mp4
    os.chdir('images')
    command = 'ffmpeg -framerate 20 -i im%03d.jpg -c:v libx264 -vf fps=20 -pix_fmt yuv420p -preset slow -crf 28 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {}.mp4'.format('out')

    #os.system(command)
    os.chdir('..')
    
    #for fn in remove_fns:
    #    os.remove(fn)



if __name__ == "__main__":
    main()

