import os
import math

import numpy as np
import matplotlib.pyplot as plt
import tifffile

from gonioanalysis.directories import ANALYSES_SAVEDIR
import gonioanalysis.coordinates as coordinates
from gonioanalysis.drosom.plotting.common import vector_plot


def _load_image(fn, roi, e):

    image = tifffile.imread(fn)
    
    x,y,w,h = [int(round(z)) for z in roi]
    
    upper = np.percentile(image[y-e:y+h+e,x-e:x+w+e], 99.5) 
    lower = np.percentile(image[y-e:y+h+e,x-e:x+w+e], 0.5)

    image = np.clip(image, lower, upper) - lower
    image = (image-np.min(image)) / np.max(image)
    image *= 255
    
    return np.array([image,image,image])


def _box(image, roi, lw, color, crop_factor=1):
    x,y,w,h = [int(round(z)) for z in roi]
    
    for i, c in enumerate(color):
        # Top
        image[i, y:y+lw,x:x+w] = c
        # Left
        image[i, y:y+h,x:x+lw] = c
        # Bottom
        image[i, y+h-lw:y+h,x:x+w] = c
        # Right
        image[i, y:y+h,x+w-lw:x+w] = c

    return image

def _crop(image, roi, factor):
    
    x,y,w,h = [int(round(z)) for z in roi]
    cp = [x+int(w/2), y+int(h/2)]

    new_image = []
    
    if factor < 1:
        h2 = int(round(factor*image.shape[1]/2))
        for i in range(len(image)):
            new_image.append( image[i, cp[1]-h2:cp[1]+h2, :] )
    elif factor > 1:
        w2 = int(round(image.shape[2]/2/factor))
        for i in range(len(image)):
            new_image.append( image[i, cp[0]-w2:cp[0]+w2, :] )
    else:
        return image

    return np.array(new_image)

def moving_rois(manalyser, roi_color='red,blue', lw=3, e=50,
        rel_rotation_time=1, crop_factor=0.5):
    '''
    Visualization video how the ROI boxes track the analyzed features,
    drawn on top of the original video frames.

    Arguments
    ---------
    roi : string
        A valid matplotlib color. If two comma separated colors
        given use the first for the left eye and the second for the right.
    lw : int
        ROI box line width, in pixels
    e : int
        Extended region for brightness normalization, in pixels
    rel_rotation_time : int or float
        Blend the last and the first next frame for "smooth"
        transition
    crop_factor : int
        If smaller than 1 then cropped in Y.
    '''
    savedir = os.path.join(ANALYSES_SAVEDIR, 'illustrate_experiments', 'moving_rois', manalyser.get_specimen_name())
    os.makedirs(savedir, exist_ok=True)
    
    colors = roi_color.split(',')
    
    image_fns, ROIs, angles = manalyser.get_time_ordered()
    N = len(image_fns)
    
    i_frame = 0

    aangles = []
    
    crop_roi = ROIs[0]

    for i_fn, (fn, roi, angle) in enumerate(zip(image_fns, ROIs, angles)):

        if i_fn+1 < len(image_fns) and angle != angles[i_fn-1]:
            aangles.append(angle)
            crop_roi = roi
        #continue

        print("{}/{}".format(i_fn+1, N))

        image = _load_image(fn, roi, e)
        
        if angle[0] > 0:
            color = (255, 0, 0)
        else:
            color = (0,0,255)

        image = _box(image, roi, lw, color=color)
        image = _crop(image, crop_roi, crop_factor)
        savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_frame))
        tifffile.imsave(savefn, image.astype(np.uint8))
        i_frame += 1

        if rel_rotation_time and angle != angles[i_fn+1]:
            next_image = _load_image(image_fns[i_fn+1], roi, e)
            if angles[i_fn+1][0] > 0:
                color = (255, 0, 0)
            else:
                color = (0,0,255)
            next_image = _box(next_image, ROIs[i_fn+1], lw, color=color)
            
            for blend in np.zeros(5).tolist() + np.linspace(0, 1, 25).tolist():
                
                im = image*(1-blend) + _crop(next_image, ROIs[i_fn+1], crop_factor)*(blend)
                savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_frame))
                tifffile.imsave(savefn, im.astype(np.uint8))
                i_frame += 1

    np.savetxt('aangles.txt', aangles)


def illustrate_experiments(manalyser, rel_rotation_time=1):
    '''
    Create a visualizing video how the vectormap is built.

    Arguments
    ---------
    rel_rotation_time : int or float
        Relative time spend on incrimentally rotating the vectormap
        between the stimuli.
    '''
    print('illustrate_experiments')
    
    savedir = os.path.join(ANALYSES_SAVEDIR, 'illustrate_experiments', manalyser.get_specimen_name())
    os.makedirs(savedir, exist_ok=True)
 
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1, projection='3d')
    plt.axis('off') 
    plt.subplots_adjust(left=-0.2, bottom=-0.2, right=1.2, top=1.2, wspace=0.0, hspace=0)


    # Points, vectors, angles
    lp, lv, la = manalyser.get_3d_vectors('left', return_angles=True)
    rp, rv, ra = manalyser.get_3d_vectors('right', return_angles=True)    
    
    image_fns, ROIs, angles = manalyser.get_time_ordered()
    arrow_artists = []

    lpoints, lvectors = [[],[]]
    rpoints, rvectors = [[],[]]

    print(len(image_fns))

    print(len(ROIs))
    print(len(angles))

    i_frame = 0
    for i_angle, (image_fn, ROI, angle) in enumerate(zip(image_fns, ROIs, angles)): 
        
        length_scale = (i_angle%20)/20 + 0.5
        final_image = (i_angle+1)%20 == 0

        #if not final_image:
        #   continue
        # Fix to account mirror_horizontal in 2D vectors
        angle = [-1*angle[0], angle[1]]


        # Calculate cameras place
        #x,y,z = coordinates.camera2Fly(*angle)
        #r, phi, theta = coordinates.to_spherical(x, y, z, return_degrees=True)

        #elev = float(90-theta)
        azim = -angle[0]+90
        
        if angle in la:
            indx = la.index(angle)
            
            nlp = coordinates.rotate_about_x(lp[indx], -angle[1])
            nlv = coordinates.rotate_about_x(lp[indx]+lv[indx], -angle[1]) - nlp

            if final_image:
                lpoints.append(lp[indx])
                lvectors.append(lv[indx])
            else:
                arrow_artists.extend(vector_plot(ax, [nlp], [length_scale*nlv], color='red'))

        
        if angle in ra:
            indx = ra.index(angle)

            nrp = coordinates.rotate_about_x(rp[indx], -angle[1])
            nrv = coordinates.rotate_about_x(rp[indx] + rv[indx], -angle[1]) - nrp

            if final_image:
                rpoints.append(rp[indx])
                rvectors.append(rv[indx])
            else:
                arrow_artists.extend(vector_plot(ax, [nrp], [length_scale*nrv], color='blue'))
        
        if lpoints:
            tmp_lpoints, tmp_lvectors = coordinates.rotate_vectors(np.array(lpoints), np.array(lvectors), 0, -math.radians(angle[1]), 0)
        
            arrow_artists.extend(vector_plot(ax, tmp_lpoints, tmp_lvectors, color='red', mutation_scale=3,
                    camerapos=[0,azim]))
            
        
        if rpoints:
            tmp_rpoints, tmp_rvectors = coordinates.rotate_vectors(np.array(rpoints), np.array(rvectors), 0, -math.radians(angle[1]), 0)
        

            arrow_artists.extend(vector_plot(ax, tmp_rpoints, tmp_rvectors, color='blue', mutation_scale=3,
                    camerapos=[0,azim]))

            
        #ax.dist = 2
        ax.view_init(elev=0, azim=azim)
        print('Horizontal {}, vertical {}'.format(*angle))

        savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_frame))
        fig.savefig(savefn, dpi=300, transparent=True)
        i_frame += 1
        
        # Rotating the saved image
        #camera_rotation = coordinates.correct_camera_rotation(*angle, return_degrees=True)
        #saved_image = Image.open(savefn)
        #saved_image.rotate(-camera_rotation).save(savefn)

        # Final image of this location, rotate the plot to the
        # new location
        if final_image and rel_rotation_time:
            next_angle = angles[i_angle+1]
            duration = 25
            hold_duration = 5
            
            for i, (h, v) in enumerate(zip(np.linspace(-angle[0], next_angle[0], duration), np.linspace(angle[1], next_angle[1], duration))):
                
                angle = [-h,v]
                azim = -angle[0]+90
                
                # FIXME The following part is copy paste from the part above
                # -> put it behind one function etc.

                # Clear arrows
                for arrow_artist in arrow_artists:
                    arrow_artist.remove()
                arrow_artists = []
                 
                # Redraw arros
                if lpoints:
                    tmp_lpoints, tmp_lvectors = coordinates.rotate_vectors(np.array(lpoints), np.array(lvectors), 0, -math.radians(angle[1]), 0)
                
                    arrow_artists.extend(vector_plot(ax, tmp_lpoints, tmp_lvectors, color='red', mutation_scale=3,
                            camerapos=[0,azim]))
                    
                
                if rpoints:
                    tmp_rpoints, tmp_rvectors = coordinates.rotate_vectors(np.array(rpoints), np.array(rvectors), 0, -math.radians(angle[1]), 0)
                

                    arrow_artists.extend(vector_plot(ax, tmp_rpoints, tmp_rvectors, color='blue', mutation_scale=3,
                            camerapos=[0,azim]))

                
                ax.view_init(elev=0, azim=azim)
                
                savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_frame))
                fig.savefig(savefn, dpi=300, transparent=True)
                i_frame += 1
  
                if i == 0:
                    for repeat in range(hold_duration):
                        savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_frame))
                        fig.savefig(savefn, dpi=300, transparent=True)
                        i_frame += 1
     


        for arrow_artist in arrow_artists:
            arrow_artist.remove()
        arrow_artists = []
    
    plt.close()



def rotation_mosaic(manalyser, imsize=(512,512),
        e=50, crop_factor=0.5):
    '''
    A mosaic (matrix) of the taken images.
    
    Arguments
    ---------
    manalyser : obj
        Analyser object
    n_vecticals : int
        How many vertical rotations rows to show
    n_horizontals : int
        How many horizontal rotation columns to show
    e, crop_factor
    '''

    # Part 1) Find rotations matching the interpolation

    rotations = manalyser.list_rotations()
    
    kdtree = KDTree(rotations)

    hrots, vrots = zip(*rotations)

    hmin, hmax = (np.min(hrots), np.max(hrots))
    vmin, vmax = (np.min(vrots), np.max(vrots))
   
    hstep = int(10 * (1024/360))
    vstep = int(10 * (1024/360))
    
    plot_data = []

    intp_v = np.arange(vmin, vmax, vstep)[::-1]
    intp_h = np.arange(hmin, hmax, hstep)

    for i_v, vrot in enumerate(intp_v):
        for i_h, hrot in enumerate(intp_h):
            
            distance, i_point = kdtree.query( (hrot, vrot), n_jobs=-1)
            
            if distance > math.sqrt((hstep/1.5)**2 + (vstep/1.5)**2):
                continue

            plot_data.append((i_v, i_h, i_point))
            

    # Part 2: Plot the images

    image_fns, ROIs, angles = manalyser.get_time_ordered(angles_in_degrees=False,
            first_frame_only=True)
    
    w_mosaic = int(imsize[0]*len(intp_h))
    h_mosaic = int(imsize[1]*len(intp_v))
    mosaic = 255 * np.ones( (h_mosaic, w_mosaic) )
    
    print('Mosaic shape {}'.format(mosaic.shape))
    
    for i_plot_data, (i_v, i_h, i_point) in enumerate(plot_data): 
        print("{}/{}".format(i_plot_data+1, len(plot_data)))

        x = i_v * imsize[0]
        y = i_h * imsize[1]

        try:
            index = angles.index(list(rotations[i_point]))
        except ValueError:
            continue
        
        try:
            image = _load_image(image_fns[index], ROIs[index], 50) 
        except:
            continue

        image = image[0, :, :]
        image = cv2.resize(image, dsize=(*imsize,))        

        mosaic[x:x+imsize[0], y:y+imsize[1]] = image
    
    drot = manalyser.get_rotstep_size()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(mosaic, cmap='gray', extent=[hmin*drot, hmax*drot, vmin*drot, vmax*drot])
    
    fig.savefig(os.path.join(ANALYSES_SAVEDIR, 'illustrate_experiments', "mosaic_"+manalyser.name+'.jpg'),
            dpi=600)

    plt.close()
