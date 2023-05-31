import os
import math
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
import tifffile
from scipy.spatial import cKDTree as KDTree
import cv2

from gonioanalysis.directories import ANALYSES_SAVEDIR
import gonioanalysis.coordinates as coordinates
from gonioanalysis.drosom.plotting.common import vector_plot
from gonioanalysis.drosom.loading import angles_from_fn
from gonioanalysis.version import used_scipy_version

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
        a = cp[1]-h2
        b = cp[1]+h2

        if a < 0:
            a = 0
            b = h2*2
        if b > image.shape[1]:
            a = image.shape[1] - h2*2
            b = image.shape[1]

        for i in range(len(image)):
            new_image.append( image[i, a:b, :] )
    elif factor > 1:
        w2 = int(round(image.shape[2]/2/factor))
        for i in range(len(image)):
            new_image.append( image[i, cp[0]-w2:cp[0]+w2, :] )
    else:
        return image

    return np.array(new_image)





def moving_rois(manalyser, roi_color='red,blue', lw=3, e=50,
        rel_rotation_time=1, crop_factor=0.5,
        _exclude_imagefolders=[], _order=None,
        _draw_arrow=False):
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
    
    RETURNS
    -------
    None
    '''
    savedir = os.path.join(ANALYSES_SAVEDIR, 'illustrate_experiments', 'moving_rois', manalyser.get_specimen_name())
    os.makedirs(savedir, exist_ok=True)
    if _draw_arrow:
        os.makedirs(os.path.join(savedir, 'inset'), exist_ok=True)

    colors = roi_color.split(',')
    
    image_fns, ROIs, angles = manalyser.get_time_ordered()

    # ------------------
    # For mosaic
    if _exclude_imagefolders:
        
        newdata = []
        for fn, ROI, angle in zip(image_fns, ROIs, angles):
            if not angles_from_fn(os.path.basename(os.path.dirname(fn))) in _exclude_imagefolders:
                newdata.append([fn, ROI, angle])
            else:
                pass
        image_fns, ROIs, angles = list(zip(*newdata))

    if _order:
        image_fns = list(image_fns)
        ROIs = list(ROIs)
        angles = list(angles)
        newdata = []

        for o in _order:
            
            indices = [i for i in range(len(image_fns)) if angles_from_fn(os.path.basename(os.path.dirname((image_fns[i])))) == o]
            
            for i in indices:
                newdata.append([image_fns[i], ROIs[i], angles[i]])

        image_fns, ROIs, angles = list(zip(*newdata))
    # End for mosaic
    # ------------------

    N = len(image_fns)
    i_frame = 0

    crop_roi = ROIs[0]
    

    if _draw_arrow:
        # Create and setup figure
        fig, ax = plt.subplots(figsize=(10,10))
        ax._myarrow = None
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_axis_off()
        
        # vector normalisation values
        normalisations = {fol: manalyser.get_displacements_from_folder(fol)[0][-1] for fol in manalyser.list_imagefolders(only_measured=True)}
        
        ax.add_patch( Circle((0,0), 1, fill=False, lw=3, color="gray") )
        

    def draw_arrow_inset(savefn, vx, vy, **kwargs):
        if ax._myarrow:
            ax._myarrow.remove()

        ax._myarrow = Arrow(0,0,vx,vy, width=0.5, **kwargs)
        ax.add_patch(ax._myarrow)
        fig.savefig(savefn, transparent=True)


    for i_fn, (fn, roi, angle) in enumerate(zip(image_fns, ROIs, angles)):

        if i_fn+1 < len(image_fns) and angle != angles[i_fn-1]:
            crop_roi = roi
        
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

        if _draw_arrow:
            vx, vy = (roi[0] - crop_roi[0], roi[1] - crop_roi[1])
            vx, vy = np.array([vx, -vy]) / normalisations[os.path.basename(os.path.dirname(fn))]
            draw_arrow_inset(os.path.join(savedir, 'inset', 'image_{:08d}.png'.format(i_frame)), vx, vy,
                    color='white')
            

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
                
                if _draw_arrow:
                    draw_arrow_inset(os.path.join(savedir, 'inset', 'image_{:08d}.png'.format(i_frame)), vx, vy,
                            color='white')

                i_frame += 1


def _get_closest(folder_a, folders_b, index=False):
    '''
    pos

    returns closets_folder, distance
    '''

    dists = []

    A = np.array( folder_a )
    for b in folders_b:
        B = np.array( b )
        dists.append( np.linalg.norm(A-B) )
    
    if index:
        return np.argmin(dists), np.min(dists)
    else:
        return folders_b[np.argmin(dists)], np.min(dists)



def moving_rois_mosaic(manalysers, common_threshold=7.5, **kwargs):
    '''
    Uses moving_rois() to make a mosaic video of the experiments, in which
    the specimens move in sync.

    The first specimen (manalyser[0]) determines the rotation order (in the
    order as it was recorded).


    ARGUMENTS
    ---------
    common_threshold : int
        In rotation stage steps, how close the recordings of different
        analysers have to be classified as the same.
    kwargs : dict
        Passed to moving_rois
    
    RETURNS
    -------
    None
    '''

    orders = {manalyser.name: [] for manalyser in manalysers}
    excludes = {}

    folders = [angles_from_fn(os.path.basename(os.path.dirname(fn))) for fn in manalysers[0].get_time_ordered(first_frame_only=True)[0]]

    has_matches = {fol: 0 for fol in folders}
    conversion = {manalyser.name: {} for manalyser in manalysers}

    for folder in folders:
        for manalyser in manalysers[1:]:
            fols = [angles_from_fn(fol) for fol in manalyser.list_imagefolders(only_measured=True)] 
            closest, distance = _get_closest(folder, fols)
            
            if distance < common_threshold:
                orders[manalyser.name].append(closest)
                has_matches[folder] += 1
                conversion[manalyser.name][closest] = folder
    
    orders[manalysers[0].name] = folders.copy()
    orders[manalysers[0].name] = [fol for fol in orders[manalysers[0].name] if has_matches[fol] == len(manalysers)-1]   
    fols = [angles_from_fn(fol) for fol in manalysers[0].list_imagefolders(only_measured=True)]
    excludes[manalysers[0].name] = [fol for fol in fols if not fol in orders[manalysers[0].name]]
    
    for manalyser in manalysers[1:]:
        orders[manalyser.name] = [fol for fol in orders[manalyser.name] if has_matches[conversion[manalyser.name][fol]] == len(manalysers)-1]
       
        fols = [angles_from_fn(fol) for fol in manalyser.list_imagefolders(only_measured=True)]
        excludes[manalyser.name] = [fol for fol in fols if not fol in orders[manalyser.name]]
 
    
    # FIXME Starts too many processes with many specimens, possibly leading to
    # out of RAM

    processes = []
    
    for manalyser in manalysers:
        p = mp.Process(target=moving_rois, args=[manalyser],
                kwargs={ **{"_exclude_imagefolders": excludes[manalyser.name],
                    "_order": orders[manalyser.name],
                    "_draw_arrow": True}, **kwargs})
        p.start()
        
        processes.append(p)
        
    for p in processes:
        p.join()



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
            
            if used_scipy_version < (1,6,0):
                distance, i_point = kdtree.query( (hrot, vrot), n_jobs=-1)
            else:
                distance, i_point = kdtree.query( (hrot, vrot), workers=-1)
            
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
