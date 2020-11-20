import os
import math

import numpy as np
import matplotlib.pyplot as plt

from pupilanalysis.directories import ANALYSES_SAVEDIR
import pupilanalysis.coordinates as coordinates
from .common import vector_plot


def illustrate_experiments(manalyser):
    '''
    Create a visualizing video how the vectormap is built.
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

        savefn = os.path.join(savedir, 'image_{:08d}.png'.format(i_angle))
        fig.savefig(savefn, dpi=300)
        
        # Rotating the saved image
        #camera_rotation = coordinates.correct_camera_rotation(*angle, return_degrees=True)
        #saved_image = Image.open(savefn)
        #saved_image.rotate(-camera_rotation).save(savefn)

        for arrow_artist in arrow_artists:
            arrow_artist.remove()
        arrow_artists = []
    
    plt.close()
