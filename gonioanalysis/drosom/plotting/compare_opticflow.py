import os
from math import radians

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation

from gonioanalysis.directories import ANALYSES_SAVEDIR, CODE_ROOTDIR

from .common import vector_plot, surface_plot, make_animation_timestep

CURRENT_ARROW_LENGTH = 1


def error_at_flight(manalyser):
    '''
    Plot 2D heatmap with vertical limits (upper_lower) as x,y axes and
    pixel color values as the on flight orientation.

    To see what (lower,upper) vertical excluding would give smallest error.
    '''

    #optimal_rot = 50
    optimal_rot = 50

    rotations = np.linspace(-180, 180, 180)

    widths = np.linspace(10, 200, 20)
    centers = np.linspace(-50, 50, 10)

    # Absolute minimum errors, no matter where happens
    min_errors = [] 
    # Distance of the minimum error from the flight position
    distances = []

    for i, width in enumerate(widths):
        print('upper {}/{}'.format(i+1, len(widths)))
        min_errors.append([])
        distances.append([])
        for center in centers:
            manalyser.set_angle_limits(va_limits=(-width/2+center, width/2+center))
            p, e = optic_flow_error(manalyser, rotations)
            e = np.mean(e, axis=1)
            
            
            error_argmin = np.argmin(e)
            distance = rotations[error_argmin] - optimal_rot

            distances[-1].append(distance)
            min_errors[-1].append(np.min(e))

    manalyser.set_angle_limits()

    for im, figname, text in zip([min_errors, distances], ['min_errors.jpg', 'distances.jpg'], ['Minimum error (between 0 and 1)','The distance of flight rotation from minimum error (degrees)']):
        
        fig = plt.figure()
        zlimit = max(abs(np.min(im)), abs(np.max(im)))
        plt.imshow(im, extent=[np.min(centers), np.max(centers), np.min(widths), np.max(widths)], cmap='seismic',vmin=-zlimit, vmax=zlimit, origin='lower')
        plt.xlabel('Center point (degrees)')
        plt.ylabel('Width (degrees)')
        cbar = plt.colorbar()
        cbar.set_label(text)

        savedir = os.path.join(ANALYSES_SAVEDIR, 'error_at_flight', manalyser.get_specimen_name())
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(os.path.join(savedir,figname))
        plt.close()



def complete_flow_analysis(manalyser, rotations, rotation_axis,
        text=True, animate=True, subdivide_regions=False, mutation_scale=3,
        dual_difference=True, elevations=[50, 0, -50]):
    '''
    Creates the combined plot of the rhabdomere movement vectormap, simulated
    optic flow and the difference between these two, rendered from 3 views
    
    PLOT
    ROW
    1       measured optic flow vectors
    2       simulated optic flow
    3       error heatplot

    INPUT ARGUMENTS
    manalyser
    rotations
    rotation_axis       Either 'yaw', 'roll', or 'pitch'
    text                wheter to render text
    dual_difference     Create a a difference plot for the opposite direction
                        as well, adding a column in the plot
    subdivide_regions   True/False, will divide the 
    

    '''
    from scipy.ndimage import rotate
    import matplotlib.image as mpli 
    from gonioanalysis.coordinates import rotate_vectors, optimal_sampling
    from gonioanalysis.drosom.optic_flow import flow_vectors, field_error

    # Parse keywork aguments
    # -----------------------
    N_fig_columns = 4
    if dual_difference:
        N_fig_columns += 1
    N_fig_rows = 3
    
    if not text:
        global CURRENT_ARROW_LENGTH
        CURRENT_ARROW_LENGTH = 1.5

    if text:
        dpi = 150
    else:
        dpi = 600

    elevation_texts = ['Dorsal\nview', 'Anterior\nview', 'Ventral\nview']

    column_texts = ['Biphasic receptive field\nmovement directions', #'Pseudopupil movement\ndirections',
            'Experienced optic flow',
            'Difference with slower phase',
            'Head orientation']
    
    if dual_difference:
        column_texts.insert(3, 'Difference with fast phase')

    if rotation_axis == 'pitch':
        zero_rot = 10
        optimal_rot = 10
        fly_image = os.path.join(CODE_ROOTDIR, 'droso6_rotated.png')
        sideflow=True
    elif rotation_axis == 'yaw':
        zero_rot = 0
        optimal_rot = 0
        fly_image = os.path.join(CODE_ROOTDIR, 'rotation_yaw.png')
        sideflow=True
    elif rotation_axis == 'roll':
        zero_rot = 0
        optimal_rot = 0
        fly_image = os.path.join(CODE_ROOTDIR, 'rotation_roll.png')
        sideflow=False


    # End parsing keyword arguments
    # ------------------------------

    lp, lvecs = manalyser.get_3d_vectors('left')
    rp, rvecs = manalyser.get_3d_vectors('right')
    
    points = np.concatenate((lp, rp))
    lrvecs = np.concatenate((lvecs, rvecs))

    # Flow field errors for each rotation
    #points, all_errors = optic_flow_error(manalyser, rotations)
    
    # Flow field vectors for each rotation
    vector_points = optimal_sampling(np.arange(-90, 90, 5), np.arange(-180, 180, 5))

    if rotation_axis == 'yaw':
        all_flow_vectors = [rotate_vectors(vector_points, flow_vectors(vector_points), -radians(rot), 0, 0) for rot in rotations]
    elif rotation_axis == 'pitch':
        all_flow_vectors = [rotate_vectors(vector_points, flow_vectors(vector_points), 0, -radians(rot), 0) for rot in rotations]
    elif rotation_axis == 'roll':
        all_flow_vectors = [rotate_vectors(vector_points, flow_vectors(vector_points), 0, 0, -radians(rot)) for rot in rotations] 

    all_errors = [field_error(points, lrvecs, *flow_vectors) for flow_vectors in all_flow_vectors]
    
    # 1D errorplot for the mean error over rotations
    #average_errors_1D = np.mean(all_errors, axis=1)
    #average_errors_1D_stds = np.std(all_errors, axis=1)
    
    
    savedir = os.path.join(ANALYSES_SAVEDIR, 'comparision_to_optic_flow', manalyser.get_specimen_name()+'_'+rotation_axis)
    os.makedirs(savedir, exist_ok=True)
    
    # SUBDIVIDE REGIONS
    # ------------------
    
    subdivide_flow_points = optimal_sampling(np.arange(-90, 90, 3), np.arange(-180, 180, 3))
    
    if subdivide_regions:
        subdivide_regions = [(-70, 70), (70, None), (None, -70)]
    else:
        subdivide_regions = [(None, None)]

    subdivide_regions_colors = ['black', 'gray', 'gray']#['pink', 'lime', 'turquoise']
    subdivide_styles = ['-', '-', '--']
    subdivide_lws = [3, 1, 1]
    subdivide_errors = []
    
    subdivide_points = []
    subdivide_vectors = []
    subdivide_flow_vectors = []

    for reg in subdivide_regions:
        manalyser.set_angle_limits(va_limits=reg)
        
        reglp, reglvecs = manalyser.get_3d_vectors('left')
        regrp, regrvecs = manalyser.get_3d_vectors('right')
        
        regpoints = np.concatenate((reglp, regrp))
        reglrvecs = np.concatenate((reglvecs, regrvecs))

        if rotation_axis == 'yaw':
            reg_all_flow_vectors = [rotate_vectors(subdivide_flow_points, flow_vectors(subdivide_flow_points), -radians(rot), 0, 0) for rot in rotations]
        elif rotation_axis == 'pitch':
            reg_all_flow_vectors = [rotate_vectors(subdivide_flow_points, flow_vectors(subdivide_flow_points), 0, -radians(rot), 0) for rot in rotations]
        elif rotation_axis == 'roll':
            reg_all_flow_vectors = [rotate_vectors(subdivide_flow_points, flow_vectors(subdivide_flow_points), 0, 0, -radians(rot)) for rot in rotations] 


        reg_all_errors = [field_error(regpoints, reglrvecs, *flow_vectors) for flow_vectors in reg_all_flow_vectors]
       
        #p, e = optic_flow_error(manalyser, all_flow_errors)
        e = np.mean(reg_all_errors, axis=1)
        subdivide_errors.append(e)

        #subdivide_flow_vectors.append( [flow_vectors(p, xrot=rot) for rot in rotations] )
        subdivide_flow_vectors.append( reg_all_flow_vectors )
        
        subdivide_points.append([reglp, regrp])
        subdivide_vectors.append([reglvecs, regrvecs])

    manalyser.set_angle_limits(va_limits=(None, None))
   
    # END OF SUBDIVIDE REGIONS
    # -------------------------

   
    
    fim = mpli.imread(fly_image)
    im_scaler = len(fim)
    

    N_steady_frames = 20*3
    steadied = False

    i_image = -1  # Saved image number
    i_steady = 0   # Steady looping
    i_rot = -1   # Determines current rotation
    while True:
        
        i_rot += 1
        rot = rotations[i_rot]

        print('Rotation {} degrees'.format(rot))

        if i_steady > N_steady_frames:
            steadied = True

        if rot > optimal_rot and not steadied:
            i_rot -= 1
            i_steady += 1
        
        if not text:
            if rot < optimal_rot:
                continue


        # Data collection part
        flow_vectors = all_flow_vectors[i_rot]
        flow_errors = all_errors[i_rot]

        i_image += 1
        fn = 'image_{:0>8}.png'.format(i_image)
        savefn = os.path.join(savedir, fn)
        
  
        # Plotting part        
        fig = plt.figure(figsize=(11.69,8.27), dpi=dpi)   
        
       

        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(N_fig_rows, N_fig_columns, N_fig_columns*i+1, projection='3d')

            for ppp, vecs, color in zip(subdivide_points, subdivide_vectors, subdivide_regions_colors): 

                lcolor = 'red'
                rcolor = 'blue'

                if color != 'black':
                    lcolor = color
                    rcolor = color

                vector_plot(ax, ppp[0], vecs[0], color=lcolor, mutation_scale=mutation_scale, animate=animate, camerapos=(elev,90))
                vector_plot(ax, ppp[1], vecs[1], color=rcolor, mutation_scale=mutation_scale, animate=animate, camerapos=(elev,90))
            
            ax.view_init(elev=elev, azim=90)
            
            ax.dist = 6
            
            if text:
                if dual_difference:
                    ax.text2D(0, 0.5, elevation_texts[i].replace('\n', ' '), transform=ax.transAxes, va='center', ha='center', rotation=90)
                else:
                    ax.text2D(-0.15, 0.5, elevation_texts[i], transform=ax.transAxes, va='center')

        


    


        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(N_fig_rows, N_fig_columns, N_fig_columns*i+2, projection='3d')
            vector_plot(ax, *flow_vectors, color='darkviolet', mutation_scale=mutation_scale, camerapos=(elev,90))
            ax.view_init(elev=elev, azim=90)

            ax.dist = 6

        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(N_fig_rows, N_fig_columns, N_fig_columns*i+3, projection='3d')
            m = surface_plot(ax, points, 1-np.array(flow_errors), cb=False)
            ax.view_init(elev=elev, azim=90)

            ax.dist = 6
        
        if dual_difference:
            for i, elev in enumerate(elevations):
                ax = fig.add_subplot(N_fig_rows, N_fig_columns, N_fig_columns*i+4, projection='3d')
                m = surface_plot(ax, points, np.array(flow_errors), cb=False)
                ax.view_init(elev=elev, azim=90)

                ax.dist = 6
            
        
        axes = fig.get_axes()
        j=0
        for ikk, ax in enumerate(axes):
            if ikk in [0,3,6,9]:
                if text:
                    ax.text2D(0.5, 1.05, column_texts[j], transform=ax.transAxes, ha='center', va='top')
                j +=1

        # Plot image of the fly
        ax = fig.add_subplot(N_fig_rows, N_fig_columns, N_fig_columns)
        ax.imshow(rotate(fim, rot, mode='nearest', reshape=False), cmap='gray')
        ax.set_frame_on(False)


        # Stop video at the optimal
        if 0<i_steady<N_steady_frames:
            rect = matplotlib.patches.Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False,
                    color='yellow', linewidth=8)
            if text:
                ax.add_patch(rect)
                ax.text(0.5, 1.1, 'Typical head tilt', ha='center', va='top', transform=ax.transAxes)

        # Add arrows
        if text:
            if sideflow == True:
                arrows = [np.array((1.05,ik))*im_scaler for ik in np.arange(0.1,0.91,0.1)]
                for x, y in arrows:
                    plt.arrow(x, y, -0.1*im_scaler, 0, width=0.01*im_scaler, color='darkviolet')
            else:
                x = []
                y = []
                for i in range(10):
                    for j in range(10):
                        x.append((0.5+i)*fim.shape[0]/10)
                        y.append((0.5+j)*fim.shape[1]/10)
                ax.scatter(x,y, marker='x', color='darkviolet')

        # Text
        ax = fig.add_subplot(N_fig_rows, N_fig_columns, 2*N_fig_columns)
        cbox = ax.get_position()
        print(cbox)
        cbox.x1 -= abs((cbox.x1 - cbox.x0))/1.1
        cbox.y0 -= 0.18*abs(cbox.y1-cbox.y0)
        cax = fig.add_axes(cbox)
        plt.colorbar(m, cax)
        if text:
            if dual_difference:
                text_x = 0.1
            else:
                text_x = 0.025
            cax.text(text_x, 0.95, 'Matching', va='top', transform=ax.transAxes)
            cax.text(text_x, 0.5, 'Perpendicular', va='center', transform=ax.transAxes)
            cax.text(text_x, 0.05, 'Opposing', va='bottom', transform=ax.transAxes)


        # Axis for the error
        ax = fig.add_subplot(N_fig_rows, N_fig_columns, 3*N_fig_columns)        
        ax_pos = ax.get_position()
        ax_pos = [ax_pos.x0+0.035, ax_pos.y0-0.08, ax_pos.width+0.022, ax_pos.height+0.05]
        ax.remove()
        ax = fig.add_axes(ax_pos)


        if text:
            ax.text(0.6, 1, 'Head tilt\n{} degrees'.format(int(rot)-zero_rot), transform=ax.transAxes, va='bottom', ha='left')
        
        
        
        # Plot the last figure, showing the mean error etc.
        if subdivide_errors:


            for subdivide_errors_1D, color, style, lw in zip(subdivide_errors, subdivide_regions_colors, subdivide_styles, subdivide_lws):
                if not text and color != 'black':
                    continue
                ax.plot(rotations-zero_rot, subdivide_errors_1D, style, lw=lw, color=color, label='Slower phase')
                # "Cursor"
                ax.scatter(rotations[i_rot]-zero_rot, subdivide_errors[0][i_rot], color='black')
                
                if dual_difference:
                    ax.plot(rotations-zero_rot, 1-subdivide_errors_1D, style, lw=lw, color='gray', label='Fast phase')
                    ax.scatter(rotations[i_rot]-zero_rot, 1-subdivide_errors[0][i_rot], color='gray')

            # Make optimal band
            ax.legend(loc=(0.39,1.2))
            left_side = -50
            right_side = 50
            p = matplotlib.patches.Rectangle((left_side, 0), right_side-left_side, 1, alpha=0.5, color='yellow')
            ax.add_patch(p)
            
            if text:
                ax.set_ylabel('Mean error')
            
            ax.set_ylim(np.min(subdivide_errors)-0.05, np.max(subdivide_errors)+0.05)
            l = matplotlib.lines.Line2D([rot-zero_rot, rot-zero_rot], [np.min(subdivide_errors)-0.05, np.max(subdivide_errors)+0.05], color='black')
            
            if text:
                ax.add_line(l)
                  
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        
        ax.set_xlim(-180-zero_rot, 180-zero_rot)
        
        
        if text:
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_xticks([-90, 0, 90])
            ax.set_xticklabels(['-90$^\circ$', '0$^\circ$','90$^\circ$'])
        else:
            ax.set_xticks([-90, 0, 90])
            ax.set_xticklabels([])
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels([])


        # Hide axis from all the figures that are 3D plots or plots of images
        axes = fig.get_axes()
        for ax in axes[0:-1]:
            ax.set_axis_off()
        
        if len(subdivide_regions) == 3:
            # Plot subdivide_regions 
            for i_ax, ax in enumerate(axes[0:8]):
                reg = subdivide_regions[0]
                color = subdivide_regions_colors[0]
                for theta in reg:
                    

                    style = '-'
                    if theta < 0 and i_ax in [0,3]:
                        style = '--'
                    elif theta > 0 and i_ax in [2, 5]:
                        style = '--'
                     
                    
                    phi = np.linspace(0,180)
                    phi = np.radians(phi)
                    theta = np.radians(90-theta)
                    x = np.cos(phi)
                    y = np.sin(phi) * np.sin(theta)
                    
                    z = np.sin(phi) * np.cos(theta)

                   
                    ax.plot(x,y,z, style, color=color, lw=1)
        

        if dual_difference:
            plt.subplots_adjust(left=0.02, bottom=0.05, right=0.95, top=0.90, wspace=0.0, hspace=0.1)
        else:
            plt.subplots_adjust(left=0.08, bottom=0, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        
        make_animation_timestep(step_size=0.025, low_val=0.7)

        if savefn:
            fig.savefig(savefn)
        plt.close()

