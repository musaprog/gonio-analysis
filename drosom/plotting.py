#!/usr/bin/env python3
'''
Plotting analysed DrosoM data.
'''
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
#from mayavi import mlab

# Plotting 3D in matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

#from pupil.coordinates import findClosest
from pupil.directories import ANALYSES_SAVEDIR
from .optic_flow import flow_direction



class MPlotter:

 
    def __init__(self):
        
        self.plots = {}
        self.savedir = os.path.join(ANALYSES_SAVEDIR, 'mplots')
        os.makedirs(self.savedir, exist_ok=True)
    
    def save(self):
        '''
        Saves all the plots made to the analysis savedirectory.
        '''

        for plot_name in self.plots:
            fn = os.path.join(self.savedir, plot_name)
            self.plots[plot_name]['fig'].savefig(fn+'.svg', format='svg')

    def setLimits(self, limit):
        '''
        Sets all the plots to the same limits

        limit       'common', 'individual' or to set fixed for all: (min_hor, max_hor, min_pit, max_pit)
        '''

        if len(limit) == 4:
            limit = tuple(limit)
        
        elif limit == 'common':
            
            manalysers = [self.plots[plot_name]['manalyser'] for plot_name in self.plots]

            limits = [1000, -1000, 1000, -1000]
            
            for manalyser in manalysers:
                for eye in ['left', 'right']:
                    angles, X, Y = manalyser.get2DVectors(eye)
                    horizontals, pitches = zip(*angles)
                    
                    #print(horizontals)
                    limits[0] = min(limits[0], np.min(horizontals))
                    limits[1] = max(limits[1], np.max(horizontals))
                    limits[2] = min(limits[2], np.min(pitches))
                    limits[3] = max(limits[3], np.max(pitches))
        
        for ax in [self.plots[plot_name]['ax'] for plot_name in self.plots]:
            ax.set_xlim(limits[0]-10, limits[1]+10)
            ax.set_ylim(limits[2]-10, limits[3]+10)
    
    
    def plot_2d_trajectories(self, manalyser):
        '''
        Plot 2D movement trajectories of the pseudopupils, separetly for each imaged position.
        '''
        
        plt_limits = [[],[],[],[]]

        figure_content = []

        for eye in ['left', 'right']:
            angles, movements = manalyser.get_raw_xy_traces(eye)
            
            for movement in movements:
                subfig_dict = {'eye': eye}
                
                x, y = [[],[]]
                for repeat in movement:
                    x.extend(repeat['x'])
                    y.extend(repeat['y'])
                    
                plt_limits[0].append(np.min(x))
                plt_limits[1].append(np.max(x))
                plt_limits[2].append(np.min(y))
                plt_limits[3].append(np.max(y))

                subfig_dict = {'x': x, 'y': y, **subfig_dict}
                figure_content.append(subfig_dict)
        
        ROWS, COLS = (8, 6)
        i_page = 0
        

        for i, data in enumerate(figure_content):
            
            if i == i_page * ROWS * COLS:
                fig = plt.figure()
                i_page += 1

        
            ax = plt.subplot(ROWS, COLS, i - ((i_page-1) * ROWS * COLS)+1)
            ax.axis('off')
            cmap = matplotlib.cm.get_cmap('inferno', len(data['x']))
            
            for i_point in range(1, len(data['x'])):
                ax.plot([data['x'][i_point], data['x'][i_point-1]], [data['y'][i_point], data['y'][i_point-1]], color=cmap(i_point/len(data['x'])))
                
                
                #ax.set_xlim([np.percentile(plt_limits[0], 99), np.percentile(plt_limits[1], 99)])
                #ax.set_ylim([np.percentile(plt_limits[2], 99), np.percentile(plt_limits[3], 99)])



            #ax.suptitle('{} eye, {}'.format(data['eye'], data['time']))
        
        plt.show()
    
    def plotDirection2D(self, manalyser):
        '''
        manalyser       Instance of MAnalyser class or MAverager class (having get2DVectors method)
        
        limits          Limits for angles, [min_hor, max_hor, min_pitch, max_pitch]
        '''
        

        fig, ax = plt.subplots()
        
   
        for color, eye in zip(['red', 'blue'], ['left', 'right']):
            angles, X, Y = manalyser.get2DVectors(eye)
            for angle, x, y in zip(angles, X, Y):
               
                horizontal, pitch = angle
                
                # If magnitude too small, its too unreliable to judge the orientation so skip over
                #movement_magnitude = math.sqrt(x**2 + y**2)
                #if movement_magnitude < 2:
                #    continue 
  
                # Vector orientation correction due to sample orientation dependent camera rotation
                #xc = x * np.cos(np.radians(pitch)) + y * np.sin(np.radians(pitch))
                #yc = x * np.sin(np.radians(pitch)) + y * np.cos(np.radians(pitch))

                # Scale all vectors to the same length
                scaler = math.sqrt(x**2 + y**2) / 5 #/ 3
                #scaler = 0
                if scaler != 0:
                    x /= scaler
                    y /= scaler /2.4    # FIXME

                #ar = matplotlib.patches.Arrow(horizontal, pitch, xc, yc)
                ar = matplotlib.patches.FancyArrowPatch((horizontal, pitch), (horizontal-x, pitch+y), mutation_scale=10, color=color, picker=True)
                #fig.canvas.mpl_connect('pick_event', self.on_pick)
                ax.add_patch(ar)
        
        ax.set_xlabel('Horizontal (degrees)')
        ax.set_ylabel('Pitch (degrees)')
        
        for key in ['top', 'right']:
            ax.spines[key].set_visible(False)
       
        plot_name = 'direction2D_{}'.format(manalyser.getFolderName())
        self.plots[plot_name] = {'fig': fig, 'ax': ax, 'manalyser': manalyser}

    def plotMagnitude2D(self, manalyser):
        '''

        TODO
        - combine eyes to yield better picture
        - axes from pixel values to actual

        '''
        
        distancef = lambda p1,p2: math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
        
        fig, ax = plt.subplots(ncols=2)
     
        for eye_i, (color, eye) in enumerate(zip(['red', 'blue'], ['left', 'right'])):
            angles, X, Y = manalyser.get2DVectors(eye)
            
                
            HOR = []
            PIT = []
            for angle, x, y in zip(angles, X, Y):
                horizontal, pitch = angle
                HOR.append(horizontal)
                PIT.append(pitch) 

            # TRY NEAREST NEIGHBOUR INTERPOLATION
            res = (50, 50)
            xi = np.linspace(np.min(HOR), np.max(HOR), res[0]) 
            yi = np.linspace(np.min(PIT), np.max(PIT), res[1]) 
            zi = np.zeros(res)
            for j in range(len(yi)):
                for i in range(len(xi)):
                    point = findClosest((xi[i], yi[j]), angles, distance_function=distancef)
                    
                    index = angles.index(point)
                    
                    zi[j][i] = (math.sqrt(X[index]**2 + Y[index]**2))


            print('{} to {}'.format(xi[0], xi[-1]))
            print('{} to {}'.format(yi[0], yi[-1]))

            im = ax[eye_i].imshow(zi, interpolation='none', extent=[xi[0], xi[-1], yi[0], yi[-1]])
            #cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax[eye_i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            
            fig.colorbar(im, cax=cax)

            ax[eye_i].title.set_text('{} eye'.format(eye.capitalize()))          

            #XYZ.append([xi,yi,zi])
    
        #fig = plotter.contourplot(XYZ, 1, 2, colorbar=True) 
        #X,Y = np.meshgrid(X, Y)
        #plt.pcolor(X,Y,Z)

        #ax.set_xlim(-np.max(HOR)-10, -np.min(HOR)+10)
        #ax.set_ylim(-np.max(PIT)-10, -np.min(PIT)+10)
        #ax.set_xlabel('Horizontal angle (degrees)')
        #ax.set_ylabel('Pitch angle (degrees)')
    

    def plotTimeCourses(self, manalyser, exposure=0.010):
        '''
        Plotting time courses

        FIXME This became dirty
        '''
        avg = []

        for eye in ['left', 'right']:
            traces = manalyser.getMagnitudeTraces(eye)
            
            for angle in traces:
                print(np.max(traces[angle]))
                trace = traces[angle]
                #trace /= np.max(traces[angle])
                if np.isnan(trace).any():
                    print('Nan')
                    continue
                avg.append(trace)
        
        for trace in avg:
            time = np.linspace(0, exposure*len(trace)*1000, len(trace))
            plt.plot(time, trace)
            #plt.show(block=False)
            #plt.pause(0.1)
            #plt.cla()
        print(len(avg))

        avg = np.mean(np.asarray(avg), axis=0)
        print(avg)
        plt.plot(time, avg, color='black')

        plt.show()



    class Arrow3D(FancyArrowPatch):
        def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            #self._verts3d = xs, ys, zs
            self._verts3d = (x0, x1), (y0, y1), (z0, z1)

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    

    def plot_3d_vectormap_mayavi(self, manalyser):
        '''
        Use mayavi to make the 3D image that then can be saved in obj file format.
        '''

        for color, eye in zip([(1.,0,0), (0,0,1.)], [('left'), 'right']):
            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True)


            N = len(vectors_3d)
            arrays = [np.zeros(N) for i in range(6)]

            for i in range(N):
                arrays[0][i] = vectors_3d[i][1][0]
                arrays[1][i] = vectors_3d[i][2][0]
                arrays[2][i] = vectors_3d[i][3][0]
                
                arrays[3][i] = vectors_3d[i][1][1] - arrays[0][i]
                arrays[4][i] = vectors_3d[i][2][1] - arrays[1][i]
                arrays[5][i] = vectors_3d[i][3][1] - arrays[2][i]

            mlab.quiver3d(*arrays, color=color)
        
        mlab.show()


    def when_moved(self, event):
        '''
        Callback to make two axes to have synced rotation when rotating
        self.axes[0].
        '''
        if event.inaxes == self.axes[0]:
            self.axes[1].view_init(elev = self.axes[0].elev, azim = self.axes[0].azim)
        self.fig.canvas.draw_idle()


    def plot_3d_vectormap(self, manalyser, with_optic_flow=False, animation=False):
        '''
        relp0   Relative zero point

        with_optic_flow         Angle in degrees. If non-false, plot also estimated optic
                                flow with this parameter
        animation           Sequence of (elevation, azimuth) points to create an
                                animation of object rotation
        '''

        fig = plt.figure(figsize=(15,15))
        
        if with_optic_flow:
            axes = []
            axes.append( fig.add_subplot(121, projection='3d') )
            axes.append( fig.add_subplot(122, projection='3d') )

        else:
            axes = [fig.add_subplot(111, projection='3d')]
        

        
        points = []
        pitches = []

    
        for color, eye in zip(['red', 'blue'], ['left', 'right']):
            
            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True)
            vector_plot(axes[0], *vectors_3d, color='blue', mutation_scale=15)

            if with_optic_flow:
                flow_vectors = [flow_direction(P0, xrot=with_optic_flow) for P0 in vectors_3d[0]]
                vector_plot(axes[1], vectors_3d[0], flow_vectors)        


            
        for ax in axes:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1,1)
            ax.set_zlim(-1, 1)
            
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
        
        
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)



        if with_optic_flow and not animation:
            connection = fig.canvas.mpl_connect('motion_notify_event', self.when_moved)
            self.axes = axes
            self.fig = fig

        if animation:
            savedir = os.path.join(self.savedir, 'vectormap_3d_anim')
            os.makedirs(savedir, exist_ok=True)

            plt.show(block=False)

            for i, (elevation, azimuth) in enumerate(animation):
                print('{} {}'.format(elevation, azimuth)) 
                for ax in axes:
                    ax.view_init(elev=elevation, azim=azimuth)
                fig.canvas.draw_idle()

                fn = 'image_{:0>8}.png'.format(i)
                fig.savefig(os.path.join(savedir, fn))
                #plt.pause(0.1)

        # make the panes transparent
        #ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        #ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #plt.savefig('vectormap.svg', transparent=True)
        
        
        plt.show()


class Arrow3D(FancyArrowPatch):
    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = (x0, x1), (y0, y1), (z0, z1)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def vector_plot(ax, points, vectors, color='black', mutation_scale=3):
    '''
    Plot vectors on ax.
    '''
    for point, vector in zip(points, vectors):
        ar = Arrow3D(*point, *(point+vector), arrowstyle="-|>", lw=1,
                mutation_scale=mutation_scale, color=color)
        ax.add_artist(ar)
    
    # With patches limits are not automatically adjusted
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1, 1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def surface_plot(ax, points, values, cb=False):
    '''
    3D surface plot of the error between the optic flow vectors and the actual
    eye-movements vector map.
    
    points
    values
    '''
    
    from coordinates import nearest_neighbour
    from matplotlib import cm
    import matplotlib.colors
    # Points where the error is "evaluated" (actually interpolated)
    
    N = 100
    phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, np.pi, N))
    X = np.sin(theta) * np.cos(phi)
    Y = np.sin(theta) * np.sin(phi)
    Z = np.cos(theta)
    

    def color_function(theta, phi):

        intp_dist = (2 * np.sin(np.radians(5)))

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        errs = np.empty_like(x)

        for i in range(x.size):
            i_point = nearest_neighbour((x.flat[i], y.flat[i], z.flat[i]), points,
                    max_distance=intp_dist)
            
            if i_point is False:
                errs.flat[i] = 0
            else:
                errs.flat[i] = values[i_point]
        return errs

    colors = color_function(theta, phi)
    
    
    #fig = plt.figure(figsize=(15,6))
        

    #for i, elev in enumerate([-50, 0, 50]):

    #     ax = fig.add_subplot(1,3,i+1, projection='3d')
    
    culurs = [(0.2, 0.1, 0),(1,0.55,0),(1,1,0.4)]
    ownmap = matplotlib.colors.LinearSegmentedColormap.from_list('ownmap', culurs, 100)
    
    ax.plot_surface(X, Y, Z, facecolors=ownmap(colors), linewidth=0, vmin=0, vmax=1)
    #ax.view_init(elev=elev, azim=90)

   
    #fig.suptitle(title)
    #plt.subplots_adjust(left=0, bottom=0.05, right=0.95, top=0.95, wspace=-0.1, hspace=-0.1)
    
    m = cm.ScalarMappable(cmap=ownmap)
    m.set_array(colors)
    return m    
    #plt.show()


def complete_flow_analysis(manalyser, rotations):
    '''
    Creates combined plot to 
    
    PLOT
    ROW
    1       measured optic flow vectors
    2       simulated optic flow
    3       error heatplot

    '''
    
    from scipy.ndimage import rotate
    import matplotlib.image as mpli

    from .new_analysing import optic_flow_error
    from .optic_flow import flow_direction, flow_vectors, field_error

    # Flow field errors for each rotation
    points, all_errors = optic_flow_error(manalyser, rotations)
    
    # Flow field vectors for each rotation
    all_flow_vectors = [flow_vectors(points, xrot=rot) for rot in rotations]
    
    fly_image = 'side_aligning_pupil_antenna_whiteBG.jpg'

    savedir = 'optic_flow_error'
    
    
    # 1D errorplot for the mean error over rotations
    average_errors_1D = np.mean(all_errors, axis=1)


    
    fim = mpli.imread(fly_image)
    im_scaler = len(fim)
    

    N_steady_frames = 20*3
    optimal_rot = 35
    steadied = False

    i_image = -1  # Saved image number
    i_steady = 0   # Steady looping
    i_rot = -1   # Determines current rotation
    while True:

        i_rot += 1
        rot = rotations[i_rot]

        if i_steady > N_steady_frames:
            steadied = True

        if rot > optimal_rot and not steadied:
            i_rot -= 1
            i_steady += 1

        # Data collection part

        points = points
        flow_vectors = all_flow_vectors[i_rot]
        flow_errors = all_errors[i_rot]

        i_image += 1
        fn = 'image_{:0>8}.png'.format(i_image)
        savefn = os.path.join(savedir, fn)
        
  
        # Plotting part

        elevations = [50, 0, -50]

        fig = plt.figure(figsize=(11.69,8.27), dpi=150)
    
        lp, lvecs = manalyser.get_3d_vectors('left')
        rp, rvecs = manalyser.get_3d_vectors('right')
        
        
        elevation_texts = ['Dorsal\nview', 'Anterior\nview', 'Ventral\nview']

        column_texts = ['Pseudopupil movement\ndirections',
                'Experienced optic flow',
                'Difference',
                'Head orientation']
        


        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(3, 4, 4*i+1, projection='3d')

    
            vector_plot(ax, lp, lvecs, color='red')
            vector_plot(ax, rp, rvecs, color='blue')
            
            ax.view_init(elev=elev, azim=90)
            
            ax.dist = 6
            
            ax.text2D(-0.15, 0.5, elevation_texts[i], transform=ax.transAxes, va='center')

        


    


        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(3, 4, 4*i+2, projection='3d')
            vector_plot(ax, points, flow_vectors, color='darkviolet')
            ax.view_init(elev=elev, azim=90)

            ax.dist = 6

        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(3, 4, 4*i+3, projection='3d')
            m = surface_plot(ax, points, 1-np.array(flow_errors), cb=False)
            ax.view_init(elev=elev, azim=90)

            ax.dist = 6
        
        
        axes = fig.get_axes()
        j=0
        for ikk, ax in enumerate(axes):
            if ikk in [0,3,6,9]:
                ax.text2D(0.5, 1.05, column_texts[j], transform=ax.transAxes, ha='center', va='top')
                j +=1

        # Plot image of the fly
        ax = fig.add_subplot(3, 4, 4)
        ax.imshow(rotate(fim, rot, mode='nearest', reshape=False), cmap='gray')
        ax.set_frame_on(False)
       
        if 0<i_steady<N_steady_frames:
            #ax.set_facecolor('yellow') 
            rect = matplotlib.patches.Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False,
                    color='yellow', linewidth=8)
            ax.add_patch(rect)
            ax.text(0.5, 1.1, 'OPTIMAL MATCH', ha='center', va='top', transform=ax.transAxes)

        # Add arrows
        arrows = [np.array((1.05,ik))*im_scaler for ik in np.arange(0.1,0.91,0.1)]
        for x, y in arrows:
            plt.arrow(x, y, -0.1*im_scaler, 0, width=0.01*im_scaler, color='darkviolet')





        # Text
        ax = fig.add_subplot(3, 4, 8)
        
        cbox = ax.get_position()
        cbox.x1 -= abs((cbox.x1 - cbox.x0))/1.1
        cbox.y0 -= 0.18*abs(cbox.y1-cbox.y0)
        cax = fig.add_axes(cbox)
        plt.colorbar(m, cax)
        cax.text(1.2, 1, 'Matching', va='top')
        cax.text(1.2, 0.5, 'Perpendicular', va='center')
        cax.text(1.2, 0, 'Opposing', va='bottom')

        ax = fig.add_subplot(3, 4, 12)
        #ax.plot(rotations, average_errors_1D)
        #ax.scatter(rot, np.mean(flow_errors))
        #ax.set_frame_on(False)

        ax.text(0, 0.5, 'Head tilt {} degrees'.format(int(rot)), transform=ax.transAxes)
       
       
        axes = fig.get_axes()
        for ax in axes:
            ax.set_axis_off()

       
            
       
        plt.subplots_adjust(left=0.08, bottom=0, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        #plt.show()
        

        if savefn:
            fig.savefig(savefn)
        plt.close() 
