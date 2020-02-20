#!/usr/bin/env python3
'''
Plotting analysed DrosoM data.
'''
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation
#from mayavi import mlab

# Plotting 3D in matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

#from pupil.coordinates import findClosest
from pupil.directories import ANALYSES_SAVEDIR
from .optic_flow import flow_direction

from pupil.coordinates import force_to_tplane

CURRENT_ARROW_LENGTH = 1

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
    
    def displacement_over_time(self, manalyser):
        
        displacement_traces = []

        for eye in ['let', 'right']:
            traces = manalyser.get_magnitude_traces(eye)
            
            for item in traces.items():
                displacement_traces.append(item)

        
        plt.plot(np.mean(displacement_traces), axis=0)
        plt.show()

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
            #plt.plot(time, trace)
            break
            #plt.show(block=False)
            #plt.pause(0.1)
            #plt.cla()
        print(len(avg))

        avg = np.mean(np.asarray(avg), axis=0)
        print(avg)
        plt.plot(time, avg)
        plt.xlabel('Time (ms)')
        plt.ylabel('Displacement (pixels)')
        plt.suptitle('Displacement over time\n{}'.format(manalyser.get_specimen_name()))
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


    def plot_3d_vectormap(self, manalyser, with_optic_flow=False, animation=False, arrow_animation=True):
        '''
        relp0   Relative zero point

        with_optic_flow         Angle in degrees. If non-false, plot also estimated optic
                                flow with this parameter
        animation           Sequence of (elevation, azimuth) points to create an
                                animation of object rotation
        '''

        fig = plt.figure(figsize=(15,15))
        fig.canvas.set_window_title(manalyser.get_specimen_name())


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
            vector_plot(axes[0], *vectors_3d, color=color, mutation_scale=15)

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
            savedir = os.path.join(self.savedir, 'vectormap_3d_anim_{}'.format(manalyser.get_specimen_name()))
            os.makedirs(savedir, exist_ok=True)

            #plt.show(block=False)
            
            try:
                video_writer = matplotlib.animation.writers['ffmpeg'](fps=20, metadata={'title':manalyser.get_specimen_name()})
                video_writer.setup(fig, os.path.join(savedir,'{}.mp4'.format(manalyser.get_specimen_name())))
            except RuntimeError:
                print('Install ffmpeg by "pip install ffmpeg" to get the video')
                video_writer = False

            doublegrab_next = False

            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
            
            for i, (elevation, azimuth) in enumerate(animation):
                
                try:
                    

                    if arrow_animation:
                        axes[0].clear()
                        for color, eye in zip(['red', 'blue'], ['left', 'right']):
                            vectors_3d = manalyser.get_3d_vectors(eye, correct_level=True)
                            vector_plot(axes[0], *vectors_3d, color=color, mutation_scale=15, animate=arrow_animation, guidance=True, camerapos=(elevation, azimuth))

                        make_animation_timestep(step_size=0.02, low_val=0.33, high_val=1)
                                    
                    print('{} {}'.format(elevation, azimuth)) 
                    for ax in axes:
                        ax.view_init(elev=elevation, azim=azimuth)
                    fig.canvas.draw_idle()
                    # for debugging here plt.show()
                    fn = 'image_{:0>8}.png'.format(i)
                    fig.savefig(os.path.join(savedir, fn))
                    if video_writer:
                         video_writer.grab_frame()
                         if doublegrab_next:
                            video_writer.grab_frame()
                            doublegrab_next = False
                    #plt.pause(0.1)

                except Exception as e:
                    print('Could not make a frame, error message on the next line')
                    print(e)
                    doublegrab_next = True
            if video_writer:
                video_writer.finish()



        else:
            #plt.show()
            pass
        # make the panes transparent
        #ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        #ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        #plt.savefig('vectormap.svg', transparent=True)
        
        


class Arrow3D(FancyArrowPatch):
    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = (x0, x1), (y0, y1), (z0, z1)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def make_animation_timestep(step_size=0.1, low_val=0.5, high_val=1.2):
    '''
    step_size between 0 and 1.
    Once total displacement 1 has been reached go back to low_value
    '''
    # Update arrow length
    global CURRENT_ARROW_LENGTH
    CURRENT_ARROW_LENGTH += step_size
    if CURRENT_ARROW_LENGTH > high_val:
        CURRENT_ARROW_LENGTH = low_val
        
def is_behind_sphere(elev, azim, point):
    '''
    Calculates wheter a point seend by observer at (elev,azim) in spehrical
    coordinates is behind a sphere (radius == point) or not.
    
    NOTICE: Elev from horizontal plane (a non-ISO convention) and azim as in ISO
    '''
    cx = np.sin(np.radians(90-elev)) * np.cos(np.radians(azim))
    cy = np.sin(np.radians(90-elev)) * np.sin(np.radians(azim))
    cz = np.cos(np.radians(90-elev))
    
    vec_cam = (cx,cy,cz)
    vec_arr = point
    angle = np.arccos(np.inner(vec_cam, vec_arr)/(np.linalg.norm(vec_cam)*np.linalg.norm(vec_arr)))
    
    if angle > np.pi/2:
        return True
    else:
        return False




def vector_plot(ax, points, vectors, color='black', mutation_scale=3,
        animate=False, guidance=False, camerapos=None):
    '''
    Plot vectors on ax.

    ax              Matplotlib ax (axes) instance
    points          Starting points 
    vectors
    color           Color of the arrows
    mutation_scale  Size of the arrow head basically
    animate         Set the size of the arrows to the current size in the animation
                    Call make_animation_timestep to go to the next step
    guidance    Add help elements to point left,right,front,back etc. and hide axe
    camerapos       (elev, axzim). Supply so vectors bending the visible himspehere can be hidden 
    '''


    # With patches limits are not automatically adjusted
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1, 1)
     
    if guidance:
        ax.set_axis_off()
        
        r = 0.9
        
        guidances = {'Right': ((r,0,0), (0.2,0,0)),
                'Left': ((-r,0,0),(-0.2,0,0)),
                '  Ventral': ((0,0,-r),(0,0,-0.3)),
                '  Dorsal': ((0,0,r),(0,0,0.3))}
        #'Antenna': ((0,0.7,0),(0,0.4,0))}

        
        for name, (point, vector) in guidances.items():
            point = np.array(point)
            vector = np.array(vector)

            if is_behind_sphere(*camerapos, point):
                zorder = 1
            else:
                zorder = 8

            ar = Arrow3D(*point, *(point+vector), mutation_scale=5, lw=0.2, color='black', zorder=zorder)
            ax.add_artist(ar)
            
            if name in ('Left', 'Right'):
                ha = 'center'
            else:
                ha = 'left'
            ax.text(*(point+vector/1.05), name, color='black', fontsize='xx-large', va='bottom', ha=ha, linespacing=1.5, zorder=zorder+1)
        
        N = 75
        phi, theta = np.meshgrid(np.linspace(0, 2*np.pi, N), np.linspace(0, np.pi, N))
        X = r * np.sin(theta) * np.cos(phi)
        Y = r * np.sin(theta) * np.sin(phi)
        Z = r * np.cos(theta)
        ax.plot_surface(X, Y, Z, color='lightgray')

    if animate:
        global CURRENT_ARROW_LENGTH
        scaler = CURRENT_ARROW_LENGTH
    else:
        scaler = 1

    for point, vector in zip(points, vectors):

        if camerapos:
            #elev, azim = camerapos
            #cx = np.sin(np.radians(90-elev)) * np.cos(np.radians(azim))
            #cy = np.sin(np.radians(90-elev)) * np.sin(np.radians(azim))
            #cz = np.cos(np.radians(90-elev))
            
            #if elev < 0:
            #    cz = -cz


            #vec_cam = (cx,cy,cz)
            vec_arr = point
            #angle = np.arccos(np.inner(vec_cam, vec_arr)/(np.linalg.norm(vec_cam)*np.linalg.norm(vec_arr)))
    

            if is_behind_sphere(*camerapos, vec_arr):
                alpha = 0
            else:
                alpha = 1
                zorder = 10

        else:
            alpha = 1
            zorder = 10

        

        ar = Arrow3D(*point, *(point+scaler*vector), arrowstyle="-|>", lw=1,
                mutation_scale=mutation_scale, color=color, alpha=alpha, zorder=10)
        ax.add_artist(ar)
    
    #try:
    #    ar = Arrow3D(*(0,0,0), *vec_cam, arrowstyle="-|>", lw=1,
    #        mutation_scale=mutation_scale, color='green', zorder=11)
    #    
    #    ax.add_artist(ar)
    #except:
    #    pass
    
           
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_zlim(-1.1, 1.1)
    

   
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


def histogram_heatmap(all_errors, nbins=20, horizontal=True, drange=None):
    '''

    all_errors      [errors_rot1, errors_rot1, ....] where
                    errors_rot_i = [error1, error2, ...], error_i is float
    '''
    N_bins = 20
    #N_bins = 3
    if drange == 'auto':
        data_range = (np.min(all_errors), np.max(all_errors))
        print('histogram_heatmap data_range {}'.format(data_range))
    elif drange != 'auto':
        data_range = drange
    else:
        data_range = (0, 1)

    image = []

    for rotation_errors in all_errors:
        hist, bin_edges = np.histogram(rotation_errors, bins=N_bins, range=data_range)
        image.append(hist)

    image = np.array(image)

    if horizontal:
        image = image.T

    return image

def complete_flow_analysis(manalyser, rotations, text=True, lastfig=True, error_heatmap=False, fitting_analysis=False):
    '''
    Creates combined plot to 
    
    PLOT
    ROW
    1       measured optic flow vectors
    2       simulated optic flow
    3       error heatplot

    INPUT ARGUMENTS
    text                wheter to render text

    '''
    #text=False 
    from scipy.ndimage import rotate
    import matplotlib.image as mpli
    
    
    import scipy.stats

    from .new_analysing import optic_flow_error
    from .optic_flow import flow_direction, flow_vectors, field_error
    from .fitting import get_reference_vector

    # Flow field errors for each rotation
    points, all_errors = optic_flow_error(manalyser, rotations)
    
    # Flow field vectors for each rotation
    all_flow_vectors = [flow_vectors(points, xrot=rot) for rot in rotations]
    
    #fly_image = 'side_aligning_pupil_antenna_whiteBG.jpg'
    fly_image = 'mikkos_alternative_fly.png'

    savedir = 'optic_flow_error'
    
    mutation_scale = 3
    animate = True

    
    # 1D errorplot for the mean error over rotations
    average_errors_1D = np.mean(all_errors, axis=1)
    average_errors_1D_stds = np.std(all_errors, axis=1)
    
    errors_circmeans = scipy.stats.circmean(all_errors, high=2,axis=1)
    errors_circstds = scipy.stats.circstd(all_errors, high=2, axis=1)



    #errors_low_percentile = np.percentile(all_errors, 25, axis=1)
    #errors_high_percentile = np.percentile(all_errors, 75, axis=1)

    #sems = scipy.stats.sem(all_errors, axis=1)

    #median_errors_1D = np.median(all_errors, axis=1)
    #median_errors_1D_mads = scipy.stats.median_absolute_deviation(all_errors, axis=1)
   
    # all errors = the residuals
    squared_errors = np.array(all_errors)**2
    mse = np.mean(squared_errors, axis=1)

    if fitting_analysis:
        movement_vectors = []
        movement_points = []
        for eye in ['left', 'right']:
            ignore, vectors = manalyser.get_3d_vectors(eye)
            movement_vectors.extend(np.array(vectors))
            movement_points.extend(np.array(ignore))
        
        
        reference_vectors = [get_reference_vector(P0) for P0 in movement_points]
        optic_flow_vectors = [[flow_direction(P0, xrot=rot) for P0 in movement_points] for rot in rotations]
        
        F = np.array( [field_error(flow, reference_vectors) for flow in optic_flow_vectors] )
        Y = np.array( field_error(movement_vectors, reference_vectors) )

        residuals = (Y[np.newaxis,:].repeat(len(rotations), axis=0) - F)
        
        errors_circmeans = scipy.stats.circmean(residuals, low=-1, high=1,axis=1)
        errors_circstds = scipy.stats.circstd(residuals, low=-1, high=1, axis=1)

       

    #ignore, vector_orientations = optic_flow_error(manalyser, rotations, self_error=True)
    #vector_orientations = np.array(vector_orientations[0]+vector_orientations[1])
    
    #plt.hist(vector_orientations)
    #plt.show()
 
    #plt.imshow(histogram_heatmap(vector_orientations, nbins=50, horizontal=True), aspect='auto')
    #plt.show()

    #print('Vector orientations max {}, min {}'.format(np.max(vector_orientations), np.min(vector_orientations)))

    #SStot = np.sum((vector_orientations - vector_orientations_mean[:, np.newaxis])**2, axis=1)
    
    if fitting_analysis:
        N_vectors = len(points)

        
        movement_vectors = []
        movement_points = []
        for eye in ['left', 'right']:
            ignore, vectors = manalyser.get_3d_vectors(eye)
            movement_vectors.extend(np.array(vectors))
            movement_points.extend(np.array(ignore))
        
        
        #reference_vectors = [force_to_tplane(point, np.array(point) + np.array([1,0,0]))-np.array(point) for point in movement_points]
        #reference_vectors = [np.array([1,0,0]) for point in movement_points]
        reference_vectors = [get_reference_vector(P0) for P0 in movement_points]
        optic_flow_vectors = [[flow_direction(P0, xrot=rot) for P0 in movement_points] for rot in rotations]
        
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(1,2,1, projection='3d')
            vector_plot(ax, movement_points, reference_vectors, color='gray')
            vector_plot(ax, movement_points, optic_flow_vectors[0], color='darkviolet')
            
            ax2 = fig.add_subplot(1,2,2, projection='3d')
            vector_plot(ax2, movement_points, reference_vectors, color='gray')
            vector_plot(ax2, movement_points, movement_vectors, color='blue')
            
            plt.show()

        F = np.array( [field_error(flow, reference_vectors) for flow in optic_flow_vectors] )
        Y = np.array( field_error(movement_vectors, reference_vectors) )
     
        
        #residuals = (Y[np.newaxis,:].repeat(len(rotations), axis=0) - F)
        #plt.imshow(histogram_heatmap(residuals, nbins=50, drange='auto'), aspect='auto', extent=(-180,180,-1,1), origin='lower')
        #plt.show()

        #SStot = np.sum( (Y - np.mean(Y))**2 )

        #SSres = np.sum( (Y[np.newaxis,:].repeat(len(rotations), axis=0) - F)**2 , axis=1)

        #R_squared = 1 - (SSres / SStot)
        
        R_squared = []
        for i_rot in range(len(rotations)):
            rr = scipy.stats.linregress(Y, F[i_rot])
            rr = rr[2]
            R_squared.append(rr)

        plt.plot(R_squared)
        plt.show()


    if error_heatmap:
        errors_image = histogram_heatmap(squared_errors)

    
    fim = mpli.imread(fly_image)
    im_scaler = len(fim)
    

    N_steady_frames = 20*3
    #optimal_rot = 35
    optimal_rot= 50
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
        
        if not text:
            if rot < optimal_rot:
                continue


        # Data collection part

        points = points
        flow_vectors = all_flow_vectors[i_rot]
        flow_errors = all_errors[i_rot]

        i_image += 1
        fn = 'image_{:0>8}.png'.format(i_image)
        savefn = os.path.join(savedir, fn)
        
  
        # Plotting part

        elevations = [50, 0, -50]
        
        if text:
            dpi = 150
        else:
            dpi = 600

        fig = plt.figure(figsize=(11.69,8.27), dpi=dpi)
    
        lp, lvecs = manalyser.get_3d_vectors('left')
        rp, rvecs = manalyser.get_3d_vectors('right')
        
        
        elevation_texts = ['Dorsal\nview', 'Anterior\nview', 'Ventral\nview']

        column_texts = ['Pseudopupil movement\ndirections',
                'Experienced optic flow',
                'Difference',
                'Head orientation']
        


        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(3, 4, 4*i+1, projection='3d')

    
            vector_plot(ax, lp, lvecs, color='red', mutation_scale=mutation_scale, animate=animate)
            vector_plot(ax, rp, rvecs, color='blue', mutation_scale=mutation_scale, animate=animate)
            
            ax.view_init(elev=elev, azim=90)
            
            ax.dist = 6
            
            if text:
                ax.text2D(-0.15, 0.5, elevation_texts[i], transform=ax.transAxes, va='center')

        


    


        for i, elev in enumerate(elevations):
            ax = fig.add_subplot(3, 4, 4*i+2, projection='3d')
            vector_plot(ax, points, flow_vectors, color='darkviolet', mutation_scale=mutation_scale)
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
                if text:
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
            if text:
                ax.add_patch(rect)
                ax.text(0.5, 1.1, 'OPTIMAL RANGE', ha='center', va='top', transform=ax.transAxes)

        # Add arrows
        if text:
            arrows = [np.array((1.05,ik))*im_scaler for ik in np.arange(0.1,0.91,0.1)]
            for x, y in arrows:
                plt.arrow(x, y, -0.1*im_scaler, 0, width=0.01*im_scaler, color='darkviolet')





        # Text
        ax = fig.add_subplot(3, 4, 8)
        cbox = ax.get_position()
        print(cbox)
        cbox.x1 -= abs((cbox.x1 - cbox.x0))/1.1
        cbox.y0 -= 0.18*abs(cbox.y1-cbox.y0)
        cax = fig.add_axes(cbox)
        plt.colorbar(m, cax)
        if text:
            cax.text(0.025, 0.95, 'Matching', va='top', transform=ax.transAxes)
            cax.text(0.025, 0.5, 'Perpendicular', va='center', transform=ax.transAxes)
            cax.text(0.025, 0.05, 'Opposing', va='bottom', transform=ax.transAxes)


        # Axis for the error
        ax = fig.add_subplot(3, 4, 12)        
        ax_pos = ax.get_position()
        ax_pos = [ax_pos.x0+0.035, ax_pos.y0-0.08, ax_pos.width+0.022, ax_pos.height+0.05]
        ax.remove()
        ax = fig.add_axes(ax_pos)


        if text:
            ax.text(0.6, 1, 'Head tilt\n{} degrees'.format(int(rot)), transform=ax.transAxes, va='bottom', ha='left')
        
        
        
        if lastfig:
            

            if error_heatmap:
                errors_imshow = ax.imshow((errors_image), aspect='auto', extent=(-180, 180, 0, 1), origin='lower')
                cax2 = fig.add_axes([ax_pos[0]+ax_pos[2], ax_pos[1], cbox.width, ax_pos[3]])
                plt.colorbar(errors_imshow, cax2)
            
            #cax3 = fig.add_axes([ax_pos[0]-cbox.width, ax_pos[1], cbox.width, ax_pos[3]])
            #plt.colorbar(m, cax3)


           
            
            
            #ax.plot(rotations, average_errors_1D + sems, color='green')
            #ax.plot(rotations, average_errors_1D - sems, color='green')
            
            #ax.plot(rotations, errors_high_percentile, color='green')
            #ax.plot(rotations, errors_low_percentile, color='green')
     

            #ax.plot(rotations, median_errors_1D, color='white')

            if fitting_analysis:
                ax.plot(rotations, mse, color='red')
                ax.plot(rotations, R_squared, color='green')
            
            
                ax.set_ylabel('MSE')
            else:
                
                ax.plot(rotations, average_errors_1D, color='black')
                #ax.plot(rotations, average_errors_1D + average_errors_1D_stds, color='gray')
                #ax.plot(rotations, average_errors_1D - average_errors_1D_stds, color='gray')
                
                #ax.plot(rotations, errors_circmeans, color='black')
                #ax.plot(rotations, errors_circmeans + errors_circstds, color='gray')
                #ax.plot(rotations, errors_circmeans - errors_circstds, color='gray')
                
                ax.scatter(rotations[i_rot], average_errors_1D[i_rot], color='black')
                
                # Make optimal band
                error_argmin = np.argmin(average_errors_1D)
                left_side = rotations[error_argmin] - optimal_rot - 5
                right_side = rotations[error_argmin] + (rotations[error_argmin]- left_side)
                p = matplotlib.patches.Rectangle((left_side, 0), right_side-left_side, 1, alpha=0.5, color='yellow')
                ax.add_patch(p)

                ax.set_ylabel('Mean error')
            #ax.scatter(squared_errors, ma)

           
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            if error_heatmap:
                ax.scatter(rotations[i_rot], average_errors_1D[i_rot], color='white')
            
            #ax.spines['bottom'].set_visible(False)
            

            
            #ax.set_xlabel('Head tilt $^\circ$')
            ax.set_xlim(-180, 180)
            
            #ax.set_ylim(0, 1)
            
            
            if text:
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                #if error_heatmap:
                #    ax.set_ylim(0, 1)
                #    ax.set_yticks([0, 0.2, 0.5, 0.8, 1])
                #    ax.set_yticklabels(['Matching', '0.2', '0.5', '0.8', 'Opposing'])
                # 
                ax.set_xticks([-100, 0, 100])
                ax.set_xticklabels(['-100$^\circ$', '0$^\circ$','100$^\circ$'])
            else:
                ax.set_xticks([-100, 0, 100])
                ax.set_xticklabels([])
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.set_yticklabels([])
        else:
            ax.set_axis_off()


        
        #ax.set_ylim(np.min(mean_squared_errors), np.max(mean_squared_errors))
        
        # Hide axis from all the figures that are 3D plots or plots of images
        axes = fig.get_axes()
        for ax in axes[0:12]:
            ax.set_axis_off()

       
            

        plt.subplots_adjust(left=0.08, bottom=0, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
        #plt.show()
        
        make_animation_timestep(step_size=0.025, low_val=0.7)

        if savefn:
            fig.savefig(savefn)
        plt.close() 
