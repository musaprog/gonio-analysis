#!/usr/bin/env python3
'''
Plotting analysed DrosoM data.
'''
import os
import math
from math import radians

import numpy as np
from scipy.spatial import cKDTree as KDTree
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

from pupilanalysis.directories import ANALYSES_SAVEDIR, CODE_ROOTDIR
from pupilanalysis.drosom.optic_flow import flow_direction
from pupilanalysis.coordinates import force_to_tplane
import pupilanalysis.coordinates as coordinates

CURRENT_ARROW_LENGTH = 1

VECTORMAP_PULSATION_PARAMETERS = {'step_size': 0.02, 'low_val': 0.33, 'high_val': 1}

class MPlotter:

 
    def __init__(self):
        
        self.plots = {}
        self.savedir = os.path.join(ANALYSES_SAVEDIR, 'mplots')
        os.makedirs(self.savedir, exist_ok=True)
    
        # Variables for figures that get plotted to the same figure between specimens
        self.magnitude_1d_figax = None


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

    
    def plot_1d_magnitude_from_folder(self, manalyser, image_folder):
        '''
        Plots 1D displacement magnitude over time.

        manalyser
        image_folde     If None, plot all image folders
        '''
        
        # Check if axes exists already
        if self.magnitude_1d_figax is None:
            fig, ax = plt.subplots()
            self.magnitude_1d_figax = (fig, ax)
        else:
            fig, ax = self.magnitude_1d_figax
        
        
        if image_folder is not None:
            movement_data = [manalyser.get_movements_from_folder(image_folder)]
        else:
            image_folders = manalyser.list_imagefolders() 
            movement_data = [manalyser.get_movements_from_folder(imf) for imf in image_folders]

        for movements_folder in movement_data:
            for eye, movements in movement_folder.items():
                for repetition in range(len(movements)):
                    mag = np.sqrt(np.array(movements[repetition]['x'])**2 + np.array(movements[repetition]['y'])**2)
                    time = np.linspace(0, 200, len(mag))
                    ax.plot(time, mag, label=str(repetition))
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Displacement (pixels)')


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
        
            ax.view_init(elev=90, azim=90)
        
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
                            vector_plot(axes[0], *vectors_3d, color=color, mutation_scale=15,
                                    animate=arrow_animation, guidance=True, camerapos=(elevation, azimuth))

                        make_animation_timestep(**VECTORMAP_PULSATION_PARAMETERS)
                    
                        
                    style = 'normal'
                    title_string = manalyser.get_short_name()
                    
                    if ';' in title_string:
                        title_string, style = title_string.split(';')

                    if title_string is '':
                        # Use full name if short name is not set
                        title_string = manalyser.get_specimen_name()

                    #axes[0].text2D(0.5, 0.85, title_string, transform=ax.transAxes,
                    #        ha='center', va='center', fontsize=38, fontstyle=style)
                    #axes[0].text2D(0.75, 0.225, "n={}".format(manalyser.get_N_specimens()),
                    #        transform=ax.transAxes, ha='center', va='center', fontsize=30)
                                    
                    print('{} {}'.format(elevation, azimuth)) 
                    for ax in axes:
                        ax.view_init(elev=elevation, azim=azimuth)
                    
                    #ax.dist = 6
                    
                    fig.canvas.draw_idle()
                    # for debugging here plt.show()
                    fn = 'image_{:0>8}.png'.format(i)
                    #fig.savefig(os.path.join(savedir, fn), bbox_inches=Bbox.from_bounds(2.75,3,10,10))
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


CURRENT_ARROW_DIRECTION = 1
def make_animation_timestep(step_size=0.1, low_val=0.5, high_val=1.2, rel_return_speed=5):
    '''
    step_size between 0 and 1.
    Once total displacement 1 has been reached go back to low_value
    '''
    # Update arrow length
    global CURRENT_ARROW_LENGTH
    global CURRENT_ARROW_DIRECTION

    step = step_size * 1.5
    if CURRENT_ARROW_DIRECTION < 0:
        step *= rel_return_speed

    CURRENT_ARROW_LENGTH += step

    if CURRENT_ARROW_DIRECTION > 0:
        if CURRENT_ARROW_LENGTH > high_val*1.5:
            CURRENT_ARROW_LENGTH = low_val*1.5
            CURRENT_ARROW_DIRECTION = -1 * CURRENT_ARROW_DIRECTION
    else:
        if CURRENT_ARROW_LENGTH > high_val * 1.5: 
            CURRENT_ARROW_LENGTH = low_val*1.5
            CURRENT_ARROW_DIRECTION = -1 * CURRENT_ARROW_DIRECTION
       

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
        animate=False, guidance=False, camerapos=None, draw_sphere=True):
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
    draw_sphere     If true draw a gray sphere
    '''
    
    global CURRENT_ARROW_DIRECTION
    
    arrow_artists = []

    #ax.scatter([x for (x,y,z) in points], [y for (x,y,z) in points],
    #        [z for (x,y,z) in points], color='black')

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
        
        if draw_sphere:
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

        
        if CURRENT_ARROW_DIRECTION > 0 or animate == False:
            A = point
            B = point+scaler*vector
        else:
            A = point
            B = point-scaler*vector
        
        ar = Arrow3D(*A, *B, arrowstyle="-|>", lw=1,
                mutation_scale=mutation_scale, color=color, alpha=alpha, zorder=10)
        ax.add_artist(ar)
        arrow_artists.append(ar)

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
    
    return arrow_artists

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
   
    kdtree = KDTree(points)

    def color_function_optimized(theta, phi):
        
        intp_dist = (2 * np.sin(np.radians(5)))

        x = (np.sin(theta) * np.cos(phi))
        y = (np.sin(theta) * np.sin(phi))
        z = np.cos(theta)
        
        errs = np.empty_like(x)
        positions = [[x.flat[i], y.flat[i], z.flat[i]] for i in range(x.size)]

        distances, i_points = kdtree.query( positions, n_jobs=2 )
        
        for i in range(errs.size):
            if distances[i] < intp_dist:
                errs.flat[i] = values[i_points[i]]
            else:
                errs.flat[i] = 0
        return errs



    colors = color_function_optimized(theta, phi)
    

    
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
    from .optic_flow import flow_direction, flow_vectors, field_error
    from .fitting import get_reference_vector
    from pupil.coordinates import rotate_vectors
    from pupil.drosom.optic_flow import flow_vectors
    from pupil.optimal_sampling import optimal as optimal_sampling

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
    average_errors_1D = np.mean(all_errors, axis=1)
    average_errors_1D_stds = np.std(all_errors, axis=1)
    
    
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
