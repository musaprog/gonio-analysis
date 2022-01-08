import os
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits.axes_grid1

from gonioanalysis.directories import ANALYSES_SAVEDIR
from gonioanalysis.drosom.optic_flow import flow_direction


from .common import make_animation_angles, vector_plot


class MPlotter:

 
    def __init__(self):
        
        self.savedir = os.path.join(ANALYSES_SAVEDIR, 'mplots')
        os.makedirs(self.savedir, exist_ok=True)
    
        # Variables for figures that get plotted to the same figure between specimens
        self.magnitude_1d_figax = None



    def plot_2d_trajectories(self, manalyser):
        '''
        Plot 2D movement trajectories of the ROIs, separetly for each imaged position.
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
        
        if animation:
            animation = make_animation_angles()

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
 
