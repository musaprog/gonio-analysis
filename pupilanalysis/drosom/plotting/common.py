'''
Common helper functions likely needed in many different plots.
'''

import os
import math
import copy
import multiprocessing

import numpy as np
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.colors
from matplotlib.patches import FancyArrowPatch, CirclePolygon
from mpl_toolkits.mplot3d import proj3d, art3d
from matplotlib import cm
 
from pupilanalysis.coordinates import nearest_neighbour, get_rotation_matrix
from pupilanalysis.directories import ANALYSES_SAVEDIR

CURRENT_ARROW_LENGTH = 1

VECTORMAP_PULSATION_PARAMETERS = {'step_size': 0.02, 'low_val': 0.33, 'high_val': 1}



# Taken from drosoeyes.blend
RHABDOMERE_LOCATIONS = [(-1.6881, 1.0273), (-1.8046, -0.9934),
        (-1.7111, -2.9717), (-0.0025, -1.9261), (1.6690, -0.9493),
        (1.6567, 0.9762), (0.0045, -0.0113)]
RHABDOMERE_DIAMETERS = [1.8627,1.8627,1.8627,1.8627,1.8627,1.8627, 1.5743]
RHABDOMERE_R3R6_ROTATION = math.radians(-49.7)



class Arrow3D(FancyArrowPatch):
    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = (x0, x1), (y0, y1), (z0, z1)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def add_line(ax, x0, y0, z0, x1, y1, z1, camerapos=None, **kwargs):
    '''
    Add a centered 3D line plot
    '''
    if camerapos and is_behind_sphere(*camerapos, (x0,y0,z0)):
        return None

    ax.plot([x0-x1/2, x0, x0+x1/2], [y0-y1/2, y0, y0+y1/2], [z0-z1/2, z0, z0+z1/2], **kwargs)


def add_rhabdomeres(ax, x0, y0, z0, x1, y1, z1, mirror_lr=False, mirror_bf=False,
        scale=0.015, camerapos=None, **kwargs):
    '''
    Add rhabdomeres R1-R7/8 patches to a 3D plot.
    
    ax : object
        Matplotlib Axes object to add attach the patches onto
    x0,y0,z0 : float
        Coordinates of the rhabdomeres center point (R6/R7)
    x1,y1,z1 : float
        Vector pointing the direction of R6R3 line
    mirror_lr, mirror_bf : True, False (or auto for bf)
        Wheter to mirror this rhabdomere or not (left/right, back/front)
    scale : float
        Scale of the rhabdemeres
    **kwargs : dict
        To matplotlib.patches.CirclePolygon

    Returns a list of matplotlib 3d patches
    '''
    
    if camerapos and is_behind_sphere(*camerapos, (x0,y0,z0)):
        return None


    #v = np.asarray([x0,y0,z0])
    #uv = v / np.linalg.norm(v)
    
    try:
        phi = math.asin(z0)
    except:
        phi = math.pi / 2

    try:
        theta = math.atan(x0/y0)
    except:
        theta = math.pi / 2
    if y0 < 0:
        theta = theta + math.pi


    # Calculate rhabdomere rotation to match x1,y1,z1 by transforming
    # point [1,0,0] to the x0,y0,z0 point and calculating angle of
    # transformed ux and x1,y1,z1
    ux = np.array([1,0,0])
    ux = get_rotation_matrix('x', phi) @ ux
    ux = get_rotation_matrix('z', -theta) @ ux
    rot = np.arccos(np.inner(ux, [x1,y1,z1])/(np.linalg.norm(ux) * np.linalg.norm([x1,y1,z1])))
    
    if z1 < 0:
        rot = -rot

    patches = []
    
    if mirror_bf == 'auto' and z0 > 0:
        mirror_bf = True
    elif mirror_bf is not True:
        mirror_bf = False

    for diameter, location in zip(RHABDOMERE_DIAMETERS, RHABDOMERE_LOCATIONS):
        
        patch = CirclePolygon((location[0]*scale, location[1]*scale), diameter/2*scale,
                **kwargs)
        patches.append(patch)
        ax.add_patch(patch)

        art3d.patch_2d_to_3d(patch)
        
        #if mirror_lr:
        #    patch._segment3d = [get_rotation_matrix('y', math.pi) @ p for p in patch._segment3d] 

        # Rotate according to the vector (x1,y1,z1)
        #   First z rotation to set initial rotation 
        patch._segment3d = [get_rotation_matrix('z', RHABDOMERE_R3R6_ROTATION) @ p for p in patch._segment3d] 
        
        #if not mirror_lr and not mirror_bf:
        #    pass

        if mirror_lr and not mirror_bf:
            patch._segment3d = [get_rotation_matrix('x', math.pi) @ p for p in patch._segment3d] 
        
        if not mirror_lr and mirror_bf:
            patch._segment3d = [get_rotation_matrix('x', math.pi) @ p for p in patch._segment3d] 
            #patch._segment3d = [get_rotation_matrix('z', math.pi) @ p for p in patch._segment3d] 

        #if not mirror_lr and mirror_bf:
        #    patch._segment3d = [get_rotation_matrix('x', math.pi) @ p for p in patch._segment3d] 
        #    patch._segment3d = [get_rotation_matrix('z', math.pi) @ p for p in patch._segment3d] 

        #if mirror_lr and mirror_bf:
        #    patch._segment3d = [get_rotation_matrix('z', math.pi) @ p for p in patch._segment3d] 
        


        patch._segment3d = [get_rotation_matrix('z', rot) @ p for p in patch._segment3d] 
        
        patch._segment3d = [get_rotation_matrix('x', math.pi/2) @ p for p in patch._segment3d] 
        

        patch._segment3d = [get_rotation_matrix('x', phi) @ p for p in patch._segment3d]
        patch._segment3d = [get_rotation_matrix('z', -theta) @ p for p in patch._segment3d]

        # Translate
        patch._segment3d = [(x+x0,y+y0,z+z0) for x,y,z in patch._segment3d]


    return patches


def make_animation_angles(step=0.5):
    '''
    Returns the matplotlib angles to rotate a 3D plot
    
    This really shouldnt be here...
    '''

    animation = []
    sidego = 30
    # go up, to dorsal
    for i in np.arange(-30,60,step):
        animation.append((i,90))
    #rotate azim
    for i in np.arange(90,90+sidego,step*2):
        animation.append((60,i))
    # go back super down, to ventral
    for i in np.arange(0,120,step):
        animation.append((60-i,90+sidego))
    # rotate -azim
    for i in np.arange(0,2*sidego,step*2): 
        animation.append((-60,90+sidego-i))
    # go up back to dorsal
    for i in np.arange(0,120, step):
        animation.append((-60+i,90-sidego))
    return animation



CURRENT_ARROW_DIRECTION = 1
def make_animation_timestep(step_size=0.075, low_val=0.6, high_val=1, rel_return_speed=3, twoway=False):
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

    if twoway:
        s = -1
    else:
        s = 1

    if CURRENT_ARROW_DIRECTION > 0:
        if CURRENT_ARROW_LENGTH > high_val*1.5:
            CURRENT_ARROW_LENGTH = low_val*1.5
            CURRENT_ARROW_DIRECTION = s * CURRENT_ARROW_DIRECTION
    else:
        if CURRENT_ARROW_LENGTH > high_val * 1.5: 
            CURRENT_ARROW_LENGTH = low_val*1.5
            CURRENT_ARROW_DIRECTION = s * CURRENT_ARROW_DIRECTION


def plot_2d_opticflow(ax, direction):
    
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    width = abs(x1-x0)
    height = abs(y1-y0)

    if direction == 'side':
        arrows = [np.array((1.05,ik))*width for ik in np.arange(0.1,0.91,0.1)]
        for x, y in arrows:
            ax.arrow(x, y, -0.1*width, 0, width=0.01*width, color='darkviolet')
    else:
        x = []
        y = []
        for i in range(10):
            for j in range(10):
                x.append((0.5+i)*height/10)
                y.append((0.5+j)*width/10)
        ax.scatter(x,y, marker='x', color='darkviolet')


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



    




def vector_plot(ax, points, vectors, color='black', mutation_scale=6,
        i_pulsframe=False, guidance=False, camerapos=None, draw_sphere=True,
        **kwargs):
    '''
    Plot vectors on ax.

    ax              Matplotlib ax (axes) instance
    points          Starting points 
    vectors
    color           Color of the arrows
    mutation_scale  Size of the arrow head basically
    i_pulsframe    Index of the pulsation animation frame, sets the length of the arrows
    guidance    Add help elements to point left,right,front,back etc. and hide axe
    camerapos       (elev, axzim). Supply so vectors bending the visible himspehere can be hidden 
    draw_sphere     If true draw a gray sphere
    **kwargs        To matplotlib FancyArrowPatch
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

            ar = Arrow3D(*point, *(point+vector), mutation_scale=mutation_scale, lw=0.2, color='black', zorder=zorder)
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

    if i_pulsframe:
        global CURRENT_ARROW_LENGTH
        scaler = CURRENT_ARROW_LENGTH
    else:
        scaler = 1.1

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

        
        if CURRENT_ARROW_DIRECTION > 0 or i_pulsframe == False:
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


def surface_plot(ax, points, values, cb=False, phi_points=None, theta_points=None):
    '''
    3D surface plot of the error between the optic flow vectors and the actual
    eye-movements vector map.
    
    Points and values have to be in the same order.

    points
    values
    '''

    if len(points) != len(values):
        raise ValueError('For sufrace_plot, points and values have to be same lenght (and in the same order)')
     
    # Points where the error is "evaluated" (actually interpolated)
    

    N = 100
    if phi_points is None:
        phi_points = np.linspace(0, 2*np.pi, N)
    
    phi, theta = np.meshgrid(phi_points, np.linspace(0, np.pi, N))
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

        distances, i_points = kdtree.query( positions, n_jobs=-1 )
        
        for i in range(errs.size):
            if distances[i] < intp_dist:
                errs.flat[i] = values[i_points[i]]
            else:
                errs.flat[i] = 0
        return errs



    colors = color_function_optimized(theta, phi)
    
    culurs = [(0.2, 0.1, 0),(1,0.55,0),(1,1,0.4)]
    ownmap = matplotlib.colors.LinearSegmentedColormap.from_list('ownmap', culurs, 100)
    
    ax.plot_surface(X, Y, Z, facecolors=ownmap(colors), linewidth=0, vmin=0, vmax=1)

    
    m = cm.ScalarMappable(cmap=ownmap)
    m.set_array(colors)
    return m    


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



def save_3d_animation(manalyser, ax=None, plot_function=None,interframe_callback=print,
        i_worker=None, N_workers=None, animation_type='rotate_plot', video_writer=False,
        *args, **kwargs):
    '''
    interframe_callback
    
    manalyser : object or list of objects
        Either one analyser or a list of analysers

    animation_type : string
        "rotate_plot" or "rotate_arrows"
    '''
    try:
        # If manalyer is a list of manalysers
        manalyser[0]
        manalysers = manalyser
    except:
        # If it is actually only one manalyser
        manalysers = [manalyser]
    
    
    fps = 30
    frameskip = False
    
    biphasic=False
    optimal_ranges = []
    
    # FIXME No predetermined optimal ranges
    if len(manalysers) > 1:
        
        if manalysers[0].manalysers[0].__class__.__name__ == 'MAnalyser':
            optimal_ranges = [[24.3-3, 24.3+3, 'Typical photoreceptor\nmovement axis']]
        elif manalysers[0].__class__.__name__ == 'FAnalyser':
            optimal_ranges = [[-80-3, -80+3, 'Typical ommatidial\nrhabdomere aligment']]
            
            if plot_function.__class__.__name__ == 'plot_3d_vectormap':
                kwargs['arrow_rotations'] = [0, 29]

    if animation_type == 'rotate_plot':
        animation = make_animation_angles(step=0.5 * (20/fps))
    elif animation_type == 'rotate_arrows':
        
        animation = np.linspace(-90, 90, 12*fps)
        
               
    elif animation_type in ['pitch_rot', 'yaw_rot', 'roll_rot']:
        biphasic = True
        animation = np.linspace(-180, 180, 16*fps)
        
        if animation_type == 'pitch_rot':
            optimal_ranges = [[0, 20, 'Typical head tilt\nrange']]
        else:
            optimal_ranges = [[-10, 10, 'Typical head tilt\nrange']]
    
    if len(optimal_ranges) > 0:
        A = optimal_ranges[0][0]
        B = optimal_ranges[-1][1]
        
        for optimal_range in optimal_ranges:
            a,b,n = optimal_range
            start, trash, end = np.split(animation, [np.where(a<=animation)[0][0], np.where(b<animation)[0][0]] )
            
            animation = np.concatenate( (start, np.linspace(a, b, 3*fps), end) )

 


    if i_worker is None:
        partname = ''
    else:
        partname = '_' + str(i_worker)
        
        worksize = math.floor(len(animation) / N_workers)

        if i_worker == N_workers-1:
            animation = animation[i_worker*worksize:]
        else:
            animation = animation[i_worker*worksize:(i_worker+1)*worksize]
        
        # Fast forward to right animation timestep
        for i_frame_before in range(i_worker*worksize):
            make_animation_timestep(step_size=0.05*(20/fps), twoway=biphasic)

    if plot_function:
        
        kwargs['animation_variable'] = animation[0]
        kwargs['animation_type'] = animation_type
        kwargs['animation'] = animation
        kwargs['optimal_ranges'] = optimal_ranges
        kwargs['pulsation_length'] = CURRENT_ARROW_LENGTH
        
        po = plot_function(*manalysers, i_frame=0, *args, **kwargs)
        ax = po[0]
        
        try:
            len(ax)
            axes = ax
        except TypeError:
            axes = [ax]

   
    title = plot_function.__name__ + '_{}_'.format(animation_type) + '_'.join([ma.manalysers[0].__class__.__name__ for ma in manalysers])
    
    savedir = os.path.join(ANALYSES_SAVEDIR, 'videos', title)
    os.makedirs(savedir, exist_ok=True)


    if video_writer:
        doublegrab_next = False
        try:
            video_writer = matplotlib.animation.writers['ffmpeg'](fps=fps, metadata={'title': title}, codec='mjpeg')
            video_writer.setup(axes[0].figure, os.path.join(savedir,'{}{}.avi'.format(title ,partname)), dpi=200)
        except RuntimeError:
            print('Install ffmpeg by "pip install ffmpeg" to get the video')
            video_writer = False

    

    for i, animation_variable in enumerate(animation[1:]):
        
        if frameskip:
            if i % frameskip != 0:
                continue

        if i_worker is None:
            i_frame = i
        else:
            i_frame = i + i_worker * worksize

        #try:
        
        if callable(plot_function):
            
            kwargs['animation_variable'] = animation_variable
            kwargs['animation_type'] = animation_type
            kwargs['animation'] = animation
            kwargs['optimal_ranges'] = optimal_ranges
            kwargs['pulsation_length'] = CURRENT_ARROW_LENGTH

            for ax in axes:
                ax.clear()
            make_animation_timestep(step_size=0.05*(20/fps), twoway=biphasic)
            
            if len(axes) > 1:
                plot_function(*manalysers, i_frame=i_frame, axes=axes, *args, **kwargs)
            else:
                plot_function(*manalysers, i_frame=i_frame, ax=axes[0], *args, **kwargs)
        
        print('Animation variable: {}'.format(animation_variable))
        
        if animation_type == 'rotate_plot':
            for ax in axes:
                ax.view_init(elev=animation_variable[0], azim=animation_variable[1])
        
        axes[0].figure.canvas.draw_idle()
        interframe_callback()

        if video_writer:
            video_writer.grab_frame()
            if doublegrab_next:
                video_writer.grab_frame()
                doublegrab_next = False
        
        axes[0].figure.savefig(os.path.join(savedir, 'frame_{0:07d}.png'.format(i_frame)))

        #except Exception as e:
        #    print('Could not make a frame, error message on the next line')
        #    print(e)
        #    doublegrab_next = True

    if video_writer:
        video_writer.finish()

