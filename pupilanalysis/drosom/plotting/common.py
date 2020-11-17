'''
Common helper functions likely needed in many different plots.
'''

import numpy as np
from scipy.spatial import cKDTree as KDTree
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


CURRENT_ARROW_LENGTH = 1

VECTORMAP_PULSATION_PARAMETERS = {'step_size': 0.02, 'low_val': 0.33, 'high_val': 1}



class Arrow3D(FancyArrowPatch):
    def __init__(self, x0, y0, z0, x1, y1, z1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = (x0, x1), (y0, y1), (z0, z1)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)



def make_animation_angles():
    '''
    Returns the matplotlib angles to rotate a 3D plot
    
    This really shouldnt be here...
    '''

    animation = []
    step = 0.5 # old 0.5
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


