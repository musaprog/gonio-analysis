'''
A short script that takes creates a pulsating arrow in sync with
the vectormap movie.
'''

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import writers as mpl_writers

from pupil.drosom.terminal import make_animation_angles
import pupil.drosom.plotting
from pupil.drosom.plotting import VECTORMAP_PULSATION_PARAMETERS, make_animation_timestep


def get_arrow_lengths():
    '''
    Returns a list of arrow lenght scalers at each time point (movie frame).
    '''
    angles = make_animation_angles()

    arrow_lengths = []

    for angle in angles:
        make_animation_timestep(**VECTORMAP_PULSATION_PARAMETERS)
        arrow_lengths.append(pupil.drosom.plotting.CURRENT_ARROW_LENGTH)
    
    return arrow_lengths

def encode_arrow_video():
    plt.rcParams['axes.facecolor']='magenta'
    plt.rcParams['savefig.facecolor']='magenta'

    fig = plt.figure()
    ax = plt.gca()
    ax.axis('off')
    ax.set_xlim(0,1)
    ax.set_ylim(-0.2, 0.2)

    video_writer = mpl_writers['ffmpeg'](fps=20)
    video_writer.setup(fig, 'arrow_video.mp4')

    arrow = FancyArrowPatch((0,0), (0,0), arrowstyle='simple', mutation_scale=100, color='white')
    ax.add_patch(arrow)

    for arrl in get_arrow_lengths():
        arrow.set_positions((0,0), (1*arrl,0))
        
        video_writer.grab_frame()

    video_writer.finish()

if __name__ == "__main__":
    encode_arrow_video()

