'''
Attribute ANALYSER_CMDS dict here contains functions that accept
the Analyser object as their only argument.
'''

import numpy as np

from gonioanalysis.drosom import plotting
from gonioanalysis.drosom.plotting.common import save_3d_animation
from gonioanalysis.drosom.plotting import basics, illustrate_experiments
from gonioanalysis.drosom.plotting import complete_flow_analysis, error_at_flight
from gonioanalysis.drosom.special.norpa_rescues import norpa_rescue_manyrepeats
from gonioanalysis.drosom.special.paired import cli_group_and_compare
import gonioanalysis.drosom.reports as reports
import gonioanalysis.drosom.export as export

I_WORKER = None
N_WORKERS = None


# Functions that take only one input argument that is the MAnalyser
ANALYSER_CMDS = {}
ANALYSER_CMDS['pass'] = print
ANALYSER_CMDS['vectormap'] = basics.plot_3d_vectormap
ANALYSER_CMDS['vectormap_video'] = lambda analyser: save_3d_animation(analyser, plot_function=basics.plot_3d_vectormap, guidance=True, i_worker=I_WORKER, N_workers=N_WORKERS) 
ANALYSER_CMDS['export_vectormap'] = export.export_vectormap
ANALYSER_CMDS['magtrace'] = basics.plot_1d_magnitude
ANALYSER_CMDS['2d_vectormap'] =  basics.plot_2d_vectormap
ANALYSER_CMDS['trajectories'] = basics.plot_xy_trajectory

# Analyser + image_folder
ANALYSER_CMDS['moving_rois_video'] = illustrate_experiments.moving_rois
ANALYSER_CMDS['illustrate_experiments_video'] = illustrate_experiments.illustrate_experiments
ANALYSER_CMDS['rotation_mosaic'] = illustrate_experiments.rotation_mosaic
ANALYSER_CMDS['norpa_rescue_manyrepeats'] = norpa_rescue_manyrepeats
ANALYSER_CMDS['compare_paired'] = cli_group_and_compare
ANALYSER_CMDS['lr_displacements'] = lambda analyser: reports.left_right_displacements(analyser, 'test')
ANALYSER_CMDS['left_right_summary'] = reports.left_right_summary
ANALYSER_CMDS['pdf_summary'] = reports.pdf_summary

rotations = np.linspace(-180,180, 360)
ANALYSER_CMDS['flow_analysis_yaw'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'yaw')
ANALYSER_CMDS['flow_analysis_roll'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'roll')
ANALYSER_CMDS['flow_analysis_pitch'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'pitch')

ANALYSER_CMDS['error_at_flight'] = error_at_flight


# Functions that take two input arguments;
# MAanalyser object and the name of the imagefolder, in this order
IMAGEFOLDER_CMDS = {}
IMAGEFOLDER_CMDS['magtrace'] = basics.plot_1d_magnitude


# Functions that take two manalyser as input arguments
DUALANALYSER_CMDS = {}
DUALANALYSER_CMDS['difference'] = basics.plot_3d_differencemap
DUALANALYSER_CMDS['compare'] = basics.compare_3d_vectormaps
DUALANALYSER_CMDS['compare_compact'] = basics.compare_3d_vectormaps_compact
DUALANALYSER_CMDS['compare_manyviews'] = basics.compare_3d_vectormaps_manyviews

DUALANALYSER_CMDS['difference_video'] = lambda analyser1, analyser2: save_3d_animation([analyser1, analyser2],
        plot_function=basics.plot_3d_differencemap, guidance=False, hide_axes=True, colorbar=False, hide_text=True,
        i_worker=I_WORKER, N_workers=N_WORKERS) 
DUALANALYSER_CMDS['export_differencemap'] = export.export_differencemap


# Manyviews videos
for animation_type in ['rotate_plot', 'rotate_arrows', 'pitch_rot', 'yaw_rot', 'roll_rot']:
    DUALANALYSER_CMDS['compare_manyviews_{}_video'.format(animation_type.replace('_',''))] = lambda an1, an2, at=animation_type: save_3d_animation([an1, an2], plot_function=basics.compare_3d_vectormaps_manyviews, animation_type=at)


# Functions that take in a list of manalysers (first positional argument)
MULTIANALYSER_CMDS = {}
MULTIANALYSER_CMDS['magnitude_probability'] = basics.plot_magnitude_probability
MULTIANALYSER_CMDS['moving_rois_mosaic'] = illustrate_experiments.moving_rois_mosaic



