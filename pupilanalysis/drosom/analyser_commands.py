'''
Attribute ANALYSER_CMDS dict here contains functions that accept
the Analyser object as their only argument.
'''

import numpy as np

from pupilanalysis.drosom import plotting
from pupilanalysis.drosom.plotting.plotter import MPlotter
from pupilanalysis.drosom.plotting import complete_flow_analysis, error_at_flight
from pupilanalysis.drosom.special.norpa_rescues import norpa_rescue_manyrepeats
from pupilanalysis.drosom.special.paired import cli_group_and_compare
import pupilanalysis.drosom.reports as reports


plotter = MPlotter()


# Functions that take only one input argument that is the MAnalyser
ANALYSER_CMDS = {}
ANALYSER_CMDS['vectormap'] = plotter.plot_3d_vectormap
ANALYSER_CMDS['vectormap_mayavi'] = plotter.plot_3d_vectormap_mayavi
ANALYSER_CMDS['vectormap_video'] = lambda analyser: plotter.plot_3d_vectormap(analyser, animation=True)
ANALYSER_CMDS['magtrace'] = plotter.plotTimeCourses
ANALYSER_CMDS['2d_vectormap'] =  plotter.plotDirection2D
ANALYSER_CMDS['trajectories'] = plotter.plot_2d_trajectories
ANALYSER_CMDS['2dmagnitude'] = plotter.plotMagnitude2D

# Analyser + image_folder
#ANALYSER_CMDS['1dmagnitude'] = plotter.plot_1d_magnitude_from_folder

ANALYSER_CMDS['illustrate_experiments_video'] = plotting.illustrate_experiments
ANALYSER_CMDS['norpa_rescue_manyrepeats'] = norpa_rescue_manyrepeats
ANALYSER_CMDS['compare_paired'] = cli_group_and_compare
ANALYSER_CMDS['left_right_summary'] = reports.left_right_summary
ANALYSER_CMDS['pdf_summary'] = reports.pdf_summary

rotations = np.linspace(-180,180, 360)
ANALYSER_CMDS['flow_analysis_yaw'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'yaw')
ANALYSER_CMDS['flow_analysis_roll'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'roll')
ANALYSER_CMDS['flow_analysis_pitch'] = lambda analyser: complete_flow_analysis(analyser, rotations, 'pitch')

ANALYSER_CMDS['error_at_flight'] = error_at_flight

ANALYSER_CMDS['export_vectormap'] = lambda analyser: analyser.export_3d_vectors()

