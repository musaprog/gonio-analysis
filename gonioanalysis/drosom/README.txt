Goniometric movement analysis

On the first run, the script makes you to select regions of interest (ROIs),
on which it applies cross-correlation movement analysis. The ROIs and
cross-correlation results are cached on disk.

You can pass any of the following options (arguments) to the script:
	
	vectormap	Interactive 3D plot of the pseudopupil movement directions
	averaged	Vectormap but averaging over the selected specimen
	trajectories	2D movement trajectories

With averaged, one can also use
	animation			Create a video rotating the 3dplot
	complete_flow_analysis		Create a video with the simulated optic flow and
					flow / vector map difference over many specimen
					rotations

Use space to separate the arguments.
Any undocumented options can be found in drosom/terminal.py
