Pseudopupil movement analysis

Independent of passed options, the script first makes you to select regions of
interest (ROIs) on which it applies cross-correlation movement analysis.
The ROIs and cross-correlation results are cached on disk.

You can pass any of the following options (arguments) to the script:
	
	vectormap	Interactive 3D plot of the movement vectors, separate
			for each selected data folder
	averaged	Averaged 3D plot (like vectormap) interpolating from
			all the selected data folders
	movements	2D movement trajectories	

With averaged, one can also use
	animation			Create a video rotating the 3dplot
	complete_flow_analysis		Create a video with the simulated optic flow and
					flow / vector map difference over many specimen
					rotations

Any undocumented options can be found in drosom/terminal.py
