Dynamic data analysis - How to use

Wheter if any extra options are passed or not, the script first makes the user
to select regions of interest (ROIs) on which the cross-correlation movement
analysis is applied.

The ROIs and cross-correlation results are then being saved on disk so that
this initial step has to be only done once.

User can pass any of the following options (arguments) to the script:
	
	3dplot		Interactive 3D plot of the movement vectors, separate
			for each selected data folder
	averaged	Averaged 3D plot (like 3dplot) but interpolating from
			all the selected data folders

With averaged, one can also use
	animation			Create a video rotating the 3dplot
	complete_flow_analysis		Create a video with the simulated optic flow and
					flow / vector map difference over many specimen
					rotations

If there are any undocumented options, these can be seen in drosom/terminal.py file
in the main method of the TerminalDrosoM class. Most of the logic is just if
statements after each other.
