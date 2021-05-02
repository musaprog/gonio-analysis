'''
Goniometric Analysis
---------------------

Python pacakge `gonioanalysis` provides an intergrated set of graphical
and command line tools to analyse goniometric imaging data.
Here goniometry means that the rotation of a sample with respect
to the imaging device is well documented while rotating the sample
between the imaging runs.

More precesily, `gonioanalysis` takes advantage of the following data structuring

    data_directory
    ├── specimen_01
    │   ├── pos(horizontal, vertical)_some-suffix-stufff
    │   │   ├── image_rep0_0.tif
    │   │   └── image_rep0_1.tif
    │   └── ...
    └── ...

Here, the rotation of the sample is encoded in the image folder name
with <em>pos</em> prefix.
Alternatively, images with a generalized directory structure can be also used

    data_directory
    ├── specimen_01
    │   ├── experiment_01
    │   │   ├── image_001.tif
    │   │   └── image_002.tif
    │   └── ...
    └── ...


Please mind that `gonioanalysis` was created to analyse <em>Drosophila</em>'s
deep pseudogonio movement and orientation data across its left and right eyes
and because of this the language used in many places can be droso-centric.


User interfaces
---------------

Currently `gonioanalysis` comes with two user interfaces, one being command line based and
the other graphical.

The documentation on this page gives out only general instrctions and ideas.
To have exact usage instructions, please refer to submodule documentation of `gonioanalysis.drosom.terminal` and
`gonioanalysis.tkgui`.


Initial analysis
----------------
The following two steps have to be performed for every imaged specimen, but luckily only once.
The first one is manual needing user intercation but can be performed quite fast depending the amount of image folders,
The second one is automated but can take a long time depending on the amount of data.

### 1) Selecting regions of interest (ROIs)

In `gonioanalysis`, rectangular ROIs are manually drawn by the user
once per each image folder.
The idea is to confine the moving feature inside a rectangle but only in the first frame. 


### 2) Measuring movements

After the ROIs have been selected, movement analysis can be run.
Using 2D cross-correlation based methods (template matching), `gonioanalysis`
automatically quntifies the movement of the ROI bound method across all the frames and repeats.

Because this part requires no user interactions and takes a while,
usually it is best to select all the ROIs for all specimens beforehand,
then batch run the movement analysis.


After initial analysis
----------------------

### Vectormap - 3D directions

The 2D data of the moving features in the camera coordinates can be transferred into a 3D coordinate
system of specimens frame of reference.
To do so, we need to have the rotation of the specimen with respect to the camera specified
in some fixed coordinate system, most naturally in the coordinate system set by the rotation stages
that are used to rotate the fly.
At the moment, `gonioanalysis` supports only one rotation stage configuration that is
a specimen fixed on a vertical rotation stage that is fixed on a horizontal rotation stage that is fixed on a table.


### Orientation analysis
To analyse directionaly of any arbitrary features (such as hair pointin directions across the head),
you can override the `gonioanalysis.drosom.analysing.MAnalyser` with `gonioanalysis.drosom.orientation_analysis.OAnalyser`,
making the movement measurement part to be a manual user drawing arrows according to the feature direction analysis.
Then later, you can use any `gonioanalysis` code paths just remembering that for example in the vectormap,
the arrows point the direction of the features, not their movement.


Exporting data
--------------

All quitely saved files are stored in a location set in `gonioanalysis.directories`.
By default, on Windows this is <em>C:/Users/USER/GonioAnalysis</em>
and on other platforms <em>/home/USER/.gonioanalysis</em>


Command line interface
----------------------

`gonioanalysis` also includes a command interface, that can be invoked by

    python -m gonioanalysis.drosom.terminal


For all different options and help use `--help` option.
In case no ROIs have been selected, the 
When elections of the ROIs cannot be done in headless environments.
'''

from .version import __version__
