<h1>GonioAnalysis - A goniometric analysis program</h1>
GonioAnalysis is a specialised spatial motion analysis software.
It is targeted mainly to analyse data produced by GonioImsoft.

In general, GonioAnalysis can be used for data following the hierarchy
```
data_directory
├── specimen_01
│   ├── experiment_01
│   │   ├── image_001.tif
│   │   └── image_002.tif
│   └── ...
└── ...
```

Tiff files or stacks are the preferred image format.


<h2>Installing</h2>

Installing with the pip command is recommended.

<h3>Option A: Using the pip-command</h3>

The latest version from [PyPi](https://pypi.org/) can be installed with the command

```
pip install gonio-analysis
```

This should install all the required dependencies. On Windows, OpenCV may require
[Visual C++ Runtime 2015](https://www.microsoft.com/download/details.aspx?id=48145) to be installed.

Some Linux distributions have separated tkinter
(Python's interface to the Tcl/Tk GUI toolkit) and pillow's
ImageTk module their own packages that are not installed by default.
Please make sure to have them installed from your package manager.

```
# Example on Debian/Ubuntu/Mint/...
sudo apt install python3-tk python3-pil.imagetk
```


Launch the program by
```
python -m gonioanalysis.tkgui
```

<h4>Managing versions using pip</h4>

Upgrade to the latest version

```
pip install --upgrade gonio-analysis
```

Downgrade to a selected version

```
pip install gonio-analysis==0.1.2
```


<h3>Option B: Installer on Windows</h3>

*There are currently some technical issues providing the installer
for recent Gonio Analysis versions. Please consider using
the pip command instead, as instructed above.*

A Windows installer bundles together the Gonio Analysis suite and all its depencies, including a complete Python runtime. It is ideal for users not having Python installed.

The installer can be found at
[Releases](https://github.com/jkemppainen/gonio-analysis/releases).

The installer creates a start menu shorcut called <em>Gonio Analysis</em>,
which can be used to launch the program.

To uninstall the program, use the Windows <em>Add or Remove programs</em>.



<h2>Usage instructions</h2>

Start by opening your data directory (this should contain the folders containing the images).
Next, select the regions of interest (ROIs), and then, run the motion analysis.
This may take a while.
The ROIs and movements are saved on disk (<em>C:\Users\USER\GonioAnalysis</em> or <em>/home/USER/.gonioanalysis</em>),
meaning that this part of the analysis (ROIs and movement analysis) has to be performed
only once per specimen (unless you want to re-analyse).

After the initial steps you, can perform further analyses in the program or
export the data by
1) copy-pasting the results to an external program
or 2) exporting CSV files.

Many other features are present but yet undocumented.


<h2>Contributing</h2>

- Any problems or missing features,
[Issues](https://github.com/jkemppainen/gonio-analysis/issues).

- For general chatting,
[Discussions](https://github.com/jkemppainen/gonio-analysis/discussions)

See also below for the project's future plans.


<h2>About the project</h2>

This program was created in the University of Sheffield
to analyse the photomechanical
photoreceptor microsaccades that occur in the insect compound eyes.
For more information, please see
our [GHS-DPP methods @ Communications Biology](https://www.nature.com/articles/s42003-022-03142-0),
or visit
[the lab's website](https://cognition.group.shef.ac.uk/).
