<h1>Goniometric Analysis suite</h1>
Specialised spatial motion analysis software for Gonio Imsoft data.

In general, can be used for data that follows hierarchy
```
data_directory
├── specimen_01
│   ├── experiment_01
│   │   ├── image_001.tif
│   │   └── image_002.tif
│   └── ...
└── ...
```

*OBS!* This is still an early development version (expect rough edges).



<h2>Installing</h2>

Currently, two installation methods are supported.

- The stand-alone installer (Windows only)
- Python packaging (all platforms)


<h3>Installer on Windows (easiest)</h3>

A Windows installer bundles together the Gonio Analysis suite and all its depencies,
including a complete Python runtime. It is ideal for users not having Python installed before.

The installer can be found in
[Releases](https://github.com/jkemppainen/gonio-analysis/releases).

The installer creates a start menu shorcut called <em>Gonio Analysis</em>,
which can be used to launch the program.

To uninstall the program, use the Windows <em>Add or Remove programs</em>.


<h3>Using pip (the python standard way)</h3>

The latest version from [PyPi](https://pypi.org/) can be installed with the command

```
pip install gonio-analysis
```

This should install all the required dependencies, except when on Windows, OpenCV may require
[Visual C++ Runtime 2015](https://www.microsoft.com/download/details.aspx?id=48145) to be installed.


Launching the program
```
python -m gonioanalysis.tkgui
```


<h4>Managing versions with pip</h4>

Upgrade to the latest version

```
pip install --upgrade gonio-analysis
```

Downgrade to a selected version

```
pip install gonio-analysis==0.1.2
```


<h2>How to use</h2>

First, open a data directory (containing the folders containing the images).
Next, select the regions of interest (ROIs), and then, run the motion analysis.
The ROIs and movements are saved on disk (<em>C:\Users\USER\GonioAnalysis</em> or <em>/home/USER/.gonioanalysis</em>),
so these steps are needed only once per specimen.

After the initial steps you, can perform further analyses in the program or
export the data by
1) copy-pasting the results to an external program
or 2) exporting CSV files.


<h2>Contributing</h2>

- Any problems or missing features,
[Issues](https://github.com/jkemppainen/gonio-analysis/issues).

- For general chatting,
[Discussions](https://github.com/jkemppainen/gonio-analysis/discussions)



<h2>About the project</h2>

This program was created in the University of Sheffield
to analyse the photomechanical
photoreceptor microsaccades that occur in the insect compound eyes.
For more information, please see
our [GHS-DPP methods @ Communications Biology](https://www.nature.com/articles/s42003-022-03142-0),
or visit
[the lab's website](https://cognition.group.shef.ac.uk/).

Currently, it is (tardily) spare-time-developed by
[jkemppainen](https://github.com/jkemppainen).

