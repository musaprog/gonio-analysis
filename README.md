<h1>Pseudopupil Analysis suite</h1>
Specialised spatial motion analysis software for Pupil Imsoft data.

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


<h2>Installing</h2>

There are two supported installation ways at the moment.
On Windows, the stand-alone installer is possibly the best option unless you feel familiar with Python.
On other platforms, use pip.

<h3>Installer on Windows (easiest)</h3>

A Windows installer that bundles together the Pseudopupil Analysis suite and all its depencies,
including a complete Python runtime, is provided at
[Releases](https://github.com/jkemppainen/pupil-analysis/releases).

The installer creates a start menu shorcut called <em>Pupil Analysis</em>,
which can be used to launch the program.

To uninstall, use <em>Add or Remove programs</em> feature in Windows.


<h3>Using pip (the python standard way)</h3>

The latest version from [PyPi](https://pypi.org/) can be installed with the command

```
pip install pupil-analysis
```

This should install all the required dependencies, except when on Windows, OpenCV may require
[Visual C++ Runtime 2015](https://www.microsoft.com/download/details.aspx?id=48145) to be installed.


Afterwards, to upgrade an existing installation to the latest version

```
pip install --upgrade pupil-analysis
```

In case of regressions, a specific version of the suite (for example 0.1.2) can be installed

```
pip install pupil-analysis==0.1.2
```

Finally, to open the program

```
python -m pupilanalysis.tkgui
```

<h2>How to use</h2>

First, open a data directory (containing the folders containing the images).
Next, select the regions of interest (ROIs) and then run the motion analysis.
The ROIs and movements are saved on disk (<em>C:\Users\USER\PupilAnalysis</em> or <em>/home/USER/.pupilanalysis</em>), so these steps are needed only once per specimen.

After the initial steps you, can perform further analyses in the program or
export the data by
1) copy-pasting to your favourite spread sheet or plotting program
or 2) exporting CSV files.

<h3>Notes</h3>
This is still an early development version (expect rough edges).

