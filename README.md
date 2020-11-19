<h1>Pseudopupil Analysis suite</h1>
Specialised spatial motion analysis software for Pupil Imsoft data.

Can be used for data that follows hierarchy
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

<h3>Installer on Windows (easiest)</h3>

Windows installers that bundle together Pseudopupil Analysis and all its depencies,
including a complete Python runtime, are provided at
[Releases](https://github.com/jkemppainen/pupil-analysis/releases).

The installer creates a start menu shorcut.

Use <em>Add or Remove programs</em> to uninstall.

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

To open the GUI from Python

```python
import pupilanalysis.tkgui as gui
gui.run()
```

or from command line
```
python -m pupilanalysis.tkgui
```
