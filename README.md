<h1>Pseudopupil analysis suite</h1>
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


<h2>Installing with pip</h2>

The latest version from PyPi can be installed with the command

```
pip install pupil-analysis
```

This should install all the required dependencies, except when on Windows, OpenCV may require
[Visual C++ Runtime 2015](https://www.microsoft.com/download/details.aspx?id=48145) to be installed.


Then to upgrade an existing installation to the latest version

```
pip install --upgrade pupil-analysis
```


<h2>Graphical user interface (GUI)</h2>

To open the GUI from Python

```python
import pupilanalysis.tkgui as gui
gui.run()
```

or from command line
```
python -m pupilanalysis.tkgui
```
