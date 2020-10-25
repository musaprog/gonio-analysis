<h1>Pseudopupil analysis suite</h1>
Specialized spatial motion analysis software for PupilImsoft data.


<h2>Installing</h2>

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
import pupilanalysis.drosom.gui as gui
gui.run()
```
