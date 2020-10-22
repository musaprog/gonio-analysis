<h1>Pseudopupil analysis suite</h1>
Specialized spatial motion analysis software for PupilImsoft data.


<h2>Installing</h2>

The latest version can be installed with the command

```
pip install https://www.github.com/jkemppainen/pseudopupil-analysis/archive/master.zip
```

This should install all the required dependencies, except when on Windows, OpenCV may require
[Visual C++ Runtime](https://www.microsoft.com/en-us/download/details.aspx?id=48145) to be installed.


<h3>Upgrading<h3>

To upgrade an existing installation to the latest

```
pip install --force-reinstall https://www.github.com/jkemppainen/pseudopupil-analysis/archive/master.zip
```


<h2>Graphical user interface (GUI)</h2>

To open the GUI from Python, type the following

```
import pupilanalysis.drosom.gui as gui
gui.run()
```
