import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Version number to __version__ variable
exec(open("pupilanalysis/version.py").read())

install_requires = [
        # As a workaround for a recent Windows numpy bug, do not use 1.19.4 or later for now 
        #https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html?page=2&pageSize=10&sort=votes&type=problem
        'numpy<=1.19.3',
        'scipy',
        'tifffile',
        'matplotlib',
        'tk-steroids>=0.3.0',
        'roimarker>=0.1.1',
        'movemeter>=0.2.0',
        'python-biosystfiles',
        ]


setuptools.setup(
    name="pupil-analysis",
    version=__version__,
    author="Joni Kemppainen",
    author_email="jjtkemppainen1@sheffield.ac.uk",
    description="Spatial motion analysis program",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkemppainen/pupil-analysis",
    packages=setuptools.find_packages(),
    install_requires=install_requires, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
