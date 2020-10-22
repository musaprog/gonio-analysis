import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
        'numpy',
        'scipy',
        'tifffile',
        'matplotlib',
        'tk_steroids-jkemppainen @ https://github.com/jkemppainen/tk_steroids/archive/master.zip',
        'roimarker-jkemppainen @ https://github.com/jkemppainen/roimarker/archive/master.zip',
        'movemeter-jkemppainen @ https://github.com/jkemppainen/movemeter/archive/master.zip',
        'biosystfiles-jkemppainen @ https://github.com/jkemppainen/python-biosystfiles/archive/master.zip',
        ]


setuptools.setup(
    name="pupilanalysis-jkemppainen", # Replace with your own username
    version="0.0.1",
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
