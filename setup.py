import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Version number to __version__ variable
exec(open("gonioanalysis/version.py").read())

install_requires = [
        'numpy',
        'scipy',
        'tifffile',
        'matplotlib',
        'tk-steroids>=0.6.0',
        'roimarker>=0.2.0',
        'movemeter>=0.4.0',
        'python-biosystfiles',
        ]


setuptools.setup(
    name="gonio-analysis",
    version=__version__,
    author="Joni Kemppainen",
    author_email="jjtkemppainen1@sheffield.ac.uk",
    description="Spatial motion analysis program",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jkemppainen/gonio-analysis",
    packages=setuptools.find_packages(),
    install_requires=install_requires, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
