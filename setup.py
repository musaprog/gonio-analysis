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
        'tk-steroids>=0.7.1',
        'roimarker>=0.2.1',
        'movemeter>=0.6.0',
        'python-biosystfiles',
        ]


setuptools.setup(
    name="gonio-analysis",
    version=__version__,
    author="Joni Kemppainen",
    author_email="joni.kemppainen@windowslive.com",
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
    # Used language features that require Python 3.6 or newer
    #   - fstrings
    #
    #
    # Python 3.8 final version supporting Windows 7
    # Python 3.4 final version supporting Windows XP
    python_requires='>=3.6',
)
