import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(
     name='Pupil',  
     version='git',
     scripts=['run_on_windows, adm'] ,
     author="Joni Kemppainen",
     author_email="jjtkemppainen1@sheffield.ac.uk",
     description="Scripts to analyse pupil_imsoft data",
     long_description=long_description,
     packages=setuptools.find_packages(),
 )
