'''
Just fast pull images from a one imaging to a single folder
'''
import os
import shutil

newfolder = 'export'

print('Type path')
folder = input()

subfolders = [os.path.join(folder, fn) for fn in os.listdir(folder) if os.path.isdir(os.path.join(folder, fn))]

for subfolder in subfolders:
    images = [os.path.join(subfolder, fn) for fn in os.listdir(subfolder)]
    for image in images:
        newfn = os.path.join(newfolder, os.path.basename(image))
        shutil.copyfile(image, newfn)

