import os
from drosom import angleFromFn

import csv

fields = [['filename', 'horizontal', 'vertical']]

folder = 'export'
fns = [fn for fn in os.listdir(folder) if fn.endswith('tiff')]
fns.sort()

for fn in fns:
    hor, ver = angleFromFn(fn)
    fields.append([fn, hor, ver])

with open('export/'+'angles_and_filenames.csv', 'w') as fp:
    writer = csv.writer(fp)
    for field in fields:
        writer.writerow(field)
        

