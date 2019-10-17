

'''
Accidentally combined binary_search analysis results from
DrosoX5 and DrosoX6, try to fix this.
'''

import json

with open('binary_search_results/resultsDrosoX6.json', 'r') as fp:
    analysed_data = json.load(fp)

with open('binary_search_results/results.json', 'r') as fp:
    wrong_data = json.load(fp)


corrected = []

for item in analysed_data:
    
    if not item in wrong_data:
        corrected.append(item)
    else:
            print('found duplicate')
with open('binary_search_results/resultsDrosoX6cor.json', 'w') as fp:
    json.dump(corrected, fp)

