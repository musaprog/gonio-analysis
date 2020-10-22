'''
Create a summary of the files
'''
import csv
import numpy as np

from pupilanalysis.drosom.kinematics import mean_max_response

def left_right_summary(manalysers):
    
    csvfile = []
    
    header = ['Specimen name', 'Left mean response (pixels)', 'Right mean response (pixels)', 'Right eye ERG response (mV)', 'Eyes with microsaccades', '"Normal" ERGs']
    
    csvfile.append(header)

    for manalyser in manalysers:
        
        print(manalyser.get_specimen_name())

        line = []

        line.append(manalyser.get_specimen_name())
        
        # Left eye

        responses = []
        
        for image_folder in manalyser.list_imagefolders(horizontal_condition=lambda h: h>=0): 
            if image_folder.endswith("green"):
                continue
            
            print(image_folder)
            mean = mean_max_response(manalyser, image_folder)
            if not np.isnan(mean):
                responses.append(mean)
        
        line.append(np.mean(responses))
        
        # Right eye
        responses = []
        
        for image_folder in manalyser.list_imagefolders(horizontal_condition=lambda h: h<0): 
            if image_folder.endswith('green'):
                continue
            mean = mean_max_response(manalyser, image_folder)
            if not np.isnan(mean):
                responses.append(mean)
        
        line.append(np.mean(responses))
        
   
        uvresp = None
        greenresp = None
        
        print(manalyser.linked_data.keys())

        if "ERGs" in manalyser.linked_data:
            data = manalyser.linked_data['ERGs']
            
            print(data[0].keys())

            for recording in data:
                print(recording.keys())
                if int(recording['N_repeats']) == 25:

                    if recording['Stimulus'] == 'uv':
                        uvresp = np.array(recording['data'])
                    if recording['Stimulus'] == 'green':
                        greenresp = np.array(recording['data'])

        

        uvresp = uvresp - uvresp[0]
        uvresp = np.mean(uvresp[500:1000])
        
        greenresp = greenresp - greenresp[0]
        greenresp = np.mean(greenresp[500:1000])
        
        line.append(uvresp)
        
        # Descide groups
        dpp_threshold = 0.9999
        if line[1] < dpp_threshold and line[2] < dpp_threshold:
            line.append('None')
        elif line[1] < dpp_threshold or line[2] < dpp_threshold:
            line.append('One')
        else:
            line.append('Two')

        erg_threshold = 0.5
        if abs(line[3]) < erg_threshold:
            line.append('No')
        else:
            line.append('Yes')


        csvfile.append(line)


    for line in csvfile:
        print(line)
    
    with open("left_right_summary.csv", 'w') as fp:
        writer = csv.writer(fp)
        for line in csvfile:
            writer.writerow(line)


    


