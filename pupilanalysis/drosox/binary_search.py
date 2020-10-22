import os
import json
import time

import matplotlib.pyplot as plt


from pupilanalysis.directories import ANALYSES_SAVEDIR
from pupilanalysis.droso import DrosoSelect
from pupilanalysis.imageshower import ImageShower



def main(xloader):
    start_time = time.time()
    
    # Letting user to select a Droso folder
    selector = DrosoSelect()
    folder = selector.askUser()
    fly = os.path.split(folder)[1]

    data = xloader.getData(folder)

    fig, ax = plt.subplots()
    shower = ImageShower(fig, ax)

    # Try to open if any previously analysed data
    analysed_data = []
    try: 
        with open(os.path.join(ANALYSES_SAVEDIR, 'binary_search', 'results_{}.json'.format(fly)), 'r') as fp:
            analysed_data = json.load(fp)
    except:
        pass
    analysed_pitches = [item['pitch'] for item in analysed_data]
   
    print('Found {} pitches of previously analysed data'.format(len(analysed_data)))

    for i, (pitch, hor_im) in enumerate(data):
        
        # Skip if this pitch is already analysed
        if pitch in analysed_pitches:
            continue

        horizontals = [x[0] for x in hor_im]
        images = [x[1] for x in hor_im]

        N = len(images)
        shower.setImages(images)

        # Ask user to determine middle, left, and right
        i_m = binarySearchMiddle(N, shower)
        if i_m == 'skip':
            continue

        i_l = binarySearchLeft(N, shower, i_m)
        i_r = binarySearchRight(N, shower, i_m)
        
        analysed_data.append({})

        analysed_data[-1]['N_images'] = N
        analysed_data[-1]['pitch'] = pitch
        analysed_data[-1]['horizontals']= horizontals
        analysed_data[-1]['image_fns']= images
       
        analysed_data[-1]['index_middle'] = i_m
        analysed_data[-1]['index_left'] = i_l
        analysed_data[-1]['index_right']= i_r
        
        analysed_data[-1]['horizontal_middle'] = horizontals[i_m]
        analysed_data[-1]['horizontal_left'] = horizontals[i_l]
        analysed_data[-1]['horizontal_right']= horizontals[i_r]
        

        print('Done {}/{} in time {} minutes'.format(i+1, len(data), int((time.time()-start_time)/60) ))
        
        # Save on every round
        with open(os.path.join(ANALYSES_SAVEDIR, 'binary_search/', 'results_{}.json'.format(fly)), 'w') as fp:
            json.dump(analysed_data, fp)


if __name__ == "__main__":
    main()
    


