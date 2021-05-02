'''
binary_search.py
Determining binocular overlap by half-interval search.

DESCRIPTION


TODO
- examination mode, fine tune changes
- combine binarySearchs to a single function for easier mainantance
- remove reverse option from binarySearch

'''
import os
import json
import time
from math import floor

import matplotlib.pyplot as plt

from gonioanalysis.directories import ANALYSES_SAVEDIR
from gonioanalysis.droso import DrosoSelect
from gonioanalysis.image_tools import ImageShower


def inputRead():
    while True:
        try:
            inpt = input('>> ')
            if inpt == 's':
                return 'skip'
            direction = int(inpt)
            break
        except ValueError:
            pass
    return direction

def calcM(L, R):
    '''
    Calculate m used in the binary search.
    '''
    return int(floor(abs(L+R)/2))


def binary_search_middle(N_images, shower, reverse=False):
    '''
    Search for the border where pseudopupils are visible simultaneously.
    Midpoint search.

    shower UI class

    '''
    DIR = [1,2]
    if reverse:
        DIR = [2,1]


    right_end = N_images-1

    R = right_end
    L = 0
    
    #print('Find midpoint. "1" to go left or "2" to go right')
    print('Binary search')
    print('  Type in 1 or 2 to rotate\n  -1 to return to the beginning\n  0 to instaselect')
    
    shower.setTitle('Midpoint')

    while L<R:
        
        m = calcM(L, R)
        shower.setImage(m)
        shower.cacheImage(calcM(L, m-1))
        shower.cacheImage(calcM(m+1, R))


        direction = inputRead()
        
        if direction == 'skip':
            return direction

        if direction == DIR[0]:
            R = m-1
        elif direction == DIR[1]:
            L = m+1
        elif direction == 0:
            break

        elif direction == -1:
            R = right_end
            L = 0
 
    return int((R+L)/2)


def binary_search_left(N_images, shower, midpoint, reverse=False):
    '''
    Search for the border where pseudopupils are visible simultaneously.
    Left side search.

    shower UI class

    '''
    DIR = [1,2]
    if reverse:
        DIR = [2,1]


    R = midpoint
    L = 0

    print('How many pupils (1 or 2)?')
    shower.setTitle('Left side')


    while L<R:
        m = calcM(L, R)
        shower.setImage(m)
        shower.cacheImage(calcM(L, m-1))
        shower.cacheImage(calcM(m+1, R))

       
        N_pupils = inputRead()

        if N_pupils == DIR[1]:
            R = m-1
        elif N_pupils == DIR[0]:
            L = m+1
            
        elif N_pupils == -1:
            R = midpoint
            L = 0

    
    return int((R+L)/2)


def binary_search_right(N_images, shower, midpoint, reverse=False):
    '''
    Search for the border where pseudopupils are visible simultaneously.
    Right side search.

    shower UI class

    '''
    DIR = [1,2]
    if reverse:
        DIR = [2,1]

    right_end = N_images-1

    R = right_end
    L = midpoint
    
    print('How many pupils (1 or 2)?')
    shower.setTitle('Right side')


    while L<R:
        m = calcM(L, R)
        shower.setImage(m)
        shower.cacheImage(calcM(L, m-1))
        shower.cacheImage(calcM(m+1, R))

        N_pupils = inputRead()

        if N_pupils == DIR[1]:
            L = m+1

        elif N_pupils == DIR[0]:
            R = m-1
            
        elif N_pupils == -1:
            R = right_end
            L = midpoint
 
    return int((R+L)/2)



def main():
    start_time = time.time()
    

    # Letting user to select a Droso folder
    selector = DrosoSelect()
    folder = selector.askUser()
    fly = os.path.split(folder)[1]


    xloader = XLoader()
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
    


