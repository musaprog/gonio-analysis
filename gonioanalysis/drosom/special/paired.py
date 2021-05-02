
import numpy as np
import matplotlib.pyplot as plt

from gonioanalysis.droso import simple_select


def cli_group_and_compare(manalysers):
    '''
    Command line user interface to create grouped data for
    compare_before_after() function.
    '''
    
    grouped_data = []

    print('For compare_before_after function, we have to group the data')

    for i_manalyser, manalyser in enumerate(manalysers):
        print('Specimen {}/{}'.format(i_manalyser+1, manalyser))


        image_folders = manalyser.list_imagefolders()
        image_folders.sort()
    
        for i_pair in range(0, int(len(image_folders)/2)):
            
            print('Select a BEFORE experiment')
            be = simple_select(image_folders)
            
            image_folders.remove(be)

            print('Select an AFTER experiment')
            af = simple_select(image_folders)
            
            image_folders.remove(af)

            grouped_data.append([manalyser, be, af])


    compare_before_after(grouped_data)


def compare_before_after(grouped):
    '''
    
    Grouped:   List where each element is
            [manalyser, image_folder_before, image_folder_after]

    '''

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    

        
    colors = plt.cm.get_cmap('Dark2', len(grouped))


    # if stands for image_folder
    for i, (manalyser, if_before, if_after) in enumerate(grouped):
        
        be = manalyser.get_displacements_from_folder(if_before)
        af = manalyser.get_displacements_from_folder(if_after)
        
        mbe = np.mean(be, axis=0)
        maf = np.mean(af, axis=0)
        
        
        axes[0].plot(mbe, color=colors(i))
        axes[1].plot(maf, color=colors(i))


    axes[0].set_title('Before')
    axes[1].set_title('After')
    





