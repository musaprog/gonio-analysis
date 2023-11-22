
import matplotlib.pyplot as plt
import matplotlib.cm

from ..kinematics import sigmoidal_fit
from ..startpos_analysis import StartposAnalyser
from ..analysing import MAnalyser

def start_heatmap(spanalyser, image_folder=None, ax=None):
    '''Plots heatmap how the feature starting position affects the movement.

    spanalyser : obj
        Movement analyser
    '''
    if isinstance(spanalyser, StartposAnalyser):
        manalyser = MAnalyser(spanalyser.data_path, spanalyser.folder)
    elif isinstance(spanalyser, MAnalyser):
        manalyser = spanalyser
        spanalyser = StartposAnalyser(manalyser.data_path, manalyser.folder)
    else:
        raise ValueError(f'Spanalyser has to be a StartposAnalyser (or MAnalyser for flip)')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
        
    if image_folder is None:
        image_folder = manalyser.list_imagefolders(only_measured=True)
    elif not isinstance(image_folder, list):
        image_folder = [image_folder]

    xys = []

    for folder in image_folder:
        amplitudes, speeds, latencies = sigmoidal_fit(
                manalyser, folder, fit_to_mean=False)

        movements = spanalyser.get_movements_from_folder(folder)
       
        for eye in movements:
            X = movements[eye][0]['x']
            Y = movements[eye][0]['y']
            print(X)
            print(Y)
            print(amplitudes)
            if len(X) != len(amplitudes):
                continue
            if len(Y) != len(amplitudes):
                continue

            ax.plot(X,Y, color='gray')
            sc = ax.scatter(X, Y, c=amplitudes)
            
            if not hasattr(ax, 'startcolorbar'):
                #ax.startcolorbar.remove()
                ax.startcolorbar = plt.colorbar(sc, ax=ax)
            
            xys.append([X,Y])
            break
 
    return xys
