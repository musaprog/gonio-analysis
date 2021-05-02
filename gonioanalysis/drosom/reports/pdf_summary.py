
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from gonioanalysis.image_tools import open_adjusted
from gonioanalysis.directories import ANALYSES_SAVEDIR


def _make_figure(manalyser, n_rows, i_pg):
    i_pg += 1

    height_ratios = [1 for i in range(n_rows)]
    if i_pg == 0:
        height_ratios[0] *= 1.4

    fig, axes = plt.subplots(n_rows ,3, figsize=(8.27, 11.69),
            gridspec_kw={'width_ratios': [0.618, 1, 1], 'height_ratios': height_ratios})

    fig.suptitle( "{}, page {}".format(manalyser.get_specimen_name(), i_pg))
    
    return fig, axes, i_pg


def _imshow(image, ax):
    ax.imshow(image, cmap='gray')
    ax.set_axis_off()


def _image_folder_info(manalyser, image_folder, ax):
    
    pass

def _specimen_info(manalyser, ax):
    
    string = ""
    
    string += manalyser.get_specimen_sex() + '\n'
    string += manalyser.get_specimen_age() + '\n'


    ax.text()


def pdf_summary(manalysers):
    '''
    Make a structured PDF plotting all the DPP data of
    the fly and any linked data as well.
    '''
    

    pdf_savedir = os.path.join(ANALYSES_SAVEDIR, 'reports')
    os.makedirs(pdf_savedir, exist_ok=True)

    # Subplot rows per page
    n_rows = 4

    with PdfPages(os.path.join(pdf_savedir, 'pdf_summary_{}.pdf'.format(datetime.datetime.now()))) as pdf:
    
        for manalyser in manalysers:
            
            # The page index for this fly
            i_pg = -1
            fig, axes, i_pg = _make_figure(manalyser, n_rows, i_pg)

            # SNAP / FACE IMAGE
            snap_fn = manalyser.get_snap_fn()
            
            if snap_fn:
                face = open_adjusted( manalyser.get_snap_fn() )
                _imshow(face, axes[0][2])

            # Information about the fly
            
            # DPP data
            for i, image_folder in enumerate(manalyser.list_imagefolders()):
                
                if i+1 >= n_rows:
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                    fig, axes, i_pg = _make_figure(manalyser, n_rows, i_pg)
                
                i_row = i+1 - i_pg*n_rows
                
                axes[i_row][1].set_title(image_folder)

                # Photo of the location
                location_im = open_adjusted( manalyser.list_images(image_folder, absolute_path=True)[0] )
                axes[i_row][0].imshow(location_im, cmap='gray')
                _imshow(location_im, axes[i_row][0])

                displacements = manalyser.get_displacements_from_folder(image_folder)
                for dis in displacements:
                    axes[i_row][1].plot(dis, lw=1)

                axes[i_row][1].plot(np.mean(displacements, axis=0), lw=3, color='black')
            

            plt.tight_layout()
            pdf.savefig()
            plt.close()





