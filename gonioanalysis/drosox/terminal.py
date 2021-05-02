import os
import argparse

import matplotlib.pyplot as plt

from gonioanalysis.droso import DrosoSelect, simple_select
from gonioanalysis.drosox.analysing import XAnalyser
from gonioanalysis.drosox.plotting import (
        plot_1d_overlap,
        plot_matrix_overlap,
        plot_experiment_illustration
)


def main():

    parser = argparse.ArgumentParser(description='DrosoX: Analyse DPP static imaging data')
    parser.add_argument('--datadir',
            help='Path to the specimens data directory or ASK')
    
    parser.add_argument('--specimens',
            help='Space separated list of specimens. Ased if not specified.',
            nargs='*')

   
    parser.add_argument('--measure-overlap',
            help='(Re)measure binocular overlap using user-assisted binary search',
            action='store_true')

    
    # Overlap data
    parser.add_argument('--print-overlap',
            help='Print overlap data as text',
            action='store_true')

    # Plotting arguments
    parser.add_argument('--plot-overlap',
            help='Plot 1D binocular overlap',
            action='store_true')
    parser.add_argument('--plot-matrix-overlap',
            help='Plot binocular overlap as discrete rectangulars',
            action='store_true')
    parser.add_argument('--plot-experiment-illustration',
            help='Plot illustrative video how the experiments were done',
            action='store_true')



    args = parser.parse_args()
    
    if args.datadir.lower() == 'ask':
        datadir = input("Input data directory >> ")
    else:
        datadir = args.datadir


    if not args.specimens:
        specimens = DrosoSelect(datadir=datadir).ask_user()
        specimens = [os.path.basename(specimen) for specimen in specimens]
    else:
        specimens = args.specimens
    

    xanalysers = [XAnalyser(datadir, specimen) for specimen in specimens]

    # Perform the tasks that have to be performed per xanalyser
    for xanalyser in xanalysers:
        print(xanalyser.fly)
        if args.print_overlap:
            xanalyser.print_overlap()

        if args.measure_overlap:
            xanalyser.measure_overlap()
        
        if args.plot_experiment_illustration:
            plot_experiment_illustration(xanalyser)

   
    if args.plot_overlap:        
        plot_1d_overlap(xanalysers)
    elif args.plot_matrix_overlap:
        plot_matrix_overlap(xanalysers)

    if not args.measure_overlap:
        plt.show()


if __name__ == "__main__":
    main()

