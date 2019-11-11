'''
drosox.py

Loading DrosoX data.

DrosoX data is about static mapping of pseudopupils in order to work
out binocular overlap between the eyes.

This module standardizes loading the data. 

'''
import os
import json
import warnings

import matplotlib.pyplot as plt

from numpy import sign, mean
import numpy as np
import scipy.interpolate

from pupil.directories import PROCESSING_TEMPDIR, ANALYSES_SAVEDIR
from pupil_imsoft.anglepairs import loadAnglePairs, toDegrees


class XLoader:
    '''
    Loading DrosoX data.

    TODO:
    - in getData, optimize speed. No image loading is done here yet,
        just rotating some python lists in an inefficient way
        (there's about 20 000 images for each droso so this has to be done properly)
    '''
    def __init__(self):
        pass
    

    def getData(self, folder, arl_fly=False):
        '''
        Loading a data folder.

        Returns a list where horizontal angles are grouped by the pitch angle
        and each horizontal angle is next to it's image's filename.

        grouped = [pit1, [[hor1, fn1], ...] ...]
        

        INPUT ARGUMENTS     DESCRIPTION
        arl_fly             Set true if normal DrosoX processing should be skipped
                                (meaning no outliner remove, pitch grouping etc...)

        '''
        fns = [os.path.join(folder,'rot',fn) for fn in os.listdir(os.path.join(folder, 'rot')) if fn.endswith('.tif')]
        fns.sort()
        
        angles = loadAnglePairs(os.path.join(folder, 'anglepairs.txt'))
        toDegrees(angles)
        
        print('Angles {} and images {}'.format(len(angles), len(fns)))
        if abs(len(angles) - len(fns)) > 10:
            warnings.warn("Large missmatch between the number of recorded the angles and images.", UserWarning)
        
        fns, angles = self.makeSameLength(fns, angles)
       
        angles_and_images = self.pitchGroupedHorizontalsAndImages(fns, angles, arl_fly=arl_fly)
        if not arl_fly:
            
            print('Determing pitches to be combined...')
            angles_to_combine = self.pitchesToBeCombined(angles_and_images, angles)
            
            #self.plotAnglesAndImages(angles_and_images)
            

            # GROUP NEAR PITCHES
            print('Combining pitches...')
            angles_and_images = self.groupPitchesNew(angles_and_images, angles_to_combine) 
            print('After grouping: {}'.format(len(angles_and_images)))
            #self.plotAnglesAndImages(angles_and_images)

            # NO PROBLEM AFTER THIS
            # -------------------------
            print('Removeing lonely outliners...')
            angles_and_images = self.removeOutliners(angles_and_images, 2)

            #angles_and_images = self.removeShorts(angles_and_images)
            
            # SORT PITCHES
            angles_and_images.sort(key=lambda x: x[0], reverse=True)
            
           
            # SORT HORIZONTALS
            for i in range(len(angles_and_images)):
                angles_and_images[i][1].sort(key=lambda x: x[0])
            
            #self.plotAnglesAndImages(angles_and_images)

        return angles_and_images
    
    def plotAnglesAndImages(self, angles_and_images):

        import matplotlib.pyplot as plt

        X = []
        Y = []

        for pitch, hor_im in angles_and_images:
            X.append([])
            Y.append([])
            for horizontal, fn in hor_im:
                X[-1].append(horizontal)
                Y[-1].append(pitch)
       
        for x, y in zip(X, Y):
            plt.scatter(x,y)

        plt.show()

    def removeOutliners(self, angles_and_images, degrees_threshold):
        '''

        '''

        for pitch, hor_im in angles_and_images:
            remove_indices = []

            for i in range(len(hor_im)):
                center = hor_im[i][0]
                try:
                    previous = hor_im[i-1][0]
                except IndexError:
                    previous = None
                try:
                    forward = hor_im[i+1][0]
                except IndexError:
                    forward = None

                if not (previous == None and forward == None):

                    if forward == None:
                        if abs(previous-center) > degrees_threshold:
                            remove_indices.append(i)
                    if previous == None:
                        if abs(forward-center) > degrees_threshold:
                            remove_indices.append(i)

                #if previous != None and forward != None:
                #    if abs(previous-center) > degrees_threshold and abs(forward-center) > degrees_threshold:
                #        remove_indices.append(i)
        
            for i in sorted(remove_indices, reverse=True):
                hor_im.pop(i)

        return angles_and_images

    @staticmethod
    def _getPitchIndex(pitch, angles_and_images):

        for i in range(len(angles_and_images)):
            if angles_and_images[i][0] == pitch:
                return i
        print('Warning: No pitch {} in angles_and_images'.format(pitch))
        return None

    def groupPitchesNew(self, angles_and_images, to_combine):
        '''
        Rotatory encoders have some uncertainty so that the pitch can "flip"
        to the next value if encoder's position in
        '''
        grouped = []

        for pitches in to_combine:
            
            combinated = []
            for pitch in pitches:
                index = self._getPitchIndex(pitch, angles_and_images)

                combinated.extend(angles_and_images.pop(index)[1])
            
            grouped.append([mean(pitches), combinated])
            
        
        angles_and_images.extend(grouped)
        
        return angles_and_images


    def makeSameLength(self, lista, listb):
        if len(lista) > len(listb):
            lista = lista[0:len(listb)]
        elif len(lista) < len(listb):
            listb = listb[0:len(lista)]
        return lista, listb


    def pitchesToBeCombined(self, angles_and_images, angles):
        '''
        Assuming pupil scanning was done keeping pitch constant while
        varying horizontal angle, it's better to group line scans together
        because there may be slight drift in the pitch angle.
        '''
        
        pitches = [[]]
        scan_direction = -10
        
        anglefied_angles_and_images = []
        for pitch, hor_im in angles_and_images:
            for horizontal, fn in hor_im:
                anglefied_angles_and_images.append([horizontal, pitch])

        # Determine pitches that should be combined
        for i in range(1, len(angles)-1):
            if angles[i] in anglefied_angles_and_images:

                direction = sign( angles[i][0] - angles[i-1][0] )
                future_direction = sign(angles[i+1][0] - angles[i][0])
                
                if direction != scan_direction and not future_direction == scan_direction:
                    pitches.append([])
                    scan_direction = direction
               
                if direction == scan_direction or (scan_direction == 0 and future_direction == scan_direction):
                    if not angles[i][1] in pitches[-1]:
                        pitches[-1].append(angles[i][1])

        
        pitches = [p for p in pitches if len(p)>=2 and len(p)<5]
        
       
        # A pitch can appear more than one time to be combined. This seems
        # usually happen in adjacent pitch groupings.
        # Here, combine [a,b] [b,c] -> [a,b,c]
        combine = []
        for i in range(len(pitches)-1):
            for j, pitch in enumerate(pitches[i]):
                if pitch in pitches[i+1]:                    
                    combine.append([i, j])

        for i, j in sorted(combine, reverse=True):
            pitches[i].pop(j)
            pitches[i] += pitches[i+1]
            pitches.pop(i+1)
        # ----------------------------------------------------- 

        print("Pitches to be combined")
        for p in pitches:
            print(p)

        return pitches
    

    def pitchGroupedHorizontalsAndImages(self, image_fns, angles, arl_fly=False):
        '''
        Returns horizontal angles grouped by pitch (as groupHorizontals)
        but also links image fn with each horizontal angle.
        
        Note: image_fns and angles must have one to one correspondence.
        
        IDEAL STRUCTURE TO WORK WITH

        grouped = [pit1, [[hor1, fn1], ...] ...]
        '''

        grouped = []
        pitches_in_grouped = []
        
        for fn, (horizontal, pitch) in zip(image_fns, angles):
            
            if not pitch in pitches_in_grouped:
                pitches_in_grouped.append(pitch)
                grouped.append([pitch, []])
            
            i = pitches_in_grouped.index(pitch)
            grouped[i][1].append([horizontal, fn])
        
        # For each pitch angle there must be more than 10 images
        # or the whole pitch is removed
        if not arl_fly:
            grouped = [x for x in grouped if len(x[1]) > 10]
        else:
            print('ARL fly, not capping of imaging row.')

        return grouped


class XAnalyser:

    def __init__(self):
        
        self.males = ['DrosoX6', 'DrosoX7', 'DrosoX8', 'DrosoX9', 'DrosoX15']
        self.females = ['DrosoX10', 'DrosoX11', 'DrosoX12', 'DrosoX13', 'DrosoX14']

        self.skip_flies = ['DrosoX14']
        
        # Set saving directories and create them
        self.savedirs = {'overlap_width_points': 'overlap_width_points'}
        for key in self.savedirs:
            self.savedirs[key] = os.path.join(ANALYSES_SAVEDIR, self.savedirs[key])
            os.makedirs(self.savedirs[key], exist_ok=True)
    
    
    def loadOverlapMarkings(self):
        '''
        Load the results of binary search.
        '''
        self.overlap_markings = {}
        results_dir = os.path.join(ANALYSES_SAVEDIR, 'binary_search')
        result_fns = os.listdir(results_dir)
        
        for fn in result_fns:
            with open(os.path.join(results_dir, fn), 'r') as fp:
                self.overlap_markings[fn.split('.')[0].split('_')[1]] = json.load(fp)
        
    def loadAntennaLevels(self):
        '''
        Load pitch points where the pseudopupils align with antenna.
        Run antenna_levels.py first to find antenna levels.
        '''
        
        self.antenna_levels = {}
        
        antenna_levels_dir = os.path.join(ANALYSES_SAVEDIR, 'antenna_levels')
        filenames = [fn for fn in os.listdir(antenna_levels_dir) if fn.endswith('.txt')]
        
        for fn in filenames:
            with open(os.path.join(antenna_levels_dir, fn), 'r') as fp:
                antenna_level = float(fp.read())
            
            self.antenna_levels[fn.split('.')[0]] = antenna_level
        
        print(self.antenna_levels)

    




    def plot1DOverlap(self, sexes='all'):
        
        X = []
        Y = []
        flies = []
        
        plt.figure()

        # ##################################
        # Raw, antenna level corrected data
        # ##################################
        
        for fly, overlap_markings in self.overlap_markings.items():
            
            if sexes == 'males' and not fly in self.males:
                continue
            if sexes == 'females' and not fly in self.females:
                continue

            X.append([])
            Y.append([])
            for marking in sorted(overlap_markings, key=lambda x: x['pitch']):
                if marking['pitch'] == 0:
                    continue

                Y[-1].append(abs(marking['horizontal_left'] - marking['horizontal_right']))
                X[-1].append(marking['pitch']-self.antenna_levels[fly])
            
            flies.append(fly)

        # ##################
        # Interpolated data
        # ##################

        intp_step = 1

        XXs_span = np.arange(int(np.min(np.min(X))/intp_step)*intp_step, int(np.max(np.max(X))/intp_step)*intp_step, intp_step)

        XX = []
        YY = []

        for fly, x, y in zip(flies, X,Y):
            xx = np.arange(int(np.min(x)/intp_step)*intp_step, int(np.max(x)/intp_step)*intp_step, intp_step)
            yy = np.interp(xx, x, y)
            plt.scatter(xx, yy, s=5)
            XX.append(xx)
            YY.append(yy)
            
            with open(os.path.join(self.savedirs['overlap_width_points'], '{}.csv'.format(fly)), 'w') as fp:
                for xxx, yyy in zip(xx, yy):
                        fp.write('{},{}\n'.format(xxx, yyy))

        # ##############
        # Averaged data
        # ##############
        mean_YY = []

        for x in XXs_span:
            
            yys_to_average = []
            #N_averages = 0
            
            for yy, xx in zip(YY, XX):
                try:
                    index = list(xx).index(x)
                except:
                    continue
                #N_averages += 1
                
                yys_to_average.append(yy[index])
            
            
            if yys_to_average:
                mean_YY.append(np.mean(yys_to_average))
            else:
                mean_YY.append(0)
        
        
        
        plt.plot(XXs_span, mean_YY, linewidth=2)
        plt.xlabel('Vertical angle (degrees)')
        plt.ylabel('Binocular overlap (degrees)')
        plt.title(sexes)

    

    def plotMatrixOverlap(self):

        
        self.hor_step = 1
        self.ver_step = 4
        self.hor_range = (-40, 40)
        self.ver_range = (-90, 90)
        
        hor_step = self.hor_step
        ver_step = self.ver_step
        hor_range = self.hor_range
        ver_range = self.ver_range

        matrices = []

        matrix_width = int((hor_range[1]-hor_range[0])/hor_step)
        matrix_height = int((ver_range[1]-ver_range[0])/ver_step)


        for fly, markings in self.overlap_markings.items():
            if fly in self.skip_flies:
                continue
            
            print(fly) 
            X = []
            Y = []
            midpoints =  [] 
            
            for marking in markings:
                
                mid = marking['horizontal_middle']
                marking['horizontal_left'] - mid
                
                row = []


                marking['horizontals'].sort()
                
               # 
               # if marking['horizontal_left'] == marking['horizontal_right']:
               #     row = [0]*marking['N_images']
               # else:
               #     for i in range(0, marking['N_images']):
               #         
               #         if marking['horizontals'][i] < marking['horizontal_left']:
               #             row.append(0)
               #         elif marking['horizontal_left'] <= marking['horizontals'][i] <= marking['horizontal_right']:
               #             if marking['horizontals'][i] == marking['horizontal_middle']:
               #                 row.append(2)
               #             else:
               #                 row.append(1)
               #         elif marking['horizontals'][i] > marking['horizontal_right']:
               #             row.append(0)
               #     
               # if len(row) != marking['N_images']:
               #     print(row)
               #     print(marking['horizontal_left'])
               #     print(marking['horizontal_right'])
               #     raise UserWarning('Row length {} but markings length {}'.format(len(row), marking['N_images']))
               # 
                if marking['horizontal_right'] - marking['horizontal_left'] < hor_step:
                    row = [0]*matrix_width
                    row[int(matrix_width/2)] = 2

                else:
                    for angle in range(hor_range[0], hor_range[1], hor_step):
                        if angle < marking['horizontal_left']:
                            row.append(0)
                        elif marking['horizontal_left'] <= angle <= marking['horizontal_right']:
                            #if marking['horizontal_middle'] - hor_step/1.5 < angle < marking['horizontal_middle'] + hor_step/1.5:
                            #    row.append(2)
                            #else:
                            row.append(1)
                        elif marking['horizontal_right'] < angle:
                            row.append(0)
                
                midpoints.append( (marking['horizontal_middle']) / (int(hor_range[1]-hor_range[0])/2) )

                if len(row) != matrix_width:
                    print(row)
                    print(marking['horizontal_left'])
                    print(marking['horizontal_right'])
                    raise UserWarning('Row length {} but matrix width {}'.format(len(row), matrix_width))
          


                X.append(row)
                Y.append(marking['pitch']-self.antenna_levels[fly])
            
            matrix = np.zeros( (int((ver_range[1]-ver_range[0])/ver_step), int((hor_range[1]-hor_range[0])/hor_step)) )
            matrix_i_midpoint = int(matrix.shape[0] / 2)
            for j, pitch in enumerate(range(*ver_range, ver_step)):

                indices = [y for y in Y if pitch-ver_step/2 <= y <= pitch+ver_step/2]
                indices = [Y.index(y) for y in indices]
                
                for index in indices:
                    i_midpoint = int((midpoints[index])*int(matrix_width/2))
                    shift = -1*(i_midpoint)
                    if shift >= 0:
                        matrix[j][shift:] += np.asarray(X[index])[0:matrix_width-shift]
                    elif shift < 0:
                        matrix[j][0:matrix_width+shift] += np.asarray(X[index])[-shift:]
                        
                    #matrix /= len(indices)
                    matrix = np.round(matrix)
                    matrix = np.clip(matrix, 0, 1) 
            matrices.append(matrix)
            
        avg_matrix = matrices[0] / len(matrices)
        for i in range(1, len(matrices)):
            avg_matrix += matrices[i] / len(matrices)
        
        
        matrix = np.round(avg_matrix)
        #matrix = np.clip(avg_matrix, 0, 1)
        
        self._plotMatrix(avg_matrix, newfig=True)
    
        for j in range(0, avg_matrix.shape[0]):
            row_max = np.max(avg_matrix[j])
            if row_max > np.min(avg_matrix[j]):
                avg_matrix[j] /= row_max
        plt.figure() 
        
        for i, matrix in enumerate(matrices):
            for j in range(0, matrix.shape[0]):
                if not np.any(matrix[j]):
                    matrix[j] += 0.5

            self._plotMatrix(matrix, subplot=int('{}{}{}'.format(3,round(len(matrices)/3),i+1)))
        
        self._plotMatrix(avg_matrix, newfig=True)
    


    def _plotMatrix(self, matrix, newfig=False, subplot=111):
        matrix_height, matrix_width = matrix.shape
        
        if newfig == True:
            plt.figure()
        plt.subplot(subplot)

        plt.imshow(matrix, cmap='coolwarm', interpolation='none', extent=(self.hor_range[0], self.hor_range[1], self.ver_range[1], self.ver_range[0]), aspect='auto')
        ax = plt.gca();
        ax.set_xticks(np.arange(self.hor_range[0]+self.hor_step, self.hor_range[1]+self.hor_step, self.hor_step), minor=True)
        ax.set_yticks(np.arange(self.ver_range[0]+self.ver_step, self.ver_range[1]+self.ver_step, self.ver_step), minor=True) 
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    def plot2DOverlap(self):
        
        Xl = []
        Yl = []
        Xr = []
        Yr = []
        Xm = []
        Ym = []



        for marking in self.overlap_markings:
            
            mid = marking['horizontal_middle']
            #mid = 0
            Xl.append(marking['horizontal_left']-mid)
            Xr.append(marking['horizontal_right']-mid)
            Xm.append(marking['horizontal_middle']-mid)
            Yl.append(marking['pitch'])
            Yr.append(marking['pitch'])
            Ym.append(marking['pitch'])
        
        plt.scatter(Xl, Yl, color='blue')
        plt.scatter(Xr, Yr, color='red')
        plt.scatter(Xm, Ym, color='yellow')


        plt.show()

       
    def plotFancy2DOverlap(self):

        X,Y,C = [[],[],[]]
        for marking in self.overlap_markings:

            mid = marking['horizontal_middle']
            
            for i in range(len(marking['horizontals'])):
                
                pitch = marking['pitch']
                horizontal = marking['horizontals'][i]
                L = min(marking['horizontal_left'], marking['horizontal_right'])
                R = max(marking['horizontal_left'], marking['horizontal_right'])
                
                if L < horizontal < R:
                    C.append(2)
                else:
                    C.append(1)
                
                X.append(horizontal-mid)
                Y.append(pitch)

        f_int = scipy.interpolate.interp2d(X, Y, C, fill_value=1)
        
        X = np.linspace(np.min(X), np.max(X), 100)
        Y = np.linspace(np.min(Y), np.max(Y), 100)
        C = f_int(X, Y)
        
        X, Y = np.meshgrid(X, Y)
        
        C = np.around(C)
        #C = np.clip(C, 1, 2)


        plt.pcolormesh(X, Y, C)

        plt.show()


def main():
    

    xanalyser = XAnalyser()
    xanalyser.loadOverlapMarkings()
    xanalyser.loadAntennaLevels()
    
    xanalyser.plot1DOverlap()
    
    xanalyser.plotMatrixOverlap()
    
    #xanalyser.plot1DOverlap(sexes='males')
    #analyser.plot1DOverlap(sexes='females')
    
    plt.show()
    #xanalyser.plot2DOverlap()
    #xanalyser.plotFancy2DOverlap()


if __name__ == "__main__":
    main()


