import os
import sys

def findFiles(fileList, keyword):
    """
    Returns a list of files that contains the keyword.
    """
    desiredFiles = []
    for filename in fileList:
        if keyword in filename:
            desiredFiles.append(filename)
    return desiredFiles


def findTxtFromPath(path):
    """
    Gets a list containing all .txt files from directory path
    """
    return filter(lambda f: f.endswith('.txt'), os.listdir(path))
#---------------------------------------------------------------------
if __name__ == '__main__':
    fileList = findTxtFromPath('.')
    keywords = set(['RHC', 'SA', 'GA','MIMIC'])
    value_statistics = {'RHC':[],'SA':[],'GA':[],'MIMIC':[]}
    for f in fileList:
        with open(f) as file:
            for line in file:
                if 'Average time for' in line:
                    break
                if any(key in line for key in keywords):
                    data_line = line.split(':')
                    value_statistics[data_line[0]].append(float(data_line[1]))

    num = len(value_statistics['RHC'])
    rhc_avg = sum(value_statistics['RHC']) / float(num)
    sa_avg = sum(value_statistics['SA']) / float(num)
    ga_avg = sum(value_statistics['GA']) / float(num)
    mimic_avg = sum(value_statistics['MIMIC']) / float(num)

    print 'Average fitness values: '
    print 'RHC:', rhc_avg
    print 'SA:', sa_avg
    print 'GA:', ga_avg
    print 'MIMIC:', mimic_avg

    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    algoName = 'RHC'
    bins = 50

    data = sorted(value_statistics[algoName])
    print data
    n, bins, patches = plt.hist(data, bins)#, 100, facecolor='green', alpha=0.75)
    plt.xlabel('Fitness Value')
    plt.ylabel('Frequency')
    plt.title('$\mathrm{Knapsack\ Problem\ (Randomized\ HC)}$')
    plt.axis([4000, 4500, 0, 10])
    plt.yticks(np.arange(0,11,1.0))
    plt.grid(True)

    plt.show()