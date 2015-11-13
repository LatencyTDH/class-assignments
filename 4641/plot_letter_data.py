import os
import sys
import numpy as np
# import matplotlib.pyplot as plt

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


def convertDataLine(infoString):
    """
    Transforms pertinent data into format that can be read later
    """
    values = infoString.split()[3:-1]
    return map(float, values)


#################################################################################################################################################
# Vary the number of hidden layers in our RO algorithms. Plot classification error vs. # iterations for hiddenLayer = {0,1,2}
def plot_n_hiddenLayers_by_iteration(n, xlabel='Iterations', ylabel='Error (%)', xmin=0, xmax=100, ymin=75, ymax=85,
                                     xincrement=1.0, yincrement=1.0):
    path = 'result_data/Run-{0}-HL/'.format(n)
    fileList = findTxtFromPath(path)
    iterationIndex = 4
    data_points_dict = get_data_points_by_iteration(fileList, path, iterationIndex)
    rhc_points = sorted(data_points_dict['RHC'])
    sa_points = sorted(data_points_dict['SA'])
    ga_points = sorted(data_points_dict['GA'])

    import matplotlib.pyplot as plt

    plt.figure()
    plt.title('Classification Error vs. No. of Iterations ({0}-HL)'.format(n))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis([xmin, xmax, ymin, ymax])

    plt.plot([x[0] for x in rhc_points], [x[1] for x in rhc_points], 'k--', label='RHC')
    plt.plot([x[0] for x in sa_points], [x[1] for x in sa_points], 'g-', label='SA')
    plt.plot([x[0] for x in ga_points], [x[1] for x in ga_points], 'b-', label='GA')
    plt.yticks(np.arange(ymin, ymax + 1, yincrement))
    plt.legend()
    plt.show()
    print fileList


def get_data_points_by_iteration(fileList, path, iterationIndex):
    search_results = {'RHC': [], 'SA': [], 'GA': []}

    for fileName in fileList:
        iteration_number = float(fileName.split('-')[iterationIndex])
        with open(path + fileName) as data_file:
            search_by_iteration_number(data_file, iteration_number, search_results)
    return search_results


def search_by_iteration_number(file, it_num, search_results={'RHC': [], 'SA': [], 'GA': []}):
    for line in file:
        if 'Results for RHC' in line:
            for a in file:
                if 'Percent correctly' in a:
                    err = 100.0 - float(a.split(':')[1].replace('%', ''))
                    search_results['RHC'].append((it_num, err))
                    break
        if 'Results for SA' in line:
            for a in file:
                if 'Percent correctly' in a:
                    err = 100.0 - float(a.split(':')[1].replace('%', ''))
                    search_results['SA'].append((it_num, err))
                    break
        if 'Results for GA' in line:
            for a in file:
                if 'Percent correctly' in a:
                    err = 100.0 - float(a.split(':')[1].replace('%', ''))
                    search_results['GA'].append((it_num, err))
                    break
    return search_results


#####################################################################################################################################################
# Plot SSE over time for each RO algorithm given our baseline NN (inputLayer = 16 nodes, hiddenLayers = 1, hiddenNodes = 16, output = 1 node
# iterations = 500)
def isfloat(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


# Plots the sse data over time for each ro alg in the file.
def plot_SSE_data(path='letter_results/Run-1-HL/',fileName='results-1-HL-iterations-500-1426221691839.txt',
                  xlabel='Iterations', ylabel='Error (%)', xmin=0, xmax=500, ymin=300, ymax=4500, xincrement=25.0,
                  yincrement=100.0):
    data_points_dict = get_SSE_data(path+fileName)
    rhc_points = sorted(data_points_dict['RHC'])
    sa_points = sorted(data_points_dict['SA'])
    ga_points = sorted(data_points_dict['GA'])

    import matplotlib.pyplot as plt
    from pylab import axis

    # plt.figure()


    plt.subplot(3, 1, 1)
    plt.title('SSE Over Time for Baseline Neural Net (1-HL)')
    axis([xmin,xmax,ymin,ymax])
    plt.plot([x[0] for x in rhc_points], [x[1] for x in rhc_points], 'k-', label='RHC')
    plt.ylabel(ylabel)
    plt.legend()

    plt.subplot(3, 1, 2)
    axis([xmin,xmax,ymin,ymax])
    plt.plot([x[0] for x in sa_points], [x[1] for x in sa_points], 'g-', label='SA')
    plt.ylabel(ylabel)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.ylabel(ylabel)
    plt.plot([x[0] for x in ga_points], [x[1] for x in ga_points], 'b-', label='GA')
    plt.xlabel(xlabel)

    plt.legend()
    axis([xmin,xmax,ymin,ymax])
    plt.show()


#relative path to file from where plot_letter_data.py is located
def get_SSE_data(fileName='letter_results/Run-1-HL/results-1-HL-iterations-500-1426221691839.txt'):
    results = {'RHC': [], 'SA': [], 'GA': []}
    with open(fileName) as file:
        consolidate_SSE(file, results)
    return results


def consolidate_SSE(file, results={'RHC': [], 'SA': [], 'GA': []}):
    """
    Looks at SSE output on each line of {RHC, SA, and GA} and constructs a collection of points p,
    where p = (time, SSE).
    """
    for line in file:
        if 'Error results for RHC' in line:
            file.next()  #skip the "-------------"
            time1 = 0
            for a in file:
                if isfloat(a):
                    time1 += 1
                    SSE = float(a)
                    results['RHC'].append((time1, SSE))
                else:
                    break
        if 'Error results for SA' in line:
            file.next()  #skip the "-------------"
            time2 = 0
            for a in file:
                if isfloat(a):
                    time2 += 1
                    SSE = float(a)
                    results['SA'].append((time2, SSE))
                else:
                    break
        if 'Error results for GA' in line:
            file.next()  #skip the "-------------"
            time3 = 0
            for a in file:
                if isfloat(a):
                    time3 += 1
                    SSE = float(a)
                    results['GA'].append((time3, SSE))
                else:
                    break
    return results


##############################################################################################
# SIMULATED ANNEALING
def get_sa_statistics(fileName='sa-cooling-0.150000-1426229592333.txt'):
    path = 'letter_results/SA/vary_coolingRate/'
    fileName = path + fileName
    sa_statistics = {'SSE_trial': []}
    with open(fileName) as file:
        for line in file:
            if "Error results for SA" in line:
                file.next()  #skip the "-------------"
                time = 0
                for a in file:
                    if isfloat(a):
                        time += 1
                        SSE = float(a)
                        sa_statistics['SSE_trial'].append((time, SSE))
                    else:
                        break
            if "Results for SA" in line:
                for a in file:
                    if 'Percent correctly' in a:
                        err = 100.0 - float(a.split(':')[1].replace('%', ''))
                        sa_statistics['classification_error'] = err
                    if 'Training' in a:
                        time = float(a.split(' ')[2])
                        sa_statistics['training_time'] = time
                    if 'Testing' in a:
                        time = float(a.split(' ')[2])
                        sa_statistics['testing_time'] = time
                    if 'inputLayer' in a:
                        sa_statistics['cooling_rate'] = float(a.split(',')[-1].split('=')[1])
    return sa_statistics


def plot_sa_by_accuracy(path='letter_results/SA/vary_coolingRate/'):
    fileList = findTxtFromPath(path)
    stat_list = []
    for fileName in fileList:
        stat_list.append(get_sa_statistics(fileName))

    accList = []
    for stat in stat_list:
        accList.append((stat['cooling_rate'], stat['classification_error']))

    import matplotlib.pyplot as plt
    # plt.figure()
    plt.title('Classification Error for Different Cooling Rates (t=1E11)')
    plt.axis([0.15, 1, 0, 100])
    plt.yticks(np.arange(0, 100, 5.0))
    plt.xticks(np.arange(0.1, .95, 0.1))
    plt.xlabel('Cooling rate')
    plt.ylabel('Error (%)')
    plt.plot([x[0] for x in accList], [x[1] for x in accList], 'g-')
    plt.legend()
    plt.show()
    # help_plot_sa(stat_list)


def help_plot_sa(stats):
    import matplotlib.pyplot as plt
    from pylab import axis

    stats_length = len(stats)
    fig = plt.figure()

    for i, stat in enumerate(stats):
        sa_points = sorted(stat['SSE_trial'])
        plt.subplot(stats_length, 1, i + 1)
        if i == 0:
            plt.title('SSE Over Time for Different Cooling Rates (t=1E11)')
        plt.plot([x[0] for x in sa_points], [x[1] for x in sa_points], 'g-',
                 label='CR={0}'.format(stat['cooling_rate']))
        plt.legend()
        if i == 2:
            plt.ylabel('SSE')
        axis([0, 500, 0, 5000])
        if i == stats_length - 1:
            plt.xlabel('Iterations')
            axis([0, 500, 0, 9000])
    plt.show()

################################################################################################################
# GENETIC ALGORITHM
##################################
def get_ga_statistics(fileName='ga-toMate-10-1426278712559.txt', path='letter_results/GA/',):
    ga_statistics = {'SSE_trial': []}

    with open(path+fileName) as file:
        for line in file:
            if "Error results for GA" in line:
                file.next()  #skip the "-------------"
                time = 0
                for a in file:
                    if isfloat(a):
                        time += 1
                        SSE = float(a)
                        ga_statistics['SSE_trial'].append((time, SSE))
                    else:
                        break
            if "Results for GA" in line:
                for a in file:
                    if 'Percent correctly' in a:
                        err = 100.0 - float(a.split(':')[1].replace('%', ''))
                        ga_statistics['classification_error'] = err
                    if 'Training' in a:
                        time = float(a.split(' ')[2])
                        ga_statistics['training_time'] = time
                    if 'Testing' in a:
                        time = float(a.split(' ')[2])
                        ga_statistics['testing_time'] = time
                    if 'inputLayer' in a:
                        ga_statistics['population'] = float(a.split(',')[4].split('=')[1])
                        ga_statistics['toMate'] = float(a.split(',')[5].split('=')[1])
                        ga_statistics['toMutate'] = float(a.split(',')[-1].split('=')[1])
    return ga_statistics

def plot_ga_by_SSE(path, title=''):
    fileList = findTxtFromPath(path)
    stat_list = []
    for fileName in fileList:
        stat_list.append(get_ga_statistics(fileName,path))
    import matplotlib.pyplot as plt
    from pylab import axis
    stats_length = len(stat_list)
    plt.figure()

    stat_list.sort(key=lambda p: p['toMate'])
    for i, stat in enumerate(stat_list):
        ga_points = sorted(stat['SSE_trial'])
        plt.subplot(stats_length, 1, i + 1)
        if i == 0:
            plt.title(title)
        plt.plot([x[0] for x in ga_points], [x[1] for x in ga_points], 'b-',
                 label='toMate={0}'.format(stat['toMate']))
        plt.legend()
        if i == 4:
            plt.ylabel('SSE')
        axis([0, 500, 0, 6000])
        if i == stats_length - 1:
            plt.xlabel('Iterations')
            # axis([0, 500, 0, 9000])
    plt.show()

    # stat_list.sort(key=lambda p: p['toMutate'])
    # for i, stat in enumerate(stat_list):
        # ga_points = sorted(stat['SSE_trial'])
        # plt.subplot(stats_length, 1, i + 1)
        # if i == 0:
            # plt.title(title)
        # plt.plot([x[0] for x in ga_points], [x[1] for x in ga_points], 'b-',
                 # label='toMutate={0}'.format(stat['toMutate']))
        # plt.legend()
        # if i == 4:
            # plt.ylabel('SSE')
        # axis([0, 500, 0, 6000])
        # if i == stats_length - 1:
            # plt.xlabel('Iterations')
            # # axis([0, 500, 0, 9000])
    # plt.show()

    # accList = []
    # for stat in stat_list:
    #     accList.append((stat['cooling_rate'], stat['classification_error']))

if __name__ == "__main__":
    path='letter_results/GA/vary_toMate/'
    fileName = "results-1-HL-iterations-1500-1426227440286.txt"
    dataSet = "LETTER"
    title = 'SSE Over Time for Different toMate Values'
    xlabel = "# Iterations"
    ylabel = "SSE"
    xmin = 0
    xmax = 1500
    xincrement = 100.0
    ymin = 0
    ymax = 9000
    yincrement = 1000.0

    if len(sys.argv) > 1:
        os.chdir(sys.argv[1])
    #Plot neural net errors for {0,1,2} hiddenLayers w/ 16 nodes in each HL
    #as a function of iterations
    # for x in range(3):
    # 	try:
    # 		plot_n_hiddenLayers_by_iteration(x,xlabel=xlabel,ylabel=ylabel,
    # 			xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax, xincrement=xincrement, yincrement=yincrement)
    # 	except Exception as e:
    # 		print 'Folder does not exist for {0} HLs!'.format(x)
    # 		continue

    # plot_SSE_data(xlabel=xlabel,ylabel=ylabel,xmin=xmin,xmax=xmax,ymin=ymin,
    # 	ymax=ymax, xincrement=xincrement, yincrement=yincrement)

    # print get_sa_statistics()
    # plot_ga_by_SSE(path,title=title)


    plot_SSE_data(fileName=fileName, xlabel=xlabel,ylabel=ylabel,xmin=xmin,xmax=xmax,ymin=ymin,
    ymax=ymax, xincrement=xincrement, yincrement=yincrement)