
import copy

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim import Optimizer

import sys
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


from scipy.special import logsumexp
from scipy.special import softmax

import matplotlib as mpl
import seaborn as sns

sns.set_style('whitegrid')
mpl.rc('text', usetex=True)
sns.set_context("notebook", font_scale=1.4)


def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data

class MutationModel(nn.Module):
    def __init__(self, M, modL=False):
        super(MutationModel, self).__init__()


        #print (modL)
        #quit()

        self.M = M
        L = 5
        #L = 10

        if modL:
            L = 5#20




        #L2 = 2
        #self.conv1 = torch.nn.Conv1d(14, mN1_1, 1)
        self.nonlin = torch.tanh

        #self.lin0 = torch.nn.Linear(L, L)

        #print (M)
        #quit()

        self.lin1 = torch.nn.Linear(M+20, L)
        self.lin2 = torch.nn.Linear(L, M)

        #self.linI = torch.nn.Linear(1, 1)

        ####self.linM = torch.nn.Linear(L, L)

        #self.linM1 = torch.nn.Linear(L1, L2)
        #self.linM2 = torch.nn.Linear(L2, L1)

        #self.linP = torch.nn.Linear(1, 20)

        #self.linSum = torch.nn.Linear(2, L)
        #self.linBaseline = torch.nn.Linear(L, 1)


    def forward(self, x):


        #print (x.shape)

        #x = self.lin1(x)
        #x = self.lin2(x)


        xSum = torch.sum(x, dim=1)#.reshape((-1, 1))

        xSum2 = torch.zeros((x.shape[0], 20))
        xSum2[np.arange(x.shape[0]), xSum.long()] = 1

        #x = x * 0

        x = torch.cat((x, xSum2), dim=1)

        x = self.lin1(x)

        x1 = x[:, 0].repeat_interleave(self.M).reshape((x.shape[0], self.M) )

        ####x = self.linM(x)
        ####x = self.nonlin(x)


        x = self.nonlin(x)

        xNP = x.data.numpy()

        #x = self.nonlin(x)

        #plt.plot(xNP)
        #plt.scatter(xNP[:, 2], xNP[:, 4])
        #plt.show()
        #quit()

        #x = self.linM(x)
        #x = self.nonlin(x)
        #x = self.linM2(x)
        #x = self.nonlin(x)
        x = self.lin2(x)

        #x = x * 0

        x = x + x1

        #shape1 = x.shape


        return x, xNP



class MutationModel2(nn.Module):
    def __init__(self, M):
        super(MutationModel2, self).__init__()
        self.M = M
        L = 10

        #self.lin1 = torch.nn.Linear(M+20, M)
        self.lin1 = torch.nn.Linear(M, M)



    def forward(self, x):

        x = x.clone()
        x = self.lin1(x)

        xNP = x.data.numpy()

        return x, xNP


def uniqueValMaker(X):

    _, vals1 = np.unique(X[:, 0], return_inverse=True)

    for a in range(1, X.shape[1]):

        vals2 = np.copy(X[:, a])
        vals2_unique, vals2 = np.unique(vals2, return_inverse=True)

        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)

    return vals1

def addFromLog(array0):

    #Just a basic function that hangles addition of logs efficiently

    array = np.array(array0)
    array_max = np.max(array, axis=0)
    for a in range(0, array.shape[0]):
        array[a] = array[a] - array_max
    array = np.exp(array)
    array = np.sum(array, axis=0)
    array = np.log(array)
    array = array + array_max

    return array

def doRECAPplot(name, doCluster=False):

    #Recap plotting compairison from the recap paper. For more information on this code, go to the github of the RECAP paper.

    import scipy

    import numpy as np #For some reaosn this needs to be re-imported
    allSaveVals = np.load('./dataNew/allSave_' + name + '.npy')
    #print (allSaveVals.shape)
    saveData = []
    for a in range(0, len(allSaveVals)):
        dataNow = allSaveVals[a]
        ar1 = dataNow[2:102]
        ar2 = dataNow[102:202]

        ar1 = ar1[ar1 != '0.0']
        ar2 = ar2[ar2 != '0.0']

        incorrect = 0
        for b in range(0, len(ar1)):
            if ar1[b] != ar2[b]:
                incorrect += 1

        accuracy = 1 - (incorrect / len(ar1))

        incorrect2 = 0
        for b in range(0, len(np.unique(ar1))):
            if np.unique(ar1)[b] != np.unique(ar2)[b]:
                incorrect2 += 1


        kTrue = np.unique(ar1).shape[0]
        kPred = np.unique(ar2).shape[0]

        #print (dataNow[0])

        mVal = int(dataNow[0].split('m')[1])

        #if mVal != 12:
        #    print (incorrect2)
        #    if incorrect2 != 0:
        #        print ("Issue")
        #        quit()

        dataNew = [dataNow[0], dataNow[1], accuracy, mVal, kTrue, kPred]

        saveData.append(np.copy(np.array(dataNew)))

    saveData = np.array(saveData)

    #quit()

    import os, sys, glob
    import math
    import numpy as np
    import pandas as pd
    #%matplotlib inline
    #%config InlineBackend.figure_format = 'svg'
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    pd.set_option('display.max_columns', None)
    sns.set_style('whitegrid')
    mpl.rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.5)


    # Read in results file for all methods
    df = pd.read_csv("./data/results.tsv", sep="\t")
    # Set output results folder
    #orf = "./simulations/pdfs/"


    # Identifying the selected k for each instance
    # For Revolver and Hintra, we only import the k those methods selected

    df["selected_k"] = [1 if (x =='HINTRA' or x =='REVOLVER') else 0 for x in df['method']]

    assert df['selected_k'].sum() == len(df[(df["method"] == 'HINTRA')].index) + len(df[(df["method"] == 'REVOLVER')].index)


    # Functions to help us select k for RECAP

    def smooth(pairs):
        smooth_pairs = []
        pairs.sort(key=lambda elem: elem[0])
        low_d = pairs[0][1]
        for k,d in pairs:
            low_d = min(low_d, d)
            smooth_pairs.append((k,low_d))

        return smooth_pairs

    def find_k(pairs, p_threshold, a_threshold):
        pairs.sort(key=lambda elem: elem[0], reverse=True)
        prev_d = pairs[0][1]
        for k,d in pairs[1:]:
            a_change = d-prev_d
            p_change = a_change/(prev_d*1.0+0.0000001)
            if (p_change >= p_threshold) and (a_change >= a_threshold):
                return k+1
            prev_d = d

        return 1

    def select_k(df, pt, at):

        # Find unique list of instances, methods, and true k
        instance_df = df.drop_duplicates(['instance','method', 'true_k'])[['instance','method', 'true_k']]
        instance_df = instance_df[instance_df['method'].isin(['RECAP-r10', 'RECAP-r50', 'RECAP-r100'])]

        assert len(instance_df.index) == 1800

        # Iterate over these instances
        for index, row in instance_df.iterrows():

            # Extract instance information
            instance = row['instance']
            method = row['method']
            true_k = row['true_k']

            # Subset data to this instance
            subset = df[(df["instance"] == instance) & (df["method"] == method) & (df["true_k"] == true_k)]

            # Compute k for this instance
            pairs = []

            for index, row in subset.iterrows():
                pairs.append((row['inferred_k'], row['PC_dist']))

            smooth_pairs = smooth(pairs)
            selected_k = find_k(smooth_pairs, pt, at)

            # Fill in df with this selected k
            df.loc[(df["instance"] == instance) & (df["method"] == method) & (df["true_k"] == true_k) & (df["inferred_k"] == selected_k), 'selected_k'] = 1

        return df





    # Fill in selected k for RECAP with given percentage and absolute thresholds for finding the elbow
    pt = .05
    at = .5

    df = select_k(df, pt, at)

    #df.append(np.zeros(21).astype(str))
    #df.append(list(np.zeros(21).astype(str)))

    keys1 = np.array(df.keys())

    argAccuracy = np.argwhere(keys1 == 'selection_accuracy')[0, 0]
    argMethod = np.argwhere(keys1 == 'method')[0, 0]
    argSelectedK = np.argwhere(keys1 == 'selected_k')[0, 0]
    argTrueK = np.argwhere(keys1 == 'true_k')[0, 0]



    '''
    dataForDf = []
    for a in range(0, len(saveData)):

        #ar1 = np.zeros(21).astype(str)
        ar1 = list(np.zeros(21))
        ar1[argAccuracy] = saveData[a, 2]
        ar1[argMethod] = 'Stefan'
        ar1[argSelectedK] = 1
        ar1[argTrueK] = 1
        #ar1 = list(ar1)

        dataForDf.append(ar1)
    '''

    #dataDF = pd.DataFrame(dataForDf, columns=df.keys())


    df = df[df["selected_k"] == 1]

    df_vals = df.values

    #dataForDf = np.array(dataForDf)
    #dataForDf = np.zeros((len(saveData), len(keys1)))
    dataForDf = df_vals[:len(saveData)]

    df_vals = np.concatenate((dataForDf, df_vals), axis=0)



    df = pd.DataFrame(df_vals, columns=df.keys())



    dataForDf = []
    for a in range(0, len(saveData)):

        #print (int(saveData[a, 4]))

        df.loc[a, 'selected_k'] = 1
        df.loc[a, 'selection_accuracy'] = float(saveData[a, 2])
        df.loc[a, 'true_k'] = int(saveData[a, 4])
        df.loc[a, 'inferred_k'] = int(saveData[a, 5])
        df.loc[a, 'method'] = 'Stefan'
        df.loc[a, 'm'] = int(saveData[a, 3])



    df = df[df["m"] == 5]

    #0.986


    #print (np.mean(df[df['method'] == "RECAP-r50"]['selection_accuracy']))
    #print (np.mean(df[df['method'] == "Stefan"]['selection_accuracy']))
    #quit()

    random1 = np.random.normal(size = df[df['method'] == "RECAP-r50"]['selection_accuracy'].shape[0]) * 0.001
    random2 = np.random.normal(size = df[df['method'] == "RECAP-r50"]['selection_accuracy'].shape[0]) * 0.001

    y1 = df[df['method'] == "RECAP-r50"]['selection_accuracy']
    y2 = df[df['method'] == "Stefan"]['selection_accuracy']

    import scipy
    from scipy import stats




    #df = df.append(dataDF, ignore_index=True)
    #df = df.append(df, ignore_index=True)

    print (df.keys())


    #assert df[df["method"] == 'RECAP-r10']['selected_k'].sum() == 600
    #assert df[df["method"] == 'RECAP-r50']['selected_k'].sum() == 600
    #assert df[df["method"] == 'RECAP-r100']['selected_k'].sum() == 600






    #methods = ["REVOLVER", "RECAP-r50"]
    #methods = ["REVOLVER", "RECAP-r50", "HINTRA"]
    methods = ["REVOLVER", "RECAP-r50", 'Stefan']

    #print (df[df["selected_k"] == 1])
    # Model condition all


    if doCluster:
        sns.stripplot(data=df, x="true_k",
                  y="inferred_k", hue="method",
                  hue_order=methods,
                  alpha=.4, dodge=True, linewidth=1, jitter=.1,)
        sns.boxplot(data=df, x="true_k",
                    y="inferred_k", hue="method",
                    hue_order=methods, showfliers=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
        #plt.gca().set_title("all model conditions")

        #['M5_m5', 'M12_m7', 'M12_m12']
        if name == 'M5_m5':
            plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        if name == 'M12_m7':
            plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        if name == 'M12_m12':
            plt.gca().set_title("$|\Sigma| = 12$, $12$ mutations per cluster")
        #plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        plt.gca().set_xlabel("simulated number $k^*$ of clusters")
        plt.gca().set_ylabel("inferred number $k$ of clusters")
        plt.gca().set_ylim((-0.05,8.5))
        plt.gcf().set_size_inches(7, 5.5)
        plt.show()

    else:

        sns.stripplot(data=df, x="true_k",
                      y="selection_accuracy", hue="method",
                      hue_order=methods,
                      alpha=.4, dodge=True, linewidth=1, jitter=.1,)
        sns.boxplot(data=df, x="true_k",
                    y="selection_accuracy", hue="method",
                    hue_order=methods,
                    showfliers=False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])

        if name == 'M5_m5':
            plt.gca().set_title("$|\Sigma| = 5$, $5$ mutations per cluster")
        if name == 'M12_m7':
            plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        if name == 'M12_m12':
            plt.gca().set_title("$|\Sigma| = 12$, $12$ mutations per cluster")


        #plt.gca().set_title("all model conditions")
        #plt.gca().set_title("$|\Sigma| = 12$, $7$ mutations per cluster")
        plt.gca().set_xlabel("simulated number $k^*$ of clusters")
        plt.gca().set_ylabel("fraction of correctly selected trees")
        #plt.gca().set_ylim((-0.05,1.05))
        plt.gca().set_ylim((0.55,1.05))
        plt.gcf().set_size_inches(7, 5.5)
        #plt.savefig(orf+"selection_accuracy_all.pdf")
        plt.show()


def toTreeMHNformat():


    #T = 4
    #T = 8
    #T = 9

    folder1 = 'negativeInter'
    T = 2

    #folder1 = 'passanger'
    #T = 2


    #dataSet = 9
    for dataSet in range(0, 1):
        print (dataSet)

        bulkTrees = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(dataSet) + '_bulkTrees.npz').astype(int)
        sampleInverse = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(dataSet) + '_bulkSample.npz').astype(int)
        treeLength = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')
        #treeLength = np.zeros(int(np.max(sampleInverse))) + 5

        #probabilityMatrix = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(dataSet) + '_prob.npz')
        #print (probabilityMatrix)
        #quit()



        newFileLines = [  ['Patient_ID', 'Tree_ID', 'Node_ID', 'Mutation_ID', 'Parent_ID'] ]

        #Only half of trees used for training, just like CloMu.
        numPatient = treeLength.shape[0] // 2

        #numPatient = 100 #For Mini
        #numPatient = 500 #For Mini

        #print (sampleInverse[:3])

        #print (sampleInverse[sampleInverse<10].shape)

        #print (sampleInverse[sampleInverse<500].shape)
        #quit(())

        #Patient_ID Tree_ID Node_ID Mutation_ID Parent_ID

        a = 0
        Tree_ID = 1
        for b in range(0, numPatient):

            #print (a, b)

            treeSize = int(treeLength[b])
            while sampleInverse[a] == b:

                tree1 = bulkTrees[a]#[:treeSize]

                #print (tree1[:treeSize])
                #quit()

                Patient_ID = b + 1

                start1 = len(newFileLines)

                newFileLines.append( [Patient_ID, Tree_ID, 1, 0, 1] )

                #if b == 1:
                #    print (treeSize)
                #    print (tree1)


                for c in range(treeSize):

                    #print ('c')

                    Node_ID = c + 2
                    Mutation_ID = tree1[c, 1] + 1

                    #Parent_ID = np.argwhere(tree1[c, :] == tree1[c, 0])#[0, 0]
                    Parent_ID = np.argwhere(tree1[:c, 1] == tree1[c, 0])#[0, 0]
                    #print (Parent_ID)
                    if Parent_ID.shape[0] == 0:
                        Parent_ID = 0
                    else:
                        Parent_ID = Parent_ID[0, 0] + 1

                    Parent_ID = Parent_ID + 1

                    #print (Parent_ID)

                    newFileLines.append( [Patient_ID, Tree_ID, Node_ID, Mutation_ID, Parent_ID] )

                #if b == 1:
                #    print (np.array(newFileLines[start1:]))
                #    quit()
                #if a == 2:
                #   for d in range(len(newFileLines)):
                #        print (newFileLines[d])
                #    quit()



                Tree_ID += 1
                a += 1

        newFileLines = np.array(newFileLines)

        #print (newFileLines[:20])
        #quit()




        #np.savetxt('./otherMethod/treeMHN/data/input/' + str(T) + '/' + str(dataSet) + '_trees.csv', newFileLines, delimiter=",", fmt='%s')
        #np.savetxt('./otherMethod/treeMHN/data/input/passenger/' + str(dataSet) + '_trees.csv', newFileLines, delimiter=",", fmt='%s')
        np.savetxt('./otherMethod/treeMHN/data/input/5now/' + str(dataSet) + '_trees.csv', newFileLines, delimiter=",", fmt='%s')

        #np.savetxt('./TreeMHN/data/input/' + str(dataSet) + '_trees_mini.csv', newFileLines, delimiter=",", fmt='%s')


#toTreeMHNformat()
#quit()


def fromTreeMHNformat():


    def converter0(tree_file):


        li = []
        for data in tree_file:
            #list1 = data.split(',')
            list1 = data
            lis = []
            for number in list1:
                lis.append(int(number))
            li.append(lis)

        sorted_trees = []
        temp_list = []
        previous_number = 1
        for edge in li:
            if edge[0] != previous_number:
                sorted_trees.append(temp_list)
                temp_list = []
            temp_list.append(edge)
            previous_number = edge[0]
            if edge[0] == 500 and edge[4] == 2:
                sorted_trees.append(temp_list)

        trees_with_mutations = []
        for tree in sorted_trees:
            trees_with_mutations.append(create_trees(tree))
        final_list = []
        for tree in trees_with_mutations:
            temp_list2 = []
            temp_list2.append(tree)
            final_list.append(temp_list2)
        # print(final_list)

        data = np.array(final_list, dtype=object)
        #np.save('./dataNew/customData/' + 'TreeMHN_simulations' + '.npy', data)
        #file = np.load('./dataNew/customData/' + 'TreeMHN_simulations' + '.npy', allow_pickle=True)
        #print(file)
        return data

    # takes the mutations that create one tree and converts them into a single tree that has the correct mutations


    def create_trees(tree_data):
        tree_with_mutations = []
        for index in range(1, len(tree_data)):
            if tree_data[index][4] == 1:
                tree_with_mutations.append(['Root', str(tree_data[index][3]) + '_' + str(tree_data[index][2] - 1)])
                # tree_with_mutations.append(['Root', mutations[tree_data[index][3] - 1]])

            else:
                tree_with_mutations.append([str(tree_data[tree_data[index][4] - 1][3]) + '_' + str(tree_data[tree_data[index][4] - 1][2] - 1),
                                            str(tree_data[index][3]) + '_' + str(tree_data[index][2] - 1)])
                # tree_with_mutations.append([mutations[tree_data[tree_data[index][4] - 1][3] - 1],
                #                            mutations[tree_data[index][3] - 1]])

        return tree_with_mutations



    '''
    #a = 0
    for a in range(100):
        print (a)
        trees = np.loadtxt('./treeMHN/data/MHNtree/trees_' + str(a) + '.csv', delimiter=",", dtype=str)
        trees = trees[1:]
        #print (trees)


        #print (trees[:5])
        data = converter0(trees)

        #print (data[0])

        np.save('./treeMHN/data/MHNtree_np/' + str(a) + '.npy', data)
    '''




    if False:
        folder1 = './treeMHN/treeMHNdata/CSV'
        saveFolder = './treeMHN/treeMHNdata/np'

        subFolders = os.listdir(folder1)
        if '.DS_Store' in subFolders:
            subFolders.remove('.DS_Store')

        existingFolders = os.listdir(saveFolder)

        for a in range(len(subFolders)):
            subFolder = subFolders[a]

            if not subFolder in existingFolders:
                newFolder1 = saveFolder + '/' + subFolder
                os.mkdir(newFolder1)

            for b in range(100):

                fileName = folder1 + '/' + subFolder + '/trees_' + str(b) + '.csv'

                trees = np.loadtxt(fileName, delimiter=",", dtype=str)
                trees = trees[1:]

                data = converter0(trees)

                np.save('./treeMHN/treeMHNdata/np/' + subFolder + '/trees_' + str(b) + '.npy', data)



#fromTreeMHNformat()
#quit()


def toRecapFormat():
    True

    #T = 4
    T = 1
    #dataSet = 9
    for dataSet in range(0, 40):
        print (dataSet)

        #baseline = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(b) + '_baseline.pt.npy')
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkSample.npz').astype(int)

        bulkTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkTrees.npz').astype(int).astype(str)
        #trueTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_trees.npz').astype(int)

        unique1 = np.unique(bulkTrees)
        for a in range(len(unique1)):
            bulkTrees[bulkTrees == unique1[a]] = 'M' + unique1[a]
        #bulkTrees[bulkTrees == 'M10'] = 'GL'
        bulkTrees[bulkTrees == 'M100'] = 'GL'

        #treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')
        treeLength = np.zeros(int(np.max(sampleInverse))) + 5


        newFileLines = []

        numPatient = treeLength.shape[0] // 2

        newFileLines.append(str(numPatient) + ' # patients')

        a = 0
        for b in range(0, numPatient):

            numTrees = 0
            a1 = a
            while sampleInverse[a1] == b:
                a1 += 1
                numTrees += 1

            newFileLines.append(str(numTrees) + ' #trees for P' + str(b))

            treeSize = int(treeLength[b])
            while sampleInverse[a] == b:

                newFileLines.append(str(treeSize) + ' #edges')

                tree1 = bulkTrees[a]#[:treeSize]

                for c in range(treeSize):
                    newFileLines.append(tree1[c, 0] + ' ' +  tree1[c, 1])
                a += 1

        #newFileLines = '\n'.join(newFileLines)

        #newFileLines = np.array(newFileLines)

        #for name in newFileLines:
        #    print (name)

        np.savetxt('./dataRecap/T_' + str(T) + '_R_' + str(dataSet) + '.txt', newFileLines, fmt='%s')

#toRecapFormat()
#quit()

def toRevolverFormat():
    True

    T = 4
    #T = 1
    #dataSet = 9
    for dataSet in range(0, 20):
        print (dataSet)

        #baseline = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(b) + '_baseline.pt.npy')
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkSample.npz').astype(int)

        bulkTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkTrees.npz').astype(int)
        #trueTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_trees.npz').astype(int)

        if T == 1:
            bulkTrees[bulkTrees == 100] = -1
        else:
            bulkTrees[bulkTrees == 10] = -1
        bulkTrees = bulkTrees + 1
        bulkTrees = bulkTrees.astype(str)

        #unique1 = np.unique(bulkTrees)
        #unique1 = unique1.astype(int)
        #print (unique1)
        #quit()
        #for a in range(len(unique1)):
        #    bulkTrees[bulkTrees == unique1[a]] = 'M' + unique1[a]
        #bulkTrees[bulkTrees == 'M10'] = 'GL'
        #bulkTrees[bulkTrees == 'M100'] = 'GL'

        treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')
        #treeLength = np.zeros(int(np.max(sampleInverse))) + 5


        newFileLines = []

        numPatient = treeLength.shape[0] // 2
        numPatient = 499
        #numPatient = 50
        #numPatient = 10

        newFileLines.append(str(numPatient) + ' # patients')

        a = 0
        for b in range(0, numPatient):

            numTrees = 0
            a1 = a
            while sampleInverse[a1] == b:
                a1 += 1
                numTrees += 1

            newFileLines.append(str(numTrees) + ' #trees for patient ' + str(b + 1))

            treeSize = int(treeLength[b])
            while sampleInverse[a] == b:

                newFileLines.append(str(treeSize) + ' #edges')

                tree1 = bulkTrees[a]#[:treeSize]

                for c in range(treeSize):
                    newFileLines.append(tree1[c, 0] + ' ' +  tree1[c, 1])
                a += 1

        #newFileLines = '\n'.join(newFileLines)

        #newFileLines = np.array(newFileLines)

        #for name in newFileLines:
        #    print (name)
        #s0_k1_n499_M5_m5_revolver_output.txt
        #s0_k1_n50_M5_m5_simulations2
        #np.savetxt('./dataBaseline/revolver/input/'+ str(T) + '/s' + str(dataSet) + '_k1_n' + str(numPatient) + '_sim.txt', newFileLines, fmt='%s')
        #np.savetxt('./dataBaseline/revolver/input/'+ str(T) + '/T_' + str(T) + '_R_' + str(dataSet) + '.txt', newFileLines, fmt='%s')
        #quit()
        np.savetxt('./dataBaseline/revolver/input/'+ str(T) + '/s' + str(dataSet) + '_k1_n' + str(numPatient) + '_M7_m7_simulations.txt', newFileLines, fmt='%s')

#toRevolverFormat()
#quit()

def toGeneAccordFormat():


    T = 4
    #T = 1
    #dataSet = 9
    for dataSet in range(0, 2):
        print (dataSet)

        bulkTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkTrees.npz').astype(int)
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkSample.npz').astype(int)
        treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')
        #treeLength = np.zeros(int(np.max(sampleInverse))) + 5

        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_prob.npz')
        print (probabilityMatrix)
        #quit()

        patientNum1 = 0
        treeNums_0 = np.arange(sampleInverse.shape[0])
        for a in range(1, sampleInverse.shape[0]):
            if patientNum1 != sampleInverse[a]:
                treeNums_0[a] = treeNums_0[a-1] + 1
            else:
                treeNums_0[a] = 0

        _, counts = np.unique(sampleInverse, return_counts=True)
        freqs_0 = 1 / counts[sampleInverse]


        numPatient = np.unique(sampleInverse).shape[0]

        clonesAll = np.zeros((0, 10)).astype(int)
        patientAll = []
        treeNums = []
        freqs = []
        for treeNum in range(bulkTrees.shape[0]):
            patientName = 'AML-' + str(int(sampleInverse[treeNum])) + '-001'


            treeSize = treeLength[sampleInverse[treeNum]]
            tree1 = bulkTrees[treeNum, :treeSize].astype(int)
            mutToClone = np.zeros(12).astype(int)
            mutToClone[tree1[:, 1].astype(int)] = np.arange(treeSize)
            cloneRef = mutToClone[tree1[:, 0].astype(int)]
            clones = np.zeros((treeSize, 10)).astype(int)
            for a in range(treeSize):
                if tree1[a, 0] != 10:
                    clones[a] = np.copy(clones[cloneRef[a]])

                patientAll.append(patientName)
                treeNums.append(treeNums_0[a])
                freqs.append(freqs_0[a])

                clones[a, tree1[a, 1]] = 1

            clonesAll = np.concatenate((clonesAll, clones), axis=0)

        fullMatrix = np.zeros((1 + clonesAll.shape[0], 3 + clonesAll.shape[1])).astype(str)
        fullMatrix[0] = np.array(['case', 'tree', 'freq', 'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'])

        fullMatrix[1:, 3:] = clonesAll

        patientAll = np.array(patientAll)
        freqs = np.array(freqs)
        treeNums = np.array(treeNums)
        fullMatrix[1:, 0] = patientAll
        fullMatrix[1:, 1] = treeNums.astype(int).astype(str)
        fullMatrix[1:, 2] = freqs.astype(str)

        #np.savetxt('./GeneAccord/data/AMLsubclones_trees.txt', fullMatrix, delimiter=" ", fmt='%s')
        np.savetxt('./dataBaseline/GeneAccord/input/' + str(dataSet) + '_trees.txt', fullMatrix, delimiter=" ", fmt='%s')


#toGeneAccordFormat()
#quit()


def fromGeneAccord():

    errorList = []

    truePos_total = 0
    trueNeg_total = 0
    falsePos_total = 0
    falseNeg_total = 0

    for dataSet in range(20):

        data = np.load('./dataBaseline/GeneAccord/output/' + str(dataSet) + '.npy')
        continue1 = True
        a = 0
        while continue1:
            if data[a][:5] == '     ':
                if a != 0:
                    continue1 = False
            a += 1
        data = data[1:a-1]

        info = np.zeros((data.shape[0], 3))
        for a in range(len(data)):

            Mstring = data[a][5:13]
            Mstring = Mstring.split('_')
            Mstring[0] = Mstring[0][-1:]
            Mstring[1] = Mstring[1][1:2]

            assert data[a][37] != '-'
            delta1 = data[a][38:50]
            #print (data[a][37:50])
            delta1 = delta1.replace(' ', '')

            info[a, 0] = int(Mstring[0])
            info[a, 1] = int(Mstring[1])
            info[a, 2] = float(delta1)


        cutOff = 0.5

        causal = info[info[:, 2] > cutOff][:, :2]
        causal = causal.astype(int)

        causalMatrix = np.zeros((10, 10))
        causalMatrix[causal[:, 0], causal[:, 1]] = 1
        causalMatrix[causal[:, 1], causal[:, 0]] = 1

        causalPart = np.zeros((10, 10))
        causalPart[causal[:, 0], causal[:, 1]] = 1


        trueMatrix_0 = loadnpz('./dataNew/specialSim/dataSets/T_4_R_' + str(dataSet) + '_prob.npz')
        trueMatrix = np.zeros((10, 10))
        trueMatrix[:5, :5] = trueMatrix_0[:5, :5]
        trueMatrix[trueMatrix > 0] = 1

        #Adjustment for lower bound
        trueMatrix_sym = trueMatrix + trueMatrix.T
        trueMatrix_sym[trueMatrix_sym > 1] = 1

        same1 = trueMatrix_sym - causalMatrix
        causalMatrix[same1 == 0] = trueMatrix[same1 == 0]
        causalMatrix[same1 != 0] = causalPart[same1 != 0]


        truePos = np.argwhere(np.logical_and(trueMatrix == 1, causalMatrix == 1)).shape[0]
        trueNeg = np.argwhere(np.logical_and(trueMatrix == 0, causalMatrix == 0)).shape[0]
        falsePos = np.argwhere(np.logical_and(trueMatrix == 0, causalMatrix == 1)).shape[0]
        falseNeg = np.argwhere(np.logical_and(trueMatrix == 1, causalMatrix == 0)).shape[0]

        trueNeg = trueNeg - 10

        truePos_total += truePos
        trueNeg_total += trueNeg
        falsePos_total += falsePos
        falseNeg_total += falseNeg

        errorList.append([truePos, trueNeg, falsePos, falseNeg])



    print ('True Positives: ' + str(truePos_total))
    print ('True Negatives: ' + str(trueNeg_total))
    print ('False Positives: ' + str(falsePos_total))
    print ('False Negatives: ' + str(falseNeg_total))

    errorList = np.array(errorList)

    np.save('./plotResult/geneAccordCausal.npy', errorList)


#fromGeneAccord()
#quit()

def fromRecapFormat(filename):

    edges = []
    with open(filename) as f:

        start1 = False
        done = False
        while not done:

            try:
                line = f.readline().rstrip("\n")

                if line[-8:] == 'patients':
                    start1 = True

                if start1:
                    if not '#' in line:
                        if not 'dummy' in line.split(' '):


                            mut1 = line.split(' ')[0]
                            mut2 = line.split(' ')[1]
                            #print (line)

                            #print (mut1, mut2)

                            edges.append([mut1, mut2])

            except:

                done = True


    edges = np.array(edges)

    return edges

def testRecapPathway(revolver=False):

    edgeList = []
    errorList = []
    pathwayNumList = []

    true_pos_list = []
    false_pos_list = []
    false_neg_list = []

    totalError = 0
    totalPathway = 0
    for dataSet in range(30):

        if revolver:
            filename = './dataBaseline/revolver/output/1/s' + str(dataSet) + '_k1_n499_M5_m5_revolver_output.txt'
        else:
            filename = './dataRecap/T1_R' + str(dataSet) + '_solution.txt'

        edges = fromRecapFormat(filename)

        if revolver:
            edges = edges.astype(int) - 1
            edges = edges.astype(str)
            unique1 = np.unique(edges)
            for a in range(len(unique1)):
                edges[edges == unique1[a]] = 'M' + unique1[a]
            edges[edges == 'M-1'] = 'GL'


        #print (edges)
        #quit()

        edges = edges[edges[:, 0] != 'GL']

        vals1 = uniqueValMaker(edges)

        _, index, count = np.unique(vals1, return_index=True, return_counts=True)
        edges = edges[index]

        pathway = loadnpz('./data/specialSim/dataSets/T_1_R_' + str(dataSet) + '_pathway.npz', allow_pickle=True)

        pathwayEdges = []
        for a in range(len(pathway)):
            for b in range(len(pathway[a]) - 1):
                for c in range(len(pathway[a][b])):
                    for d in range(len(pathway[a][b+1])):
                        mut1 = pathway[a][b][c]
                        mut2 = pathway[a][b+1][d]

                        #print (mut1, mut2)
                        mut1 = 'M' + str(int(mut1))
                        mut2 = 'M' + str(int(mut2))

                        pathwayEdges.append([mut1, mut2])


        pathwayEdges = np.array(pathwayEdges)

        #countMin = 10
        #countMin = 15
        countMin = 20
        #countMin = 25
        #countMin = 30
        edges = edges[count >= countMin]



        vals1 = uniqueValMaker(np.concatenate((pathwayEdges, edges), axis=0))
        #print (type(vals1))

        pathway_vals = vals1[:pathwayEdges.shape[0]]
        edges_vals = vals1[pathwayEdges.shape[0]:]

        #print (pathway_vals.shape)
        #print (edges_vals.shape)

        inter_num = np.intersect1d(pathway_vals, edges_vals).shape[0]

        true_pos = inter_num#.shape[0]
        false_pos = edges_vals[np.isin(edges_vals, pathway_vals) == False].shape[0]
        false_neg = pathway_vals[np.isin(pathway_vals, edges_vals) == False].shape[0]

        true_pos_list.append(true_pos)
        false_pos_list.append(false_pos)
        false_neg_list.append(false_neg)


        error_num = pathway_vals.shape[0] + edges_vals.shape[0] - (2 * inter_num)

        #totalPathway += pathway_vals.shape[0]
        #totalError += error_num

        edgeList.append(pathway_vals.shape[0])
        errorList.append(error_num)
        pathwayNumList.append(len(pathway))

        #print (totalError, totalPathway)

    #print (pathwayNumList)

    x = np.array([errorList, edgeList, pathwayNumList, true_pos_list, false_pos_list, false_neg_list]).T

    if revolver:
        np.save('./plotResult/revolverPathway.npy', x)
    else:
        np.save('./plotResult/recapPathway.npy', x)


#testRecapPathway(revolver=False)
#quit()

def testRecapCausal(revolver=False):

    truePos_total = 0
    trueNeg_total = 0
    falsePos_total = 0
    falseNeg_total = 0

    errorList = []


    for dataSet in range(20):

        if revolver:
            #s0_k1_n499_M5_m5_revolver_output.txt


            #s0_k1_n499_M5_m5_revolver_output.txt
            filename = './dataBaseline/revolver/output/4/s' + str(dataSet) + '_k1_n499_M7_m7_revolver_output.txt'
        else:
            filename = './dataRecap/T4_R' + str(dataSet) + '_solution.txt'

        edges = fromRecapFormat(filename)

        if revolver:
            #print (edges[:10])
            edges = edges.astype(int) - 1
            edges = edges.astype(str)
            unique1 = np.unique(edges)
            for a in range(len(unique1)):
                edges[edges == unique1[a]] = 'M' + unique1[a]
            edges[edges == 'M-1'] = 'GL'

        edges = edges[edges[:, 0] != 'GL']

        for num in range(10):
            edges[edges == ('M' + str(num))] = num
        edges = edges.astype(int)

        vals1 = uniqueValMaker(edges)
        _, index, count = np.unique(vals1, return_index=True, return_counts=True)
        edges = edges[index]

        matrix = np.zeros((10, 10))
        matrix[edges[:, 0], edges[:, 1]] = count
        #matrix1 = np.zeros()

        countMin = 40
        matrix[matrix < countMin] = 0
        matrix[matrix >= countMin] = 1


        trueMatrix_0 = loadnpz('./dataNew/specialSim/dataSets/T_4_R_' + str(dataSet) + '_prob.npz')
        trueMatrix = np.zeros((10, 10))
        trueMatrix[:5, :5] = trueMatrix_0[:5, :5]
        trueMatrix[trueMatrix > 0] = 1
        #print (trueMatrix)

        truePos = np.argwhere(np.logical_and(trueMatrix == 1, matrix == 1)).shape[0]
        trueNeg = np.argwhere(np.logical_and(trueMatrix == 0, matrix == 0)).shape[0]
        falsePos = np.argwhere(np.logical_and(trueMatrix == 0, matrix == 1)).shape[0]
        falseNeg = np.argwhere(np.logical_and(trueMatrix == 1, matrix == 0)).shape[0]

        trueNeg = trueNeg - 10

        truePos_total += truePos
        trueNeg_total += trueNeg
        falsePos_total += falsePos
        falseNeg_total += falseNeg

        errorList.append([truePos, trueNeg, falsePos, falseNeg])


    print ('True Positives: ' + str(truePos_total))
    print ('True Negatives: ' + str(trueNeg_total))
    print ('False Positives: ' + str(falsePos_total))
    print ('False Negatives: ' + str(falseNeg_total))

    errorList = np.array(errorList)

    if revolver:
        np.save('./plotResult/revolverCausal.npy', errorList)
    else:
        np.save('./plotResult/recapCausal.npy', errorList)




#testRecapCausal(revolver=True)
#quit()


def processTreeData(maxM, fileIn, fullDir=False):

    #This processes the data from a form using lists of trees to
    #a form using numpy tensors.
    #It also processes the mutation names into mutation numbers.

    if fileIn in ['./dataNew/manualCancer.npy', './dataNew/breastCancer.npy']:
        treeData = np.load(fileIn, allow_pickle=True)
    else:

        if fullDir:
            treeData = np.load(fileIn + '.npy', allow_pickle=True)
        else:
            treeData = np.load('./dataNew/customData/' + fileIn + '.npy', allow_pickle=True)
        #print (treeData[0])
        #print (len(treeData))

        #print (treeData[0])
        #print (treeData[1])
        #print (treeData[2])
        #print (treeData[3])
        #print (treeData[4])
        #quit()




    MVal = 100

    sampleInverse = np.zeros(100000)
    treeLength = np.zeros(100000)
    newTrees = np.zeros((100000, maxM, 2)).astype(str)
    lastName = 'ZZZZZZZZZZZZZZZZ'
    firstName = 'ZZZZZZZZZZ'
    newTrees[:] = lastName

    count1 = 0
    for a in range(0, len(treeData)):
        treeList = treeData[a]
        treeList = np.array(list(treeList))

        #print (treeList)
        #quit()

        if treeList.shape[1] <= maxM:
            size1 = treeList.shape[0]

            #print (treeList)

            newTrees[count1:count1+size1, :treeList.shape[1]] = treeList
            treeLength[count1:count1+size1] = treeList.shape[1]
            sampleInverse[count1:count1+size1] = a
            count1 += size1






    newTrees = newTrees[:count1]
    newTrees[newTrees == 'Root'] = 'GL'
    if ('0' in newTrees) and not ('GL' in newTrees):
        newTrees[newTrees == '0'] = firstName
    else:
        newTrees[newTrees == 'GL'] = firstName
    treeLength = treeLength[:count1]
    sampleInverse = sampleInverse[:count1]
    shape1 = newTrees.shape
    newTrees = newTrees.reshape((newTrees.size,))




    #uniqueMutation =  np.unique(newTrees)
    #for name in uniqueMutation:
    #    name1 = name.split('_')[0]
    #    #print (name, name1)
    #    newTrees[newTrees == name] = name1
    #quit()


    uniqueMutation, newTrees = np.unique(newTrees, return_inverse=True)


    uniqueMutation2 = []
    for name in uniqueMutation:
        name1 = name.split('_')[0]
        uniqueMutation2.append(name1)
    uniqueMutation2 = np.array(uniqueMutation2)
    uniqueMutation2, mutationCategory = np.unique(uniqueMutation2, return_inverse=True)

    #print (uniqueMutation2)
    #quit()

    #uniqueMutation2 = uniqueMutation2[:-2]

    if not fullDir:

        if fileIn == './dataNew/manualCancer.npy':
            np.save('./dataNew/mutationNames.npy', uniqueMutation)
            np.save('./dataNew/categoryNames.npy', uniqueMutation2[:-2])
        elif fileIn == './dataNew/breastCancer.npy':
            #print ("Hi")
            #print (len(uniqueMutation))
            np.save('./dataNew/mutationNamesBreast.npy', uniqueMutation)
            #np.save('./data/mutationNamesBreastLarge.npy', uniqueMutation)
            True
        else:
            np.save('./dataNew/mutationNames_' + fileIn + '.npy', uniqueMutation)
            np.save('./dataNew/categoryNames_' + fileIn + '.npy', uniqueMutation2)


    #print (uniqueMutation)
    #print (uniqueMutation2)
    #quit()
    newTrees = newTrees.reshape(shape1)
    M = uniqueMutation.shape[0] - 2

    #print (M)
    #quit()

    if (lastName in uniqueMutation) and (lastName != uniqueMutation[-1]):
        print ("Error in Mutation Name")
        quit()


    return newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M


def doChoice(x):

    #This is a simple function that selects an option from a probability distribution given by x.


    x = np.cumsum(x, axis=1) #This makes the probability cummulative

    #x = x / x[:, -1].reshape((-1, 1))

    #if np.min(x[:, -1]) != 1:
    #    print (x[:, -1])
    #    print (np.min(x[:, -1]))
    #    quit()

    #assert np.min(x[:, -1]) >= 0.9999
    #assert np.max(x[:, -1]) <= 1.0001

    #print (x[:, -1])
    #quit()

    randVal = np.random.random(size=x.shape[0])
    randVal = randVal.repeat(x.shape[1]).reshape(x.shape)

    x = randVal - x
    x2 = np.zeros(x.shape)
    x2[x > 0] = 1
    x2[x <= 0] = 0
    x2[:, -1] = 0

    x2 = np.sum(x2, axis=1)

    return x2

def transformManualData():

    #This code processes the leukemia data which I entered manually into a usable format.
    #This code does not need to be understood by the user, since it is related to
    #the details of how I manually recorded the data.

    def doUniqueEdge(tree1):
        tree1 = np.array(tree1)
        edges2 = []
        for b in range(0, tree1.shape[0]):
            edges2.append(str(tree1[b][0]) + ':' + str(tree1[b][1]) )
        edges2 = np.array(edges2)
        edges2, index = np.unique(edges2, return_index=True)
        tree1 = tree1[index]
        return tree1


    #print ("Hi")
    #data = np.loadtxt('./data/manualCan1.txt', dtype=str)

    file1 = open('./data/manualCan3.txt', 'r')
    data1 = file1.readlines()
    file2 = open('./data/manualCan2.txt', 'r')
    data2 = file2.readlines()
    file3 = open('./data/manualCan1.txt', 'r')
    data3 = file3.readlines()
    data1 = data1 + data2 + data3

    fullTrees = []
    patientsDone = []
    treeNow = []
    patientTrees = []
    treeDict = {}
    treeNum = -1
    doingData = False
    toAddTree = False
    isNumBefore = False
    for a in range(0, len(data1)):
        string1 = data1[a]
        string1 = string1.replace('\n', '')
        string1 = string1.replace(']', '')
        string1 = string1.replace('[', '')
        string1 = string1.replace(' ', '')
        string1 = string1.replace('’', '')
        string1 = string1.replace("'", '')
        string1 = string1.replace("‘", '')


        #print (string1)

        isData = False
        isNum = False
        isSpace = False
        if ',' in string1:
            isData = True
        try:
            int(string1)
            isNum = True
            doingData = False
        except:
            True

        if string1 == '':
            isSpace = True
            doingData = False

        #print (string1)


        if (isData == False) and (isNum == False) and (isSpace == False):
            print ("ERROR")
            quit()


        if isNum:
            treeNum = int(string1)
            if not (treeNum in patientsDone):
                toAddTree = True
                patientsDone.append(treeNum)
            else:
                toAddTree = False

        #print (patientsDone)
        #print (toAddTree)

        if toAddTree:


            if isData:
                if doingData == False:
                    #Add tree!!
                    if a != 1:
                        #print (treeNow)

                        treeNow = doUniqueEdge(treeNow)

                        patientTrees.append(copy.deepcopy(treeNow))
                        treeNow = []
                        treeDict = {}

                doingData = True
                locData = string1.split(',')
                #print (locData)
                for b in range(0, len(locData) - 1):

                    if not (locData[b+1] in treeDict.keys()):
                        treeDict[locData[b+1]] = locData[b]
                    else:
                        if treeDict[locData[b+1]] != locData[b]:
                            print ("Issue")
                            print (treeDict[locData[b+1]])
                            print (locData[b+1], locData[b])
                            quit()


                    edge = [locData[b], locData[b+1]]
                    treeNow.append(copy.copy(edge))

        #if len(patientsDone) == 1:
        #    print (string1)

        if isNumBefore:
            #if a != 0:
            #print (patientTrees)
            #quit()
            if False:#len(patientsDone) == 2:
                print ("Hi", treeNum)
                print (a, len(fullTrees))

                print ('patientTrees ', len(patientTrees))
                print (np.unique(np.array(patientTrees[0])))
                print (np.unique(np.array(patientTrees[1])))
                print (np.unique(np.array(patientTrees[2])))
                #print (np.unique(np.array(patientTrees[0])))
                #print (np.unique(np.array(patientTrees[1])))
                #quit()
                print ('')
                print ('')
                print ('')
                #print
            fullTrees.append(copy.deepcopy(patientTrees))
            patientTrees = []

        if toAddTree and (a != 0):
            isNumBefore = isNum

        #if a == len(data1) - 1:
        #    print ("Info")
        #    print (string1)

    patientTrees.append(copy.deepcopy(treeNow))
    fullTrees.append(copy.deepcopy(patientTrees))

    #print (len(fullTrees))
    #print (len(patientsDone))
    #quit()
    numberOfTrees = 0

    mutAll1 = np.array([])
    treeNumAll = np.array([])
    for a in range(0, len(fullTrees)):

        mutAll0 = np.array([])
        N1 = len(fullTrees[a])
        numberOfTrees += N1
        mutFull = []
        for b in range(0, N1):
            #print (fullTrees[a][b])
            #quit()
            mutations = []
            for c in range(0, len(fullTrees[a][b])):
                mutations.append(fullTrees[a][b][c][0])
                mutations.append(fullTrees[a][b][c][1])
            mutations = np.array(np.unique(mutations))
            mutAll0 = np.concatenate((mutAll0, mutations ))
            mutFull.append(np.copy(mutations))

        for b in range(0, N1):
            for c in range(0, N1):
                s1, s2, s3 = mutFull[b].shape[0], mutFull[c].shape[0], np.intersect1d(mutFull[b], mutFull[c]).shape[0]
                if (s3 != s1) or (s3 != s2):

                    print (a, patientsDone[a])
                    #print (b, c)
                    print (s1, s2, s3)
                    print (mutFull[b][np.isin(mutFull[b], mutFull[c]) == False])
                    print (mutFull[c][np.isin(mutFull[c], mutFull[b]) == False])
                    print ("Issue2")
                    quit()

        #mutAll0 = np.unique(mutAll0)
        if mutAll0.shape[0] == 0:
            print (a)
            print ("Issue3")
            quit()

        if 'Root' not in mutAll0:
            print ("Issue4")
            quit()
        mutAll1 = np.concatenate((mutAll1, mutAll0))
        treeNumAll = np.concatenate((treeNumAll, np.zeros(mutAll0.shape[0]) + patientsDone[a] ))

    #unique1, indices1, counts1 = np.unique(mutAll1, return_counts=True, return_index=True)
    #for a in range(0, unique1.shape[0]):
    #    print (unique1[a], counts1[a], treeNumAll[indices1[a]])
    patientsDone = np.array(patientsDone).astype(int)
    fullTrees2 = []
    for a in range(0, len(patientsDone)):
        #print (a+1)
        #print ( np.argwhere(patientsDone == (a+1) ))
        #print (np.unique(patientsDone))
        arg1 = np.argwhere(patientsDone == (a+1) )[0, 0]
        fullTrees2.append(copy.deepcopy(fullTrees[a]))

    print (numberOfTrees)

    np.save('./data/manualCancer.npy', fullTrees2)

def saveTreeList(fileIn, fileOut, doSolution=False):

    #This code converts the format of the tree data to a more usable format by my algorithm.
    #This does not need to be understood if one's data is already in a usable format.

    dataList = []
    patientIdx = 0
    #for filename in ['./breast_Razavi.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k1_n50_M5_m5_simulations.txt']:
    #for filename in ['./data/M5_m5/simulations_input/s0_k2_n100_M5_m5_simulations.txt']:
    for filename in [fileIn]:
        with open(filename) as f:

            if doSolution:
                notPatient = True
                while notPatient:
                    line = f.readline().rstrip("\n")
                    if line[-8:] == 'patients':
                        notPatient = False

            else:
                line = f.readline().rstrip("\n")

            #print (line)

            numPatient = int(line.split()[0])

            for loop1 in range(0, numPatient):
                #print (loop1)
                #for loop1 in range(0, 3):
                treeList = []

                if doSolution:
                    line = f.readline().rstrip("\n")

                line = f.readline().rstrip("\n")

                #print (line)

                #quit()

                #print ("Base Line")
                #print (line)

                #if loop1:
                #    print (line)
                #    quit()

                if line == "":
                    continue

                numTrees = int(line.split()[0])

                if doSolution:
                    numTrees = 1


                #if numTrees > 10000:
                #    continue
                #if numTrees == 0:
                #    continue



                #print (numTrees)

                line2 = line
                ok = True
                for treeIdx in range(numTrees):
                    #print ("Base Tree")
                    line = f.readline().rstrip("\n")

                    assert 'edges' in line

                    numEdges = int(line.split()[0])

                    #print (numEdges)

                    #print (numEdges)
                    if numEdges == 0:
                        ok = False
                        continue
                    #if treeIdx == 0:
                    #    print (line2, "for patient", os.path.basename(filename).rstrip(".trees"))

                    #print (numEdges + 1, "#edges, tree", treeIdx)
                    tree = []
                    vertices = set([])
                    inner_vertices = set([])
                    for edgeIdx in range(numEdges):
                        line = f.readline().rstrip("\n")
                        #print (line)
                        s = line.split()
                        tree.append([s[0].split(":")[-1], s[1].split(":")[-1]])
                        vertices.add(tree[-1][0])
                        vertices.add(tree[-1][1])
                        inner_vertices.add(tree[-1][1])

                    if len(set.difference(vertices, inner_vertices)) != 1:
                        print (vertices)
                        print (inner_vertices)
                        print (set.difference(vertices, inner_vertices))
                    assert len(set.difference(vertices, inner_vertices)) == 1
                    root = list(set.difference(vertices, inner_vertices))[0]
                    #for edge in tree:
                    #    print (edge[0], edge[1])
                    #print ("GL", root)

                    treeList.append(tree)
                if ok:
                    patientIdx += 1


                dataList.append(treeList)


    #####np.save('./treeData.npy', dataList)
    #####np.save('./treeDataSim2.npy', dataList)

    dataList1 = np.array(dataList, dtype=object)

    np.save(fileOut, dataList1)

def saveTreeListSim():
    import os

    #This converts the format of the data for all of the RECAP simulations data.

    names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    for name in names1[:1]:
        arr = os.listdir('./data/' + name + '/simulations_input')
        for name2 in arr:
            fileIn = './data/' + name + '/simulations_input/' + name2
            fileOut = './data/p_' + name + '/simulations_input/' + name2
            saveTreeList(fileIn, fileOut)

#saveTreeListSim()
#quit()

def saveAllTrees(N):

    #This code saves all of the trees of size N.
    #This list of trees can then be used to find all
    #trees of size N that fit some measurement data.

    #First an initial set of trees of length 0 is created.
    L = 101
    trees = np.zeros((1, N+1, 3)).astype(int)
    trees[:] = L - 1

    for a in range(N):

        #Next, the set of trees is expanded by adding every possible next node.
        size1 = a+1
        trees2 = trees.reshape((trees.size,)).repeat(N*size1)
        trees2 = trees2.reshape((trees.shape[0], trees.shape[1], trees.shape[2], N, size1 ))

        for b in range(size1):
            trees2[:, size1, 0, :, b] = np.copy(trees2[:, b, 1, :, 0])
            trees2[:, size1, 2, :, b] = b
        for b in range(N):
            #print (b)
            trees2[:, size1, 1, b, :] = b
            #trees2[:, size1, 1, :, b] = trees2[:, a, 0, :, b]

        trees2 = np.swapaxes(trees2, 1, 3)
        trees2 = np.swapaxes(trees2, 2, 4)

        trees2 = trees2.reshape((trees.shape[0] * N * size1, trees.shape[1], trees.shape[2] ))


        #Then duplicate mutations are eliminated, so each tree has at most one of each mutation.
        tree_end = np.copy(trees2[:, :a+2, 1])
        tree_end = np.sort(tree_end, axis=1)
        tree_end = tree_end[:, 1:] - tree_end[:, :-1]
        tree_end = np.min(np.abs(tree_end), axis=1)
        trees2 = trees2[tree_end != 0]






        #This reduces the trees to the set of unique trees.
        #Specifically, some of the trees prior to this are just different representations of
        #the same tree. For instance, [(0, 1), (0, 2)] and [(0, 2), (0, 1)] are different
        #representations of the same tree.
        #It does this by comparising the sets of edges and making sure the sets of edges
        #are different.
        trees2_edgenum = (trees2[:, :, 0] * L) + trees2[:, :, 1]
        trees2_edgenum = np.sort(trees2_edgenum, axis=1)

        trees2_str = trees2_edgenum.astype(str)
        trees2_str_sep = np.zeros(trees2_str.shape).astype(str)
        trees2_str_sep[:] = ':'

        trees2_str = np.char.add(trees2_str, trees2_str_sep)

        trees2_str_full = np.zeros(trees2_str.shape[0]).astype(str)
        trees2_str_full[:] = ''
        for b in range(trees2_str.shape[1]):
            trees2_str_full = np.char.add(trees2_str_full, np.copy(trees2_str[:, b]))

        _, uniqueArgs = np.unique(trees2_str_full, return_index=True)
        trees2 = trees2[uniqueArgs]

        trees = np.copy(trees2)

    trees = trees[:, 1:]

    #This gives the clones that are formed by these trees.
    clones = np.zeros((trees.shape[0], N+1, N))

    for a in range(N):

        clonesNow = clones[np.arange(trees.shape[0]), trees[:, a, 2] ]
        clonesNow[np.arange(trees.shape[0]), trees[:, a, 1]] = 1
        clones[:, a+1] = np.copy(clonesNow)

    clones = clones[:, 1:]

    #print (clones.shape[0])

    np.savez_compressed('./data/allTrees/edges/' + str(N) + '.npz', trees)
    np.savez_compressed('./data/allTrees/clones/' + str(N) + '.npz', clones)

def bulkFrequencyPossible(freqs, M, specialM=101):

    #This code determines the set of trees that are consistent with bulk frequency measurements.
    #It does this by determining whether or not a tree is constistent with the frequency measurement
    #for every possible tree of the correct length.

    clones = loadnpz('./extra/allTrees/clones/' + str(M) + '.npz')

    clones_0 = np.copy(clones)
    clones = np.swapaxes(clones, 1, 2)
    clones_shape = clones.shape

    #print (clones)


    freq_shape = freqs.shape

    freqs = freqs.reshape((freq_shape[0]*freq_shape[1], freq_shape[2]))
    freqs = freqs.T


    clones = np.linalg.inv(clones)

    clones = clones.reshape((clones_shape[0] * clones_shape[1], clones_shape[2]))

    #This matrix multiplication determines the mixture of clones that would be required to
    #produce these frequency measurements.
    mixture = np.matmul(clones, freqs)


    mixture = mixture.reshape((clones_shape[0], clones_shape[1], freq_shape[0], freq_shape[1]))
    #mixture = mixture.reshape((clones_shape[0], clones_shape[1], freq_shape[1] * freq_shape[0]))



    mixture_sum = np.sum(np.copy(mixture), axis=1)
    mixture_sum = np.max(mixture_sum, axis=2).T

    mixture = np.min(mixture, axis=1)
    mixture = np.min(mixture, axis=2).T


    #If mixture < 0, the this implies one would need a negative qualtity of some clone
    #in order to create the proper mixture, which is physically impossible.
    #If mixture_sum > 1, it implies that the solution requires to total percentage of all
    #of the included clones to be over 100%, which is physically imposisble.
    #If neither of these conditions are tree, then there is a valid mixture of clones that
    #gives rise to the bulk frquency measurement.
    ep = 1e-6
    argsGood = np.argwhere(np.logical_and(mixture >= 0 - ep, mixture_sum <= 1 + ep))

    #This gives information of which sample each possible tree corresponds to.
    sampleInverse = np.copy(argsGood[:, 0])

    trees = loadnpz('./extra/allTrees/edges/' + str(M) + '.npz')
    trees = trees[argsGood[:, 1]][:, :, :2]

    #print (trees[0])
    #quit()

    if specialM != 101:
        trees[trees == 100] = specialM - 1


    return trees, sampleInverse

def simulationBulkFrequency(clones, M, S, specialM=101):

    #This function takes in clones, a the number of mutations, and the number
    #of samples, and (1) finds frequency measurements for those clones and
    #(2) calculates the set of possible trees for those measurements.

    #print (clones.shape)
    #print (np.mean(clones, axis=(0, 1)))
    #quit()

    if True:
        clones_sum = np.sum(clones, axis=1)
        clones_argsort = np.argsort(clones_sum, axis=1)
        clones_argsort = clones_argsort[:, -1::-1]


    #clones_sum = np.sum(clones, axis=1)
    #print (clones_sum[:10])
    #clones_sum[clones_sum >= 1] = -1
    #clones_argsort = np.argsort(clones_sum, axis=1)

    #print (clones_sum[:10])
    #print (clones_argsort[:10])
    #quit()


    #print (np.unique(clones_argsort))
    #quit()

    allArgs = np.argwhere(clones > -100)
    allArgs[:, 2] = clones_argsort[allArgs[:, 0], allArgs[:, 2]]


    clones_flat = clones[allArgs[:, 0], allArgs[:, 1], allArgs[:, 2]]
    clones = clones_flat.reshape(clones.shape)



    clones = clones[:, 1:, :M]

    N = clones.shape[0]

    clones_shape = clones.shape
    clones = clones.reshape((clones.size,))
    clones = clones.repeat(S)
    clones = clones.reshape((clones_shape[0], clones_shape[1] * clones_shape[2], S))
    clones = np.swapaxes(clones, 1, 2)
    clones = clones.reshape((clones_shape[0] * S, clones_shape[1], clones_shape[2]))

    #freqs gives random sampling from the clones in the phylogeny tree.
    freqs = np.random.random((clones.shape[0], clones.shape[1]))
    freqs_sum = np.sum(freqs, axis=1).repeat(clones.shape[1]).reshape(freqs.shape)
    freqs = freqs / freqs_sum
    freqs = freqs.reshape((freqs.size,)).repeat(clones.shape[2]).reshape(clones.shape)
    freqs = np.sum(freqs * clones, axis=1) #This computation determines the frequency of
    #different mutations based on the proportion of each clone.

    #print (freqs.shape)
    #print (np.mean(freqs, axis=0))
    #print (clones.shape)
    #print (np.mean(clones, axis=(0, 1)))
    #quit()


    freqs = freqs.reshape((N, S, M))

    trees, sampleInverse = bulkFrequencyPossible(freqs, M, specialM=specialM) #This calculates the possible trees from the bulk frequency measurement.

    sampleInverse_reshape = sampleInverse.repeat(trees.shape[1]).repeat(trees.shape[2]).reshape(trees.shape)


    trees_copy = np.copy(trees)
    trees[trees == specialM-1] = -1
    trees = clones_argsort[sampleInverse_reshape, trees]
    trees[trees_copy == specialM-1] = specialM-1


    return trees, sampleInverse

def multisizeBulkFrequency(clones, treeSizes, S, specialM=101):

    #This simulates bulk frequency measurements and determining the set of possible trees
    #when the number of mutations is different for different patients.
    #Specifically, it seperates it into subsets of patients with the same number
    #of mutations, and calculates it for these subsets. Then it combines this data and
    #reformats as needed.

    uniqueSizes = np.unique(treeSizes)

    maxSize = int(np.max(uniqueSizes))
    fullTree = np.zeros((0, maxSize, 2))
    fullSampleInverse = np.zeros(0)

    for a in range(len(uniqueSizes)):

        argSize = np.argwhere(uniqueSizes[a] == treeSizes)[:, 0]

        #
        #print (treeSizes[:10])
        #print (argSize[:10])
        #quit()

        clonesNow = clones[argSize]

        clonesNow[:, int(uniqueSizes[a]+1):] = 0
        #print (clonesNow.shape)
        #quit()


        #print (clonesNow[0])
        #print (uniqueSizes[a])

        treesNow, sampleInverseNow = simulationBulkFrequency(clonesNow, uniqueSizes[a], S, specialM=specialM)

        #if uniqueSizes[a] == 6:
        #    print (clonesNow[0])
        #    print (treesNow[0])
        #    quit()
        #print (treesNow[0])
        #quit()

        treesNow2 = np.zeros((treesNow.shape[0], maxSize, 2))
        treesNow2[:] = specialM
        treesNow2[:, :treesNow.shape[1], :] = treesNow
        fullTree = np.concatenate((fullTree, np.copy(treesNow2)))

        sampleInverseNow = argSize[sampleInverseNow.astype(int)]

        fullSampleInverse = np.concatenate((fullSampleInverse, np.copy(sampleInverseNow)))

    fullSampleInverse_argsort = np.argsort(fullSampleInverse)

    fullTree = fullTree[fullSampleInverse_argsort]
    fullSampleInverse = fullSampleInverse[fullSampleInverse_argsort]


    return fullTree, fullSampleInverse

def makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize, maxLogProb=1000, useRequirement=False, reqMatrix=False, doNonClone=False, nonCloneMatrix=False):

    #This simulates cancer evolution given a a set of assumed properties.
    #It assumes there are different mutation types and the probability of a new mutation
    #is depend on the existing mutation types in the clone and the mutation type of the new mutation.
    #Additionally, it assumes linear effects on the log probability from each new mutation type.
    #This code can be used to simulate many different types of possible patterns in cancer evolution since
    #it is very general.



    #N = 100
    clones = np.zeros((N, treeSize+1, M))

    edges = np.zeros((N, treeSize+1, 2)).astype(int)
    edges[:] = M

    for a in range(0, treeSize):

        treePos = a + 1

        clonesNow = clones[:, :treePos]
        clonesNow = clonesNow.reshape((N*treePos, M))

        if doNonClone:
            interfere = np.matmul(clonesNow, nonCloneMatrix)
            interfere_size1 = interfere.shape[1]
            interfere = interfere.reshape((N, treePos, interfere_size1))
            interfere = np.mean(interfere, axis=1)
            interfere = interfere.repeat(treePos).reshape((N, interfere_size1, treePos))
            interfere = np.swapaxes(interfere, 1, 2)
            interfere = interfere.reshape((N*treePos, interfere_size1))


        #if a == 5:
        #    print (clonesNow[:20])

        if a == treeSize - 1:
            True
            #print (np.round(clonesNow[:20]))

        #This computation determines which mutation types are present in the clone.
        clonesNow = np.matmul(clonesNow, mutationTypeMatrix_extended)
        clonesNow[:, K] = 1
        clonesNow[clonesNow > 1] = 1

        #if a == 5:
        #    print (clonesNow[:20])
        #    print (reqMatrix)



        if useRequirement:
            #if "useRequirement" is true, this makes it so effects of mutations work by satisfying mutation requirements of effects,
            #rather than working addatively. In other words, having two mutations of the same effect won't do anything additional if one is satisfactory.
            #However, having two mutations that are required together won't do anything if they are not both together.

            clonesNow = np.matmul(clonesNow, reqMatrix)
            clonesNow[clonesNow < 1] = 0
            clonesNow[clonesNow >= 1] = 1

        #if a == 5:
        #    print (clonesNow[:20])
        #    print ("A")
        #    print (probabilityMatrix)
        #    print ("AB")
        #    #quit()





        if a == treeSize - 1:
            True
            #print (np.round(clonesNow[:20]))

        #This calculates the probability of mutations of certian types
        #given which types of mutations already exist on the clone.
        clonesNow = np.matmul(clonesNow, probabilityMatrix)
        clonesNow[clonesNow > maxLogProb] = maxLogProb
        #This gives the probability of new mutations given the probability of mutation types,
        #and which mutations are of each mutation type.


        if a == treeSize - 1:
            True
            #print (np.round(clonesNow[:20]*10))
            #quit()


        #rint (clonesNow.shape)
        #print (mutationTypeMatrix.shape)
        #quit()

        clonesNow = np.matmul(clonesNow, mutationTypeMatrix.T)




        if doNonClone:
            clonesNow = clonesNow + interfere



        clonesNow = clonesNow.reshape((N, treePos*M))

        #print (clonesNow[0])
        #quit()

        clonesNow_max = np.max(clonesNow, axis=1)
        clonesNow_max = clonesNow_max.repeat(clonesNow.shape[1]).reshape(clonesNow.shape)
        clonesNow = clonesNow - clonesNow_max
        clonesNow = np.exp(clonesNow)

        #print (clonesNow[0])
        #quit()

        clonesNow = clonesNow.reshape((N, treePos, M))
        for b in range(a):
            for c in range(a+1):
                clonesNow[np.arange(N), c, edges[:, b+1, 1]] = 0
        clonesNow = clonesNow.reshape((N, treePos*M))

        #This modifies the probabilities, given that it is impossible to have the same
        #mutation occur multiple times in one patient.
        clonesNow_sum = np.sum(clonesNow, axis=1)
        clonesNow_sum = clonesNow_sum.repeat(clonesNow.shape[1]).reshape(clonesNow.shape)
        clonesNow = clonesNow / clonesNow_sum


        #This randomly chooses which mutation to be added next given the probability of each
        #new mutation being added.
        choicePoint = doChoice(clonesNow)


        #quit()

        #This determines which clone the mutation will be added to
        clonePoint = (choicePoint // M).astype(int)
        #This determnes which mutation will be added to the clone.
        mutPoint = (choicePoint % M).astype(int)

        #print (np.unique(mutPoint, return_counts=True))

        #This adds to the tree the fact that mutation mutPoint is added to the clone clonePoint
        edges[:, a+1, 1] = np.copy(mutPoint)
        edges[:, a+1, 0] = np.copy(edges[np.arange(N), clonePoint, 1])

        #This adds the new clone to the list of clones.
        cloneSelect = np.copy(clones[np.arange(N), clonePoint])
        cloneSelect[np.arange(N), mutPoint] = 1
        clones[:, a+1] = np.copy(cloneSelect)

    #This remove the trivial edge associated with the clone with no mutations.
    edges = edges[:, 1:]


    #print (np.mean(clones[:, :6], axis=(0, 1)))
    #quit()

    #quit()

    return edges, clones

def makeSimulation():


    #This runs a simple simulation of causal relationships.

    S = 3

    M = 10
    K = 6

    #This makes it so there are 10 total mutations, 5 of which are "interesting"
    #mutations of there own type. The remaining 5 are "boring" and are all considered the same type of mutation.
    mutationType = np.arange(M)
    mutationType[mutationType >= K] = K - 1

    mutationTypeMatrix_extended = np.zeros((M, K+1))
    mutationTypeMatrix_extended[np.arange(M), mutationType] = 1

    mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])

    #This gives random causal relationships between all of the interesting mutations.
    #Note, the causal relationship from mutation A to mutation B is independent
    #of the relationship from B to A.
    probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))
    probabilityMatrix = (probabilityMatrix - 0.5) * 6

    probabilityMatrix[K] = 0
    probabilityMatrix[K-1, :] = 0
    probabilityMatrix[:, K-1] = 0


    #This makes sure the trees are not larger than the number of mutations. Otherwise,
    #it sets the tree size to a default value of 5.
    treeSize = min(5, M)
    N = 1000

    #This runs the simulation given the causal relationships, mutations, and mutation types.
    edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)


    np.savez_compressed('./data/specialSim/temp_trees_0.npz', edges)
    np.savez_compressed('./data/specialSim/temp_mutationType_0.npz', mutationType)
    np.savez_compressed('./data/specialSim/temp_prob_0.npz', probabilityMatrix)

    #This runs bulk frequency measurements on the clonal data.
    trees, sampleInverse = simulationBulkFrequency(clones, treeSize, S)

    np.savez_compressed('./data/specialSim/temp_bulkTrees_0.npz', trees)
    np.savez_compressed('./data/specialSim/temp_bulkSample_0.npz', sampleInverse)

def makeOccurSimulation():

    for a in range(0, 20):

        #This runs a simple simulation of causal relationships.

        #np.random.seed(1)

        print (a)

        #S 5, min(7, M), T = 4.

        #S = 3
        S = 5 #5 samples in the bulk frequency sampling

        M = 10 #10 total mutations
        K = 6 #6 types of mutations. 5 interesting mutations, and the remaining 5 mutations are "boring" mutations.
        T = 14#7 Sept 1 2022

        if T in [8, 11, 13, 14]:
            M = 10

        if T in [9, 12]:
            M = 15


        #This makes it so there are 10 total mutations, 5 of which are "interesting"
        #mutations of there own type. The remaining 5 are "boring" and are all considered the same type of mutation.
        if T in [8, 11]:
            mutationType = np.arange(M) // 2#5
        elif T in [9, 12]:
            mutationType = np.arange(M) // 3#5
        else:
            mutationType = np.arange(M)
            mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
        #mutationTypeMatrix_extended[:, K] = 1

        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])

        #This gives random causal relationships between all of the interesting mutations.
        #Note, the causal relationship from mutation A to mutation B is independent
        #of the relationship from B to A.

        if T in [10, 11, 12]:
            probabilityMatrix = np.random.randint(3, size=(K+1) *  K).reshape((K+1, K)) - 1
        else:
            probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))

        nonCloneMatrix = False
        if T == 14:
            nonCloneMatrix = np.zeros((M, M))
            nonCloneMatrix[:5, :5] = probabilityMatrix[:5, :5] * np.log(2)


        probabilityMatrix = probabilityMatrix * np.log(11)


        probabilityMatrix[K] = 0
        probabilityMatrix[K-1, :] = 0
        probabilityMatrix[:, K-1] = 0


        if T == 13:
            nonCloneMatrix = np.random.randint(3, size=M*M).reshape((M, M)) - 1
            nonCloneMatrix = nonCloneMatrix * np.log(2)


        #print (nonCloneMatrix.shape)
        #quit()


        #probabilityMatrix = np.zeros((K+1, K))
        #probabilityMatrix[:, :3] = 10

        #probabilityMatrix[:, :K-1] = probabilityMatrix[:, :K-1] + 4

        #This makes sure the trees are not larger than the number of mutations. Otherwise,
        #it sets the tree size to a default value of 7.
        treeSize = min(7, M) #7

        if T in [8]:#, 11]:
            N = 400
        elif T in [9]:#, 12]:
            N = 600
        elif T == 10:
            N = 400
        elif T == 11:
            N = 800
        elif T == 12:
            N = 1200
        elif T in [13, 14]:
            N = 400

        else:
            N = 200

        #N = 100

        if T in [13, 14]:
            doNonClone = True

        #This runs the simulation given the causal relationships, mutations, and mutation types.
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize, doNonClone=doNonClone, nonCloneMatrix=nonCloneMatrix)


        #This randomizes the tree sizes to be between 5 and 7.
        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5
        #treeSizes[:] = 5 #Temp oct 13
        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M + 1
            clones[b, size1+1:] = 0

        #print (edges[0])

        if doNonClone:
            np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_interfereProb.npz', nonCloneMatrix)

        #'''
        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz', treeSizes)
        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_prob.npz', probabilityMatrix)
        #'''
        #print (edges[0])
        #print (clones[0])



        #This runs bulk frequency measurements on the clonal data.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        #for c in range(100):
        #    print ('')
        #    print (trees[c])
        #quit()


        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz', sampleInverse)

        #print (sampleInverse[:10])

        #print (sampleInverse[0])
        #print (clones[0])
        #print ('')
        #print (edges[0])
        #print (trees[0])
        #print (trees[1])
        #print (trees[2])
        print ("Saved")
        #quit()

#makeOccurSimulation()
#quit()


def makeNewOccur():

    for a in range(0, 1):

        #This runs a simple simulation of causal relationships.

        #T = 3
        #T = 0

        print (a)

        #folder1 = 'passanger'
        folder1 = 'negativeInter'
        #folder1 = 'lowPatient'
        #folder1 = 'fewSamples'
        #T = 2
        T = 3

        #numInter = 3
        #numInter = 1
        #numInter = 5
        numInter = 10

        if folder1 == 'negativeInter':
            assert numInter >= 2
        else:
            assert numInter == 1

        if numInter == 2:
            T = 0
        if numInter == 3:
            T = 1


        S = 5 #5 samples in the bulk frequency sampling

        if folder1 == 'fewSamples':
            M = 10

        if numInter in [1, 2]:
            M = 10
        if numInter == 3:
            M = 15
        if numInter == 5:
            M = 25
        if numInter == 10:
            M = 50


        if folder1 == 'passanger':
            M = 22
            T = 0


        K = 6 #6 types of mutations. 5 interesting mutations, and the remaining 5 mutations are "boring" mutations.
        #T = 14#7 Sept 1 2022



        #This makes it so there are 10 total mutations, 5 of which are "interesting"
        #mutations of there own type. The remaining 5 are "boring" and are all considered the same type of mutation.
        if numInter == 2:
            mutationType = np.arange(M) // 2#5
        elif numInter == 3:
            mutationType = np.arange(M) // 3#5
        else:
            mutationType = np.arange(M)
            mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1


        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])

        #This gives random causal relationships between all of the interesting mutations.
        #Note, the causal relationship from mutation A to mutation B is independent
        #of the relationship from B to A.

        if numInter != 1:
            probabilityMatrix = np.random.randint(3, size=(K+1) *  K).reshape((K+1, K)) - 1
        else:
            probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))


        if folder1 == 'negativeInter':
            probabilityMatrix[np.arange(5), np.arange(5)] = -1



        probabilityMatrix = probabilityMatrix * np.log(11)


        probabilityMatrix[K] = 0
        probabilityMatrix[K-1, :] = 0
        probabilityMatrix[:, K-1] = 0





        #This makes sure the trees are not larger than the number of mutations. Otherwise,
        #it sets the tree size to a default value of 7.
        treeSize = min(7, M) #7

        skipSizes = False
        if folder1 == 'lowPatient':
            if T == 0:
                N = 200

            if T == 1:
                N = 100

            if T == 4:
                N = 600

            if T in [2, 3]:

                if T == 2:
                    treeSizes = loadnpz('./extra/AML_treeSizes.npz')
                #if T == 3:
                #    treeSizes = loadnpz('./extra/breast_treeSizes.npz')

                #plt.hist(treeSizes, bins=100)
                #plt.show()
                #quit()

                treeSizes = treeSizes.astype(int)
                treeSizes[treeSizes>7] = 7


                #print (np.unique(treeSizes, return_counts=True))
                #quit()

                N = 200

                rand1 = np.random.choice(treeSizes.shape[0], size=N)

                treeSizes = treeSizes[rand1]




        if numInter == 2:
            N = 800

        if numInter == 3:
            N = 1200

        if numInter == 5:
            N = 1000
        if numInter == 10:
            N = 1000



        if folder1 == 'passanger':
            N = 1000


        if folder1 == 'fewSamples':
            N = 1000
            S = 2 #2 bulk sequencing samples




        #This runs the simulation given the causal relationships, mutations, and mutation types.
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)

        if not skipSizes:
            #This randomizes the tree sizes to be between 5 and 7.
            treeSizes = np.random.randint(3, size=edges.shape[0]) + 5


        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M + 1
            clones[b, size1+1:] = 0


        #if folder1 != 'passanger':
        #    folder1 = 'lowPatient'
        #    if numInter != 1:
        #        folder1 = 'negativeInter'



        #print (folder1)
        #print (T)
        #print (mutationType.shape)
        #quit()

        #'''
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz', treeSizes)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz', probabilityMatrix)
        #'''


        #This runs bulk frequency measurements on the clonal data.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)



        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz', sampleInverse)


        print ("Saved")
        #quit()

#makeNewOccur()
#quit()



def makeBootstrap():


    #dataName = 'AML'
    dataName = 'breast'
    #dataName = 'I-a'


    if dataName == 'AML':
        maxM = 10
        trees, sampleInverse, mutationCategory, treeSizes, uniqueMutation, M = processTreeData(maxM, './data/realData/AML', fullDir=True)
    elif dataName == 'breast':
        maxM = 9
        trees, sampleInverse, mutationCategory, treeSizes, uniqueMutation, M = processTreeData(maxM, './data/realData/breastCancer', fullDir=True)

    if dataName in ['AML', 'breast']:
        _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

        _, index1 = np.unique(sampleInverse, return_index=True)
        treeSizes = treeSizes[index1]


    if dataName == 'I-a':
        folder1 = 'I-a'

        T = 4
        R = 19
        treeSizes = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(R) + '_treeSizes.npz')
        #edges = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz')
        #mutationType = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz')
        trees = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(R) + '_bulkTrees.npz')
        sampleInverse = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(R) + '_bulkSample.npz')


    for R2 in range(20, 100):

        print (R2)

        if dataName == 'I-a':
            N = 500
            bootstrap = np.random.choice(N, size=1000, replace=True)
        else:
            N = int(np.max(sampleInverse+1))
            bootstrap = np.random.choice(N, size=N*2, replace=True)

        #print (np.max(bootstrap))
        #quit()


        treeSizes2 = treeSizes[bootstrap]

        sampleInverse2 = np.zeros(sampleInverse.shape[0]*20, dtype=int)
        newArg = np.zeros(sampleInverse.shape[0]*20, dtype=int)
        count1 = 0
        for a in range(bootstrap.shape[0]):

            args1 = np.argwhere(sampleInverse == bootstrap[a])[:, 0]
            size1 = args1.shape[0]

            newArg[count1:count1+size1] = np.copy(args1)
            sampleInverse2[count1:count1+size1] = a
            count1 += size1

        newArg = newArg[:count1]
        sampleInverse2 = sampleInverse2[:count1]


        ######sampleInverse2 = sampleInverse[newArg]
        trees2 = trees[newArg]


        if dataName == 'I-a':
            folder2 = 'bootstrap'
            T2 = 0

            probabilityMatrix = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(R) + '_prob.npz')
            np.savez_compressed('./data/simulations/' + folder2 + '/T_' + str(T2) + '_R_' + str(R2) + '_prob.npz', probabilityMatrix)

        if dataName == 'AML':
            folder2 = 'realBoostrap'
            T2 = 0

        if dataName == 'breast':
            folder2 = 'realBoostrap'
            T2 = 1




        quit()
        #
        np.savez_compressed('./data/simulations/' + folder2 + '/T_' + str(T2) + '_R_' + str(R2) + '_treeSizes.npz', treeSizes2)
        np.savez_compressed('./data/simulations/' + folder2 + '/T_' + str(T2) + '_R_' + str(R2) + '_bulkTrees.npz', trees2)
        np.savez_compressed('./data/simulations/' + folder2 + '/T_' + str(T2) + '_R_' + str(R2) + '_bulkSample.npz', sampleInverse2)


#makeBootstrap()
#quit()


def makePathwaySimulation():

    for saveNum in range(100):

        #This function creates simulations of evolutionary pathways.


        #S = 3
        S = 5

        M = 20
        #K = 4
        #L = 3


        pathways = []


        #This randomly generates the lengths of the pathways
        sizes = np.random.randint(3, size=6) + 1

        #This randomly generates the number of pathways.
        pathwayNum = np.random.randint(4) + 1
        if pathwayNum > 2:
            pathwayNum = 2


        #This builds the pathways. The mutation numbers are sorted such that they automatically
        #assend along the pathway. This is fine since the model is mutation number invariant.
        count1 = 0
        for b in range(pathwayNum):
            pathways.append([])
            for c in range(3):
                size1 = sizes[(b * 3) + c]
                pathways[b].append([])
                for d in range(size1):
                    pathways[b][c].append(count1 + d)
                count1 += size1


        mutationType = (np.zeros(M) - 1).astype(int)
        #mutationType[mutationType >= L] = L - 1

        reducedPathways = []
        c = 0
        for a in range(len(pathways)):
            reducedPathways.append([])
            for b in range(len(pathways[a])):
                reducedPathways[a].append(c)
                mutationType[np.array(pathways[a][b]).astype(int)] = c
                c += 1

        K = c + 1
        mutationType[mutationType == -1] = K - 1

        #This matrix says which mutations are required to reach a certian level of the pathway.
        #For instance, to reach the third step in a pathway one needs to have the mutations of the first two steps.
        propertyRequirement = np.zeros((K+1, K+1))
        propertyRequirement[:, K] = 1

        #This matrix gives the probability of new mutation types based on the existing mutation types in the clone.
        probabilityMatrix = np.zeros((K+1, K))

        #This gives the probability of adding new mutations given the existing new mutations.
        #Specifically, it makes it so that if you are at one step in an evolutionary pathway you
        #have a high liklyhood of adding a mutation in the next step of the evolutionary pathway.
        Amplify = 21# * 100
        probabilityMatrix[K, reducedPathways[0][0]] = np.log(6)
        if len(reducedPathways) >= 2:
            probabilityMatrix[K, reducedPathways[1][0]] = np.log(6)


        for a in range(len(reducedPathways)):
            pathway1 = np.array(reducedPathways[a]).astype(int)
            for b in range(len(pathway1) - 1):
                in1 = pathway1[:b+1]
                out1 = pathway1[b+1]
                outRem1 = pathway1[b]

                propertyRequirement[in1, out1] = (1 / in1.shape[0]) + 1e-2

                #probabilityMatrix[out1, out1] = np.log(11)
                if True:#a == 1: #TODO REMOVE!
                    #print (in1)
                    #print (out1)
                    probabilityMatrix[out1, out1] = np.log(Amplify) #* (b + 1)
                    probabilityMatrix[out1, outRem1] = -1 * np.log(Amplify) #* (b + 1)



        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1
        #mutationTypeMatrix_extended[:, K] = 1

        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


        treeSize = min(7, M)
        N = 1000

        #This simulates the cancer evolution given the probability information implied by the pathways.
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize, maxLogProb=np.log(10000), useRequirement=True, reqMatrix=propertyRequirement)

        #This makes the tree sizes between 5 and 7 randomly.

        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M
            clones[b, size1+1:] = 0


        pathways_save = np.array(pathways, dtype=object)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_pathway.npz', pathways_save)


        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_treeSizes.npz', treeSizes) #Was 4 instead of 5. saved over at least 3

        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_trees.npz', edges)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_prob.npz', probabilityMatrix)

        #This simulates bulk frequency measurements.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/specialSim/dataSets/T_6_R_' + str(saveNum) + '_bulkSample.npz', sampleInverse)


def makeNonlinSimulation():




    for saveNum in range(0, 1):

        print (saveNum)

        #This function creates simulations of evolutionary pathways.


        #S = 3
        S = 5

        M = 10
        #K = 4
        #L = 3

        T = 0
        #T = 1


        Nrelationship = 1
        Ndrive = Nrelationship * 2


        Npassenger = M - Ndrive



        mutationTypeMatrix_extended = np.zeros((M, M+1))
        mutationTypeMatrix_extended[np.arange(M), np.arange(M)] = 1
        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])


        #print (mutationTypeMatrix_extended)
        #quit()

        pairs = np.random.permutation(Ndrive)
        pairs = pairs.reshape((Nrelationship, 2))


        propertyRequirement = np.zeros((M+1, (Nrelationship*1)+2))
        for a in range(Nrelationship):
            propertyRequirement[pairs[a, 0], a] = (1/2) + 1e-3
            propertyRequirement[pairs[a, 1], a] = (1/2) + 1e-3



        propertyRequirement[:, -2] = 1 + 1e-3
        propertyRequirement[-1, -1] = 1 + 1e-3


        probabilityMatrix = np.zeros(((Nrelationship*1)+2, M+1))



        randEffect = np.random.randint(2, size=M-Ndrive)
        #randEffect = randEffect - 1
        randEffect = (randEffect * 2) - 1

        for b in range(randEffect.shape[0]):
            probabilityMatrix[:Nrelationship, b+Ndrive] = np.log(10) * randEffect[b]



        probabilityMatrix[-2, :Ndrive] = np.log(5)


        #print (probabilityMatrix)
        #quit()



        treeSize = min(7, M)
        N = 1000


        #edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, Ndrive, treeSize, maxLogProb=np.log(10000), useRequirement=True, reqMatrix=propertyRequirement)
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix_extended, mutationTypeMatrix_extended, N, M, Ndrive, treeSize, maxLogProb=np.log(10000), useRequirement=True, reqMatrix=propertyRequirement)


        #This makes the tree sizes between 5 and 7 randomly.

        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M


        #quit()
        #quit()


        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_treeSizes.npz', treeSizes) #Was 4 instead of 5. saved over at least 3

        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_trees.npz', edges)

        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_causes.npz', randEffect)
        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_nonlinPairs.npz', pairs)

        #This simulates bulk frequency measurements.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(saveNum) + '_bulkSample.npz', sampleInverse)


def makeLatentSimulation():



    for saveNum in range(3, 20):

        #This function creates simulations of evolutionary pathways.


        #S = 3
        S = 5

        M = 20
        #K = 4
        #L = 3


        #plt.plot(np.sin(np.arange(100) * (np.pi * 2 / 100)  ))
        #plt.show()
        #quit()


        pathways = []

        Mtype = M // 2


        mutationTypeMatrix_extended = np.zeros((M, M+1))
        mutationTypeMatrix_extended[np.arange(M), np.arange(M)] = 1
        mutationTypeMatrix = mutationTypeMatrix_extended[:M, :M]


        theta = np.random.random(M) * np.pi * 2
        scale = (np.random.random(M) + 1) / 2


        catA = np.cos(theta)
        catB = np.sin(theta)

        propertyA = scale * catA
        propertyB = scale * catB
        propertyBoth = np.array([propertyA, propertyB]).T


        probabilityMatrix = np.zeros((M+1, M))
        probabilityMatrix[:-1, :] =  propertyA.reshape((-1, 1)) * catA.reshape((1, -1))
        probabilityMatrix[:-1, :] += propertyB.reshape((-1, 1)) * catB.reshape((1, -1))
        probabilityMatrix = probabilityMatrix * np.log(10)



        treeSize = min(7, M)
        N = 1000
        #N = 800


        #edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, Ndrive, treeSize, maxLogProb=np.log(10000), useRequirement=True, reqMatrix=propertyRequirement)
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, M, treeSize, maxLogProb=np.log(10000))


        #This makes the tree sizes between 5 and 7 randomly.

        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M


        np.savez_compressed('./data/simulations/latent/T_0_R_' + str(saveNum) + '_treeSizes.npz', treeSizes) #Was 4 instead of 5. saved over at least 3

        np.savez_compressed('./data/simulations/latent/T_0_R_' + str(saveNum) + '_trees.npz', edges)

        np.savez_compressed('./data/simulations/latent/T_0_R_' + str(saveNum) + '_property.npz', propertyBoth)

        #This simulates bulk frequency measurements.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S)

        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        np.savez_compressed('./data/simulations/latent/T_0_R_' + str(saveNum) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/simulations/latent/T_0_R_' + str(saveNum) + '_bulkSample.npz', sampleInverse)


def makeSimBasedReal():

    for a in range(0, 20):

        #This runs a simple simulation of causal relationships.

        T = 0

        print (a)


        #folder1 = 'effectSize'

        folder1 = 'passanger'



        numInter = 1

        #S 5, min(7, M), T = 4.

        #S = 3
        S = 5 #5 samples in the bulk frequency sampling


        K = 6 #6 types of mutations. 5 interesting mutations, and the remaining 5 mutations are "boring" mutations.
        #T = 14#7 Sept 1 2022


        if folder1 == 'effectSize':
            #model = torch.load('./Models/realData/savedModel_breast.pt')
            model = torch.load('./Models/realData/savedModel_AML.pt')
            b = 0
            for param in model.parameters():
                #print (param.shape)
                #quit()
                if b == 0:
                    M = param.shape[1] - 20
                b += 1
            X = torch.zeros((M, M))
            X[np.arange(M), np.arange(M)] = 1


            pred0, _ = model( torch.zeros((1, M)) )
            pred0 = pred0.data.numpy()

            pred, _ = model(X)
            pred = pred.data.numpy()

            pred = pred - pred0

            #pred_adj = pred - np.median(pred)

            #plt.hist(pred_adj.reshape(( pred.shape[0]*pred.shape[1], ))  , bins=100 )
            #plt.show()

            effectSize = np.max(pred, axis=1)
            argDrive = np.argsort(effectSize)[-1::-1][:5]

            pred = pred[argDrive]

            pred = np.sort(pred, axis=1)[:, -5:]

            M = 10


            distribution = pred.reshape((pred.shape[0]*pred.shape[1],))

            #np.savez_compressed('./plotData/effectSizeDistribution.npz', distribution)

            #plt.hist(distribution)
            #plt.show()

        #quit()

        if folder1 == 'passanger':


            #fileIn  = './data/realData/breastCancer.npy'
            #treeData = np.load(fileIn, allow_pickle=True)

            file1 = './data/realData/breastCancer'
            file2 = './extra/Breast_treeSizes.npz'

            #maxM = 100
            maxM = 10
            newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)

            _, index1 = np.unique(sampleInverse, return_index=True)
            newTrees = newTrees[index1]
            newTrees = newTrees[:, :, 1]

            unique1, count1 = np.unique(newTrees, return_counts=True)

            unique1, count1 = unique1[:-1], count1[:-1]

            np.savez_compressed('./plotData/mutationCountsBreast.npz', count1)

            rates = np.sort(count1)[-1::-1].astype(float)
            rates = rates / np.sum(rates)

            M = rates.shape[0]
            K = M


            #print (M)
            #quit()


        quit()



        #This makes it so there are 10 total mutations, 5 of which are "interesting"
        #mutations of there own type. The remaining 5 are "boring" and are all considered the same type of mutation.

        mutationType = np.arange(M)
        mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1


        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])

        #This gives random causal relationships between all of the interesting mutations.
        #Note, the causal relationship from mutation A to mutation B is independent
        #of the relationship from B to A.





        if folder1 == 'effectSize':

            probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))

            probabilityMatrix[K] = 0
            probabilityMatrix[K-1, :] = 0
            probabilityMatrix[:, K-1] = 0

            for p0 in range(probabilityMatrix.shape[0]):
                for p1 in range(probabilityMatrix.shape[1]):
                    if probabilityMatrix[p0, p1] > 0:
                        rand1 = np.random.randint(distribution.shape[0])
                        probabilityMatrix[p0, p1] = distribution[rand1]

        if folder1 == 'passanger':

            Ndrive = 5
            probabilityMatrix = np.zeros((M+1, M))
            probabilityMatrix[:Ndrive, :Ndrive] = np.random.randint(2, size= Ndrive ** 2 ).reshape((Ndrive, Ndrive))
            probabilityMatrix = probabilityMatrix * np.log(10)

            probabilityMatrix[M] = np.log(np.copy(rates))



        #plt.imshow(probabilityMatrix)
        #plt.show()
        #quit()



        #This makes sure the trees are not larger than the number of mutations. Otherwise,
        #it sets the tree size to a default value of 7.
        treeSize = min(7, M) #7

        if folder1 == 'effectSize':
            N = 600

        if folder1 == 'passanger':
            N = 1000


        #This runs the simulation given the causal relationships, mutations, and mutation types.
        edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)


        #This randomizes the tree sizes to be between 5 and 7.
        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5

        ###treeSizes = treeSizes - 2 #TODO THIS IS TEMP. REMOVE!


        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            edges[b, size1:] = M + 1
            clones[b, size1+1:] = 0


        quit()


        #'''
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz', treeSizes)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz', probabilityMatrix)
        #'''

        specialM = 101
        if folder1 == 'passanger':
            specialM = 1001
        #This runs bulk frequency measurements on the clonal data.
        trees, sampleInverse = multisizeBulkFrequency(clones, treeSizes, S, specialM=specialM)


        #_, index1 =np.unique(sampleInverse, return_index=True)
        #treeLength = treeSizes[sampleInverse.astype(int)]
        #maxList = []
        #for index1 in range(treeLength.shape[0]):
        #    max1 = np.max(np.unique(trees[index1, :treeLength[a]]))
        #    maxList.append(max1)
        #print (np.unique(np.array(maxList)))
        #quit()




        trees[trees == specialM-1] = M
        trees[trees == specialM] = M + 1

        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz', sampleInverse)


        print ("Saved")
        #quit()



def mapTrees(trees1, trees2, treeSizes):

    for a in range(trees1.shape[0]):

        treeSize = int(treeSizes[a])

        tree1 = trees1[a, :treeSize]
        tree1_flat = tree1.reshape((tree1.shape[0]*tree1.shape[1],))
        unique1 = np.unique(tree1_flat)

        tree2 = trees2[a, :treeSize]
        tree2_flat = tree2.reshape((tree2.shape[0]*tree2.shape[1],))
        _, inverse2 = np.unique(tree2_flat, return_inverse=True)
        tree2_new = unique1[inverse2]
        tree2_new = tree2_new.reshape(tree1.shape)


        trees2[a,  :treeSize] = np.copy(tree2_new)

    return trees2


def treePermuation(trees1, treeSizes):

    trees2 = np.copy(trees1)

    for a in range(trees1.shape[0]):
        treeSize = int(treeSizes[a])
        tree1 = trees1[a, :treeSize]
        tree1_flat = tree1.reshape((tree1.shape[0]*tree1.shape[1],))

        unique1, inverse1 = np.unique(tree1_flat, return_inverse=True)

        perm1 = np.random.permutation(unique1.shape[0] - 1)
        unique1_perm = np.copy(unique1)
        unique1_perm[:perm1.shape[0]] = unique1_perm[perm1]

        tree2_flat = unique1_perm[inverse1]
        tree2 = tree2_flat.reshape(tree1.shape)

        #print (tree1)
        #print (tree2)
        #quit()

        trees2[a, :treeSize] = np.copy(tree2)

    return trees2



def makeWithRandomOccur():

    for a in range(0, 20):

        #This runs a simple simulation of causal relationships.



        print (a)


        folder1 = 'random'
        T = 4


        numInter = 1

        S = 5 #5 samples in the bulk frequency sampling
        M = 10



        K = 6 #6 types of mutations. 5 interesting mutations, and the remaining 5 mutations are "boring" mutations.


        mutationType = np.arange(M)
        mutationType[mutationType >= K] = K - 1

        mutationTypeMatrix_extended = np.zeros((M, K+1))
        mutationTypeMatrix_extended[np.arange(M), mutationType] = 1


        mutationTypeMatrix = np.copy(mutationTypeMatrix_extended[:, :-1])

        #This gives random causal relationships between all of the interesting mutations.
        #Note, the causal relationship from mutation A to mutation B is independent
        #of the relationship from B to A.
        probabilityMatrix = np.random.randint(2, size=(K+1) *  K).reshape((K+1, K))


        if folder1 == 'negativeInter':
            probabilityMatrix[np.arange(5), np.arange(5)] = -1


        probabilityMatrix = probabilityMatrix * np.log(11)
        probabilityMatrix[K] = 0
        probabilityMatrix[K-1, :] = 0
        probabilityMatrix[:, K-1] = 0


        #This makes sure the trees are not larger than the number of mutations. Otherwise,
        #it sets the tree size to a default value of 7.
        treeSize = min(7, M) #7

        skipSizes = False
        N = 1000






        if not T in [2]:
            #This runs the simulation given the causal relationships, mutations, and mutation types.
            edges, clones = makePartSimulation(probabilityMatrix, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)

        probabilityMatrix_random = probabilityMatrix * 0
        if T in [2]:
            edges, clones= makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)

        treeSizes = np.random.randint(3, size=edges.shape[0]) + 5



        if T in [0, 1, 2, 3, 4]:
            edges_random2, clones_random2 = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
            edges_random2 = mapTrees(edges, edges_random2, treeSizes)

        #if T in [3, 4]:
        #    edges_random2 = treePermuation(edges, treeSizes)

        if T in [3, 4]:
            edges_random3, clones_random3 = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
            edges_random3 = mapTrees(edges, edges_random3, treeSizes)

        if T in [4]:
            edges_random4, _ = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
            edges_random4 = mapTrees(edges, edges_random4, treeSizes)

            edges_random5, _ = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
            edges_random5 = mapTrees(edges, edges_random5, treeSizes)

            edges_random6, _ = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
            edges_random6 = mapTrees(edges, edges_random6, treeSizes)



        #quit()

        #edges_random3, clones_random3 = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
        #edges_random4, clones_random4 = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)
        #edges_random5, clones_random5 = makePartSimulation(probabilityMatrix_random, mutationTypeMatrix, mutationTypeMatrix_extended, N, M, K, treeSize)




        for b in range(len(treeSizes)):
            size1 = treeSizes[b]
            #if T != 2:
            edges[b, size1:] = M + 1
            clones[b, size1+1:] = 0

            #edges_random[b, size1:] = M + 1

            #if T == [0, 1]:
            edges_random2[b, size1:] = M + 1
            if T in [3, 4]:
                edges_random3[b, size1:] = M + 1
            if T == 4:
                edges_random4[b, size1:] = M + 1
                edges_random5[b, size1:] = M + 1
                edges_random6[b, size1:] = M + 1








        if T in [0, 1, 2]:
            trees = np.concatenate((edges, edges_random2), axis=0)
        if T == 3:
            trees = np.concatenate((edges, edges_random2, edges_random3), axis=0)
        if T == 4:
            trees = np.concatenate((edges, edges_random2, edges_random3), axis=0)
        #    trees = np.concatenate((edges, edges_random2, edges_random3, edges_random4, edges_random5, edges_random6), axis=0)


        #sampleInverse = np.arange(trees.shape[0]) % edges.shape[0]

        #if T == 1:
        #    sampleInverse = np.concatenate((   np.arange(N), np.arange(N//2)*2  ))
        #else:
        sampleInverse = np.arange(trees.shape[0]) % edges.shape[0]

        #sample_argsort = np.argsort(sampleInverse)

        if T in [0, 1, 2]:
            sample_argsort = np.array([np.arange(edges.shape[0]), np.arange(edges.shape[0])+edges.shape[0]]).T
        if T in [3]:
            sample_argsort = np.array([np.arange(edges.shape[0]), np.arange(edges.shape[0])+edges.shape[0], np.arange(edges.shape[0])+(edges.shape[0] * 2)  ] ).T
        if T in [4]:
            #sample_argsort = np.array([np.arange(edges.shape[0]), np.arange(edges.shape[0])+edges.shape[0], np.arange(edges.shape[0])+(edges.shape[0] * 2)  ] ).T
            sample_argsort = np.zeros((edges.shape[0], 3), dtype=int)
            for c in range(3):
                sample_argsort[:, c] = np.copy(  np.arange(edges.shape[0])+(edges.shape[0] * c)   )

        sample_argsort = sample_argsort.reshape((trees.shape[0],))

        #plt.plot(sample_argsort)
        #plt.show()
        #quit()
        trees = trees[sample_argsort]
        sampleInverse = sampleInverse[sample_argsort]

        trees[trees == 100] = M
        trees[trees == 101] = M + 1

        print ('T', T)
        quit()


        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz', trees)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz', sampleInverse)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz', treeSizes)
        #if T != 2:
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz', probabilityMatrix)

        '''
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz', treeSizes)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz', edges)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_mutationType.npz', mutationType)
        np.savez_compressed('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz', probabilityMatrix)
        #'''

        print ("Saved")



#makeWithRandomOccur()
#quit()




def plotEffectSize():

    distribution = loadnpz('./plotData/effectSizeDistribution.npz')

    plt.hist(distribution, color='orange')
    plt.xlabel('effect size (absolute causality)')
    plt.ylabel('count')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/effectSizeDistribution.pdf')
    plt.show()

def plotMutationFrequency():

    file1 = './data/realData/breastCancer'
    maxM = 10
    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)

    Npatient = np.unique(sampleInverse).shape[0]

    print (Npatient)
    quit()

    counts = loadnpz('./plotData/mutationCountsBreast.npz')
    counts = np.sort(counts)[-1::-1]

    N = counts.shape[0]

    plt.plot(counts / float(Npatient), color='orange')
    plt.ylabel('fraction of patients')
    plt.xlabel('mutation')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/mutationFrequency.pdf')
    plt.show()

def saveTreeSizes():

    if False:
        file1 = './data/realData/AML'
        file2 = './extra/AML_treeSizes.npz'
    else:
        file1 = './data/realData/breastCancer'
        file2 = './extra/Breast_treeSizes.npz'

    maxM = 10
    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
    #maxM = 9
    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')

    _, index1 = np.unique(sampleInverse, return_index=True)
    treeSizes = treeLength[index1]


    np.savez_compressed(file2, treeSizes)

def plotTreeSizes():

    treeSizes = loadnpz('./extra/AML_treeSizes.npz')
    #treeSizes = loadnpz('./extra/breast_treeSizes.npz')

    treeSizes[treeSizes>7] = 7

    plt.hist([5, 6, 7], bins=6, range=(1.5, 7.5), density=True, color='blue', alpha=0.7)
    plt.hist(treeSizes, bins=6, range=(1.5, 7.5), density=True, color='orange', alpha=0.7)
    plt.ylim(0, 0.35)
    plt.xlabel('number of mutations')
    plt.ylabel('fraction of patients')
    plt.legend(['original mutations', 'fewer mutations'], loc='lower left')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/treeSizesAML.pdf')
    plt.show()



    #plt.ylim(0, 0.35)
    #plt.xlabel('number of mutations')
    #plt.ylabel('fraction of patients')
    #plt.gcf().set_size_inches(8, 6)
    #plt.savefig('./images/treeSizesDefault.pdf')
    #plt.show()

    quit()

def trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False, regularizeFactor=0.02, nonLin=True):


    #This function trains a model to predict the probability of new mutations being added to clones,
    #and therefore the probability of phylogeny trees generated through this process.
    #This function differs from trainModelTree in that it uses groups of mutations rather than purely individual mutations.
    #Because multiple mutations in the same group can be in a single tree, it must allow for a mutation group to be introduced
    #multiple times in a single tree. This requires modifications of the trainModelTree code.

    doTrainSet = not (type(trainSet) == type(False))

    N1 = newTrees.shape[0]
    N2 = int(np.max(sampleInverse) + 1)

    M2 = np.unique(mutationCategory).shape[0] - 2

    #This calculates the test set patients, as well as the training set trees (trainSet2)
    if doTrainSet:
        #trainSet = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]
        testSet = np.argwhere(np.isin(np.arange(N2), trainSet) == False)[:, 0]

        trainSet2 = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]

    torch.autograd.set_detect_anomaly(True)

    #nonLin = True

    #torch.autograd.set_detect_anomaly(True)
    #model = MutationModel(M)

    if nonLin:
        model = MutationModel(M2)
    else:
        model = MutationModel2(M2)
    #model = torch.load('./Models/savedModel.pt')

    #model = torch.load('./Models/savedModel24.pt')

    #N1 = 10000

    nPrint = 100
    #if adjustProbability:
        #learningRate = 1e1
    #    learningRate = 1e0
        #learningRate = 1e-1#-1 #1 #2
    #else:
    #learningRate = 1e-1#-2 #1
    #learningRate = 2e0

    #learningRate = 1e0
    #learningRate = 2e-1
    if nonLin:
        learningRate = 3e-1
    else:
        #learningRate = 5e-1
        learningRate = 1e0 #Standard version sep 22 2022
    #learningRate = 2e0

    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    #learningRate = 1e-2#-1
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    if adjustProbability or True:
        baseLine = np.ones(N1) * 0
        #baseLine = np.load('./Models/baseline1.npy')
        baseN = 10

    accuracies = []

    #iterNum = 20000
    if nonLin:
        #iterNum = 4000
        iterNum = 10000 #Modified Oct 12 2022
    else:
        #iterNum = 1000 #Only for TreeMHN #Standard version sep 22 2022
        iterNum = 10000
    #iterNum = 2000
    #iterNum = 4000
    #iterNum = 5000

    for iter in range(0, iterNum):#301): #3000
        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print ('starting iteration ' + str(iter))





        #This is initializing the edges of the generated trees
        Edges = np.zeros((N1, maxM+1, 2))
        Edges[:, 0, 1] = M
        #This is initializing the clones for each possible phylogeny tree.
        clones =  torch.zeros((N1, maxM+1, M2))

        #The edges remaining represent the edges which still need to be added
        #in order to generate the correct tree. It is initialized as all
        #of the edges in the tree.
        edgesRemaining = np.copy(newTrees)
        #edgesRemainingGroup is the remaining edges in terms of mutation groups rather than in terms
        #of individual mutations.
        edgesRemainingGroup = mutationCategory[edgesRemaining.astype(int)]

        #This converts them to numerical incoding of edges.
        edgesRemainingNum = (edgesRemaining[:, :, 0] * (M + 2)) + edgesRemainingGroup[:, :, 1]

        #These are the log probability of the generation of the tree (theoretically)
        #as well as the log probability of this way of generatating the tree,
        #when we restrict the model to only generating this correct tree.
        probLog1 = torch.zeros(N1)
        probLog2 = torch.zeros(N1)

        for a in range(0, maxM):




            argsLength = np.argwhere(treeLength >= (a + 1))[:, 0]

            M1 = a + 1
            counter = np.arange(N1)

            #This calculates the output of the model given the clones that exist.
            clones1 = clones[:, :M1].reshape((N1 * M1, M2))

            output, _ = model(clones1)
            output = output.reshape((N1, M1 * M2))
            output = torch.softmax(output, dim=1)

            #This calculates the possible new mutations and clones for new mutations to be added to.
            newStart = Edges[:, :M1, 1].repeat(M2).reshape((N1, M1 * M2))
            newStartClone = np.arange(M1).repeat(N1*M2).reshape((M1, N1, M2))
            newStartClone = np.swapaxes(newStartClone, 0, 1).reshape((N1, M1 * M2))

            newEnd = np.arange(M2).repeat(N1*M1).reshape((M2, N1*M1)).T.reshape((N1, M1 * M2))

            edgeNums = (newStart * (M + 2)) + newEnd

            #This makes it so you can only add edges which are present in the correct tree.
            validEndMask = np.zeros((N1, M1 * M2))
            for b in range(0, N1):
                validEndMask[b, np.isin(edgeNums[b], edgesRemainingNum[b])] = 1

            #This removes the impossible choices, and then adjust the probability to still sum to 1.
            output2 = output * torch.tensor(validEndMask).float()
            output2_sum = torch.sum(output2, dim=1).repeat_interleave(M1*M2).reshape((N1, M1*M2))
            output2 = output2 / output2_sum

            #This makes a choice of clone to add a mutation as well as the mutation to be added based on the probabilities.
            choiceNow = doChoice(output2.data.numpy()).astype(int)

            printNum = 10

            #This determines the probability of this specific step in the generation process, given this tree as
            #the correct tree to be generated.
            sampleProbability = output2[counter, choiceNow]

            #This gives the probability of this tree generation process in general, not assuming
            #the correct tree must be generated.
            theoryProbability = output[counter, choiceNow]

            #This is the numerical representation of the edge which is added to the tree.
            edgeChoice = edgeNums[counter, choiceNow]
            newStartClone = newStartClone[counter, choiceNow]

            edgeChoice_end_individual = np.zeros(N1)


            #This updates the remaining edges which need to be added based on the edges which were just added.
            for b in range(0, N1):
                #print (edgeChoice[b])
                #print (edgesRemainingNum[b])
                argIn1 = np.argwhere(edgesRemainingNum[b] == edgeChoice[b])
                if argIn1.shape[0] != 0:
                    argIn1 = argIn1[0, 0]
                    edgesRemainingNum[b, argIn1] = (M + 2)**2
                    argIn1 = edgesRemaining[b, argIn1, 1]
                    edgeChoice_end_individual[b] = argIn1

            #This gives the first and second node on the new edge added
            edgeChoice_start = edgeChoice // (M + 2)
            edgeChoice_end = edgeChoice % (M + 2)

            #This adds the new clone to the clones in this phylogeny tree.
            clones[counter, a+1] = clones[counter, newStartClone].clone()
            clones[counter, a+1, edgeChoice_end] = clones[counter, a+1, edgeChoice_end] + 1

            #This adds the new edge to the list of edges in the tree.
            Edges[:, M1, 0] = edgeChoice_start
            Edges[:, M1, 1] = edgeChoice_end_individual


            #This adds the theoryProbability and sampleProbability (described earlier) for this edge
            #to there respective sums.
            probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-12)
            probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-12)




        probLog1_np = probLog1.data.numpy()
        probLog2_np = probLog2.data.numpy()

        #This adjusts the probabiltiy baseline for each tree.
        baseLine = baseLine * ((baseN - 1) / baseN)
        baseLine = baseLine + ((1 / baseN) * np.exp(probLog1_np - probLog2_np)   )

        #adjustProbability means that the algorithm optimizes to accuratly represent the probability
        #of different trees, rather than just trying to maximize the probability of the very most
        #likely trees (which is useful in some situations not discussed in the paper)
        if adjustProbability:
            baseLineLog = np.log(baseLine)


            #This is the reinforcement learning loss function, before some adjustments for sampling frequency
            loss_array = probLog1 / (torch.exp(probLog2.detach()) + 1e-10)
            loss_array = loss_array / maxM

            sampleUnique, sampleIndex = np.unique(sampleInverse, return_index=True)

            #This will give some adjustement terms associated with sampling frequency.
            #Specifically, adjustments for the fact that things are not sampled exactly proportional to
            #there liklyhood according to the model. For more detailed information, read the paper.
            prob_adjustment = np.zeros(sampleInverse.shape[0])

            baseLineMean = np.zeros(int(np.max(sampleInverse) + 1)) + 1

            for b in range(0, sampleIndex.shape[0]):
                start1 = sampleIndex[b]
                if b == sampleIndex.shape[0] - 1:
                    end1 = N1
                else:
                    end1 = sampleIndex[b+1]

                argsLocal = np.arange(end1 - start1) + start1
                localProb = probLog1_np[argsLocal]
                localBaseline = baseLineLog[argsLocal]
                #maxLogProb = max(np.max(localBaseline), np.max(localProb))
                maxLogProb = np.max(localBaseline)
                localProb = localProb - maxLogProb
                localBaseline = localBaseline - maxLogProb

                #localProb_0 = np.copy(localProb)

                localProb = np.exp(localProb) / (np.sum(np.exp(localBaseline)) + 1e-5)

                #if np.max(localProb) > 1:
                #    print ('Hi')
                #    print (np.exp(localProb+maxLogProb))
                #    print (np.exp(localBaseline+maxLogProb))
                #    quit()

                prob_adjustment[argsLocal] = np.copy(localProb)

                #baseLineMean[b] = np.sum(baseLine[argsLocal])
                baseLineMean[int(sampleUnique[b])] = np.sum(baseLine[argsLocal])



            #This applies the adjustment to the loss function
            loss_array = loss_array * torch.tensor(prob_adjustment)

            #This takes the loss on the training set trees.
            loss_array = loss_array[trainSet2]

            #Thiscalculates the unsupervised learning log liklyhood loss.
            # Note, this is not the same as the reinforcement learning reward function.
            score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-6))
            score_test = np.mean(np.log(baseLineMean[testSet] + 1e-6))


        else:


            loss_array = torch.exp( probLog1 - probLog2.detach() )
            loss_array = loss_array[trainSet2]

        #This gives a minus sign, since we minimize the negative of the reward function mean.
        loss = -1 * torch.mean(loss_array)





        #This adds regularization.
        #There are some small subset of parameters where regularization is not useful for
        #preventing overfitting or increasing interpretability
        regularization = 0
        numHigh = 0
        numAll = 0
        c1 = 0
        if nonLin:
            for param in model.parameters():
                #print (param.shape)
                if c1 in [0, 2, 3]:

                    #regularization = regularization + torch.sum(torch.abs(param))
                    #regularization = regularization + torch.sum( torch.abs(param) - ( 0.9 * torch.relu( torch.abs(param) - 0.2 ))       )
                    regularization = regularization + torch.sum( torch.abs(param) - ( 0.9 * torch.relu( torch.abs(param) - 0.1 ))       ) #STANDARD!!! (before Oct 13)
                    #regularization = regularization + torch.sum( torch.abs(param) - ( 0.8 * torch.relu( torch.abs(param) - 0.1 ))       ) #Trying Oct 13
                    #regularization = regularization + (torch.sum( 1 - torch.exp(-torch.abs(param) * 10)   ) * 0.1)
                    numHigh += np.argwhere(np.abs(np.abs(param.data.numpy()) < 0.01)).shape[0]
                    numAll += np.argwhere(np.abs(np.abs(param.data.numpy()) > -1)).shape[0]
                    #numAll += param.size
                c1 += 1

            #quit()
        else:

            for param in model.parameters():
                #regularization = regularization + torch.sum(torch.abs(param))
                regularization = regularization + torch.sum(param**2)
                c1 += 1


        if regularizeFactor == 0.02:
            #regularization = regularization * 0.02#0.0001#
            regularization = regularization * 0.02
            #regularization = regularization * 0.05
            #regularization = regularization * 0.002
        else:
            regularization = regularization * regularizeFactor

        #Adding regularization to the loss
        loss = loss + regularization




        if doPrint:


            #print ("")
            print ('Mean Probability: ', np.mean(baseLine))
            print ('Training Score: ', score_train, 'Testing Score:', score_test)
            print ('Loss: ', loss.data.numpy())

            #Saving the probabilities and model.
            if baselineSave and fileSave:
                torch.save(model, fileSave)
                np.save(baselineSave, baseLine)




        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #quit()

def trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False, modL=False, regFactor=0.0002, Niter=1000):


    #This function trains a model to predict the probability of new mutations being added to clones,
    #and therefore the probability of phylogeny trees generated through this process.

    excludeSameMut = True

    doTrainSet = not (type(trainSet) == type(False))

    N1 = newTrees.shape[0]
    N2 = int(np.max(sampleInverse) + 1)

    #M2 = np.unique(mutationCategory).shape[0]

    #This calculates the test set patients, as well as the training set trees (trainSet2)
    if doTrainSet:
        #trainSet = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]
        testSet = np.argwhere(np.isin(np.arange(N2), trainSet) == False)[:, 0]

        trainSet2 = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]


    model = MutationModel(M, modL=modL)
    #model = torch.load('./Models/savedModel23.pt')

    #N1 = 10000

    nPrint = 1#00


    #learningRate = 1e0

    #learningRate = 1e0 #Typically used May 22
    learningRate = 1e1
    #learningRate = 1e2

    optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
    #learningRate = 1e-1
    #learningRate = 1e-2
    #learningRate = 1e-3
    #optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)

    if adjustProbability or True:
        #This is the baseline probability that each tree is generated.
        baseLine = np.zeros(N1) #+ 0.1
        #baseLine = np.load('./Models/baseline1.npy')
        #baseN = 100
        baseN = 10

    accuracies = []

    recordBase = np.zeros((100000, N1))
    recordSamp = np.zeros((100000, N1))

    print ("This runs for 1000 iterations.")
    print ("The user can stop the code at any time if the testing loss has ")
    print ("converged sufficiently close to the optimum for the user's applicaiton. ")

    #print (Niter)
    #quit()


    for iter in range(0, Niter):#301): #3000


        #if True:
        #    baseN = (min(iter, 1000) + 10)

        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print ('starting iteration ' + str(iter))


        #This is initializing the edges of the generated trees
        Edges = np.zeros((N1, maxM+1, 2))
        Edges[:, 0, 1] = M
        #This is initializing the clones for each possible phylogeny tree.
        clones =  torch.zeros((N1, maxM+1, M))
        #The edges remaining represent the edges which still need to be added
        #in order to generate the correct tree. It is initialized as all
        #of the edges in the tree.
        edgesRemaining = np.copy(newTrees)
        #This converts them to numerical incoding of edges.
        edgesRemaining = (edgesRemaining[:, :, 0] * (M + 2)) + edgesRemaining[:, :, 1]

        #These are the log probability of the generation of the tree (theoretically)
        #as well as the log probability of this way of generatating the tree,
        #when we restrict the model to only generating this correct tree.
        probLog1 = torch.zeros(N1)
        probLog2 = torch.zeros(N1)

        #Looping over the edges in the tree.
        for a in range(0, maxM):

            #These are the trees which have an "a+1"th edge, since there length
            #is larger than a+1.
            argsLength = np.argwhere(treeLength >= (a + 1))[:, 0]

            #The "a+1"th edge should only be ran if at least some tree has an "a+1"th edge.
            if argsLength.shape[0] != 0:


                M1 = a + 1
                counter = np.arange(N1)

                #This calculates the output of the model given the clones that exist.
                clones1 = clones[:, :M1].reshape((N1 * M1, M))
                output, _ = model(clones1)
                output = output.reshape((N1, M1 * M))
                output = torch.softmax(output, dim=1)

                #This calculates the possible new mutations and clones for new mutations to be added to.
                newStart = Edges[:, :M1, 1].repeat(M).reshape((N1, M1 * M))
                newStartClone = np.arange(M1).repeat(N1*M).reshape((M1, N1, M))
                newStartClone = np.swapaxes(newStartClone, 0, 1).reshape((N1, M1 * M))

                newEnd = np.arange(M).repeat(N1*M1).reshape((M, N1*M1)).T.reshape((N1, M1 * M))

                edgeNums = (newStart * (M + 2)) + newEnd

                #This makes it so the same mutation can not be added multiple times.
                #Specifically, it assigns a probability of zero to this case.
                if excludeSameMut:
                    notAlreadyUsedMask = np.zeros((N1, M1 * M))
                    for b in range(0, N1):


                        notAlreadyUsedMask[b, np.isin(newEnd[b], Edges[b, :M1, 1]) == False]


                        notAlreadyUsedMask[b, np.isin(newEnd[b], Edges[b, :M1, 1]) == False] = 1

                    output = output * torch.tensor(notAlreadyUsedMask).float()
                    output_sum = torch.sum(output, dim=1).repeat_interleave(M1*M).reshape((N1, M1*M))
                    output = output / output_sum


                #This makes it so you can only add edges which are present in the correct tree.
                validEndMask = np.zeros((N1, M1 * M))
                for b in range(0, N1):
                    validEndMask[b, np.isin(edgeNums[b], edgesRemaining[b])] = 1

                #This removes the impossible choices, and then adjust the probability to still sum to 1.
                output2 = output * torch.tensor(validEndMask).float()
                output2_sum = torch.sum(output2, dim=1).repeat_interleave(M1*M).reshape((N1, M1*M))
                output2 = output2 / output2_sum




                #This makes a choice of clone to add a mutation as well as the mutation to be added based on the probabilities.
                choiceNow = doChoice(output2.data.numpy()).astype(int)


                printNum = 10


                #This determines the probability of this specific step in the generation process, given this tree as
                #the correct tree to be generated.
                sampleProbability = output2[counter, choiceNow]

                #This gives the probability of this tree generation process in general, not assuming
                #the correct tree must be generated.
                theoryProbability = output[counter, choiceNow]
                #This is the numerical representation of the edge which is added to the tree.
                edgeChoice = edgeNums[counter, choiceNow]
                newStartClone = newStartClone[counter, choiceNow]

                #This updates the remaining edges which need to be added based on the edges which were just added.
                for b in range(0, N1):
                    argsNotRemaining = np.argwhere(edgesRemaining[b] == edgeChoice[b])[:, 0]
                    edgesRemaining[b, argsNotRemaining] = (M + 2) ** 2

                #This gives the first and second node on the new edge added
                edgeChoice_start = edgeChoice // (M + 2)
                edgeChoice_end = edgeChoice % (M + 2)

                #This adds the new clone to the clones in this phylogeny tree.
                clones[counter, a+1] = clones[counter, newStartClone].clone()
                clones[counter, a+1, edgeChoice_end] = clones[counter, a+1, edgeChoice_end] + 1


                #This adds the new edge to the list of edges in the tree.
                Edges[:, M1, 0] = edgeChoice_start
                Edges[:, M1, 1] = edgeChoice_end

                #This adds the theoryProbability and sampleProbability (described earlier) for this edge
                #to there respective sums.
                probLog1[argsLength] += torch.log(theoryProbability[argsLength]+1e-12)
                probLog2[argsLength] += torch.log(sampleProbability[argsLength]+1e-12)

                #print ('hi')
                #print (torch.mean(sampleProbability))
                #print (torch.mean(torch.log(sampleProbability[argsLength]+1e-12)))
                #quit()








        probLog1_np = probLog1.data.numpy()
        probLog2_np = probLog2.data.numpy()


        #This adjusts the probabiltiy baseline for each tree.
        baseLine = baseLine * ((baseN - 1) / baseN)
        baseLine = baseLine + ((1 / baseN) * np.exp(probLog1_np - probLog2_np)   )


        #plt.plot(probLog2_np  * 0)
        #plt.show()
        #quit()



        #This just records data for analysis
        recordBase[iter] = np.copy(probLog1_np)
        recordSamp[iter] = np.copy(probLog2_np)




        #adjustProbability means that the algorithm optimizes to accuratly represent the probability
        #of different trees, rather than just trying to maximize the probability of the very most
        #likely trees (which is useful in some situations not discussed in the paper)
        if adjustProbability:
            baseLineLog = np.log(baseLine)
            #baseLineLog = np.copy(baseLine)


            #print (probLog2[:10])

            #This is the reinforcement learning loss function, before some adjustments for sampling frequency
            loss_array = probLog1 / (torch.exp(probLog2.detach()) + 1e-10)

            loss_array = loss_array / maxM


            sampleUnique, sampleIndex = np.unique(sampleInverse, return_index=True)

            #This will give some adjustement terms associated with sampling frequency.
            #Specifically, adjustments for the fact that things are not sampled exactly proportional to
            #there liklyhood according to the model. For more detailed information, read the paper.
            prob_adjustment = np.zeros(sampleInverse.shape[0])

            baseLineMean = np.zeros(int(np.max(sampleInverse) + 1)) + 1

            for b in range(0, sampleIndex.shape[0]):
                start1 = sampleIndex[b]
                if b == sampleIndex.shape[0] - 1:
                    end1 = N1
                else:
                    end1 = sampleIndex[b+1]

                argsLocal = np.arange(end1 - start1) + start1
                localProb = probLog1_np[argsLocal]
                localBaseline = baseLineLog[argsLocal]
                #maxLogProb = max(np.max(localBaseline), np.max(localProb))
                maxLogProb = np.max(localBaseline)
                localProb = localProb - maxLogProb
                localBaseline = localBaseline - maxLogProb



                localProb = np.exp(localProb) / (np.sum(np.exp(localBaseline)) + 1e-5)


                prob_adjustment[argsLocal] = np.copy(localProb)


                baseLineMean[int(sampleUnique[b])] = np.sum(baseLine[argsLocal])

            #This applies the adjustment to the loss function
            loss_array = loss_array * torch.tensor(prob_adjustment)

            #This takes the loss on the training set trees.
            loss_array = loss_array[trainSet2]


            #Thiscalculates the unsupervised learning log liklyhood loss.
            # Note, this is not the same as the reinforcement learning reward function.
            score_train = np.mean(np.log(baseLineMean[trainSet] + 1e-20))
            score_test = np.mean(np.log(baseLineMean[testSet] + 1e-20))

            #quit()

        else:


            loss_array = torch.exp( probLog1 - probLog2.detach() )
            loss_array = loss_array[trainSet2]


        #This gives a minus sign, since we minimize the negative of the reward function mean.
        loss = -1 * torch.mean(loss_array)


        #This adds regularization.
        #There are some small subset of parameters where regularization is not useful for
        #preventing overfitting or increasing interpretability
        regularization = 0
        numHigh = 0
        numAll = 0
        c1 = 0
        for param in model.parameters():
            if c1 in [0, 2, 3]:
                regularization = regularization + torch.sum( torch.abs(param) ) #- ( 0.9 * torch.relu( torch.abs(param) - 0.1 ))       )
                numHigh += np.argwhere(np.abs(np.abs(param.data.numpy()) < 0.01)).shape[0] #Just recording information
                numAll += np.argwhere(np.abs(np.abs(param.data.numpy()) > -1)).shape[0] #Just recording information
                #numAll += param.size
            c1 += 1

        #regularization = regularization * 0.0001
        #regularization = regularization * 0.0002 #Best for breast cancer
        #regularization = regularization * 0.002 #Used for our occurance simulation as well
        regularization = regularization * regFactor

        #Adding regularization to the loss
        loss = loss + regularization

        #Printing information about training
        if doPrint:

            #print (baseLine)
            #quit()

            #print ("")
            print ('Mean Probability: ', np.mean(baseLine))
            print ('Training Score: ', score_train, 'Testing Score:', score_test)
            print ('Loss: ', loss.data.numpy())

            #Saving the probabilities and model.
            if baselineSave and fileSave:
                torch.save(model, fileSave)
                np.save(baselineSave, baseLine)


        if iter == 500:
            #This allows for initially having a higher learning rate and then moving to a lower learning rate.
            optimizer = torch.optim.SGD(model.parameters(), lr = 1e0)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def trainRealData(dataName, maxM=10, trainPer=0.666):

    #This trains on real data sets or custum data sets.
    #Simulated data sets in the paper are given there own individual functions for training the model.


    #This loads in the data.
    if dataName == 'AML':
        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/manualCancer.npy')
    elif dataName == 'breast':
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')
    else:
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, dataName)
        runInfo = [maxM]
        np.save('./Models/runInfo_' + dataName + '.npy', runInfo)


    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/lungData/processed.npy')




    #This code creates a training set test set split using a random state.
    rng = np.random.RandomState(2)

    N2 = int(np.max(sampleInverse)+1)
    trainSet = rng.permutation(N2)

    N3 = int(np.floor(trainPer * N2))
    trainSet = trainSet[:N3]


    print (N2)
    print (np.unique(sampleInverse).shape)
    quit()
    quit()

    ''''
    #This code actually trains the model using the data.
    if dataName == 'AML':
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_manual_PartialTrain_ex.pt', baselineSave='./Models/baseline_manual_Partial.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.015)#, regularizeFactor=0.005)#, regularizeFactor=0.01)
    elif dataName == 'breast':
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_breast_ex.pt', baselineSave='./Models/baseline_breast_ex.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
    else:
        #trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_' + dataName + '.pt', baselineSave='./Models/baseline_' + dataName + '.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_' + dataName + '.pt', baselineSave='./Models/baseline_' + dataName + '.npy',
                            adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.0005) #0.00005 #0.0005 #too small 0.00001
    #'''

def doProportionAnalysis():

    import matplotlib.pyplot as plt

    def makeLabel(lengthList, Labels):


        label1 = np.zeros( int(np.sum(np.array(lengthList))), dtype=int )

        posList = np.cumsum(np.array(lengthList))
        posList = np.concatenate(( np.zeros(1, dtype=int), posList ))

        for a in range(len(lengthList)):
            label1[posList[a]:posList[a+1]] = Labels[a]

        return label1






    #This function does a simple analysis of the proportion based fitness of different mutations,
    #based on data that I manually recorded. This proportion based fitness can then be compared
    #to the mutation based fitness found by CloMu.


    #import scipy.stats.sem
    from scipy.stats import sem

    'NPM1'
    ar1 = [[30.7, 62.1], [69.8, 21.8], [14.0, 29.1], [23.7, 66.5], [19.9, 66.3],
           [24.0, 16.4], [12.0, 16.3], [12.0, 16.3], [21.7, 13.5], [13.7, 21.4], [18.1, 62.0], [27.7, 20.3],
           [17.1, 72.0], [12.6, 23.5], [23.8, 67.9], [20.7, 64.9], [2.5, 9.3], [14.4, 76.6], [20.5, 67.2],
           [7.9, 29.9], [19.6, 40.4], [32.8, 33.8], [2.9, 5.1], [15.3, 37.4], [18.1, 50.2], [22.4, 68.1]]
    'ASXL1'
    ar2 = [[29.2, 45.0], [27.3, 55.3], [6.8, 26.3], [19.1, 48.1], [24.2, 26.9], [0.9, 3.5], [12.0, 26.1], [9.9, 80.9]]
    'DNMT3A'
    ar3 = [[3.7, 6.9], [8.4, 34.4], [16.5, 37.4], [6.6, 19.9], [4.2, 13.7],
           [3.5, 23.5], [12.1, 15.6], [15.1, 52.6], [6.3, 19.8], [2.2, 6.9]]
    'NRAS'
    ar4 = [[4.2, 1.0], [23.2, 13.2], [23.2, 13.3], [18.6, 17.4], [18.6, 6.6], [16.4, 40.5],
           [16.3, 71.7], [17.3, 12.1], [17.3, 9.0], [17.3, 9.0], [34.9, 34.0], [13.5, 8.9],
           [20.3, 14.5], [17.0, 13.7], [18.0, 19.5], [26.3, 21.3], [4.6, 17.1], [5.4, 5.0],
           [5.4, 10.6], [23.5, 27.6], [27.6, 5.9],  [33.5, 17.8], [33.5, 5.8], [48.1, 7.6],
           [48.1, 6.3], [17.4, 13.4], [54.0, 6.5], [29.9, 5.2], [29.9, 3.8], [27.6, 5.6],
           [11.1, 12.9], [56.6, 4.7], [33.8, 34.0], [6.6, 38.7], [42.6, 3.2], [19.9, 15.5],
           [6.9, 44.0], [44.0, 8.0], [44.0, 8.8], [80.9, 6.6] ]
    'FLT3'
    ar5 = [[3.2, 15.9], [4.2, 0.7], [25.5, 9.9], [30.7, 10.6], [14.9, 8.6],
           [14.9, 1.7], [45.4, 15.1], [28.4, 1.6], [13.5, 15.2], [21.7, 9.3],
           [8.3, 66.1], [36.4, 3.1], [23.5, 4.8], [23.5, 10.3], [5.8, 20.7],
           [29.9, 28.7], [50.8, 43.9], [44.0, 7.5], [44.0, 6.1]]
    'IDH1'
    ar6 = [[4.2, 0.0], [1.0, 0.1], [42.4, 13.5], [46.4, 4.0], [14.5, 13.8],
           [16.7, 23.0], [1.7, 5.4], [1.7, 5.9], [4.3, 15.6], [3.5, 0.9],
           [15.6, 27.1], [4.2, 15.3], [21.9, 5.6], [1.4, 9.9]]
    'PTPN11'
    ar7 = [[70.3, 17.2], [45.4, 22.3], [18.6, 22.0], [19.5, 12.8], [21.4, 32.4],
            [5.4, 1.7], [5.9, 7.0], [7.0, 0.8], [33.8, 5.6], [33.5, 4.9],
            [11.1, 21.9], [11.1, 4.2], [5.1, 40.6], [5.1, 33.3], [5.1, 1.4], [16.0, 33.8], [19.9, 19.3] ]
    'FLT3-ITD'
    ar8 = [[4.2, 70.3], [25.5, 12.3], [23.2, 17.7], [14.9, 69.6], [26.3, 21.0],
            [16.3, 0.0], [28.4, 28.6], [44.7, 4.3], [4.5, 6.3], [42.4, 17.1],
            [18.0, 13.9], [8.3, 11.3], [32.4, 21.1], [5.4, 21.9], [27.6, 6.5],
            [3.1, 9.3], [46.6, 7.9], [4.7, 14.4], [29.9, 17.4], [40.2, 14.2],
            [27.2, 24.9], [49.0, 10.1], [5.1, 6.6], [37.4, 25.0], [22.0, 62.9]]


    mutNames = ['NPM1', 'ASXL1', 'DNMT3A', 'NRAS', 'FLT3', 'IDH1', 'PTPN11', 'FLT3-ITD']


    ar1 = np.array(ar1) + 1
    ar2 = np.array(ar2) + 1
    ar3 = np.array(ar3) + 1
    ar4 = np.array(ar4) + 1
    ar5 = np.array(ar5) + 1
    ar6 = np.array(ar6) + 1
    ar7 = np.array(ar7) + 1
    ar8 = np.array(ar8) + 1




    arList = [ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8]

    lengthList = [ar1.shape[0], ar2.shape[0], ar3.shape[0], ar4.shape[0], ar5.shape[0], ar6.shape[0], ar7.shape[0], ar8.shape[0]]
    label1 = makeLabel(lengthList, [0, 1, 2, 3, 4, 5, 6, 7])
    arList = np.concatenate((ar1, ar2, ar3, ar4, ar5, ar6, ar7, ar8))

    prop = np.log(arList[:, 1] / arList[:, 0])


    print (np.mean(  np.log(ar1[:, 1] / ar1[:, 0]) )) #NP
    print (np.mean(  np.log(ar2[:, 1] / ar2[:, 0]) )) #AS
    print (np.mean(  np.log(ar3[:, 1] / ar3[:, 0]) )) #DMT
    print (np.median(  np.log(ar4[:, 1] / ar4[:, 0]) ))
    print (np.median(  np.log(ar5[:, 1] / ar5[:, 0]) ))
    print (np.median(  np.log(ar6[:, 1] / ar6[:, 0]) ))
    print (np.median(  np.log(ar7[:, 1] / ar7[:, 0]) ))
    print (np.median(  np.log(ar8[:, 1] / ar8[:, 0]) ))


    v1 = np.mean(  np.log(ar1[:, 1] / ar1[:, 0]) ) #NP
    v2 = np.mean(  np.log(ar2[:, 1] / ar2[:, 0]) ) #AS
    v3 = np.mean(  np.log(ar3[:, 1] / ar3[:, 0]) ) #DMT
    v4 = np.median(  np.log(ar4[:, 1] / ar4[:, 0]) )
    v5 = np.median(  np.log(ar5[:, 1] / ar5[:, 0]) )
    v6 = np.median(  np.log(ar6[:, 1] / ar6[:, 0]) )
    v7 = np.median(  np.log(ar7[:, 1] / ar7[:, 0]) )
    v8 = np.median(  np.log(ar8[:, 1] / ar8[:, 0]) )

    vHigh = np.array([v1, v2, v3])
    vLow = np.array([v4, v5, v6, v7, v8])

    print (np.mean(vHigh))
    print (np.mean(vLow))
    quit()



    #print (np.median(prop[label1== 5]))
    quit()

    np.save('./sending/proportion/propFitness.npy', prop)
    np.save('./sending/proportion/MutationNames.npy', np.array(mutNames)[label1] )
    #np.save('./sending/proportion/OurFitness.npy',  )


    plt.scatter(label1, prop)
    plt.show()
    quit()


    for a in range(len(arList)):
        #print (a)
        print (mutNames[a])

        mean1 = np.mean(np.log(arList[a][:, 1] / arList[a][:, 0]))
        sig1 = scipy.stats.sem(np.log(arList[a][:, 1] / arList[a][:, 0]))

        print (str(mean1)[:5] + str(' +- ') + str(sig1)[:5])


#doProportionAnalysis()
#quit()



def trainMHNsim():

    folder1 = './treeMHN/treeMHNdata/np'
    saveFolder = './treeMHN/treeMHNdata/models'

    subFolders = os.listdir(folder1)
    if '.DS_Store' in subFolders:
        subFolders.remove('.DS_Store')

    #print (subFolders)
    #quit()

    done1 = False
    existingFolders = os.listdir(saveFolder)

    for a in range(0, len(subFolders)):
        subFolder = subFolders[a]

        if not subFolder in existingFolders:
            newFolder1 = saveFolder + '/' + subFolder
            os.mkdir(newFolder1)

        print (a)
        print (subFolder)

        Mname = int(subFolder.split('_')[0][1:])
        Nname = int(subFolder.split('_')[1][1:])

        if True:# Mname == 10:
            if Nname == 300:#(Nname in [100, 300]):
                #done1 = True


                start1 = 0
                #if False:
                for b in range(start1, 20):##range(100):

                    #if (a != 0) or (b != 0):

                    fullSaveFolder = saveFolder + '/' + subFolder + '/'

                    mutNum = subFolder.split('_')[0]
                    mutNum = int(mutNum[1:])


                    #maxM = 10
                    maxM = mutNum

                    dataName = folder1 + '/' + subFolder + '/trees_' + str(b)

                    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, dataName, fullDir=True)

                    rng = np.random.RandomState(2)

                    N2 = int(np.max(sampleInverse)+1)
                    trainSet = rng.permutation(N2)

                    trainPer = 1.0

                    N3 = int(np.floor(trainPer * N2))
                    trainSet = trainSet[:N3]


                    trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=fullSaveFolder + str(b) + '_12.pt', baselineSave=fullSaveFolder + 'baseline_' +  str(b) + '.npy',
                                        adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.0005 * 2)


def trainBootAML():


    maxM = 10
    trees, sampleInverse, mutationCategory, treeSizes, uniqueMutation, M = processTreeData(maxM, './data/realData/AML', fullDir=True)

    T = 0
    folder1 = './data/simulations/realBoostrap'
    folder2 = './Models/simulations/realBoostrap'

    for a in range(20):


        newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
        newTrees = newTrees.astype(int)
        sampleInverse = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
        #mutationCategory = ''

        maxM = newTrees.shape[1]
        M = int(np.max(newTrees+1)) - 2

        #This loads the length of each tree
        treeLength = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
        treeLength = treeLength[sampleInverse]

        N2 = int(np.max(sampleInverse)+1)

        trainSet = np.arange(N2)
        trainSet = trainSet[:N2//2]


        #Preparing the files to save the model and tree probabilities
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        baselineFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_baseline.pt'


        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelFile, baselineSave=baselineFile,
                            adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.01, nonLin=True) #0.02


def trainNewSimulations(folder1, folder2, T, N, modL=False, regFactor=0.0002, Niter=1000):

    #This function trains models on the new simulated data sets formed by this paper.


    for a in range(0, 20):



        #Loading in the saved data sets
        newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
        newTrees = newTrees.astype(int)
        sampleInverse = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
        mutationCategory = ''

        #print (newTrees.shape)
        #quit()

        #print (np.unique(sampleInverse).shape)
        #quit()

        maxM = newTrees.shape[1]
        M = int(np.max(newTrees+1)) - 2


        #print (treeLength.shape[0])
        #print (newTrees.shape)

        #quit()



        #print (sampleInverse.shape)
        #print (newTrees.shape)
        #print (np.unique(sampleInverse).shape)
        #quit()
        #print (np.unique(newTrees))
        #quit()

        #This loads the length of each tree
        treeLength = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
        treeLength = treeLength[sampleInverse]

        #print (treeLength.sh)


        #Creating a training set test set split
        #rng = np.random.RandomState(2)

        #This calculates the number of patients in the simulation
        N2 = int(np.max(sampleInverse)+1)
        #trainSet = np.random.permutation(N2)
        #trainSet = rng.permutation(N2)

        trainSet = np.arange(N2)

        #if folder1 != './data/simulations/passanger':
        trainSet = trainSet[:N2//2]


        #Preparing the files to save the model and tree probabilities
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        baselineFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_baseline.pt'

        #modelFile = './temp/temp.pt'
        #baselineFile = './temp/temp.npy'


        #Training the model
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelFile, baselineSave=baselineFile, adjustProbability=True, trainSet=trainSet, unknownRoot=True, regFactor=regFactor, Niter=Niter)




#import tracemalloc
#import time

#tracemalloc.start()
#time1 = time.time()


folder1 = './data/simulations/random'
folder2 = './Models/simulations/random'
#trainNewSimulations(folder1, folder2, 3, 1)
#quit()


#folder1 = './data/simulations/fewSamples'
#folder2 = './Models/simulations/fewSamples'
#############trainNewSimulations(folder1, folder2, 0, 1)
#quit()



folder1 = './data/simulations/I-a'
folder2 = './Models/simulations/I-a'
########trainNewSimulations(folder1, folder2, 4, 1)
#quit()


folder1 = './data/simulations/latent'
folder2 = './Models/simulations/latent'
##############trainNewSimulations(folder1, folder2, 0, 1, regFactor=0.001)
#quit()
folder1 = './data/simulations/nonlinear'
folder2 = './Models/simulations/nonlinear'
##############trainNewSimulations(folder1, folder2, 0, 1) #
#quit()


folder1 = './data/simulations/negativeInter'
folder2 = './Models/simulations/negativeInter'
###########trainNewSimulations(folder1, folder2, 0, 1)
#quit()

folder1 = './data/simulations/bootstrap'
folder2 = './Models/simulations/bootstrap'
##############trainNewSimulations(folder1, folder2, 0, 1)
#quit()


folder1 = './data/simulations/effectSize'
folder2 = './Models/simulations/effectSize'
##############trainNewSimulations(folder1, folder2, 0, 1)
#quit()

folder1 = './data/simulations/passanger'
folder2 = './Models/simulations/passanger'
##############trainNewSimulations(folder1, folder2, 0, 1)
#quit()


folder1 = './data/simulations/realBoostrap'
folder2 = './Models/simulations/realBoostrap'
##############trainNewSimulations(folder1, folder2, 1, 1, regFactor = 0.0002, Niter=100)#, regFactor = 0.001) 0.0005 #0.0002
#quit()



folder1 = './data/simulations/lowPatient'
folder2 = './Models/simulations/lowPatient'
##############trainNewSimulations(folder1, folder2, 4, 1) #300
##############trainNewSimulations(folder1, folder2, 0, 1, regFactor=0.003) #0, 1, 2


#print(tracemalloc.get_traced_memory())
#tracemalloc.stop()

#print ('time')
#print (time.time() - time1)
#quit()



def testNonlinearSimulations(folder1, folder2, T=4, N=20):


    def calculateErrors(cutOff, theta_true, pred_now, mask2):


        pred_now_copy = np.copy(pred_now)



        #pred_now_copy[pred_now_copy> cutOff] = 1
        #pred_now_copy[pred_now_copy< (-1 * cutOff)] = -1
        #pred_now_copy[np.abs(pred_now_copy) < cutOff] = 0


        print (cutOff)
        #figure, axis = plt.subplots(1, 3)
        #axis[0].imshow(theta_true)
        #axis[1].imshow(pred_now)
        #axis[2].imshow(pred_now_copy)
        #plt.show()

        #choiceShow = 5

        #pred_now = pred_now[argTop][:, argTop]
        #pred_now = pred_now.T
        pred_now = pred_now[mask2 == 1]

        argAbove = np.argwhere(pred_now > cutOff)[:, 0]
        argBelow = np.argwhere(pred_now < (-1*cutOff))[:, 0]
        pred_now[:] = 0
        pred_now[argAbove] = 1
        pred_now[argBelow] = -1

        #pred_now[pred_now> cutOff] = 1
        #pred_now[pred_now< (-1 * cutOff)] = -1
        #pred_now[np.abs(pred_now) < cutOff] = 0


        #theta_true = theta_true[argTop][:, argTop]

        theta_true = theta_true[mask2 == 1]
        #theta_true = theta_true.reshape((M*M,))
        theta_true[theta_true> 0.01] = 1
        theta_true[theta_true<-0.01] = -1
        theta_true[np.abs(theta_true) < 0.02] = 0






        FalseArg = np.argwhere(  (pred_now - theta_true) != 0 )[:, 0]
        TrueArg = np.argwhere(  (pred_now - theta_true) == 0 )[:, 0]

        #print (FalseArg.shape)
        #print (TrueArg.shape)
        #quit()

        TruePosArg = TrueArg[theta_true[TrueArg] != 0]
        TrueNegArg = TrueArg[theta_true[TrueArg] == 0]

        truePosNum = TruePosArg.shape[0]
        trueNegNum = TrueNegArg.shape[0]

        falseNegNum = np.argwhere( np.logical_and(  pred_now == 0, theta_true != 0   )).shape[0]

        falsePosNum = FalseArg[pred_now[FalseArg] != 0].shape[0]

        #print (truePosNum, falsePosNum, trueNegNum, falseNegNum)
        #quit()

        #return (truePosNum, falsePosNum, trueNegNum, falseNegNum)

        #precision1 = truePosNum

        precision1 = float(truePosNum) / (float(falsePosNum) + float(truePosNum) + 1e-10)
        recall1 = float(truePosNum) / (float(falseNegNum) + float(truePosNum))

        return precision1, recall1





    import matplotlib.pyplot as plt

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships


    folder1 = './data/simulations/nonlinear'
    folder2 = './Models/simulations/nonlinear'


    #T = 4
    #T = 0
    #N = 100

    N = 20

    #T == 4


    M = 10

    if T in [9, 12]:
        M = 15

    errorList = []

    cutOffs = [0.1, 0.2, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]

    our_acc = np.zeros((20, len(cutOffs), 2))

    savedRelationships = np.zeros((20, 4, 8))


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    #for a in range(20, N):
    for a in range(0, 20):

        print (a)

        pairFile = folder1 + '/T_' + str(T) + '_R_' + str(a) + '_nonlinPairs.npz'
        pairs = loadnpz(pairFile)
        #print (pairs)

        randEffect = loadnpz('./data/simulations/nonlinear/T_' + str(T) + '_R_' + str(a) + '_causes.npz')


        #This loads in the model
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)


        X = torch.zeros((M**2, M))
        X[np.arange(M**2)  , np.arange(M**2)//M  ] = 1
        X[np.arange(M**2)  , np.arange(M**2)%M  ] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)


        X_simple = torch.zeros((M, M))
        X_simple[np.arange(M), np.arange(M)] = 1
        output_simple, _, = model(X_simple)
        output_simple_np = output_simple.data.numpy()

        #output = torch.log(torch.softmax(output, axis=1))


        #New
        X_normal = torch.zeros((1, M))
        output_normal, _ = model(X_normal)

        #output_normal = torch.log(torch.softmax(output_normal, axis=1))



        #This makes the probabilities a zero mean numpy array
        output_np = output.data.numpy()
        #output_np = output_np - np.mean(output_np)


        X1 = torch.zeros((M, M))
        X1[np.arange(M)  , M - 1 - np.arange(M)] = 1
        output1, _ = model(X1)



        output_normal = output_normal.data.numpy()
        for b in range(output.shape[1]):
            output_np[:, b] = output_np[:, b] - output_normal[0, b]
            output_simple_np[:, b] = output_simple_np[:, b] - output_normal[0, b]


        mask1 = np.ones(output_np.shape)
        mask1 = mask1.reshape((M, M, M))
        for b0 in range(M):
            mask1[b0, b0:] = 0 #b0, b0 to remove same mutation. b0, b0+1: to remove duplicates (unorder vs order pair).
            mask1[b0, :, b0] = 0
            mask1[:, b0, b0] = 0
        #for b0 in range(M-1):
        #    mask1[b0, b0:] = 0

        mask1 = mask1.reshape((M*M, M))



        savedRelationships[a, 0, :] = (randEffect + 1) / 2
        savedRelationships[a, 1, :] = np.copy(output_np[1, 2:])
        savedRelationships[a, 2, :] = np.copy(output_simple_np[0, 2:])
        savedRelationships[a, 3, :] = np.copy(output_simple_np[1, 2:])



        prob_true = np.zeros(output_np.shape)
        prob_true[1, 2:] = np.copy(randEffect)
        prob_true[10, 2:] = np.copy(randEffect)


        #plt.imshow(output_np)
        #plt.show()
        #quit()



        for cutOff0 in range(len(cutOffs)):

            cutOff = cutOffs[cutOff0]
            #quit()


            #if folder1 != './data/simulations/negativeInter':
            #    precision1, recall1 = simple_calculateErrors(cutOff, prob_true, output_np, mask1)
            #else:
            precision1, recall1 = calculateErrors(cutOff, prob_true, output_np, mask1)

            print (precision1, recall1)

            our_acc[a, cutOff0, 0] = precision1
            our_acc[a, cutOff0, 1] = recall1
        #quit()


        #plt.imshow(output1.data.numpy())
        #plt.show()
        #quit()

        #plt.imshow(output_simple_np)
        #plt.show()
        #quit()

        #plt.hist(output_np.reshape((1000,)), bins=100)
        #plt.show()

        #plt.imshow(output_np[:50])
        #plt.imshow(output_np[:100])
        #plt.show()


        #output_np = np.sum(output_np, axis=1)

        output_np = logsumexp(output_np[:, 5:], axis=1)


        #plt.imshow(output_np.reshape((M, M)))
        #plt.show()
        #quit()


    np.savez_compressed('./plotData/nonlinearComponenet.npz', savedRelationships)

    F1score = (2 * our_acc[:, :, 0] * our_acc[:, :, 1]) / (our_acc[:, :, 0] + our_acc[:, :, 1])
    F1score = np.mean(F1score, axis=0)

    #print ('F1', np.max(F1score))
    #quit()

    #quit()

    plt.plot(np.mean(our_acc[4:, :, 1], axis=0), np.mean(our_acc[4:, :, 0], axis=0))
    plt.scatter(np.mean(our_acc[4:, :, 1], axis=0), np.mean(our_acc[4:, :, 0], axis=0))
    #plt.legend(['original simulation', 'modified effect sizes'])
    plt.xlabel("multi-mutation causality recall")
    plt.ylabel("multi-mutation causality precision")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/nonlinearCurve.pdf')
    plt.show()

    quit()


def plotNonlinearComponent():

    savedRelationships = loadnpz('./plotData/nonlinearComponenet.npz')

    #savedRelationships = savedRelationships[1:]


    nonlinearEffect = savedRelationships[:, 1]
    linearEffect = savedRelationships[:, 2] + savedRelationships[:, 3]

    #nonlinearEffect = nonlinearEffect[savedRelationships[:, 0] == 0]
    #linearEffect = linearEffect[savedRelationships[:, 0] == 0]

    min1 = np.min(linearEffect)
    max1 = np.max(nonlinearEffect)

    diff0 = nonlinearEffect - linearEffect
    diff1 = diff0[savedRelationships[:, 0] == 0]
    diff2 = diff0[savedRelationships[:, 0] == 1]

    if True:
        ar1 = nonlinearEffect[savedRelationships[:, 0] == 0]
        ar2 = linearEffect[savedRelationships[:, 0] == 0]
        min1 = min(np.min(ar1), np.min(ar2))
        max1 = max(np.max(ar1), np.max(ar2))

        plt.hist(ar1, alpha=0.5, range=(min1, max1))
        plt.hist(ar2, alpha=0.5, range=(min1, max1))
        plt.gcf().set_size_inches(8, 6)
        plt.ylabel('count')
        plt.xlabel('effect strength')
        plt.legend(['multicausality', 'linear predicted causality'])
        #plt.savefig('./images/nonlinearNegative.pdf')
        plt.show()

        ar1 = nonlinearEffect[savedRelationships[:, 0] == 1]
        ar2 = linearEffect[savedRelationships[:, 0] == 1]
        min1 = min(np.min(ar1), np.min(ar2))
        max1 = max(np.max(ar1), np.max(ar2))

        plt.hist(ar1, alpha=0.5, range=(min1, max1))
        plt.hist(ar2, alpha=0.5, range=(min1, max1))
        plt.gcf().set_size_inches(8, 6)
        plt.ylabel('count')
        plt.xlabel('effect strength')
        plt.legend(['multicausality', 'linear predicted causality'])
        #plt.savefig('./images/nonlinearPositive.pdf')
        plt.show()

    #quit()

    if False:
        plt.hist(diff1, bins=100, range=(np.min(diff1), np.max(diff1)))
        plt.xlabel('nonlinear effect')
        plt.ylabel('count')
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/nonlinearNegative.pdf')
        plt.show()

        plt.hist(diff2, bins=100, range=(np.min(diff2), np.max(diff2)))
        plt.xlabel('nonlinear effect')
        plt.ylabel('count')
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/nonlinearPositive.pdf')
        plt.show()

        #plt.hist(nonlinearEffect - linearEffect, bins=100)
        #plt.show()


def testLatentSimulation(folder1, folder2, T=4, N=20):


    def giveTransformMatrix(theta, flip0):

        matrix1 = np.zeros((2, 2))

        matrix1[0, 0] = np.cos(theta)
        matrix1[1, 1] = np.cos(theta)
        matrix1[0, 1] = np.sin(theta)
        matrix1[1, 0] = -np.sin(theta)

        matrix2 = np.zeros((2, 2))
        matrix2[0, 0] = 1
        matrix2[1, 1] = 1
        if flip0 == 1:
            matrix2[1, 1] = -1

        matrix3 = np.matmul(matrix2, matrix1)

        return matrix3

    def checkError(latent2, props2, theta, flip0):

        matrix3 = giveTransformMatrix(theta, flip0)

        latentShift = np.matmul(latent2, matrix3)


        error = (latentShift - props2) ** 2
        error = np.sum(error)

        return error

    import matplotlib.pyplot as plt

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships

    #T = 4
    #T = 0
    #N = 100

    N = 20

    #T == 4


    M = 20



    errorList = []


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)


    property_all = np.zeros((20, 2, 20, 2))

    pairList = np.argwhere(np.ones((20, 20)) == 1)
    pairList = pairList[pairList[:, 0] - pairList[:, 1] > 0]

    dist1_all = np.zeros(20*pairList.shape[0])
    dist2_all = np.zeros(20*pairList.shape[0])

    error_all = np.zeros(20*20)
    norm_all = np.zeros(20*20)


    #for a in range(20, N):
    for a in range(0, 20):

        print (a)

        propertyFile = folder1 + '/T_' + str(0) + '_R_' + str(a) + '_property.npz'
        props = loadnpz(propertyFile)




        propertyA = np.copy(props[:, 0])
        propertyB = np.copy(props[:, 1])
        catA = propertyA / (np.sum(props ** 2, axis=1) ** 0.5)
        catB = propertyB / (np.sum(props ** 2, axis=1) ** 0.5)
        probabilityMatrix = np.zeros((M, M))
        probabilityMatrix[:, :] =  propertyA.reshape((-1, 1)) * catA.reshape((1, -1))
        probabilityMatrix[:, :] += propertyB.reshape((-1, 1)) * catB.reshape((1, -1))
        probabilityMatrix = probabilityMatrix * np.log(10)








        #This loads in the model
        modelFile = folder2 + '/T_' + str(0) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M) ] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, xLatent = model(X)
        output = output.data.numpy()




        #fig, axs = plt.subplots(2)
        #axs[0].imshow(output, cmap='bwr')
        #axs[1].imshow(probabilityMatrix, cmap='bwr')
        #plt.show()

        if False:#a == 0:

            probabilityMatrix_plot = np.copy(probabilityMatrix)
            probabilityMatrix_plot[np.arange(M), np.arange(M)] = 0

            sns.set_style('white')
            max1 = np.max(np.abs(probabilityMatrix_plot))

            plt.imshow(probabilityMatrix_plot, cmap='bwr')
            plt.clim(-max1, max1)
            plt.xlabel('target mutation $t$')
            plt.ylabel('source mutation $s$')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.gcf().set_size_inches(8, 6)
            plt.savefig('./images/latentCausalExample.pdf')
            plt.show()

            quit()


        #xLatent = xLatent / (np.mean(xLatent ** 2) ** 0.5)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        latent2 = pca.fit_transform(xLatent)
        latent2 = latent2 / ((np.mean(latent2**2)*2)**0.5)

        #props2 = pca.fit_transform(props)
        #norm1 = (np.mean(props ** 2) * 4) ** 0.5
        #props2 = props / norm1
        props2 = np.copy(props)
        props2[:, 0] = props2[:, 0] - np.mean(props2[:, 0])
        props2[:, 1] = props2[:, 1] - np.mean(props2[:, 1])
        props2 = props2 / ((np.mean(props2**2)*2)**0.5)

        #print (np.mean(latent2**2))
        #quit()


        Ntheta = 100
        errorList = np.zeros((Ntheta, 2))
        for theta0 in range(Ntheta):
            for flip0 in range(2):

                theta = (theta0/Ntheta) * np.pi * 2

                error = checkError(latent2, props2, theta, flip0)

                errorList[theta0, flip0] = error


        min1 = np.min(errorList)
        argmin = np.argwhere(errorList == min1)[0]

        theta0, flip0 = argmin[0], argmin[1]
        theta = (theta0/Ntheta) * np.pi * 2
        matrix3 = giveTransformMatrix(theta, flip0)

        latentShift = np.matmul(latent2, matrix3)


        error_all[a*20:(a+1)*20] = np.sum((latentShift - props2) ** 2, axis=1) ** 0.5
        norm_all[a*20:(a+1)*20] = np.sum(props2 ** 2, axis=1) ** 0.5









        change1 = np.array([props2, latentShift])


        property_all[a, 0] = np.copy(props2)
        property_all[a, 1] = np.copy(latentShift)

        if True:
            plt.scatter(props2[:, 0], props2[:, 1])
            plt.scatter(latentShift[:, 0], latentShift[:, 1])
            plt.plot(change1[:, :, 0], change1[:, :, 1], color='gray')
            plt.legend(['true property value', 'predicted property value'])
            plt.xlabel('property 1')
            plt.ylabel('property 2')
            plt.gcf().set_size_inches(8, 6)
            #plt.savefig('./images/latentExample.pdf')
            plt.show()


        dist1 = np.sum( (xLatent[pairList[:, 0]] - xLatent[pairList[:, 1]]) ** 2, axis=1)
        dist2 = np.sum( (props[pairList[:, 0]] - props[pairList[:, 1]]) ** 2, axis=1)
        size1 = pairList.shape[0]

        dist1_all[size1*a:size1*(a+1)] = np.copy(dist1)
        dist2_all[size1*a:size1*(a+1)] = np.copy(dist2)


    if True:
        max1 = max(np.max(norm_all), np.max(error_all))
        plt.hist(norm_all, bins=50, range=(0, max1), alpha=0.5)#histtype='step')
        plt.hist(error_all, bins=50, range=(0, max1), alpha=0.5)#histtype='step')
        plt.xlabel('Euclidean distance')
        plt.ylabel('count')
        plt.legend(['property vector norm', 'distance from prediction'])
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/latentHist.pdf')
        plt.show()


    if False:
        #import scipy
        #from scipy import stats
        #print (scipy.stats.pearsonr(dist1_all, dist2_all))

        #subset1 = np.random.permutation(dist1_all.shape[0])[:1000]
        #plt.scatter(dist1_all[:], dist2_all[:], s=3)
        #plt.xlabel('distance between latent representations')
        #plt.ylabel('distance between mutation properties')
        #plt.gcf().set_size_inches(8, 6)
        #plt.savefig('./images/latentScatter.pdf')
        #plt.show()

        #sns.displot(x=dist1_all, y=dist2_all)
        #plt.show()
        True


def testOccurSimulations(folder1, folder2, T):

    import matplotlib.pyplot as plt

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships

    #T = 0

    N = 20

    #T == 4


    M = 10

    if T in [9, 12]:
        M = 15




    errorList = []


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    #for a in range(20, N):
    for a in range(2, 20):


        if folder1 == './data/simulations/passanger':
            newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
            newTrees = newTrees.astype(int)
            M = int(np.max(newTrees+1)) - 2

        if False:#folder1 == './data/simulations/passanger':

            newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
            newTrees = newTrees.astype(int)
            M = int(np.max(newTrees+1)) - 2

            #print (M)



            unique1 = np.unique(newTrees)
            inverse1 = np.zeros(int(np.max(newTrees+1)))
            inverse1[unique1] = np.arange(unique1.shape[0])
            newTrees = inverse1[newTrees]

            M = int(np.max(newTrees+1)) - 2

            #print (M)

            #print (np.unique(newTrees).shape)
            #quit()
            #quit()

        #This matrix is the set of true probability of true causal relationships
        probabilityMatrix = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz')

        probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0

        #This matrix is a binary matrix of true causal relationships
        prob_true = np.zeros((M, M))
        prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])
        prob_true[prob_true > 0.01] = 1



        #This loads in the model
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        #X = X[:1]
        #X[:] = 0

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)



        #New
        X_normal = torch.zeros((1, M))
        output_normal, _ = model(X_normal)

        #output_normal = torch.softmax(output_normal, axis=1)
        #output = torch.softmax(output, axis=1)
        #output_normal = np.log(output_normal.data.numpy())
        #output = np.log(output.data.numpy())

        #This makes the probabilities a zero mean numpy array
        output_np = output.data.numpy()
        #output_np = output_np - np.mean(output_np)



        output_normal = output_normal.data.numpy()
        for b in range(output.shape[1]):
            output_np[:, b] = output_np[:, b] - output_normal[0, b]
        #output_np = output



        #plt.imshow(output_np)
        #plt.show()
        #quit()


        #This converts the output probabilities into a binary of predicted causal relationships
        #by using a cut off probability.
        if True:
            if T == 4:
                #output_np = output_np - (np.max(output_np) * 0.2) #0.4 #Before Oct 13 2021
                output_np = output_np - 1.1

            elif T == 7:
                output_np = output_np - (np.max(output_np) * 0.35) #0.4 #Before Oct 13 2021

            elif folder1 == './data/simulations/lowPatient':

                output_np = output_np - (np.max(output_np) * 0.4)
            elif folder1 == './data/simulations/passanger':
                output_np = output_np - (np.max(output_np) * 0.4)

            else:
                output_np = output_np - (np.max(output_np) * 0.4)
            output_np[np.arange(M), np.arange(M)] = 0
            output_np_bool = np.copy(output_np)
            output_np_bool[output_np_bool > 0] = 1
            output_np_bool[output_np_bool < 0] = 0



        categories_update = np.zeros((2, 2))

        #This uses the predicted causal relationships and real causal relationships to
        #determine the rate of true positives, false positives, true negatives and false negatives
        for b in range(output_np.shape[0]):
            for c in range(output_np.shape[1]):
                if b != c:

                    int(prob_true[b, c])
                    int(output_np_bool[b, c])

                    categories[int(prob_true[b, c]), int(output_np_bool[b, c])] += 1
                    categories_update[int(prob_true[b, c]), int(output_np_bool[b, c])] += 1

        errorList.append([categories_update[1, 1], categories_update[0, 0], categories_update[0, 1], categories_update[1, 0] ])


        print ('')
        print ('True Positives: ' + str(categories_update[1, 1]))
        print ('True Negatives: ' + str(categories_update[0, 0]))
        print ('False Positives: ' + str(categories_update[0, 1]))
        print ('False Negatives: ' + str(categories_update[1, 0]))
        print ('')

        #plt.imshow(output_np)
        #plt.show()

    #This prints the final information on the accuracy of the method.
    print ('True Positives: ' + str(categories[1, 1]))
    print ('True Negatives: ' + str(categories[0, 0]))
    print ('False Positives: ' + str(categories[0, 1]))
    print ('False Negatives: ' + str(categories[1, 0]))

    errorList = np.array(errorList).astype(int)
    if T == 4:
        #np.save('./plotResult/cloMuCausal.npy', errorList)
        True


def testRealSubtreePrediction():



    def pickRandomPoint(rand1, treeLength, newTrees, mutationCategory, b):


        treeLength1 = treeLength[rand1[b]]

        treePoint1 = np.random.randint(treeLength1)
        treePoint[b] = treePoint1

        newMut = newTrees[rand1[b], treePoint1, 1]
        newMuation[b] = mutationCategory[newMut]
        #print (newMut)
        arg1 = np.argwhere(newTrees[rand1[b], :, 1] == newMut)
        while arg1.shape[0] > 0:
            newMut = newTrees[rand1[b], arg1[0, 0], 0]
            arg1 = np.argwhere(newTrees[rand1[b], :, 1] == newMut)
            if arg1.shape[0] > 0:
                #print (newMut)
                clones[b, mutationCategory[newMut]] = 1

        return clones, newMuation


    def deleteRandomFromTree(tree1):


        argDelete = np.argwhere( np.isin( tree1[:, 1] ,  tree1[:, 0] ) ==False)[:, 0]

        deleteChoice = np.random.randint(argDelete.shape[0])
        deleteChoice = tree1[argDelete[deleteChoice], 1]

        #print (deleteChoice)

        tree1 = tree1[tree1[:, 1] != deleteChoice ]

        return tree1


    def reOrderTree(tree1):

        firstClone = np.argwhere(np.isin(tree1[:, 0], tree1[:, 1]) == False)[0, 0]

        order1 = np.array([]).astype(int)
        mutInclude = np.array([tree1[firstClone, 0]])


        while order1.shape[0] < tree1.shape[0]:

            toInclude = np.argwhere( np.logical_and(  np.isin(tree1[:, 0], mutInclude  ), np.isin(tree1[:, 1],  mutInclude ) == False   ) )[:, 0]
            order1 = np.concatenate((order1, toInclude))
            mutInclude = np.concatenate((mutInclude,  tree1[toInclude, 1] ))

        tree1 = tree1[order1]

        assert np.unique(order1).shape[0] == order1.shape[0]

        return tree1


    def findClones(tree1, mutationCategory, Mcat):

        clones = np.zeros((tree1.shape[0]+1, Mcat))

        #print ("A")
        #print (tree1)
        #print (tree1)
        #print ("B")

        for a in range(0, tree1.shape[0]):
            #print (tree1)
            #print (tree1[a])


            argBefore = np.argwhere(tree1[:, 1] == tree1[a, 0])
            if argBefore.shape[0] > 0:
                argBefore = argBefore[0, 0] + 1
                clones[a+1] = np.copy(clones[argBefore])

            clones[a+1, mutationCategory[ tree1[a, 1] ]] = 1

        return clones


    def findToAdd(tree1, toAdd, mutationCategory, Mcat):

        addBool = np.zeros((tree1.shape[0]+1, Mcat))

        for a in range(toAdd.shape[0]):
            pos1 = np.argwhere(tree1[:, 1] == toAdd[a, 0])#[0, 0]
            if pos1.shape[0] > 0:
                pos1 = pos1[0, 0]
                addBool[pos1+1, mutationCategory[toAdd[a, 1]]] = 1
            else:
                assert toAdd[a, 0] == tree1[0, 0]
                addBool[0, mutationCategory[toAdd[a, 1]]] = 1

        return addBool






    def pickRandomSubtree(rand1, treeLength, newTrees, mutationCategory, b, Mcat):



        treeLength1 = int(treeLength[rand1[b]])

        #treePoint1 = np.random.randint(treeLength1)
        #treePoint[b] = treePoint1

        subsetSize = np.random.randint(treeLength1-1) + 1

        removeNum = treeLength1 - subsetSize


        tree0 = newTrees[rand1[b], :treeLength1]

        tree0 = reOrderTree(tree0)

        tree1 = np.copy(tree0)
        #print (tree1)

        for b in range(removeNum):
            tree1 = deleteRandomFromTree(tree1)
            #print (tree1)


        cloneIncluded = np.concatenate( (tree1[:, 1], np.array([tree0[0, 0]])) , axis=0 )


        toAdd = np.argwhere( np.logical_and(  np.isin(tree0[:, 0], cloneIncluded)  , np.isin(tree0[:, 1], tree1[:, 1]) == False ) )[:, 0]
        toAdd = tree0[toAdd]

        #toAdd_unique = np.unique(mutationCategory[toAdd])

        #toAdd_unique, toAdd_count = np.unique( mutationCategory[toAdd], return_counts=True )

        #nextBool = np.zeros(Mcat)
        #nextBool[toAdd_unique] = 1




        clones = findClones(tree1, mutationCategory, Mcat)



        addBool = findToAdd(tree1, toAdd, mutationCategory, Mcat)

        return clones, addBool


        #return tree1, toAdd



    def getTreeMHNtheta(dataName, mutationCategory, uniqueMutation):


        if dataName == 'AML':
            theta = np.loadtxt('./otherMethod/AML_Theta.csv', delimiter=",", dtype=str)
        if dataName == 'breast':
            theta = np.loadtxt('./otherMethod/breast_Theta.csv', delimiter=",", dtype=str)
        theta = np.array(theta)[1:]
        theta = theta.astype(float)

        rateTheta = theta[np.arange(theta.shape[0]), np.arange(theta.shape[0])]
        rateTheta_argsort = np.argsort(rateTheta)[-1::-1]
        theta = theta[rateTheta_argsort]
        theta = theta[:, rateTheta_argsort]


        if dataName == 'AML':
            mutationNames = ['DNMT3A', 'IDH2', 'FLT3', 'NRAS', 'NPM1', 'TET2', 'KRAS', 'TP53', 'WT1', 'RUNX1',
                             'B1', 'IDH1', 'ASXL', 'PTPN11', 'SRSF2', 'GATA2', 'U2AF1', 'PPM1D', 'JAK2',
                             'MYC', 'KIT', 'EZH2', 'CBL', 'STAG2', 'BCOR', 'SETBP1', 'CSF3R', 'ETV6',
                             'PHF6', 'MPL', 'SMC3'] #B1 = SF3B1

        if dataName == 'breast':
            mutationNames = ['TP53', 'PIK3CA', 'CDH1', 'GATA3', 'MAP3K1', 'ESR1', 'PTEN', 'KMT2C', 'FOXA1',
                             'NF1', 'RB1', 'KMT2D', 'PIK3R1', 'RHOA', 'EPHA7', 'TSC2', 'CD79A', 'PRDM1']

        #quit()

        _, index1 = np.unique(mutationCategory, return_index=True)
        uniqueMutation = uniqueMutation[index1]

        position = np.zeros(uniqueMutation.shape[0], dtype=int) - 1
        for a in range(uniqueMutation.shape[0]):
            for b in range(len(mutationNames)):
                if mutationNames[b] in uniqueMutation[a]:
                    position[a] = b
                    uniqueMutation[a] = mutationNames[b]

        theta_extra = np.zeros((theta.shape[0]+1, theta.shape[0]+1))
        theta_extra[:theta.shape[0], :theta.shape[0]] = np.copy(theta)
        rateMin = np.min(theta[np.arange(theta.shape[0]), np.arange(theta.shape[0])])
        theta_extra[-1, -1] = rateMin

        theta = np.copy(theta_extra)


        theta = theta[position]
        theta = theta[:, position]

        for a in range(position.shape[0]):
            if position[a] != -1:
                for b in range(position.shape[0]):
                    if position[b] != -1:
                        if a != b:
                            if position[a] == position[b]:
                                theta[a, b] = 0


        theta = theta.T #To have correct convention with ours

        return theta



    dataName = 'AML'
    #dataName = 'breast'



    #This loads in the data.
    if dataName == 'AML':
        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/realData/AML', fullDir=True)
        #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/manualCancer.npy')

        mutationCategory = mutationCategory[:-2]
        #quit()

        model = torch.load('./Models/realData/savedModel_AML.pt')

    elif dataName == 'breast':
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/realData/breastCancer', fullDir=True)
        #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')


        mutationCategory = mutationCategory[:-2]

        model = torch.load('./Models/realData/savedModel_breast.pt')


    Mcat = np.unique(mutationCategory).shape[0]

    if True:

        theta = getTreeMHNtheta(dataName, mutationCategory, uniqueMutation)

        theta_matrix = np.copy(theta)
        theta_matrix[np.arange(theta_matrix.shape[0]), np.arange(theta_matrix.shape[0])] = 0
        theta_basline = theta[np.arange(theta_matrix.shape[0]), np.arange(theta_matrix.shape[0])]


            #plt.imshow(theta)
            #plt.show()

        #quit()





    #print (Mcat)
    #quit()


    if True:

        _, index1 = np.unique(sampleInverse, return_index=True)
        count1 = np.zeros(Mcat, dtype=int)

        for b in range(index1.shape[0]):
            treeSize = int(treeLength[index1[b]])
            unique1 = np.unique(newTrees[index1[b], :treeSize])
            unique1 = unique1[:-1]

            unique2, count2 = np.unique(mutationCategory[unique1], return_counts=True)

            count1[unique2] = count1[unique2] + count2


        rates = count1.astype(float)
        rates = rates / np.sum(rates)


        #plt.plot(np.log(rates))
        #lt.plot(theta_basline)
        #plt.show()
        #quit()




    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

    _, counts = np.unique(sampleInverse, return_counts=True)

    prob1 = 1 / counts.astype(float)
    prob1 = prob1 / np.sum(prob1)


    N = np.unique(sampleInverse)
    #batchSize = 10000
    #numRun = 1000000
    numRun = 100000

    #probOurs = np.zeros(numRun)
    #probBaseline = np.zeros(numRun)
    #probMHN = np.zeros(numRun)

    probAll = np.zeros((numRun, 3))


    rand1 = np.random.choice(N, size=numRun, replace=True, p=prob1).astype(int)

    for b in range(numRun):

        if b % 1000 == 0:
            print (b // 1000)

        clones, addBool = pickRandomSubtree(rand1, treeLength, newTrees, mutationCategory, b, Mcat)

        mask1 = np.ones(addBool.shape)
        sum1 = np.ones(addBool.shape[1])
        if dataName == 'breast':
            sum1 = np.sum(clones, axis=0)
            sum1 = 1 - sum1
            sum1[sum1<=0] = 0
            sum1 = sum1.reshape((1, -1))
            mask1 = sum1[np.zeros(clones.shape[0], dtype=int)]

        #mask1 = 1 - clones


        pred_MHN = np.matmul(clones, theta_matrix)
        pred_MHN = pred_MHN + theta_basline.reshape((1, -1))  #np.log(rates)#
        pred_MHN = torch.tensor(pred_MHN).float()
        pred_MHN = pred_MHN.reshape((1, pred_MHN.shape[0]*pred_MHN.shape[1]  ))
        pred_MHN = torch.softmax(pred_MHN, axis=1)
        pred_MHN = pred_MHN.data.numpy()
        pred_MHN = pred_MHN.reshape(addBool.shape)
        pred_MHN = (pred_MHN * mask1)
        pred_MHN = pred_MHN / np.sum(pred_MHN)


        #plt.imshow(pred_MHN)
        #plt.show()
        prob_MHN_now = np.sum(pred_MHN * addBool)
        #probMHN[b] = prob_MHN_now
        probAll[b, 2] = prob_MHN_now



        clones = torch.tensor(clones).float()
        pred, _ = model(clones)
        pred = pred.reshape((1, pred.shape[0]*pred.shape[1]  ))
        pred = torch.softmax(pred, axis=1)
        pred = pred.data.numpy()
        pred = pred.reshape(addBool.shape)
        pred = (pred * mask1)
        pred = pred / np.sum(pred)


        prob = np.sum(pred * addBool)

        #probOurs[b] = prob
        probAll[b, 1] = prob

        rates = rates * sum1
        rates = rates / np.sum(rates)
        probBase = rates * np.mean(addBool, axis=0)
        probBase = np.sum(probBase)

        #probBaseline[b] = probBase
        probAll[b, 0] = probBase

        if (b + 1) % 1000 == 0:

            #print (np.mean(probAll[:b], axis=0))

            print (np.mean(np.log(probAll[:b]+0.001), axis=0))


        #print (np.mean(np.log(probBase+0.001)), np.mean(np.log(prob+0.001)), np.mean(np.log(probMHN+0.001)))

        #print (np.log(1 / rates.shape[0]),  np.mean(np.log(probBase)), np.mean(np.log(prob)))

    #quit()
    if dataName == 'AML':
        np.savez_compressed('./results/realData/predSubtree_AML.npz', probAll)
    else:
        np.savez_compressed('./results/realData/predSubtree_breast.npz', probAll)


def plotRealSubtree():

    #dataName = 'AML'
    dataName = 'breast'

    if dataName == 'AML':
        probAll = loadnpz('./results/realData/predSubtree_AML.npz')
        saveName = './images/predMutationAML.pdf'
    else:
        probAll = loadnpz('./results/realData/predSubtree_breast.npz')
        saveName = './images/predMutationBreast.pdf'


    probAll = np.log(probAll+ 0.001)

    probAll = probAll.reshape((100, 1000, 3))
    probAll = np.mean(probAll, axis=1)

    min1 = np.min(probAll)
    max1 = np.max(probAll)

    plt.hist(probAll[:, 0], bins=20, alpha=0.5, range=(min1, max1))#histtype='step')
    plt.hist(probAll[:, 1], bins=20, alpha=0.5, range=(min1, max1))#histtype='step')
    plt.hist(probAll[:, 2], bins=20, alpha=0.5, range=(min1, max1))#histtype='step')
    plt.legend(['Baseline', 'CloMu', 'TreeMHN'])
    plt.ylabel('count')
    plt.xlabel('log probability')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig(saveName)
    plt.show()


def testOccurFitness(T, N):

    import matplotlib.pyplot as plt

    #This tests the models ability to determine fitness in
    #the simulated data set of causal relationships

    #T = 4
    #T = 0
    #N = 100

    #N = 20

    M = 10

    if T in [9, 12]:
        M = 15

    errorList = []

    allPred = np.zeros((N, M))
    allTrue = np.zeros((N, M))
    allDriver = np.zeros((N, M))

    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    #for a in range(20, N):
    for a in range(0, N):

        #This matrix is the set of true probability of true causal relationships
        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_prob.npz')

        probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0

        if T == 4:
            trueFitness = np.zeros((M, M))
            trueFitness[:5, :5] = np.copy(probabilityMatrix[:5, :5])
        else:
            trueFitness = np.copy(probabilityMatrix)

        trueFitness = np.exp(trueFitness)
        #trueFitness = trueFitness / np.sum(trueFitness)
        trueFitness = np.sum(trueFitness, axis=1)


        #This matrix is a binary matrix of true causal relationships
        prob_true = np.zeros((M, M))
        prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])
        prob_true[prob_true > 0.01] = 1

        sum1 = np.sum(prob_true.astype(int), axis=1)
        sum1[sum1!=0] = 1
        allDriver[a] = np.copy(sum1)



        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)


        X_normal = torch.zeros((1, M))
        output_normal, _ = model(X_normal)

        fitness = output.data.numpy()
        output_normal = output_normal[0].data.numpy()

        for b in range(M):
            fitness[:, b] = fitness[:, b] - output_normal[b]




        #fitness = output.data.numpy()
        #fitness = fitness - np.mean(fitness)
        fitness = np.exp(fitness)
        fitness = np.sum(fitness, axis=1)
        #fitness = torch.softmax(fitness, axis=0)
        #fitness = fitness.reshape((M, M))
        #fitness = torch.sum(fitness, axis=1)
        #fitness = fitness.data.numpy()

        #print (fitness)
        #print (trueFitness)
        #quit()


        allPred[a] = np.copy(fitness)
        allTrue[a] = np.copy(trueFitness)

        #print (trueFitness)
        #print (sum1)
        #quit()


    #print (allPred.shape)
    #quit()

    #plt.scatter( allPred.flatten() , allTrue.flatten() )
    #plt.show()
    #quit()


    np.save('./sending/fitness/prediction.npy', allPred)
    np.save('./sending/fitness/true.npy', allTrue)
    np.save('./sending/fitness/driver.npy', allDriver)

    #plt.plot(allPred.T, c='b')
    #plt.plot(allTrue.T, c='r')
    #plt.show()



def showRandomSelectionProb():

    T = 3

    N = 20

    accuracyList = []

    probAll = []
    probAll2 = []

    for a in range(0, N):
        baselineFile = './Models/simulations/random/T_' + str(T) + '_R_' + str(a) + '_baseline.pt.npy'
        ar = np.load(baselineFile)

        skip1 = 2
        if T == 3:
            skip1 = 3

        ar = ar[:(ar.shape[0] // 2)]
        folder1 = 'random'
        #trees = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
        #sampleInverse = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz')
        #edges = loadnpz('./data/simulations/' + folder1 + '/T_' + str(T) + '_R_' + str(a) + '_trees.npz')



        ar = np.log(ar)

        min1 = np.min(ar)
        max1 = np.max(ar)

        ar = ar[:500*skip1]


        ar = ar.reshape((500, skip1))

        argmax = np.argmax(ar, axis=1)

        accuracy = np.argwhere(argmax == 0).shape[0] / argmax.shape[0]
        accuracyList.append(accuracy)

        if True:#skip1 == 2:



            #ar1 = ar[:, 0]
            #ar2 = ar[:, 1]
            #diff1 = ar1 - ar2

            #accuracy = np.argwhere(diff1 > 0).shape[0] / diff1.shape[0]
            #accuracyList.append(accuracy)

            #prob = np.exp(diff1) / (np.exp(diff1) + 1.0)

            prob = np.exp(ar[:, 0]) / np.sum(np.exp(ar), axis=1)

            prob2 = np.exp(ar[:, 1]) / np.sum(np.exp(ar), axis=1)

            #prob = np.exp(diff1)
            #prob = diff1

            probAll = probAll + list(prob)

            probAll2 = probAll2 + list(prob2)


    print (np.mean(np.array(accuracyList)))

    probAll = np.array(probAll)
    probAll2 = np.array(probAll2)

    #print (np.argwhere(probAll > 0.5).shape[0] / probAll.shape[0])
    #quit()



    print (probAll.shape)

    plt.hist(probAll, bins=10)
    plt.hist(probAll2, bins=10, alpha=0.5)
    #plt.xlabel('relative probability of correct tree')
    plt.xlabel('relative probability')
    plt.ylabel('count')
    plt.legend(['correct tree', 'incorrect tree'])
    plt.tight_layout()
    #plt.xscale('log')
    if T == 1:
        plt.savefig('./images/new1/trueTreeProb.pdf')
        np.savez_compressed('./plotData/trueTreeProb.npz', probAll)
    if T == 2:
        plt.savefig('./images/new1/randomTreeProb.pdf')
        np.savez_compressed('./plotData/randomTreeProb.npz', probAll)
    if T == 3:
        plt.savefig('./images/new1/trueTreeProb2R.pdf')
        #np.savez_compressed('./plotData/trueTreeProb2R.npz', probAll)

    plt.show()


#showRandomSelectionProb()
#quit()


def plotRandomSelectionProb():

    probAll1 = loadnpz('./plotData/trueTreeProb.npz')
    probAll2 = loadnpz('./plotData/randomTreeProb.npz')

    plt.hist(probAll1, bins=10, alpha=0.5)
    plt.hist(probAll2, bins=10, alpha=0.5)
    plt.legend(['with causal relationships', 'with only random trees'])
    plt.xlabel('relative probability of correct tree')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig('./images/new1/randomAndTrueProb.pdf')
    plt.show()


#plotRandomSelectionProb()
#quit()



def compareOccurSimulations(folder1, folder2, T):


    def simple_calculateErrors(cutOff, prob_true, theta, mask1):




        pred = theta[mask1 == 1].flatten()
        trueVal = prob_true[mask1 == 1].flatten()

        pred_bool = np.copy(pred)
        argBelow = np.argwhere(pred_bool < cutOff)[:, 0]
        argAbove = np.argwhere(pred_bool >= cutOff)[:, 0]
        pred_bool[argBelow] = 0
        pred_bool[argAbove] = 1

        TruePos = np.argwhere(np.logical_and(pred_bool == 1, trueVal == 1)).shape[0]
        predTrue = np.argwhere(pred_bool == 1).shape[0]
        realTrue = np.argwhere(trueVal == 1).shape[0]

        #print (predTrue, realTrue)

        precision1 = float(TruePos) / (float(predTrue) + 1e-5)
        recall1 = float(TruePos) / float(realTrue)

        return precision1, recall1


    def calculateErrors(cutOff, theta_true, pred_now, mask2):



        #choiceShow = 5

        #pred_now = pred_now[argTop][:, argTop]
        #pred_now = pred_now.T
        pred_now = pred_now[mask2 == 1]
        #pred_now = pred_now.reshape((M*M,))

        pred_now[pred_now> cutOff] = 1
        pred_now[pred_now< (-1 * cutOff)] = -1
        pred_now[np.abs(pred_now) < cutOff] = 0


        #theta_true = theta_true[argTop][:, argTop]

        theta_true = theta_true[mask2 == 1]
        #theta_true = theta_true.reshape((M*M,))
        theta_true[theta_true> 0.01] = 1
        theta_true[theta_true<-0.01] = -1
        theta_true[np.abs(theta_true) < 0.02] = 0



        #figure, axis = plt.subplots(3)
        #axis[0].imshow(theta_true)
        #axis[1].imshow(pred_noSS[0].T)
        #axis[2].imshow(pred_now)
        #plt.show()


        FalseArg = np.argwhere(  (pred_now - theta_true) != 0 )[:, 0]
        TrueArg = np.argwhere(  (pred_now - theta_true) == 0 )[:, 0]

        #print (FalseArg.shape)
        #print (TrueArg.shape)
        #quit()

        TruePosArg = TrueArg[theta_true[TrueArg] != 0]
        TrueNegArg = TrueArg[theta_true[TrueArg] == 0]

        truePosNum = TruePosArg.shape[0]
        trueNegNum = TrueNegArg.shape[0]

        falseNegNum = np.argwhere( np.logical_and(  pred_now == 0, theta_true != 0   )).shape[0]

        falsePosNum = FalseArg[pred_now[FalseArg] != 0].shape[0]

        #print (truePosNum, falsePosNum, trueNegNum, falseNegNum)
        #quit()

        #return (truePosNum, falsePosNum, trueNegNum, falseNegNum)

        #precision1 = truePosNum

        precision1 = float(truePosNum) / (float(falsePosNum) + float(truePosNum) + 1e-10)
        recall1 = float(truePosNum) / (float(falseNegNum) + float(truePosNum))

        return precision1, recall1


    import matplotlib.pyplot as plt


    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships


    errorList = []




    #T = 2



    #Loading in the saved data sets
    newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkTrees.npz')
    newTrees = newTrees.astype(int)
    sampleInverse = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkSample.npz').astype(int)
    #mutationCategory = ''
    maxM = 7
    M = int(np.max(newTrees+1)) - 2



    N = 20

    maxList = []

    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    if folder1 == './data/simulations/lowPatient':
        cutOffs_MHN = []
        if T == 0:
            cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]

        if T == 1:
            cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]

        if T == 2:
            cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]

        if T == 4:
            cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]

    if folder1 == './data/simulations/OLD-passanger':
        cutOffs_MHN = []
        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]#, 1.001]

    if folder1 == './data/simulations/passanger':
        cutOffs_MHN = [0.0, 0.0001, 0.005, 0.01, 0.04, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 4.0]
        #cutOffs_MHN = [0.0, 0.0001, 0.001]


        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.7]

    if folder1 == './data/simulations/effectSize':
        #cutOffs_MHN = []

        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]

    if folder1 == './data/simulations/bootstrap':
        cutOffs_MHN = []
        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    if folder1 == './data/simulations/I-a':
        cutOffs_MHN = []
        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]
        #cutOffs = [0.8, 1.0, 1.2]



    if folder1 == './data/simulations/negativeInter':
        #cutOffs_MHN = [0.0, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02]
        cutOffs_MHN = [0.0, 0.0001, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        #cutOffs = [-2, -1.5, -1.0, -0.5, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        #cutOffs = [0, 0.2, 0.3, 0.5, 0.6, 0.8, 1]
        cutOffs = [0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.2, 0.3, 0.5, 0.6, 0.8, 1]


    if folder1 == './data/simulations/latent':
        cutOffs_MHN = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.005]
        cutOffs = [0, 0.2, 0.3, 0.5, 0.6, 0.8, 1]


    if folder1 == './data/simulations/fewSamples':
        cutOffs_MHN = []
        cutOffs = [0, 0.2, 0.3, 0.5, 0.6, 0.8, 1]


    if folder1 == './data/simulations/random':
        cutOffs_MHN = []

        if T == 2:
            cutOffs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if T == 3:
            #cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]
            cutOffs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if T == 1:
            cutOffs = [0.6, 0.7, 0.8, 0.85, 1.0, 1.2, 1.3, 1.4, 1.5]

    #if T == 11:
    #    cutOffs_MHN = [0, 1e-5, 0.0001, 0.001]
    #    cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    #if T == 4:
    #    cutOffs_MHN = [0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]
    #    cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]

    #if T == 14:
    #    cutOffs = [1.2, 1.5, 1.7, 2.0, 2.1, 2.2]

    valList = []


    baseline_acc = np.zeros((N, len(cutOffs_MHN), 2))
    our_acc = np.zeros((N, len(cutOffs), 2))

    #for a in range(20, N):
    for a in range(0, N):#20):#N):

        print (a)
        if folder1 == './data/simulations/negativeInter':
            thetaFolder = './otherMethod/treeMHN/data/output/'
            thetaFile = thetaFolder + '/' + str(T) + '/' + str(a) + '_MHN_sim.csv'
            theta = np.loadtxt(thetaFile, delimiter=",", dtype=str)
            #theta = np.loadtxt(folder1 + '/' + str(T) + '/' + str(a) + '_MHN_sim.csv', delimiter=",", dtype=str)

            theta = theta[1:].astype(float)
            theta = theta.T



        if folder1 == './data/simulations/passanger':
            thetaFolder = './otherMethod/treeMHN/data/output/passanger/'
            thetaFile = thetaFolder + str(a) + '_MHN_sim.csv'
            theta = np.loadtxt(thetaFile, delimiter=",", dtype=str)

            theta = theta[1:].astype(float)
            theta = theta.T

            theta[np.arange(M), np.arange(M)] = 0

            #plt.imshow(theta)
            #plt.show()



        if folder1 == './data/simulations/latent':
            thetaFolder = './otherMethod/treeMHN/data/output/'
            thetaFile = thetaFolder + '/latent/' + str(a) + '_MHN_sim.csv'
            theta = np.loadtxt(thetaFile, delimiter=",", dtype=str)

            theta = theta[1:].astype(float)
            theta = theta.T



        if folder1 == './data/simulations/latent':
            propertyFile = folder1 + '/T_' + str(0) + '_R_' + str(a) + '_property.npz'
            props = loadnpz(propertyFile)

            propertyA = np.copy(props[:, 0])
            propertyB = np.copy(props[:, 1])
            catA = propertyA / (np.sum(props ** 2, axis=1) ** 0.5)
            catB = propertyB / (np.sum(props ** 2, axis=1) ** 0.5)
            probabilityMatrix = np.zeros((M, M))
            probabilityMatrix[:, :] =  propertyA.reshape((-1, 1)) * catA.reshape((1, -1))
            probabilityMatrix[:, :] += propertyB.reshape((-1, 1)) * catB.reshape((1, -1))
            probabilityMatrix = probabilityMatrix * np.log(10)
        else:


            #This matrix is the set of true probability of true causal relationships
            probabilityMatrix = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz')
            #probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0
            #probabilityMatrix[probabilityMatrix > 0.01] = 1

        if folder1 != './data/simulations/negativeInter':
            probabilityMatrix[probabilityMatrix > 0.01] = 1
            probabilityMatrix[probabilityMatrix < 0.01] = 0
        else:
            probabilityMatrix[probabilityMatrix < -0.01] = -1
            probabilityMatrix[probabilityMatrix > 0.01] = 1
            probabilityMatrix[np.abs(probabilityMatrix) < 0.01] = 0







        if folder1 == './data/simulations/negativeInter':
            if T == 0:
                numInter = 2
            if T == 1:
                numInter = 3
            if T == 2:
                numInter = 5

            prob_true = probabilityMatrix[:5, :5]
            prob_true = prob_true[np.arange(M) // numInter]
            prob_true = prob_true[:, np.arange(M) // numInter]

        elif folder1 == './data/simulations/latent':
            prob_true = np.copy(probabilityMatrix[:M, :M])

        else:

            #This matrix is a binary matrix of true causal relationships
            prob_true = np.zeros((M, M))
            prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])



        #print (prob_true)
        #quit()


        #This loads in the model
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)

        #This makes the probabilities a zero mean numpy array
        output_np = output.data.numpy()
        #output_np = output_np - np.mean(output_np)


        #New
        X_normal = torch.zeros((1, M))
        output_normal, _ = model(X_normal)
        output_np = output.data.numpy()
        output_normal = output_normal.data.numpy()
        for b in range(output.shape[1]):
            output_np[:, b] = output_np[:, b] - output_normal[0, b]


        #output_np[output_np>0.5] = 1
        #output_np[output_np<-0.5] = -1

        #print (np.max(output_np))

        maxList.append(np.max(output_np))

        #plt.imshow(output_np)
        #plt.show()
        #quit()

        print (output_np)



        mask1 = np.ones(output_np.shape)
        mask1[np.arange(M), np.arange(M)] = 0


        valList = valList + list(output_np[mask1 == 1])


        #fig, axs = plt.subplots(2)
        #axs[0].imshow(prob_true)
        #axs[1].imshow(output_np)
        #plt.show()
        #quit()

        for cutOff0 in range(len(cutOffs_MHN)):

            cutOff = cutOffs_MHN[cutOff0]

            if folder1 not in ['./data/simulations/negativeInter', './data/simulations/latent']:
                precision1, recall1 = simple_calculateErrors(cutOff, prob_true, theta, mask1)
            else:
                #precision1, recall1 = calculateErrors(cutOff, prob_true, theta, mask1)
                precision1, recall1 = calculateErrors(cutOff, prob_true, theta, mask1)

            #print (precision1, recall1)

            baseline_acc[a, cutOff0, 0] = precision1
            baseline_acc[a, cutOff0, 1] = recall1

        for cutOff0 in range(len(cutOffs)):

            cutOff = cutOffs[cutOff0]
            #quit()


            if folder1 not in ['./data/simulations/negativeInter', './data/simulations/latent']:
                precision1, recall1 = simple_calculateErrors(cutOff, prob_true, output_np, mask1)
            else:
                precision1, recall1 = calculateErrors(cutOff, prob_true, output_np, mask1)

            print (precision1, recall1)

            our_acc[a, cutOff0, 0] = precision1
            our_acc[a, cutOff0, 1] = recall1

        #quit()



    #np.savetxt('./sending/ours_forMEK_prec.csv', our_acc[:, :, 0], delimiter=',')
    #np.savetxt('./sending/ours_forMEK_rec.csv', our_acc[:, :, 1], delimiter=',')
    #np.savetxt('./sending/MHN_forMEK_prec.csv', baseline_acc[:, :, 0], delimiter=',')
    #np.savetxt('./sending/MHN_forMEK_rec.csv', baseline_acc[:, :, 1], delimiter=',')

    valList = np.array(valList)

    #print ('0.8', valList[valList > 0.8].shape)
    print ('0.4', valList[valList > 0.4].shape)
    print ('0.3', valList[valList > 0.3].shape)

    #doSave = False
    saveName = ''

    if folder1 == './data/simulations/lowPatient':
        if T == 0:
            saveName = './images/100_patients.pdf'
            np.savez_compressed('./plotData/100_patients.npz', our_acc)
        if T == 1:
            saveName = './images/50_patients.pdf'
            np.savez_compressed('./plotData/50_patients.npz', our_acc)
        if T == 2:
            saveName = './images/100_patients_AML_tree_sizes.pdf'
            np.savez_compressed('./plotData/tree_sizes_100_patients.npz', our_acc)

        if T == 4:
            #saveName = './images/300_patients.pdf'
            np.savez_compressed('./plotData/300_patients.npz', our_acc)


    if folder1 == './data/simulations/effectSize':
        saveName = './images/effectSize.pdf'
        np.savez_compressed('./plotData/effectSize.npz', our_acc)

    if folder1 == './data/simulations/passanger':
        saveName = './images/passanger.pdf'
        #np.savez_compressed('./plotData/passanger.npz', our_acc)
        True

    if False:#folder1 == './data/simulations/negativeInter':
        if T == 0:
            saveName = './images/negativeInter2.pdf'
            np.savez_compressed('./plotData/negativeInter2.npz', our_acc)
            np.savez_compressed('./plotData/negativeInter2_baseline.npz', baseline_acc)
        if T == 1:
            saveName = './images/negativeInter3.pdf'
            np.savez_compressed('./plotData/negativeInter3.npz', our_acc)
            np.savez_compressed('./plotData/negativeInter3_baseline.npz', baseline_acc)

    if folder1 == './data/simulations/I-a':
        np.savez_compressed('./plotData/I-a.npz', our_acc)

    if folder1 == './data/simulations/random':
        if T == 1:
            np.savez_compressed('./plotData/random_1.npz', our_acc)
        if T == 3:
            np.savez_compressed('./plotData/random_3.npz', our_acc)

    if folder1 == './data/simulations/latent':
        np.savez_compressed('./plotData/latentCause.npz', our_acc)
        np.savez_compressed('./plotData/latentCause_baseline.npz', baseline_acc)

    #print (our_acc.shape)

    plt.plot(np.mean(our_acc[0:N, :, 1], axis=0), np.mean(our_acc[0:N, :, 0], axis=0))

    if folder1 in ['./data/simulations/negativeInter', './data/simulations/latent', './data/simulations/passanger']:


        #plt.scatter(np.mean(baseline_acc[0:2, :, 1], axis=0), np.mean(baseline_acc[0:2, :, 0], axis=0))
        #plt.plot(np.mean(baseline_acc[0:2, :, 1], axis=0), np.mean(baseline_acc[0:2, :, 0], axis=0))
        #plt.legend(['CloMu', 'TreeMHN'])
        True

    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    plt.xlabel("recall")
    plt.ylabel("precision")
    #plt.legend(['CloMu', 'TreeMHN'])
    #plt.title(title1)

    #if saveName != '':
    #    plt.savefig(saveName)

    #plt.savefig('./images/betweenClone.pdf')
    plt.show()



#folder1 = './data/simulations/effectSize'
#folder2 = './Models/simulations/effectSize'
#folder1 = './data/simulations/lowPatient'
#folder2 = './Models/simulations/lowPatient'
#folder1 = './data/simulations/negativeInter'
#folder2 = './Models/simulations/negativeInter'
#folder1 = './data/simulations/passanger'
#folder2 = './Models/simulations/passanger'
#folder1 = './data/simulations/bootstrap'
#folder2 = './Models/simulations/bootstrap'
#folder1 = './data/simulations/I-a'
#folder2 = './Models/simulations/I-a'
#folder1 = './data/simulations/latent'
#folder2 = './Models/simulations/latent'

#folder1 = './data/simulations/fewSamples'
#folder2 = './Models/simulations/fewSamples'

folder1 = './data/simulations/random'
folder2 = './Models/simulations/random'

#compareOccurSimulations(folder1, folder2, 2)
#quit()


def comparePlot():


    if True:
        our_random = loadnpz('./plotData/random_1.npz')
        our_random2 = loadnpz('./plotData/random_3.npz')

        print (np.mean(our_random2[:, :, 1], axis=0), np.mean(our_random2[:, :, 0], axis=0))
        quit()

        print (our_random.shape)
        plt.plot(np.mean(our_random[:, :, 1], axis=0), np.mean(our_random[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.plot(np.mean(our_random2[:, :, 1], axis=0), np.mean(our_random2[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.scatter(np.mean(our_random[:, :, 1], axis=0), np.mean(our_random[:, :, 0], axis=0))
        plt.scatter(np.mean(our_random2[:, :, 1], axis=0), np.mean(our_random2[:, :, 0], axis=0))
        #plt.legend(['500 patients', '300 patients', '100 patients', '50 patients'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.legend(['one random tree per patient', 'two random trees per patient'])
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/new1/causalityRandom_both.pdf')
        plt.show()
        quit()



    if False:
        our_500 = loadnpz('./plotData/I-a.npz')
        our_300 = loadnpz('./plotData/300_patients.npz')
        our_100 = loadnpz('./plotData/100_patients.npz')
        our_50 = loadnpz('./plotData/50_patients.npz')


        plt.plot(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.plot(np.mean(our_300[:, :, 1], axis=0), np.mean(our_300[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.plot(np.mean(our_100[:, :, 1], axis=0), np.mean(our_100[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.plot(np.mean(our_50[:, :, 1], axis=0), np.mean(our_50[:, :, 0], axis=0), alpha=0.5, lw=3)
        plt.scatter(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0))
        plt.scatter(np.mean(our_300[:, :, 1], axis=0), np.mean(our_300[:, :, 0], axis=0))
        plt.scatter(np.mean(our_100[:, :, 1], axis=0), np.mean(our_100[:, :, 0], axis=0))
        plt.scatter(np.mean(our_50[:, :, 1], axis=0), np.mean(our_50[:, :, 0], axis=0))
        plt.legend(['500 patients', '300 patients', '100 patients', '50 patients'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/bothPatientSizes.pdf')
        plt.show()
        quit()



    if False:
        our_100 = loadnpz('./plotData/100_patients.npz')
        our_sizes = loadnpz('./plotData/tree_sizes_100_patients.npz')


        plt.plot(np.mean(our_100[:, :, 1], axis=0), np.mean(our_100[:, :, 0], axis=0))
        plt.plot(np.mean(our_sizes[:, :, 1], axis=0), np.mean(our_sizes[:, :, 0], axis=0))
        plt.scatter(np.mean(our_100[:, :, 1], axis=0), np.mean(our_100[:, :, 0], axis=0))
        plt.scatter(np.mean(our_sizes[:, :, 1], axis=0), np.mean(our_sizes[:, :, 0], axis=0))
        plt.legend(['original mutations', 'fewer mutations'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/100_patients_AML_tree_sizes.pdf')
        plt.show()
        quit()

    if False:
        our_500 = loadnpz('./plotData/300_patients.npz')
        our_effect = loadnpz('./plotData/effectSize.npz')

        plt.plot(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0))
        plt.plot(np.mean(our_effect[:, :, 1], axis=0), np.mean(our_effect[:, :, 0], axis=0))
        plt.scatter(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0))
        plt.scatter(np.mean(our_effect[:, :, 1], axis=0), np.mean(our_effect[:, :, 0], axis=0))
        plt.legend(['original effect sizes', 'modified effect sizes'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/effectSize.pdf')
        plt.show()


    if False:
        our_500 = loadnpz('./plotData/I-a.npz')
        our_effect = loadnpz('./plotData/passanger.npz')

        plt.plot(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0))
        plt.plot(np.mean(our_effect[:, :, 1], axis=0), np.mean(our_effect[:, :, 0], axis=0))
        plt.scatter(np.mean(our_500[:, :, 1], axis=0), np.mean(our_500[:, :, 0], axis=0))
        plt.scatter(np.mean(our_effect[:, :, 1], axis=0), np.mean(our_effect[:, :, 0], axis=0))
        plt.legend(['original simulation', 'many passenger mutations'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/passanger.pdf')
        plt.show()


    if False:

        our_acc = loadnpz('./plotData/negativeInter2.npz')
        baseline_acc = loadnpz('./plotData/negativeInter2_baseline.npz')

        plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.legend(['CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/negativeInter2.pdf')
        plt.show()

    if False:

        our_acc = loadnpz('./plotData/negativeInter3.npz')
        baseline_acc = loadnpz('./plotData/negativeInter3_baseline.npz')

        plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.legend(['CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/negativeInter3.pdf')
        plt.show()


    if True:

        our_acc2 = loadnpz('./plotData/negativeInter2.npz')
        baseline_acc2 = loadnpz('./plotData/negativeInter2_baseline.npz')
        our_acc3 = loadnpz('./plotData/negativeInter3.npz')
        baseline_acc3 = loadnpz('./plotData/negativeInter3_baseline.npz')


        plt.scatter(np.mean(our_acc2[:, :, 1], axis=0), np.mean(our_acc2[:, :, 0], axis=0), color='blue')
        plt.scatter(np.mean(our_acc3[:, :, 1], axis=0), np.mean(our_acc3[:, :, 0], axis=0), marker='x', color='blue')
        plt.scatter(np.mean(baseline_acc2[:, :, 1], axis=0), np.mean(baseline_acc2[:, :, 0], axis=0), color='orange')
        plt.scatter(np.mean(baseline_acc3[:, :, 1], axis=0), np.mean(baseline_acc3[:, :, 0], axis=0), marker='x', color='orange')
        plt.plot(np.mean(our_acc2[:, :, 1], axis=0), np.mean(our_acc2[:, :, 0], axis=0), color='blue')
        plt.plot(np.mean(baseline_acc2[:, :, 1], axis=0), np.mean(baseline_acc2[:, :, 0], axis=0), color='orange')
        plt.plot(np.mean(our_acc3[:, :, 1], axis=0), np.mean(our_acc3[:, :, 0], axis=0), color='blue')
        plt.plot(np.mean(baseline_acc3[:, :, 1], axis=0), np.mean(baseline_acc3[:, :, 0], axis=0), color='orange')
        plt.legend(['CloMu 2 Int. Mutations', 'CloMu 3 Int. Mutations', 'TreeMHN 2 Int. Mutations', 'TreeMHN 3 Int. Mutations'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/negativeInterBoth.pdf')
        plt.show()


    if False:

        our_acc = loadnpz('./plotData/latentCause.npz')
        baseline_acc = loadnpz('./plotData/latentCause_baseline.npz')

        plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
        plt.legend(['CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/latentPrecRec.pdf')
        plt.show()



    if False:

        baseline_acc_10 = np.load('./extra/treeMHNplot/n10_N300_baseline_acc.npy')
        linear_acc_10 = np.load('./extra/treeMHNplot/n10_N300_our_acc_11.npy')[:, :-2, :]
        neural_acc_10 = np.load('./extra/treeMHNplot/n10_N300_our_acc_12.npy')[:, :-1, :]

        #print (baseline_acc_10.shape)
        #print (linear_acc_10.shape)
        #quit()


        plt.plot(np.mean(linear_acc_10[:, :, 1], axis=0), np.mean(linear_acc_10[:, :, 0], axis=0))
        plt.plot(np.mean(neural_acc_10[:, :, 1], axis=0), np.mean(neural_acc_10[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc_10[:, :, 1], axis=0), np.mean(baseline_acc_10[:, :, 0], axis=0))
        plt.scatter(np.mean(linear_acc_10[:, :, 1], axis=0), np.mean(linear_acc_10[:, :, 0], axis=0))
        plt.scatter(np.mean(neural_acc_10[:, :, 1], axis=0), np.mean(neural_acc_10[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc_10[:, :, 1], axis=0), np.mean(baseline_acc_10[:, :, 0], axis=0))
        plt.legend(['CloMu', 'linear CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/multi/new_TreeMHN_data_10.pdf')
        plt.show()


        baseline_acc_15 = np.load('./extra/treeMHNplot/n15_N300_baseline_acc.npy')
        linear_acc_15 = np.load('./extra/treeMHNplot/n15_N300_our_acc_11.npy')[:, :-2, :]
        neural_acc_15 = np.load('./extra/treeMHNplot/n15_N300_our_acc_12.npy')[:, :-2, :]

        plt.plot(np.mean(linear_acc_15[:, :, 1], axis=0), np.mean(linear_acc_15[:, :, 0], axis=0))
        plt.plot(np.mean(neural_acc_15[:, :, 1], axis=0), np.mean(neural_acc_15[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc_15[:, :, 1], axis=0), np.mean(baseline_acc_15[:, :, 0], axis=0))
        plt.scatter(np.mean(linear_acc_15[:, :, 1], axis=0), np.mean(linear_acc_15[:, :, 0], axis=0))
        plt.scatter(np.mean(neural_acc_15[:, :, 1], axis=0), np.mean(neural_acc_15[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc_15[:, :, 1], axis=0), np.mean(baseline_acc_15[:, :, 0], axis=0))
        plt.legend(['CloMu', 'linear CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/multi/new_TreeMHN_data_15.pdf')
        plt.show()



        baseline_acc_20 = np.load('./extra/treeMHNplot/n20_N300_baseline_acc.npy')
        linear_acc_20 = np.load('./extra/treeMHNplot/n20_N300_our_acc_11.npy')[:, :-3, :]
        neural_acc_20 = np.load('./extra/treeMHNplot/n20_N300_our_acc_12.npy')[:, :-2, :]

        plt.plot(np.mean(linear_acc_20[:, :, 1], axis=0), np.mean(linear_acc_20[:, :, 0], axis=0))
        plt.plot(np.mean(neural_acc_20[:, :, 1], axis=0), np.mean(neural_acc_20[:, :, 0], axis=0))
        plt.plot(np.mean(baseline_acc_20[:, :, 1], axis=0), np.mean(baseline_acc_20[:, :, 0], axis=0))
        plt.scatter(np.mean(linear_acc_20[:, :, 1], axis=0), np.mean(linear_acc_20[:, :, 0], axis=0))
        plt.scatter(np.mean(neural_acc_20[:, :, 1], axis=0), np.mean(neural_acc_20[:, :, 0], axis=0))
        plt.scatter(np.mean(baseline_acc_20[:, :, 1], axis=0), np.mean(baseline_acc_20[:, :, 0], axis=0))
        plt.legend(['CloMu', 'linear CloMu', 'TreeMHN'])
        plt.xlabel("causality recall")
        plt.ylabel("causality precision")
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/multi/new_TreeMHN_data_20.pdf')
        plt.show()
        quit()






#comparePlot()
#quit()



def boostrapPlot(folder1, folder2):



    def simMakePlot(image1, saveName, vmin, vmax, doV=True):


        fig, ax = plt.subplots()

        if not doV:
            med1 = np.median(image1)
            for a in range(image1.shape[0]):
                image1[a, a] = med1

            vmin=np.min(image1)
            vmax=np.max(image1)
            vavg = (vmin + vmax) / 2

            for a in range(image1.shape[0]):
                image1[a, a] = vavg

        image2 = np.copy(image1)
        image2[image2<vmin] = vmin
        image2[image2>vmax] = vmax

        im = ax.imshow(image2, vmin=vmin, vmax=vmax, cmap='bwr')

        arange1 = np.arange(10)
        arange2 = arange1.astype(str)

        #plt.imshow(output_np_original, vmin=vmin, vmax=vmax, cmap='bwr')
        ax.set_xticks(arange1, labels=arange2)
        ax.set_yticks(arange1, labels=arange2)
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #     rotation_mode="anchor")

        if True:
            for i in range(10):
                for j in range(10):
                    if i != j:

                        imgText = np.round_(image1[i, j], decimals=3)
                        imgText = str(imgText)[:-1]
                        if float(imgText) == 0:
                            if imgText[0] == '-':
                                imgText = imgText[1:]


                        #print (image1[i, j])
                        text = ax.text(j, i, imgText, ha="center", va="center", color="black")

        plt.gcf().set_size_inches(8, 6)
        plt.tight_layout()
        if saveName != '':
            plt.savefig(saveName)
        plt.show()



    import matplotlib.pyplot as plt


    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships


    doRank = False

    errorList = []




    T = 0
    #Loading in the saved data sets
    newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkTrees.npz')
    newTrees = newTrees.astype(int)
    sampleInverse = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkSample.npz').astype(int)
    #mutationCategory = ''
    maxM = 7
    M = int(np.max(newTrees+1)) - 2


    sns.set_style('white')

    N = 100

    output_all = np.zeros((N, M, M))

    #for a in range(20, N):
    for a in range(0, N):#N):

        print (a)


        #This matrix is the set of true probability of true causal relationships
        probabilityMatrix = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(a) + '_prob.npz')
        #probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0
        #probabilityMatrix[probabilityMatrix > 0.01] = 1



        #This loads in the model
        modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)

        #This makes the probabilities a zero mean numpy array
        output_np = output.data.numpy()
        #output_np = output_np - np.mean(output_np)


        #New
        X_normal = torch.zeros((1, M))
        output_normal, _ = model(X_normal)
        output_np = output.data.numpy()
        output_normal = output_normal.data.numpy()
        for b in range(output.shape[1]):
            output_np[:, b] = output_np[:, b] - output_normal[0, b]



        if doRank:

            #plt.imshow(output_np)
            ##plt.show()

            mask1 = np.zeros((M, M))
            mask1[np.arange(M), np.arange(M)] = 1
            args1 = np.argwhere(mask1 == 0)
            output_np2 = output_np[args1[:, 0], args1[:, 1]]
            output_np2 = np.argsort(np.argsort(output_np2)) + 1 #two argsorts gives ranking
            output_np[args1[:, 0], args1[:, 1]] = np.copy(output_np2)



            #plt.imshow(output_np)
            #plt.show()
            #quit()


        output_all[a] = np.copy(output_np)

    #plt.imshow(np.mean(output_all, axis=0))
    #plt.show()

    cutOff = 1.0
    #rangeSize = 0.8
    for a in range(M):
        output_all[:, a, a] = cutOff

    #for a in range(5):
    #    probabilityMatrix[a, a] = (np.min(probabilityMatrix)+np.max(probabilityMatrix)) / 2




    output_mean = np.mean(output_all, axis=0)
    output_mean_reshape = output_mean.reshape((1, M, M))
    #output_sigma = (output_all - output_mean_reshape)
    #output_sigma = np.mean(output_sigma ** 2, axis=0) ** 0.5
    #output_sigma = output_sigma * (output_mean.shape[0] / (output_mean.shape[0]-1))

    if doRank:
        #output_upper = np.max(output_all, axis=0)
        #output_lower = np.min(output_all, axis=0)
        output_upper = np.sort(output_all, axis=0)[-6]
        output_lower = np.sort(output_all, axis=0)[5]
    else:
        #output_upper = np.max(output_all, axis=0)
        #output_lower = np.min(output_all, axis=0)
        output_upper = np.sort(output_all, axis=0)[-6]
        output_lower = np.sort(output_all, axis=0)[5]

    #output_upper = np.sort(output_all, axis=0)[-6]
    #output_lower = np.sort(output_all, axis=0)[5]

    #output_upper = np.sort(output_all, axis=0)[-3]
    #output_lower = np.sort(output_all, axis=0)[2]


    #output_sigma = output_sigma * 1

    maxDiff1 = np.max(output_upper) - cutOff
    maxDiff2 = cutOff - np.min(output_lower)
    rangeSize = max(maxDiff1, maxDiff2)
    #rangeSize =

    probabilityMatrix_plot = np.zeros((M, M))
    probabilityMatrix_plot[:probabilityMatrix.shape[0], :probabilityMatrix.shape[1]] = np.copy(probabilityMatrix)
    probabilityMatrix_plot[np.arange(M), np.arange(M)] = cutOff #cutOff instead of 0 may 1 2023


    vmin = cutOff - rangeSize
    vmax = cutOff + rangeSize

    if doRank:

        vmin = 80 - 4
        vmax = 81 + 4

        probabilityMatrix_plot[probabilityMatrix_plot>2] = 81
        probabilityMatrix_plot[probabilityMatrix_plot<2] = 80

        simMakePlot(probabilityMatrix_plot, './images/new1/simCauseTrue_rank.pdf', vmin, vmax, doV=True)
        simMakePlot(output_mean, './images/new1/simCauseMean_rank.pdf', vmin, vmax, doV=True)
        simMakePlot(output_lower, './images/new1/simCauseLower_rank.pdf', vmin, vmax, doV=True)
        simMakePlot(output_upper, './images/new1/simCauseUpper_rank.pdf', vmin, vmax, doV=True)

    else:

        simMakePlot(probabilityMatrix_plot, './images/new1/simCauseTrue.pdf', vmin, vmax)
        simMakePlot(output_mean, './images/new1/simCauseMean.pdf', vmin, vmax)
        simMakePlot(output_lower, './images/new1/simCauseLower.pdf', vmin, vmax)
        simMakePlot(output_upper, './images/new1/simCauseUpper.pdf', vmin, vmax)

    quit()

    #plt.imshow(output_mean, vmin=vmin, vmax=vmax, cmap='bwr')
    #plt.show()
    #quit()

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(probabilityMatrix_plot, cmap='bwr')
    axs[0, 1].imshow(output_mean, vmin=vmin, vmax=vmax, cmap='bwr')
    axs[1, 0].imshow(output_mean-output_sigma, vmin=vmin, vmax=vmax, cmap='bwr')
    axs[1, 1].imshow(output_mean+output_sigma, vmin=vmin, vmax=vmax, cmap='bwr')

    #plt.ylim(0, 1.6)
    #plt.vmin(0)
    #plt.vmax(1.6)
    #fig.set_clim(vmin=0,vmax=1.6)
    plt.show()



    quit()


    #quit()
    title1 = ''
    if T in [4, 10, 13]:
        title1 = 'No Interchangable Mutations'
        imgName = 'causalNoInt'
    if T in [8, 11]:
        title1 = 'Two Interchangable Mutations Per Type'
        imgName = 'causal2Int'
    if T in [9, 12]:
        title1 = 'Three Interchangable Mutations Per Type'
        imgName = 'causal3Int'
    if T in [14]:
        title1 = ''
        imgName = 'betweenClone'

    #print (np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))

    '''
    if T == 4:
        np.save('./sending/causal_inter/noInterchangable_CloMu.npy', our_acc)
        np.save('./sending/causal_inter/noInterchangable_TreeMHN.npy', baseline_acc)
    elif T == 11:
        np.save('./sending/causal_inter/2Interchangable_CloMu.npy', our_acc)
        np.save('./sending/causal_inter/2Interchangable_TreeMHN.npy', baseline_acc)
    elif T == 12:
        np.save('./sending/causal_inter/3Interchangable_CloMu.npy', our_acc)
        np.save('./sending/causal_inter/3Interchangable_TreeMHN.npy', baseline_acc)
    #'''

    #np.savetxt('./sending/ours_forMEK_prec.csv', our_acc[:, :, 0], delimiter=',')
    #np.savetxt('./sending/ours_forMEK_rec.csv', our_acc[:, :, 1], delimiter=',')
    #np.savetxt('./sending/MHN_forMEK_prec.csv', baseline_acc[:, :, 0], delimiter=',')
    #np.savetxt('./sending/MHN_forMEK_rec.csv', baseline_acc[:, :, 1], delimiter=',')

    #quit()

    #doSave = False
    saveName = ''

    if folder1 == './data/simulations/lowPatient':
        if T == 0:
            saveName = './images/100_patients.pdf'
        if T == 1:
            saveName = './images/50_patients.pdf'
        if T == 2:
            saveName = './images/100_patients_AML_tree_sizes.pdf'


    if folder1 == './data/simulations/effectSize':
        saveName = './images/effectSize.pdf'

    if folder1 == './data/simulations/passanger':
        saveName = './images/passanger.pdf'


    plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    #plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    plt.xlabel("recall")
    plt.ylabel("precision")
    #plt.legend(['CloMu', 'TreeMHN'])
    plt.title(title1)

    if saveName != '':
        plt.savefig(saveName)

    #plt.savefig('./images/betweenClone.pdf')
    plt.show()



#folder1 = './data/simulations/bootstrap'
#folder2 = './Models/simulations/bootstrap'
#boostrapPlot(folder1, folder2)
#quit()





def realBootstrap():



    def breastMakePlot(image1, toShow, saveName):


        fig, ax = plt.subplots()
        im = ax.imshow(image1, vmin=vmin, vmax=vmax, cmap='bwr')

        #plt.imshow(output_np_original, vmin=vmin, vmax=vmax, cmap='bwr')
        ax.set_xticks(np.arange(5), labels=toShow)
        ax.set_yticks(np.arange(5), labels=toShow)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

        for i in range(5):
            for j in range(5):
                if i != j:

                    imgText = np.round_(image1[i, j], decimals=3)
                    imgText = str(imgText)[:-1]
                    if float(imgText) == 0:
                        if imgText[0] == '-':
                            imgText = imgText[1:]


                    #print (image1[i, j])
                    text = ax.text(j, i, imgText, ha="center", va="center", color="black")

        plt.gcf().set_size_inches(8, 6)
        plt.tight_layout()
        plt.savefig(saveName)
        plt.show()




    folder1 = './data/simulations/realBoostrap'
    folder2 = './Models/simulations/realBoostrap'

    import matplotlib.pyplot as plt


    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships


    sns.set_style('white')

    errorList = []


    file1 = './data/realData/breastCancer'
    file2 = './extra/Breast_treeSizes.npz'
    maxM = 9
    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
    mutationCategory = mutationCategory[:-2]
    uniqueMutation = uniqueMutation[:-2]



    T = 1
    #Loading in the saved data sets
    newTrees = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkTrees.npz')
    newTrees = newTrees.astype(int)
    sampleInverse = loadnpz(folder1 + '/T_' + str(T) + '_R_' + str(0) + '_bulkSample.npz').astype(int)
    #mutationCategory = ''
    maxM = 7
    M = int(np.max(newTrees+1)) - 2


    #doRanking = True
    doRanking = False

    N = 100

    output_all = np.zeros((N, 5, 5))
    fitness_all = np.zeros((N, M))


    #for a in range(6, 7):
    for a in range(-1, 100):


        if a == -1:
            #This loads in the model
            modelFile = './Models/realData/savedModel_breast.pt'
        else:

            #This loads in the model
            modelFile = folder2 + '/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, _ = model(X)

        output_np = output.data.numpy()

        #plt.imshow(output_np)
        #plt.show()
        #quit()


        fitness = output.reshape((1, M*M))
        fitness = torch.softmax(fitness, axis=1)
        fitness = fitness.reshape((M, M))
        fitness = fitness.data.numpy()
        fitness = np.sum(fitness, axis=1)




        toShow = np.array(['TP53', 'CDH1', 'GATA3', 'MAP3K1', 'PIK3CA' ])
        order1 = []
        for b in range(len(toShow)):
            arg1 = np.argwhere(uniqueMutation == toShow[b])[0, 0]
            order1.append(arg1)
        order1 = np.array(order1)





        #argHigh = np.argwhere( np.isin(uniqueMutation,   ) )

        #print (np.mean())

        #print (uniqueMutation[fitness > np.median(fitness)*1.4])

        #plt.plot(fitness)
        #plt.show()

        output_np = softmax(output_np, axis=1)
        #output_np[np.arange(M), np.arange(M)] = 0
        #output_np = output_np / np.sum(output_np, axis=1).reshape((M, 1))
        #output_np[np.arange(M), np.arange(M)] = 1
        output_np = np.log(output_np)

        #output_np = np.log(softmax(output_np, axis=1))





        for b in range(output.shape[1]):
            output_np[:, b] = output_np[:, b] - np.mean(output_np[:, b])


        #output_all[a] = np.copy(output_np)
        #'''


        if False: #This would be ranking all mutations
            M1 = output_np.shape[0]
            mask1 = np.zeros((M1, M1))
            mask1[np.arange(M1), np.arange(M1)] = 1
            args1 = np.argwhere(mask1 == 0)
            output_np2 = output_np[args1[:, 0], args1[:, 1]]
            output_np2 = np.argsort(np.argsort(output_np2)) #two argsorts gives ranking
            output_np[args1[:, 0], args1[:, 1]] = np.copy(output_np2)


        output_np = output_np[order1][:, order1]
        output_np[np.arange(output_np.shape[0]), np.arange(output_np.shape[0])] = 0

        #plt.imshow(output_np, cmap='bwr')
        #plt.clim(-1, 1)
        #plt.show()



        if doRanking:

            #plt.imshow(output_np)
            ##plt.show()

            M1 = output_np.shape[0]
            mask1 = np.zeros((M1, M1))
            mask1[np.arange(M1), np.arange(M1)] = 1
            args1 = np.argwhere(mask1 == 0)
            output_np2 = output_np[args1[:, 0], args1[:, 1]]
            output_np2 = np.argsort(np.argsort(output_np2)) + 1 #two argsorts gives ranking
            output_np[args1[:, 0], args1[:, 1]] = np.copy(output_np2)



            #plt.imshow(output_np)
            #plt.show()
            #quit()



        if a != -1:
            fitness_all[a] = np.copy(fitness)
            output_all[a] = np.copy(output_np)
        else:
            output_np_original = np.copy(output_np)
            fitness_original = np.copy(fitness)

            miniMax = np.max(np.abs(output_np_original))
            #plt.imshow(output_np_original, vmin=-miniMax, vmax=miniMax, cmap='bwr')
            #plt.show()




    if False:
        #fitness_original_log = np.log(fitness_original)
        #fitness_all_log = np.log(fitness_all)

        #fitness_all_log = fitness_all
        #fitness_mean = np.mean(fitness_all_log, axis=0)
        fitness_mean = np.median(fitness_all, axis=0)
        fitness_mean_reshape = fitness_mean.reshape((1, M))
        #fitness_sigma = (fitness_all_log - fitness_mean_reshape)
        #fitness_sigma = np.mean(fitness_sigma ** 2, axis=0) ** 0.5

        #fitness_upp = np.sort(fitness_all, axis=0)[-1]#[-2]
        #fitness_low = np.sort(fitness_all, axis=0)[0]#[1]

        #fitness_upp = np.sort(fitness_all, axis=0)[-3]#[-2]
        #fitness_low = np.sort(fitness_all, axis=0)[2]#[1]

        fitness_upp = np.sort(fitness_all, axis=0)[-6]#[-2]
        fitness_low = np.sort(fitness_all, axis=0)[5]#[1]

        fitMax = max(np.max(fitness_upp), np.max(fitness_original)) * 1.02
        fitMin = min(np.min(fitness_mean), np.min(fitness_original)) * 0.98



        toShowFit = np.array(['TP53', 'CDH1', 'GATA3', 'MAP3K1', 'PIK3CA', 'KMT2C', 'ESR1', 'ARID1A'])
        order1 = []
        for b in range(len(toShowFit)):
            arg1 = np.argwhere(uniqueMutation == toShowFit[b])[0, 0]
            order1.append(arg1)
        order1 = np.array(order1)

        #print (fitness_sigma)

        yerr2 = np.zeros((2, order1.shape[0]))
        yerr2[1] = fitness_upp[order1] - fitness_mean[order1]
        yerr2[0] = fitness_mean[order1] - fitness_low[order1]

        #sns.set_style('whitegrid')


        plt.errorbar(order1, fitness_mean[order1], yerr=  yerr2 , capsize=10, ls='none', color='black')
        plt.scatter(order1, fitness_mean[order1], color='black')
        #plt.scatter(order1, fitness_original_log[order1], color='blue')
        plt.plot(fitness_mean, color='red', alpha=0.8)

        plt.scatter(order1, fitness_original[order1], color='black', marker='x', s=100)

        plt.yscale('log')
        for i in range(len(order1)):
            name = toShowFit[i]
            pos1 = order1[i] + 3
            pos2 = fitness_mean[order1[i]]
            if name == 'ESR1':
                pos2 = pos2 * (1-0.03)

            name = '\emph{' + name + '}'

            plt.annotate(name, (pos1 , pos2 ))
            #plt.annotate(name, (order1[i]  , np.exp(fitness_mean[order1[i]]) + (fitMax / 100)    ))
        plt.ylim(fitMin, fitMax)
        plt.xlabel('mutation')
        plt.ylabel('fitness')
        plt.grid(True,which="both",c='lightgray')
        plt.tight_layout()
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/new1/fitBoot.pdf')
        plt.show()

        quit()




        plt.scatter(order1, fitness_original[order1], color='red')
        plt.plot(fitness_original, color='red')
        plt.yscale('log')
        plt.ylim(fitMin, fitMax)

        for i in range(len(order1)):
            name = toShowFit[i]
            plt.annotate(name, (order1[i] + 3 , fitness_original[order1[i]])    )
        plt.grid(True,which="both",c='lightgray')

        plt.gcf().set_size_inches(8, 6)
        #plt.savefig('./images/multi/fitOriginal.pdf')
        plt.show()




    #plt.plot(fitness_all.T)
    #plt.yscale('log')
    #plt.show()

    #quit()

    M = 5

    cutOff = 0.0
    #rangeSize = 0.8
    for a in range(M):
        output_all[:, a, a] = cutOff
    #for a in range(5):
    #    probabilityMatrix[a, a] = (np.min(probabilityMatrix)+np.max(probabilityMatrix)) / 2


    output_mean = np.median(output_all, axis=0)
    output_mean_reshape = output_mean.reshape((1, M, M))
    #output_sigma = (output_all - output_mean_reshape)
    #output_sigma = np.mean(output_sigma ** 2, axis=0) ** 0.5
    output_upp = np.sort(output_all, axis=0)[-3]
    output_low = np.sort(output_all, axis=0)[2]


    maxDiff1 = np.max(output_upp) - cutOff
    maxDiff2 = cutOff - np.min(output_low)
    rangeSize = max(maxDiff1, maxDiff2)
    #rangeSize =

    #print (output_sigma)
    #quit()
    #0.8

    vmin = cutOff - rangeSize
    vmax = cutOff + rangeSize

    #plt.imshow(output_mean, vmin=vmin, vmax=vmax, cmap='bwr')
    #plt.show()
    #quit()


    output_all_reshape = output_all.reshape((N * M * M,))
    output_nums = np.arange(M*M).reshape((1, M, M))[np.zeros(N, dtype=int)]
    output_nums = output_nums.reshape((N*M*M,))

    mask1 = (output_nums % M) - (output_nums // M)
    mask2 = (np.arange(M*M) % M) - (np.arange(M*M) // M)

    output_all_reshape = output_all_reshape[mask1 != 0]
    output_nums = output_nums[mask1 != 0]
    output_np_original_mod = output_np_original.reshape((M*M,))[mask2 != 0]




    import matplotlib as mpl
    import pandas as pd

    labels = []
    for a in range(len(toShow)):
        for b in range(len(toShow)):
            #if a != b:
            #label1 = toShow[a] + ' to ' + toShow[b]
            label1 = '\emph{' + toShow[a] + '} to \emph{' + toShow[b] + '}'
            labels.append(label1)
    labels = np.array(labels)


    sns.set_style('whitegrid')

    sns.boxplot(x=labels[output_nums.astype(int)],
                y=output_all_reshape, showfliers=False)

    plt.scatter(np.arange(20), output_np_original_mod , color='black', marker='x', s=80)
    if doRanking:
        plt.ylabel('relative causality rank')
    else:
        plt.ylabel('relative causality')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.gcf().set_size_inches(10, 7)
    if doRanking:
        plt.savefig('./images/new1/breastBootBox_rank.pdf')
    else:
        plt.savefig('./images/new1/breastBootBox.pdf')
    plt.show()
    quit()


    #df = pd.DataFrame(x, columns = ['Number of Pathways','Number of Errors','Method'])


    #sns.stripplot(data=df, x="Number of Pathways",
    #          y="Number of Errors", hue="Method",
    #          hue_order=methods,
    #          alpha=.4, dodge=True, linewidth=1, jitter=.1,)
    #sns.boxplot(data=df, x="num",
    #            y="Number of Errors", hue="Method", showfliers=False)

    #plt.show()


    #quit()



    #'''
    saveName = './images/multi/breastBootOriginal.pdf'
    breastMakePlot(output_np_original, toShow, saveName)
    saveName = './images/multi/breastBootMean.pdf'
    breastMakePlot(output_mean, toShow, saveName)
    saveName = './images/multi/breastBootUpper.pdf'
    breastMakePlot(output_upp, toShow, saveName)
    saveName = './images/multi/breastBootLower.pdf'
    breastMakePlot(output_low, toShow, saveName)
    quit()
    #'''



    #quit()



    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(output_np_original, vmin=vmin, vmax=vmax, cmap='bwr')
    axs[0, 1].imshow(output_mean, vmin=vmin, vmax=vmax, cmap='bwr')
    axs[1, 0].imshow(output_mean-output_sigma, vmin=vmin, vmax=vmax, cmap='bwr')
    axs[1, 1].imshow(output_mean+output_sigma, vmin=vmin, vmax=vmax, cmap='bwr')

    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()

    #plt.ylim(0, 1.6)
    #plt.vmin(0)
    #plt.vmax(1.6)
    #fig.set_clim(vmin=0,vmax=1.6)
    plt.show()



#realBootstrap()
#quit()




def findSubsetPhylogeny():

    maxM = 7

    def treeToString(choiceTrees):

        #This function converts trees to unique strings, which is useful for other tasks.
        #For instance, it allows to apply np.unique capabilities to trees.
        #The format used is also useful for going back to tree data.

        choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]

        choiceNums = np.sort(choiceNums, axis=1)

        choiceString = np.zeros(choiceNums.shape[0]).astype(str)
        for a in range(0, choiceNums.shape[0]):
            #choiceString[a] = str(choiceNums[a, 0]) + ':' + str(choiceNums[a, 1]) + ':' + str(choiceNums[a, 2]) + ':' + str(choiceNums[a, 3])
            choiceString1 = str(choiceNums[a, 0])
            for b in range(1, choiceNums.shape[1]):
                choiceString1 = choiceString1 + ':' + str(choiceNums[a, b])
            choiceString[a] = choiceString1

        return choiceString

    doPath0 = False
    doSelection = True
    if doSelection:
        cutOffs = np.array([0.9])
        doPathList = [doPath0]
    else:
        cutOffs = np.array([0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
        #doPathList = [False, True]
        doPathList = [False]


    reduction = np.zeros((cutOffs.shape[0], 2))
    accuracyKeep = np.zeros((cutOffs.shape[0], 2))

    dataNum = 20
    reductionAll = np.zeros((cutOffs.shape[0], dataNum, 2))
    accuracyKeepAll = np.zeros((cutOffs.shape[0], dataNum, 2))

    accuracyList = []

    for doPath in doPathList:
        for cutOffArg in range(cutOffs.shape[0]):
            patientNum = 0
            correctPatientNum = 0
            extraNum = 0
            extraKeep = 0
            probMissed = 0



            if doPath:
                T = 1
                dataNum = 30
            else:
                T = 4
                N = 10
                dataNum = 20

            for b in range(0, dataNum):
                #This loads in the model
                baseline = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(b) + '_baseline.pt.npy')
                sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_bulkSample.npz').astype(int)

                bulkTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_bulkTrees.npz').astype(int)
                trueTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_trees.npz').astype(int)

                if doPath:
                    treeLength = np.zeros(sampleInverse.shape) + 5
                    bulkTrees[bulkTrees == 100] = 20
                else:
                    treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(b) + '_treeSizes.npz')

                #print (treeLength.shape)
                #quit()


                #print (trueTrees.shape)
                #print (sampleInverse.shape)
                #print (np.unique(sampleInverse).shape)
                #quit()

                #print (treeLength.shape)
                #quit()


                bulkTrees = treeToString(bulkTrees)
                trueTrees = treeToString(trueTrees)
                #print (sampleInverse[:10])
                #print (bulkTrees[:6])
                #print (trueTrees[0])

                #print (bulkTrees[0])
                #print (trueTrees[0])
                #quit()

                printNum = 0
                args1 = np.argwhere(sampleInverse == printNum)[:, 0]

                #print (trueTrees[printNum])
                #print (bulkTrees[args1[0]])

                #print (np.unique(trueTrees[printNum]))
                #print (np.unique(bulkTrees[args1[0]]))
                #quit()



                #print (np.unique(sampleInverse).shape)
                #quit()


                validSubset = np.zeros(trueTrees.shape[0])
                sumKeep = 0


                _, sampleInverse = np.unique(sampleInverse, return_inverse=True)
                _, startIndex = np.unique(sampleInverse, return_index=True)
                startIndex = np.concatenate((startIndex, np.array( [sampleInverse.shape[0]] )))
                for a in range(0 , len(startIndex)-1):
                    args1 = np.arange(startIndex[a+1] - startIndex[a]) + startIndex[a]
                    baseline[args1] = baseline[args1] / np.sum(baseline[args1])

                    #treeSize1 = treeLength[a]


                    args1 = args1[np.argsort(baseline[args1])][-1::-1]

                    #print (baseline[args1])

                    #probSum = np.cumsum(baseline[args1]) - baseline[args1]

                    #size1 = (args1.shape[0] + 1) // 2
                    #args2 = args1[:size1]

                    #args2 = args1[probSum < 0.85]

                    cutOff = cutOffs[cutOffArg]

                    cutOff1 = min(cutOff, np.max(baseline[args1]) - 0.0001)

                    args2 = args1[baseline[args1] > cutOff1]

                    probMissed += np.sum(baseline[args1][baseline[args1] <=cutOff1])

                    sumKeep += (args2.shape[0] - 1)

                    if trueTrees[a] in bulkTrees[args2]:
                        validSubset[a] = 1



                    #print (bulkTrees_now)
                    #print (trueTree_now)
                    #quit()

                    #sizeFull.append(args1.shape[0])

                    #cumsum1 = np.cumsum(np.sort(baseline[args1]))
                    #sizeHalf[0].append(np.argwhere(cumsum1 > 0.1).shape[0])
                    #sizeHalf[1].append(np.argwhere(cumsum1 > 0.2).shape[0])
                    #sizeHalf[2].append(np.argwhere(cumsum1 > 0.5).shape[0])

                patientNum += validSubset.shape[0]
                correctPatientNum += np.sum(validSubset)

                #print (validSubset.shape)
                #quit()

                size1 = validSubset.shape[0] // 2


                #accuracyList.append(np.sum(validSubset)/validSubset.shape[0])
                accuracyList.append(np.sum(validSubset[:size1])/size1)

                #correctPatientNum/ patientNum

                extraNum += baseline.shape[0] - trueTrees.shape[0]
                extraKeep += sumKeep


                reductionAll[cutOffArg, b, 0] = sumKeep / (baseline.shape[0] - trueTrees.shape[0])
                accuracyKeepAll[cutOffArg, b, 0] = np.sum(validSubset) / validSubset.shape[0]


                #print ("Hi")
                #print (extraNum / patientNum)
                #print (correctPatientNum/ patientNum)
                #print (extraKeep / extraNum)
                #print (probMissed / patientNum)

            doPathNum = 0
            if doPath:
                doPathNum = 1
            reduction[cutOffArg, doPathNum] = extraKeep / extraNum
            accuracyKeep[cutOffArg, doPathNum] = correctPatientNum/ patientNum


    import matplotlib.pyplot as plt

    #print (np.median(accuracyList))
    #quit()

    if doSelection:

        if doPath:
            print (accuracyList)
            np.save('./plotResult/cloMuSelectPath.npy', accuracyList)
        else:
            print (accuracyList)
            np.save('./plotResult/cloMuSelect.npy', accuracyList)


    else:


        plt.plot(1 - reductionAll[:, :, 0], accuracyKeepAll[:, :, 0], c='C9', linewidth=1)
        plt.plot(1 - reduction[:, 0], accuracyKeep[:, 0], c='C0', linewidth=3)
        plt.title("Phylogeny Set Reduction")
        plt.xlabel("Proportion of Extra Trees Removed")
        plt.ylabel("Proportion of Patients with True Tree")
        plt.show()
        quit()


        print (reduction)
        print (accuracyKeep)
        plt.plot(1 - reduction[:, 0], accuracyKeep[:, 0])
        plt.title("Phylogeny Set Reduction")
        plt.xlabel("Proportion of Extra Trees Removed")
        plt.ylabel("Proportion of Patients with True Tree")
        plt.show()


        if False:
            print (reduction)
            print (accuracyKeep)
            plt.plot(1 - reduction[:, 0], accuracyKeep[:, 0])
            plt.plot(1 - reduction[:, 1], accuracyKeep[:, 1])
            plt.legend(['Causal Data Set', 'Pathway Data Set'])
            plt.title("Phylogeny Set Reduction")
            plt.xlabel("Proportion of Extra Trees Removed")
            plt.ylabel("Proportion of Patients with True Tree")
            plt.show()


#findSubsetPhylogeny()
#quit()


def fromRecapTrees(filename, maxM):

    trees = []

    #edges = []
    with open(filename) as f:

        start1 = False
        done = False
        hashline = False
        patNum = -1
        edgeNum = 0
        while not done:

            try:
                #if True:
                line = f.readline().rstrip("\n")
                #print (line)

                if line[-8:] == 'patients':
                    start1 = True

                if start1:
                    addTree = False
                    if ('patient 0' in line) and (patNum > 2):
                        done = True
                        #quit()
                    if '#' in line:
                        hashline = True
                        #edgeNum += 1
                    else:
                        if hashline == True:
                            edgeNum = 0
                            patNum += 1
                            if patNum != 0:
                                trees.append(np.copy(Tree1))
                            Tree1 = np.zeros((maxM, 2)).astype(int).astype(str)
                            Tree1[:] = '-1'
                        hashline =  False
                        if not 'dummy' in line.split(' '):

                            mut1 = line.split(' ')[0]
                            mut2 = line.split(' ')[1]

                            Tree1[edgeNum][0] = mut1
                            Tree1[edgeNum][1] = mut2

                            #print (Tree1)
                            #print (line)

                            #print (mut1, mut2)

                            #edges.append([mut1, mut2])

                            edgeNum += 1



            except:
                done = True

    trees = np.array(trees)

    #print (trees[:10])
    #print (trees.shape)
    #edges = np.array(edges)

    return trees


def singleTreePrediction(revolver=False):

    def treeToString(choiceTrees):

        #This function converts trees to unique strings, which is useful for other tasks.
        #For instance, it allows to apply np.unique capabilities to trees.
        #The format used is also useful for going back to tree data.

        choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]

        choiceNums = np.sort(choiceNums, axis=1)

        choiceString = np.zeros(choiceNums.shape[0]).astype(str)
        for a in range(0, choiceNums.shape[0]):
            #choiceString[a] = str(choiceNums[a, 0]) + ':' + str(choiceNums[a, 1]) + ':' + str(choiceNums[a, 2]) + ':' + str(choiceNums[a, 3])
            choiceString1 = str(choiceNums[a, 0])
            for b in range(1, choiceNums.shape[1]):
                choiceString1 = choiceString1 + ':' + str(choiceNums[a, b])
            choiceString[a] = choiceString1

        return choiceString

    accuracyList = []

    doPath = True

    if doPath:
        dataNum = 30
        T = 1
    else:
        dataNum = 20
        T = 4

    for dataSet in range(dataNum):

        if doPath:
            if revolver:
                filename = './dataBaseline/revolver/output/1/s' + str(dataSet) + '_k1_n499_M5_m5_revolver_output.txt'
            else:
                filename = './dataRecap/T1_R' + str(dataSet) + '_solution.txt'
        else:
            if revolver:
                filename = './dataBaseline/revolver/output/4/s' + str(dataSet) + '_k1_n499_M7_m7_revolver_output.txt'
            else:
                filename = './dataRecap/T4_R' + str(dataSet) + '_solution.txt'

        maxM = 7
        trees = fromRecapTrees(filename, maxM)
        #print (trees.shape)
        #quit()
        trueTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_trees.npz').astype(int)
        #if doPath:
        #    treeLength = np.zeros(true)
        #else:
        #    treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')

        if revolver:
            trees[trees == '-1'] = 12
            trees = trees.astype(int)
            trees = trees - 1
            trees[trees == -1] = 10
            #print (trees[0])
            #quit()
        else:
            trees[trees == '-1'] = 11
            for a in range(20):
                trees[trees == ('M' + str(a))] = a
            trees[trees == 'GL'] = 10
            trees = trees.astype(int)

        trueTrees = trueTrees[:trees.shape[0]]

        trees = treeToString(trees)
        trueTrees = treeToString(trueTrees)


        errorList = []
        for a in range(trees.shape[0]):
            error1 = 0
            if trees[a] != trueTrees[a]:
                error1 = 1
            errorList.append(error1)
        errorList = np.array(errorList)

        accuracyRate = 1 - np.mean(errorList)

        accuracyList.append(accuracyRate)

    accuracyList = np.array(accuracyList)
    #print (accuracyList)

    #print (accuracyList.shape)
    #quit()

    if doPath:
        if revolver:
            np.save('./plotResult/revolverSelectPath.npy', accuracyList)
        else:
            np.save('./plotResult/recapSelectPath.npy', accuracyList)
    else:
        if revolver:
            np.save('./plotResult/revolverSelect.npy', accuracyList)
        else:
            np.save('./plotResult/recapSelect.npy', accuracyList)

def savePathwaySimulationPredictions():

    #This determines the predicted evolutionary pathways based on the model on the simulated pathway data set.
    #There is another function with very similar code which predicts evolutionary pathways on generic data sets.
    #One difference between this functon and that function, is that this function tries to accuratly predict the evolutionary
    #pathways on data sets with strong true evolutionary pathways, and the other function tries to detect possible
    #evolutionary pathways on real data sets with weak ambigous evolutionary pathways.

    def pathwayTheoryProb(pathway, prob_assume):

        #This calculates the probability of an evolutionary pathway simply based
        #on the frequency of the different mutations that occur in the pathway,
        #completely ignoring the relationships between mutations.

        probLog = 0
        pathwayLength = len(pathway)
        for a in range(0, pathwayLength):
            pathwaySet = pathway[a]
            if len(pathwaySet) == 0:
                #probLog += -1000
                True
            else:
                prob = np.sum(prob_assume[np.array(pathwaySet)])
                probLog += np.log(prob)

        return probLog

    def pathwayRealProb(pathway, prob2_Adj_3D):

        #This calculates the probability of an evolutionary pathway
        #according to the model.

        subset = prob2_Adj_3D

        if min(min(len(pathway[0]), len(pathway[1])), len(pathway[2]) ) == 0:
            subset_sum = np.log(1e-50)
        else:

            subset = subset[np.array(pathway[0])]
            subset = subset[:, np.array(pathway[1])]
            subset = subset[:, :, np.array(pathway[2])]

            subset_max = np.max(subset)
            subset = subset - subset_max
            subset_sum = np.sum(np.exp(subset))
            subset_sum = np.log(subset_sum+1e-50)
            subset_sum = subset_sum + subset_max

        return subset_sum



    def evaluatePathway(pathway, prob2_Adj_3D, prob_assume, includeProb=False):

        #This determines how "good" or "valid" and evolutionary pathway is
        #based on its frequency of occuring and its frequency of occuring beyond the expected
        #probability from the frequency of mutations in the pathway.

        probTheory = pathwayTheoryProb(pathway, prob_assume)
        probReal = pathwayRealProb(pathway, prob2_Adj_3D)

        probDiff = probReal - probTheory


        score = (probReal * 0.4) + probDiff

        if includeProb:
            return score, probReal, probDiff
        else:
            return score



    def singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume):

        #This tries modifying a single mutation in the evolutionary pathway and then
        #scoring the results

        pathway2 = copy.deepcopy(pathway)
        set1 = pathway2[step]
        set1 = np.array(set1)

        if doAdd:
            set1 = np.concatenate((set1, np.array([position])))
            set1 = np.sort(set1)
        else:
            set1 = set1[set1 != position]

        #pathway2[step] = set1.astype(int)
        pathway2[step] = set1.astype(int)

        score= evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

        return score, pathway2

    def stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume):

        #This tries every possibile modification of a particular step in an evolutionary pathway

        M = prob_assume.shape[0]

        set1 = copy.deepcopy(pathway[step])
        if doAdd or (len(set1) > 1):
            set2 = copy.deepcopy(superPathway[step])
            set3 = set2[np.isin(set2, set1) != doAdd]

            pathways = []
            scores = []
            for position in set3:
                score, pathway2 = singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume)

                pathways.append(copy.deepcopy(pathway2))
                scores.append(score)

            return pathways, scores

        else:

            return [], []


    def iterOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        #This function does a single optimization step in optimizing an evolutionary pathway

        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]
        for doAdd in doAddList:
            for step in stepList:
                pathways, scores = stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume)
                pathways2 = pathways2 + pathways
                scores2 = scores2 + scores





        for step in stepList:
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    pathway2[step] = list(np.arange(M))
                else:
                    pathway2[step] = [pos_add]

                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))






        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def iterOptimizePathway2(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        #This function does a single optimization step in optimizing an evolutionary pathway
        #It also allows for steps of the evolutionary pathway to be set to "all mutations" or
        #"only one mutation", which helps it break out of local minima.

        M = prob_assume.shape[0]

        #This scores the original pathway
        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]

        #This iterates through the steps in the pathway
        for step in stepList:

            #This finds the mutations at this step in the current pathway
            set1 = copy.deepcopy(pathway[step])
            #This finds the mutations which are allowed to be used.
            #Currently this is set to the set of all mutations, but can be modified by the user.
            set2 = copy.deepcopy(superPathway[step])


            #This finds the mutations which are allowed to be added.
            set_add = set2[np.isin(set2, set1) != True]
            #This finds the mutations which are allowed to be removed.
            set_rem = set2[np.isin(set2, set1) != False]

            #This "-1 mutation" represents doing no modificiation.
            set_add = [-1] + list(set_add)
            set_rem = [-1] + list(set_rem)

            #This iterates through the mutations which can be added
            for pos_add in set_add:
                pathway2 = copy.deepcopy(pathway)

                #This modifies the pathway by adding the mutation pos_add and scores it.
                if pos_add >= 0:
                    score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                #This records the new pathway and its score
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

            #This iterates through the mutations which can be removed
            for pos_rem in set_rem:
                pathway2 = copy.deepcopy(pathway)

                #This modifies the pathway by removing the mutation pos_rem and scores it.
                if pos_rem >= 0:
                    score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)

                #This records the new pathway and its score
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        #This part of the code tries setting a step in the pathway to the set of all mutations or the
        #set with only one mutation. This major modification to the pathway helps prevent the pathway optimization
        #from getting stuck in a local minima.
        for step in stepList:
            #The "-1 mutation" represents adding all mutations.
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    #If pos_add == -1, then it sets this step in the pathway to the set of all mutations
                    pathway2[step] = list(np.arange(M))
                else:
                    #It sets this step in the pathway to only being the mutation pos_add
                    pathway2[step] = [pos_add]

                #This scores the pathway created.
                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                #This records the modified pathway and its score.
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        #This finds the modified pathway with the highest score.
        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob_Adj_3D, prob_assume):

        #This function fully optimizes an evolutionary pathway.

        pathway2 = copy.deepcopy(pathway)
        bestScoreBefore = -10000
        notDone = True
        while notDone:
            #pathway2, bestScore = iterOptimizePathway(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)
            pathway2, bestScore = iterOptimizePathway2(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)




            if bestScoreBefore == bestScore:
                notDone = False
            bestScoreBefore = bestScore

        #print (evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True))

        return pathway2



    def removePathFromProb(prob, pathway):

        #This removes the some pathway from the tensor of probabilities of different
        #evolutionary trajectories.
        #This makes it so the same pathway will not be found repeatedly,
        #since the probabilities of a pathway that has already been found is set to zero.
        #It also makes it unlikly for the method to find similar pathways repeatedly.

        set1, set2, set3 = np.array(pathway[0]), np.array(pathway[1]), np.array(pathway[2])
        M = prob.shape[0]
        modMask = np.zeros(prob.shape)
        modMask[np.arange(M)[np.isin(np.arange(M), set1) == False]] = 1
        modMask[:, np.arange(M)[np.isin(np.arange(M), set2) == False]] = 1
        modMask[:, :, np.arange(M)[np.isin(np.arange(M), set3) == False]] = 1

        prob[modMask == 0] = -1000

        return prob


    def doMax(ar1):

        #Calculating the maximum of an array while avoiding errors if
        #the array is empty.

        if ar1.size == 0:
            #print ("A")
            return 0
        else:
            #print ("B")
            return np.max(ar1)



    def doProbabilityFind(M, model, argsInteresting, mutationName):

        #This function uses the model to calculate the probability of different
        #evolutionary trajectories.
        #It also calculates the frequency at which different mutations occur within evolutionary trajectories.
        #These can then be easily used to find the probability of evolutionary pathways,
        #which are just a certian kind of sets of evolutionary trajectories.

        #X0 is one empty clone. It is used to get the probability of the initial mutation.
        #X1 is are clones with 1 mutation. It is used to get the probability of the second mutation.
        #X2 is clones with 2 mutations. It is used to get the probability of the third mutation.
        X0 = torch.zeros((1, M))
        X1 = torch.zeros((M, M))
        X2 = torch.zeros((M*M, M))

        arange0 = np.arange(M)
        arange1 = np.arange(M*M)

        X1[arange0, arange0] = 1

        X2[arange1, arange1 % M] = X2[arange1, arange1 % M] + 1
        X2[arange1, arange1 // M] = X2[arange1, arange1 // M] + 1

        #pred0, pred1, and pred2 give the probaility weights for the first second and third mutations.
        pred0, _ = model(X0)
        pred1, xLatent1 = model(X1)
        pred2, _ = model(X2)

        #This removes the possibility of an evolutionary trajectory having the same mutation multiple times.
        pred2 = pred2.reshape((M, M, M))
        pred1[arange0, arange0] = -1000
        pred2[arange0, arange0, :] = -1000
        pred2[:, arange0, arange0] = -1000
        pred2[arange0, :, arange0] = -1000
        pred2 = pred2.reshape((M * M, M))






        #This gives the probability for the first, second, and third mutations.
        prob0 = torch.softmax(pred0, dim=1)
        prob1 = torch.softmax(pred1, dim=1)
        prob2 = torch.softmax(pred2, dim=1)


        #This uses the probability for the first tree mutations to give the probability of
        #every trajectory of three consecutive mutations.
        prob0 = prob0.data.numpy()
        prob1 = prob1.data.numpy()
        prob2 = prob2.data.numpy()
        prob0_Adj = np.log(np.copy(prob0) + 1e-10)
        outputProb0 = np.log(prob0[0] + 1e-10)
        outputProb0 = outputProb0.repeat(M).reshape((M, M))
        prob1_Adj = outputProb0 + np.log(prob1 + 1e-10)
        outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
        prob2_Adj = outputProb1 + np.log(prob2 + 1e-10)




        if False:
            #This unused code is an alternative method of defining the probability of evolutionary trajectories based on the
            #expected rate in which they occur.
            #With the default method, the first mutation in a trajectory being more fit does not necesarily make the trajectory more likly.
            #However, with this method, the first mutation being more fit will mean the next mutations will occur more quickly and therefore
            #the trajectory occurs at a higher rate.
            prob0 = pred0.data.numpy() * -1
            prob1 = pred1.data.numpy() * -1
            prob2 = pred2.data.numpy() * -1
            prob0_Adj = np.copy(prob0)
            outputProb0 = prob0[0]
            outputProb0 = outputProb0.repeat(M).reshape((M, M))
            prob1_Adj = addFromLog([outputProb0, prob1])
            outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
            prob2_Adj = addFromLog([outputProb1, prob2])
            prob2_Adj = prob2_Adj * -1




        prob2_Adj_3D = prob2_Adj.reshape((M, M, M))

        #This ensures the total probability of all trajectories is 1, despite certian
        #trajectories having their probability set to zero due to being impossible.
        prob2_Adj_3D = prob2_Adj_3D - np.log(np.sum(np.exp(prob2_Adj_3D)))


        M = argsInteresting.shape[0] #+ 1


        prob2_Adj = prob2_Adj_3D.reshape((M*M, M))


        #This first calculates the frequency at which each mutation occurs as the first mutation, second mutation, and third mutation.
        prob2_sum0 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=0)
        prob2_sum1 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=1), axis=1)
        prob2_sum2 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=1)
        #Now this averages those three calculations to give the general frequency at which each mutation occurs.
        prob2_Assume = (prob2_sum0 + prob2_sum1 + prob2_sum2) / 3


        return prob2_Assume, prob2_Adj, prob2_Adj_3D, mutationName









    #32

    T = 1
    #T = 5
    #T = 6

    #This iterates through the simulated data sets.
    for Rnow in range(32):

        print (Rnow)

        #This loads in the data
        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(Rnow) + '_pathway.npz', allow_pickle=True)

        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_model.pt'
        model = torch.load(modelFile)

        #This says all mutations should be analyzed as possibly interesting mutations, and
        #gives integer names to the mutations.
        M = 20
        argsInteresting = np.arange(M)
        mutationName = np.arange(M)

        #This calculates the probability of different evolutionary trajectories and the frequency of different
        #mutations, both according to the model.
        prob2_Assume, prob2_Adj, prob2_Adj_3D, mutationName = doProbabilityFind(M, model, argsInteresting, mutationName)

        #This gives the initial pathway to start the optimization from, which is set to
        #the pathway of all mutations at all steps.
        pathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]
        #pathway = [[], np.arange(M-1), np.arange(M-1)]
        #pathway = [[], [] , np.arange(M-1)]

        #This limits the pathways to a subset of mutations at each step called a superPathway here.
        #However, in this case the superPathway is the superPathway of all mutations at all steps.
        superPathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]

        #This copies prob2_Adj_3D so that it can be modified at each step by removing pathways that have
        #already been found from the probabilities.
        prob2_Adj_3D_mod = np.copy(prob2_Adj_3D)



        #These arrays store the pathways found as well as there probability/score data.
        predictedPathways = []
        pathwayScoreList = []


        for a in range(0, 4): #It runs 4 iterations because there were never more than 4 pathways found.

            #score = evaluatePathway(pathway, prob2_Adj_3D, prob2_Assume)
            #doAdd = True
            #step = 0
            #position = 0
            doAddList = [True, False]
            stepList = [0, 1, 2]

            #This finds the optimal pathway
            pathway2 = singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D_mod, prob2_Assume)

            #This adds the pathway found to the list of pathways found
            predictedPathways.append(copy.copy(pathway2))

            #This scores the probabilities of the optimal pathway
            pathwayScores = evaluatePathway(pathway2, prob2_Adj_3D_mod, prob2_Assume, includeProb=True)

            #This adds the scores to the list of scores
            pathwayScoreList.append(copy.copy( [ pathwayScores[0], pathwayScores[1], pathwayScores[2]  ]  ))


            #This removes all the mutations in this pathway from the set of mutations with
            #nonzero probabilities. -1000 is used since there are log values.
            pathway2_full = np.concatenate((pathway2[0], pathway2[1], pathway2[2])).astype(int)
            prob2_Adj_3D_mod[pathway2_full] = -1000
            prob2_Adj_3D_mod[:, pathway2_full] = -1000
            prob2_Adj_3D_mod[:, :, pathway2_full] = -1000


        #This converts predicted pathways to a savable object
        predictedPathways = np.array(predictedPathways, dtype=object)

        #This saves the results.
        np.save('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_predictedPathway.npy', predictedPathways)
        np.save('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_pathwayScore.npy', pathwayScoreList)

def testPathwaySimulation():

    #This function determines how accurately the model reconstructed the evolutionary pathways
    #from the evolutionary pathway simulations.

    def givePathwayEdges(pathway):

        pathwayEdges = []
        for a in range(len(pathway)):
            for b in range(len(pathway[a]) - 1):
                for c in range(len(pathway[a][b])):
                    for d in range(len(pathway[a][b+1])):
                        mut1 = pathway[a][b][c]
                        mut2 = pathway[a][b+1][d]

                        pathwayEdges.append([mut1, mut2])


        pathwayEdges = np.array(pathwayEdges)

        return pathwayEdges

    def calculateEdgeError(predPathway, truePathway):

        predEdges = givePathwayEdges(predPathway)
        trueEdges = givePathwayEdges(truePathway)

        allEdges = np.concatenate((predEdges, trueEdges), axis=0)

        vals1 = uniqueValMaker(allEdges)

        predEdges = vals1[:predEdges.shape[0]]
        trueEdges = vals1[predEdges.shape[0]:]

        inter_num = np.intersect1d(predEdges, trueEdges).shape[0]

        error_num = predEdges.shape[0] + trueEdges.shape[0] - (2 * inter_num)

        true_pos = inter_num#.shape[0]
        false_pos = predEdges[np.isin(predEdges, trueEdges) == False].shape[0]
        false_neg = trueEdges[np.isin(trueEdges, predEdges) == False].shape[0]


        return error_num, trueEdges.shape[0], true_pos, false_pos, false_neg


    numberPathwaysErrors = 0

    pathwaySetError = np.zeros((2, 2))

    pathwayError = 0
    mutationsUsed = 0
    mutationTotal = 0

    T = 1
    #T = 5
    #T = 6

    errorRECAP = 0

    totalEdgeError = 0
    totalEdge = 0

    edgeList = []
    errorList = []
    pathwayNumList = []
    true_pos_list = []
    false_pos_list = []
    false_neg_list = []


    #This iterates through the data sets.
    for Rnow in range(0, 30):

        M = 20

        #Loading in the true pathways
        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(Rnow) + '_pathway.npz', allow_pickle=True)

        #Keeping track of the errors of on selecting a correct chain within the pathway
        for a in range(len(pathways_true)):
            for b in range(len(pathways_true[a])):
                errorRECAP += (len(pathways_true[a][b]) - 1)


        #Loading in the predicted pathways
        predictedPathways = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_predictedPathway.npy', allow_pickle=True)

        #Loading in the scores for the found pathways.
        pathwayScores = np.load('./dataNew/specialSim/results/T_' + str(T) + '_R_' + str(Rnow) + '_pathwayScore.npy')
        pathwayScores = np.array(pathwayScores)
        bestArgs = np.argwhere(pathwayScores[:, 0] > 0)[:, 0]

        if len(bestArgs) != len(pathways_true):

            numberPathwaysErrors += 1
            print (pathways_true)
            print (Rnow)
            quit()

        #This finds the set of mutations for all of the true pathways
        pathwaySets_true = []

        for b in range(len(pathways_true)):
            pathwaySets_true.append([])
            for c in range(len(pathways_true[b])):
                pathwaySets_true[b] = pathwaySets_true[b] + list(pathways_true[b][c])

        #This finds the set of mutations for all of the predicted pathways
        pathwaySets_pred = []

        for b in range(len(bestArgs)):
            pathwaySets_pred.append([])
            for c in range(len(predictedPathways[b])):
                pathwaySets_pred[b] = pathwaySets_pred[b] + list(predictedPathways[b][c])

        #This calculates the overlap between the sets of mutations at each step of the pathway.
        #This essentially calculates the similarity between the predicted pathways and true pathways.
        #Additionally, this is done for all pairs of a true pathway and predicted pathway, in orde r
        overlapScores = np.zeros((len(pathwaySets_true), len(pathwaySets_pred) ))
        overlapScoresBoth = np.zeros((len(pathwaySets_true), len(pathwaySets_pred), 2 ))
        for b in range(len(pathwaySets_true)):
            for c in range(len(pathwaySets_pred)):

                #These are the predicted and true sets of mutations at this step in the pathway.
                set_true = np.array(pathwaySets_true[b])
                set_pred = np.array(pathwaySets_pred[c])

                #This determines the number of differences between these two sets.
                overlapScore = (set_true.shape[0] + set_pred.shape[0]) - (2 * np.intersect1d(set_true, set_pred).shape[0])

                overlapScores[b][c] = overlapScore

                #This records the false positive and false negative scores seperately
                overlapScoresBoth[b][c][0] = set_true.shape[0] -    np.intersect1d(set_true, set_pred).shape[0]
                overlapScoresBoth[b][c][1] = set_pred.shape[0] -    np.intersect1d(set_true, set_pred).shape[0]

        #This finds which predicted pathway is closest to each true pathway.
        #Note, this is neccesary since pathways are not naturally "ordered" so we need to make sure
        #the correct predicted pathway is being compared to the correct true pathway.
        overlapBestArg = np.argsort(overlapScores, axis=0)[0, :]
        #This gives the score of the predicted pathway which is closest to the true pathway.
        overlapBest = np.min(overlapScores, axis=0)

        #This gives the seperate false positive and false negative scores.
        overlapScoresBoth = overlapScoresBoth[overlapBestArg, np.arange(overlapBestArg.shape[0])]
        #print (overlapScoresBoth)

        #This re-orders the predicted pathways to match the ordering of the true pathways to make
        #the compairison easier.
        predictedPathways_copy = []
        for b in range(len(overlapBestArg)):
            predictedPathways_copy.append(copy.copy(  predictedPathways[overlapBestArg[b]]   ))
        predictedPathways = predictedPathways_copy

        num_error, num_edge, true_pos_edge, false_pos_edge, false_neg_edge = calculateEdgeError(predictedPathways, pathways_true)

        totalEdgeError += num_error
        totalEdge += num_edge

        edgeList.append(num_edge)
        errorList.append(num_error)
        pathwayNumList.append(len(pathways_true))
        true_pos_list.append(true_pos_edge)
        false_pos_list.append(false_pos_edge)
        false_neg_list.append(false_neg_edge)


        #This records to true Positive, false Positive, true Negative, false negative data for all of the pathways
        if len(bestArgs) == len(pathways_true):
            for b in range(len(predictedPathways)):

                set_true = np.array(pathwaySets_true[b])
                set_pred = np.array(pathwaySets_pred[overlapBestArg[b]])

                #print (set_true)
                #print (set_pred)

                truePos = np.intersect1d(set_true, set_pred).shape[0]
                falsePos = set_pred.shape[0] - truePos
                falseNeg = set_true.shape[0] - truePos
                trueNeg = M - truePos - falsePos - falseNeg



                pathwaySetError[0][0] += truePos
                pathwaySetError[0][1] += falsePos
                pathwaySetError[1][0] += falseNeg
                pathwaySetError[1][1] += trueNeg

                #print (pathwaySetError)


            mutationErrorSet = []
            for b in range(len(predictedPathways)):
                for c in range(len(predictedPathways[b])):
                    set_true = np.array(pathways_true[b][c])
                    set_pred = np.array(predictedPathways[b][c])

                    mutationsUsed += set_true.shape[0]

                    mutationErrorSet = mutationErrorSet + list(set_true[np.isin(set_true, set_pred) == False])
                    mutationErrorSet = mutationErrorSet + list(set_pred[np.isin(set_pred, set_true) == False])
            mutationErrorSet = np.array(mutationErrorSet)


            mutationErrorSet = np.unique(mutationErrorSet)

            pathwayError += mutationErrorSet.shape[0]
            mutationTotal += 20


    #This prints the final accuracy results.
    print ("Number of Mutations: " + str( (Rnow+1) * M ))
    print ("Mutations Used In Pathways : " + str( mutationsUsed ))
    print ("Number Errors: " + str( pathwayError ))
    print ('')
    print ("Edges Used In Pathways : " + str( totalEdge ))
    print ("Number Edge Errors: " + str( totalEdgeError ))


    x = np.array([errorList, edgeList, pathwayNumList, true_pos_list, false_pos_list, false_neg_list]).T

    np.save('./plotResult/cloMuPathway.npy', x)

#testPathwaySimulation()
#quit()

def testLatentPathway():

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships

    T = 1
    #T = 4
    N = 30

    M = 20
    #M = 10

    errorList = []


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    diff_inter = []
    diff_other = []
    sim_inter = []
    sim_other = []

    #for a in range(20, N):
    for a in range(0, N):

        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_pathway.npz', allow_pickle=True)

        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))

        X[np.arange(M), np.arange(M)] = 1

        #This gives the predicted probability of new mutations on the clones.
        output, latentX = model(X)

        #latentX = np.arctanh(latentX * 0.9)

        for b in range(5):
            latentX[:, b] = latentX[:, b] - np.median(latentX[:, b])

        #print (pathways_true)

        import matplotlib.pyplot as plt





        if False:
            print (pathways_true)
            #plotArgs = np.array([0, 1, 2, 7, 8, 9, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).astype(int)
            #max1 = np.max(np.abs(latentX))
            min1 = np.min(latentX)
            max1 = np.max(latentX)
            plt.plot(latentX[:])
            #plt.vlines([-0.5, 2.5, 6.5, 9.5], -1 * max2, max2, colors='black', linestyles='dashed')
            plt.vlines([-0.5, 2.5, 6.5, 9.5], -10, 10, colors='grey', linestyles='dashed')
            plt.ylim(min1 - 0.05, max1 + 0.05)
            plt.xlabel('Mutation Number')
            plt.ylabel("Latent Representation Value")
            plt.show()



        layerNum = 0

        for b in range(len(pathways_true)):
        #for b in [0]:
            if len(pathways_true[b][layerNum]) > 1:

                for c in range( len(pathways_true[b][layerNum]) ):
                    mut1 = pathways_true[b][layerNum][c]


                    mutList = pathways_true[b][layerNum]
                    mutList = np.array(mutList)
                    mutList0 = mutList[mutList!=c]

                    #mutListBoth = pathways_true[b][0] + pathways_true[b][1]
                    #mutListBoth = list(np.arange(20)[pathways_true[b][1][-1]+1:]) + list(pathways_true[b][layerNum])

                    #size1 = np.sum(latentX[mut1] ** 2)

                    latentX_1 = latentX[mut1].repeat(M).reshape((5, M)).T
                    #latentX_2 = latentX[mut2].repeat(M).reshape((5, M)).T

                    diff = latentX - latentX_1

                    diff = np.sum(diff**2, axis=1) ** 0.5
                    cutOff = 0.9


                    diff_inter += list(diff[mutList0])
                    diff_other += list(diff[np.isin(np.arange(diff.shape[0]), mutList)==False])

                    size_other = diff[np.isin(np.arange(diff.shape[0]), mutList)==False].shape[0]
                    sim_inter += list(a + np.zeros(len(mutList0), dtype=int))
                    sim_other += list(a + np.zeros(size_other, dtype=int))


                    argNear = np.argwhere(diff < cutOff)[:, 0]

                    truePos = np.intersect1d(argNear, mutList0).shape[0]
                    falsePos = argNear[np.isin(argNear, mutList) == False].shape[0]
                    trueNeg = np.arange(M)
                    trueNeg = trueNeg[np.isin(trueNeg, mutList) == False]
                    trueNeg = trueNeg[np.isin(trueNeg, argNear) == False]
                    trueNeg = trueNeg.shape[0]
                    falseNeg = mutList0[np.isin(mutList0, argNear) == False].shape[0]

                    errorList.append([truePos, trueNeg, falsePos, falseNeg ])

                    print (errorList[-1])

    errorList = np.array(errorList)

    #print (np.sum(errorList, axis=0))

    diff_inter = np.array(diff_inter)
    diff_other = np.array(diff_other)
    sim_inter = np.array(sim_inter)
    sim_other = np.array(sim_other)

    np.save('./sending/latentDist/distance_interchangable.npy', diff_inter)
    np.save('./sending/latentDist/distance_not_interchangable.npy', diff_other)
    np.save('./sending/latentDist/simulationNumber_interchangable.npy', sim_inter)
    np.save('./sending/latentDist/simulationNumber_not_interchangable.npy', sim_other)
    quit()
    #print (diff_inter.shape)
    #quit()

    plt.hist(diff_inter, bins=200, range=(0, 10), histtype='step')
    plt.hist(diff_other, bins=200, range=(0, 10), histtype='step')
    plt.xlabel("Euclidean Distance Between Representations")
    plt.ylabel("Number of Mutations")
    plt.legend(['Interchangable Mutations', 'Non-Interchangable Mutations'])
    plt.show()

#testLatentPathway()
#quit()



def testLatentPathwaySecound():

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships

    T = 1
    #T = 4
    N = 30

    M = 20
    #M = 10

    errorList = []


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    diff_inter = []
    diff_other = []

    #for a in range(20, N):
    for a in range(0, N):

        pathways_true = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_pathway.npz', allow_pickle=True)

        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)





        layerNum = 1

        for b in range(len(pathways_true)):

            if len(pathways_true[b][layerNum]) > 1:

                mutList_first = pathways_true[b][0]

                diff_list = []

                #for firstNum in mutList_first:
                #for firstNum in [100, mutList_first[0]]:
                for firstNum in ([100] +  list(mutList_first)):

                    #This prepares M clones, where the ith clone has only mutation i.
                    X = torch.zeros((M, M))

                    #for d in mutList_first:
                    #    X[:, d] = 1 / (len(mutList_first))

                    if firstNum != 100:
                        X[:, firstNum] = 1

                    X[np.arange(M), np.arange(M)] = 1


                    #This gives the predicted probability of new mutations on the clones.
                    output, latentX = model(X)

                    #latentX = np.arctanh(latentX * 0.9)

                    for d in range(5):
                        latentX[:, d] = latentX[:, d] - np.median(latentX[:, d])

                    #print (pathways_true)

                    import matplotlib.pyplot as plt




                    for c in range( len(pathways_true[b][layerNum]) ):
                        mut1 = pathways_true[b][layerNum][c]


                        latentX_1 = latentX[mut1].repeat(M).reshape((5, M)).T
                        #latentX_2 = latentX[mut2].repeat(M).reshape((5, M)).T

                        diff = latentX - latentX_1

                        diff = np.sum(diff**2, axis=1) ** 0.5

                        diff_list.append(np.copy(diff))

                diff_list = np.array(diff_list)
                #diff = np.max(diff_list, axis=0)
                diff = np.mean(diff_list, axis=0)

                for c in range( len(pathways_true[b][layerNum]) ):
                    mut1 = pathways_true[b][layerNum][c]


                    mutList = pathways_true[b][layerNum]
                    mutList = np.array(mutList)
                    mutList0 = mutList[mutList!=c]

                    #cutOff = 0.7
                    cutOff = 0.55


                    diff_inter += list(diff[mutList0])
                    diff_other += list(diff[np.isin(np.arange(diff.shape[0]), mutList)==False])


                    argNear = np.argwhere(diff < cutOff)[:, 0]

                    truePos = np.intersect1d(argNear, mutList0).shape[0]
                    falsePos = argNear[np.isin(argNear, mutList) == False].shape[0]
                    trueNeg = np.arange(M)
                    trueNeg = trueNeg[np.isin(trueNeg, mutList) == False]
                    trueNeg = trueNeg[np.isin(trueNeg, argNear) == False]
                    trueNeg = trueNeg.shape[0]
                    falseNeg = mutList0[np.isin(mutList0, argNear) == False].shape[0]

                    errorList.append([truePos, trueNeg, falsePos, falseNeg ])

                    print (errorList[-1])

    errorList = np.array(errorList)

    print (np.sum(errorList, axis=0))

    diff_inter = np.array(diff_inter)
    diff_other = np.array(diff_other)

    plt.hist(diff_inter, bins=200, range=(0, 10), histtype='step')
    plt.hist(diff_other, bins=200, range=(0, 10), histtype='step')
    plt.xlabel("Euclidean Distance Between Representations")
    plt.ylabel("Number of Mutations")
    plt.legend(['Interchangable Mutations', 'Non-Interchangable Mutations'])
    plt.show()

#testLatentPathwaySecound()
#quit()


def plotPathwaySimulation():

    #errorList, edgeList, pathwayNumList, true_pos_list, false_pos_list, false_neg_list

    x1 = np.load('./plotResult/cloMuPathway.npy')
    x2 = np.load('./plotResult/recapPathway.npy')
    x3 = np.load('./plotResult/revolverPathway.npy')

    x_now = x3
    precision1 = x_now[:, 3] / (x_now[:, 3] + x_now[:, 4])
    recall1 = x_now[:, 3] / (x_now[:, 3] + x_now[:, 5])


    print (np.median(precision1))
    print (np.median(recall1))
    quit()

    np.save('./sending/Pathway/CloMu.npy', x1[:, 2:])
    np.save('./sending/Pathway/RECAP.npy', x2[:, 2:])
    np.save('./sending/Pathway/REVOLVER.npy', x3[:, 2:])
    quit()




    x1 = np.load('./plotResult/cloMuPathway.npy')[:, :-3]
    x2 = np.load('./plotResult/recapPathway.npy')[:, :-3]
    x3 = np.load('./plotResult/revolverPathway.npy')[:, :-3]


    np.save('./sending/Pathway/CloMu.npy', x1)
    np.save('./sending/Pathway/RECAP.npy', x2)
    np.save('./sending/Pathway/REVOLVER.npy', x3)
    #quit()

    x = np.concatenate((x1, x2, x3), axis=0)
    x = x[:, np.array([2, 0, 1])]
    x[:x1.shape[0], 2] = 0
    x[x1.shape[0]:x1.shape[0] + x2.shape[0], 2] = 1
    x[x1.shape[0] + x2.shape[0]:, 2] = 2

    #x = np.concatenate(())

    # Import libraries
    #import matplotlib.pyplot as plt

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(x, columns = ['Number of Pathways','Number of Errors','Method'])

    df['Method'][df['Method'] == 0] = 'CloMu'
    df['Method'][df['Method'] == 1] = 'RECAP'
    df['Method'][df['Method'] == 2] = 'REVOLVER'

    #methods = [0, 1]
    methods = ['CloMu', 'RECAP', 'REVOLVER']

    sns.stripplot(data=df, x="Number of Pathways",
              y="Number of Errors", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1,)
    sns.boxplot(data=df, x="Number of Pathways",
                y="Number of Errors", hue="Method",
                hue_order=methods, showfliers=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])

    plt.show()


#plotPathwaySimulation()
#quit()


def plotDoublePathway():

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    x1 = np.load('./plotResult/cloMuPathway.npy')
    x2 = np.load('./plotResult/recapPathway.npy')
    x3 = np.load('./plotResult/revolverPathway.npy')


    x = np.concatenate((x1, x2, x3), axis=0).astype(float)

    precision = x[:, 3] / (x[:, 3] + x[:, 4])

    #plt.hist(precision)
    #plt.show()
    #quit()
    recall = x[:, 3] / (x[:, 3] + x[:, 5])

    x[:, 0] = precision
    x[:, 1] = recall
    x[:x1.shape[0], 2] = 0
    x[x1.shape[0]:x1.shape[0] + x2.shape[0], 2] = 1
    x[x1.shape[0] + x2.shape[0]:, 2] = 2
    x = x[:, :3]



    fig, axs = plt.subplots(2, 2)

    df = pd.DataFrame(x, columns = ['Precision','Recall','Method'])

    df['Method'][df['Method'] == 0] = 'CloMu'
    df['Method'][df['Method'] == 1] = 'RECAP'
    df['Method'][df['Method'] == 2] = 'REVOLVER'

    #df['Number of Pathways'][df['Number of Pathways'] == 1] = '1'
    #df['Number of Pathways'][df['Number of Pathways'] == 2] = '2'


    #methods = [0, 1]
    methods = ['CloMu', 'RECAP', 'REVOLVER']

    sns.stripplot(data=df, x="Precision",
              y="Method", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1, ax=axs[0, 1],)
    sns.boxplot(data=df, x="Precision",
                y="Method", hue="Method",
                hue_order=methods, showfliers=False, ax=axs[0, 1])


    sns.stripplot(data=df, x="Method",
              y="Recall", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1, ax=axs[1, 0],)
    sns.boxplot(data=df, y="Recall",
                x="Method", hue="Method",
                hue_order=methods, showfliers=False, ax=axs[1, 0])


    #size1 = df.shape[1]
    #df['Precision'] = df['Precision'] + np.random.random()


    sns.scatterplot(data=df, x="Precision", y="Recall", hue="Method")

    #handles, labels = plt.gca().get_legend_handles_labels()
    #plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
    #axs.legend(handles[0:len(methods)], labels[0:len(methods)])
    #axs[0, 0].gca().legend(handles[0:len(methods)], labels[0:len(methods)])

    #axs[0, 1].legend(labels=[])
    #axs[1, 0].legend(labels=[])

    axs[0, 1].get_legend().remove()
    axs[1, 0].get_legend().remove()
    axs[1, 1].get_legend().remove()

    #axs.set_xlabel('common xlabel')
    #axs.set_ylabel('common ylabel')
    #axs[0, 1].ylabel('')
    #plt.setp(axs[-1, :], xlabel='x axis label')

    #plt.setp(axs[-1, :], xlabel='')
    plt.setp(axs[:, :], xlabel='')
    plt.setp(axs[:, :], ylabel='')
    plt.show()


    quit()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, y)
    axs[0, 0].set_title('Axis [0, 0]')
    axs[0, 1].plot(x, y, 'tab:orange')
    axs[0, 1].set_title('Axis [0, 1]')
    axs[1, 0].plot(x, -y, 'tab:green')
    axs[1, 0].set_title('Axis [1, 0]')
    axs[1, 1].plot(x, -y, 'tab:red')
    axs[1, 1].set_title('Axis [1, 1]')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()


#plotDoublePathway()
#quit()


def plotDoublePathway2():

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from matplotlib import gridspec

    x1 = np.load('./plotResult/cloMuPathway.npy')
    x2 = np.load('./plotResult/recapPathway.npy')
    x3 = np.load('./plotResult/revolverPathway.npy')


    x = np.concatenate((x1, x2, x3), axis=0).astype(float)

    precision = x[:, 3] / (x[:, 3] + x[:, 4])

    #plt.hist(precision)
    #plt.show()
    #quit()
    recall = x[:, 3] / (x[:, 3] + x[:, 5])

    x[:, 0] = precision
    x[:, 1] = recall
    x[:x1.shape[0], 2] = 0
    x[x1.shape[0]:x1.shape[0] + x2.shape[0], 2] = 1
    x[x1.shape[0] + x2.shape[0]:, 2] = 2
    x = x[:, :3]

    df_combine = pd.DataFrame(x, columns = ['precision','recall','method'])

    df_combine['method'][df_combine['method'] == 0] = 'CloMu'
    df_combine['method'][df_combine['method'] == 1] = 'RECAP'
    df_combine['method'][df_combine['method'] == 2] = 'REVOLVER'






    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(6, 6)

    axes = []
    ax = fig.add_subplot(gs[3:, 3:])
    axes.append(ax)
    sns.kdeplot(
        data=df_combine,
        x="precision",
        y="recall",
        hue="method",
        fill=True,
        levels=7,
        alpha=0.5,
        cut=0,
        legend=False,
        ax=ax,
    )
    df_combine_jitter = df_combine.copy()
    np.random.seed(0)
    df_combine_jitter['recall'] = df_combine_jitter['recall'] + 0.005*np.random.randn(len(df_combine_jitter))
    df_combine_jitter['precision'] = df_combine_jitter['precision'] + 0.005*np.random.randn(len(df_combine_jitter))
    sns.scatterplot(
        data=df_combine_jitter,
        x="precision",
        y="recall",
        hue="method",
        style="method",
        alpha=0.4,
        s=50,
        ax=ax,
    )
    ax.set_xlim((-.01,1.02))
    ax.set_ylim((-.01,1.02))
    ax.set_ylabel("Recall")
    ax.set_xlabel('Precision')
    # ax.set_xticks([])
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position("top")
    ax.legend(title=None, loc="lower right")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    # ax.set_aspect('equal', adjustable='box')

    ax = fig.add_subplot(gs[:3, 3:])
    axes.append(ax)
    sns.boxplot(y="method", x="precision", data=df_combine, showfliers=False, orient="h", ax=ax)
    sns.stripplot(y="method", x="precision", data=df_combine, jitter=0.3, alpha=.1,
                  linewidth=0.5, edgecolor="gray", ax=ax, palette='dark')
    ax.set_xlabel("Precision")
    ax.set_xlim((-.01,1.02))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.yaxis.tick_right()
    ax.set_yticklabels([t._text for t in ax.yaxis.get_ticklabels()], rotation=-90, va="center")
    ax.set_ylabel(None)
    ax.yaxis.set_label_position("right")

    ax = fig.add_subplot(gs[3:, :3])
    axes.append(ax)
    sns.boxplot(x="method", y="recall", data=df_combine, showfliers=False, ax=ax)
    sns.stripplot(x="method", y="recall", data=df_combine, jitter=0.3, alpha=.1, linewidth=0.5,
                  edgecolor="gray", ax=ax, palette='dark')
    # ax.set_xticks([])
    ax.set_ylabel("Recall")
    ax.set_ylim((-.01,1.02))
    ax.set_xlabel(None)

    for ax in axes:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)
    plt.tight_layout()

    plt.show()





def testMHN():


    def calculateErrors(theta_true, pred_now):


        baselineRate = theta_true[np.arange(M), np.arange(M)]
        #argTop = np.argsort(-1 * baselineRate)[:(baselineRate.shape[0] // 2)]
        argTop = np.arange(M)


        mask2 = mask1[argTop][:, argTop]


        #choiceShow = 5

        pred_now = pred_now[argTop][:, argTop]
        #pred_now = pred_now.T
        pred_now = pred_now[mask2 == 1]
        #pred_now = pred_now.reshape((M*M,))

        pred_now[pred_now> 0.01] = 1
        pred_now[pred_now<-0.01] = -1
        pred_now[np.abs(pred_now) < 0.02] = 0


        theta_true = theta_true[argTop][:, argTop]

        theta_true = theta_true[mask2 == 1]
        #theta_true = theta_true.reshape((M*M,))
        theta_true[theta_true> 0.01] = 1
        theta_true[theta_true<-0.01] = -1
        theta_true[np.abs(theta_true) < 0.02] = 0

        #figure, axis = plt.subplots(3)
        #axis[0].imshow(theta_true)
        #axis[1].imshow(pred_noSS[0].T)
        #axis[2].imshow(pred_now)
        #plt.show()


        FalseArg = np.argwhere(  (pred_now - theta_true) != 0 )[:, 0]
        TrueArg = np.argwhere(  (pred_now - theta_true) == 0 )[:, 0]

        #print (FalseArg.shape)
        #print (TrueArg.shape)
        #quit()

        TruePosArg = TrueArg[theta_true[TrueArg] != 0]
        TrueNegArg = TrueArg[theta_true[TrueArg] == 0]

        truePosNum = TruePosArg.shape[0]
        trueNegNum = TrueNegArg.shape[0]

        falseNegNum = np.argwhere( np.logical_and(  pred_now == 0, theta_true != 0   )).shape[0]

        falsePosNum = FalseArg[pred_now[FalseArg] != 0].shape[0]

        #print (truePosNum, falsePosNum, trueNegNum, falseNegNum)
        #quit()

        return (truePosNum, falsePosNum, trueNegNum, falseNegNum)







    #dataName = './treeMHN/data/MHNtree_np/' + str(a) #+ '.npy'

    #trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./treeMHN/data/models/' + str(a) + '.pt', baselineSave='./treeMHN/data/models/baseline_' +  str(a) + '.npy',
    #                    adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.0005) #0.00005 #0.0005 #too small 0.00001


    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib.pyplot as plt

    #folderName = 'n15_N300_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1'
    #folderName = 'n15_N100_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1'
    #folderName = 'n15_N300_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1'
    folderName = 'n20_N300_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1'

    #folderName = 'n20_N100_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1'

    nameStart = folderName.split('s0')[0][:-1]

    #Nsim = 100
    #Nsim = 46
    #Nsim = 4# 10
    Nsim = 20

    modelNum = 12

    baseline = np.zeros((Nsim, 8, 4))

    baseline_acc = np.zeros((Nsim, 8, 2))

    cutOffList = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
    #cutOffList = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    #cutOffList = [5, 10, 15, 20, 25, 30]

    our_acc = np.zeros((Nsim, len(cutOffList), 2))


    for a in range(0, Nsim):

        #M1 = 15

        M = int(folderName.split('_')[0][1:])

        dataName = './treeMHN/treeMHNdata/np/' + folderName + '/trees_' + str(a) + '.npy'
        dataTree = np.load(dataName, allow_pickle=True)
        mutUsed = []
        for c0 in range(len(dataTree)):
            for c1 in range(len(dataTree[c0])):
                for c2 in range(len(dataTree[c0][c1])):
                    for c3 in range(len(dataTree[c0][c1][c2])):
                        str1 = dataTree[c0][c1][c2][c3]
                        if '_' in str1:
                            str1 = str1.split('_')[0]
                            str1 = int(str1)
                            if not str1 in mutUsed:
                                mutUsed.append(str1)
        mutUsed = np.sort(np.array(mutUsed)) - 1
        #print (mutUsed)
        #quit()


        theta = np.loadtxt('./treeMHN/treeMHNdata/CSV/' + folderName + '/theta_' + str(a) + '.csv', delimiter=",", dtype=str)
        theta = theta[1:].astype(float)

        #print (a)
        #plt.hist(theta.flatten(), bins=100)
        #plt.show()


        #name1 = "./treeMHN/treeMHNdata/CSV/" + folderName + "/predNo_" + str(a) + ".csv"
        name1 = "./treeMHN/treeMHNdata/CSV/" + folderName + "/predSS_" + str(a) + ".csv"
        pred_noSS = np.loadtxt(name1, delimiter=",", dtype=str)
        #N = int(pred_noSS.shape[1] ** 0.5)
        pred_noSS = pred_noSS.reshape((pred_noSS.shape[0], M, M))
        pred_noSS = pred_noSS[1:]
        pred_noSS = pred_noSS.astype(float)

        mask1 = np.ones((M, M))
        mask1[np.arange(M), np.arange(M)] = 0



        if True:
            model = torch.load('./treeMHN/treeMHNdata/models/' + folderName + '/' + str(a) + '_' + str(modelNum) + '.pt')

            lastSize = 0
            for param in model.parameters():
                lastSize = param.shape[0]
                #print (param.shape)
            M1 = lastSize

            #print (M)
            #quit()
            print (lastSize)
            #quit()
            #mutationName = np.load('./dataNew/categoryNames_' + modelName + '.npy')
            #mutationName = np.arange(10).astype(str)
            mutationName = (np.arange(M1)+1).astype(str)

            X_zero = torch.zeros((M1, M1))
            predZero, _ = model(X_zero)

            #This creates a matrix representing all of the clones with only one mutation.
            X = torch.zeros((M1, M1))
            X[np.arange(M1), np.arange(M1)] = 1

            #This gives the predicted probability weights and the predicted latent variables.
            pred, xNP = model(X)

            pred = pred - predZero


            #order1 = [1, 10, 2, 3, 4, 5, 6, 7, 8, 9]
            #order1 = np.array(order1)
            order1 = (np.arange(M1).astype(int) + 1)
            order1 = order1.astype(str)
            order1 = np.sort(order1)
            order1 = order1.astype(int)

            pred_np = pred.data.numpy()

            #This calculates the relative probability of each mutation, for each clone representing a possible initial mutation.
            #The fitness of the clone is irrelevent to this probability.
            prob = torch.softmax(pred, dim=1)
            #This calculates the relative probability of each mutation clone pair. More fit clones will yeild higher probabilities.

            prob_np = prob.data.numpy()

            #This calculates the total probability that a mutation will be added for each clone.
            #This is a measurement of the fitness of each clone.

            #print (pred_np.shape)
            #print (mask1.shape)
            #quit()

            pred_np = pred_np[np.argsort(order1)][:, np.argsort(order1)].T

            #print (M1, M)

            if M1 < M:
                mutNotUsed = np.argwhere(np.isin(np.arange(M), mutUsed) == False)[:, 0]
                mutShuffle = np.concatenate((mutUsed, mutNotUsed))

                #print (mutShuffle)

                #pred_np[:, :] = 1


                #print (M)
                pred_np_new = np.zeros((M, M))
                pred_np_new[:M1, :M1] = pred_np
                pred_np_new[mutShuffle] = np.copy(pred_np_new)
                pred_np_new[:, mutShuffle] = np.copy(pred_np_new)



                pred_np = pred_np_new

                pred_np[np.arange(mask1.shape[0]), np.arange(mask1.shape[0])] = 0

                #print (pred_np_new)
                #quit()




        if True:
            for choiceShow in range(0, pred_noSS.shape[0]):

                pred_now = pred_noSS[choiceShow].T
                theta_true = np.copy(theta)

                (truePosNum, falsePosNum, trueNegNum, falseNegNum) = calculateErrors(theta_true, pred_now)

                baseline[a, choiceShow, 0] = truePosNum
                baseline[a, choiceShow, 1] = falsePosNum
                baseline[a, choiceShow, 2] = trueNegNum
                baseline[a, choiceShow, 3] = falseNegNum

                #print (truePosNum, falsePosNum, trueNegNum, falseNegNum)


                baseline_acc[a, choiceShow, 0] = float(truePosNum) / (float(falsePosNum) + float(truePosNum) + 1e-10)
                baseline_acc[a, choiceShow, 1] = float(truePosNum) / (float(falseNegNum) + float(truePosNum))


                #figure, axis = plt.subplots(2)
                #axis[0].imshow(theta)
                #axis[1].imshow(pred_noSS[choiceShow])
                #axis[2].imshow(pred_np)
                #plt.show()


        #plt.hist(pred_np.flatten(), bins=100)
        #plt.show()

        for cutArg in range(0, len(cutOffList)):




            cutOff = cutOffList[cutArg]

            #predAbsSort = np.sort(np.abs(pred_np.flatten()))
            #cutOff = predAbsSort[-cutOff]

            #predSort = np.sort(pred_np.flatten())
            #cutOff1 = predSort[cutOff+1]
            #cutOff2 = predSort[-cutOff]
            #pred_np = pred_np - (( cutOff1 + cutOff2 ) / 2)
            #cutOff = (cutOff2 - cutOff1) / 2

            #print (cutOff1, cutOff2, cutOff)



            pred_now = np.copy(pred_np)
            pred_now[np.abs(pred_now) < cutOff] = 0.0

            theta_true = np.copy(theta)

            #print (pred_now)
            #quit()


            (truePosNum, falsePosNum, trueNegNum, falseNegNum) = calculateErrors(theta_true, pred_now)

            #print (truePosNum, falsePosNum, trueNegNum, falseNegNum)

            #baseline[a, choiceShow, 0] = truePosNum
            #baseline[a, choiceShow, 1] = falsePosNum
            #baseline[a, choiceShow, 2] = trueNegNum
            #baseline[a, choiceShow, 3] = falseNegNum


            our_acc[a, cutArg, 0] = float(truePosNum) / (float(falsePosNum) + float(truePosNum) + 1e-10)
            our_acc[a, cutArg, 1] = float(truePosNum) / (float(falseNegNum) + float(truePosNum))



        #quit()

        #accuracy_list1 = np.array(accuracy_list1)

        #plt.plot(accuracy_list1[:, 1], accuracy_list1[:, 0])
        #plt.show()
        #quit()

            #theta




        #theta[np.arange(theta.shape[0]), np.arange(theta.shape[0])] = 0
        #figure, axis = plt.subplots(3)
        #axis[0].imshow(theta)
        #axis[1].imshow(pred_noSS)
        #axis[2].imshow(pred_np)
        #plt.show()


    np.save('./treeMHN/plot/' + nameStart + '_baseline_acc.npy', baseline_acc)
    np.save('./treeMHN/plot/' + nameStart + '_our_acc_' + str(modelNum) + '.npy', our_acc)



    baseline = np.sum(baseline, axis=0)

    #precision = baseline[:, 0] / (baseline[:, 0] + baseline[:, 1])
    #recall = baseline[:, 0] / (baseline[:, 0] + baseline[:, 3])

    #print (baseline)

    #print (our_acc[1])
    #quit()


    #quit()

    #plt.plot(recall, precision)
    #plt.show()

    #9

    #plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
    #for a in range(10):
    #    print (a)
    #    plt.plot(baseline_acc[a, :, 1], baseline_acc[a, :, 0])
    #    plt.plot(our_acc[a, :, 1], our_acc[a, :, 0])
    #    plt.xlabel("recall")
    #    plt.ylabel("precision")
    #    plt.show()


    #print (np.mean(our_acc, axis=0))

    #plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
    #plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    #plt.xlabel("recall")
    #plt.ylabel("precision")
    #plt.show()


    quit()

    if True:

        #pred_np[np.abs(pred_np) < 1.05] = 0
        pred_np[np.abs(pred_np) < 1.4] = 0
        #pred_np[np.abs(pred_np) < 1] = 0
        #pred_np[np.abs(pred_np) < 2.05] = 0

        #This is a plot of the causal relationship between all of the interesting mutations,
        #with the names of the mutations labeled.
        fig, ax = plt.subplots(1,1)

        plt.imshow(pred_np[argsInteresting][:, argsInteresting].T, cmap='bwr_r')
        #img = ax.imshow(prob_np_inter)
        img = ax.imshow(pred_np[argsInteresting][:, argsInteresting].T, cmap='bwr_r') #TODO UNDO Jul 25 2022

        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName[argsInteresting])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName[argsInteresting])


        plt.xticks(rotation = 90)
        #plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()




#testMHN()
#quit()


def plotMHN():

    import matplotlib.pyplot as plt


    #'''
    baseline_acc_10 = np.load('./treeMHN/plot/n10_N300_baseline_acc.npy')
    linear_acc_10 = np.load('./treeMHN/plot/n10_N300_our_acc_11.npy')[:, :-2, :]
    neural_acc_10 = np.load('./treeMHN/plot/n10_N300_our_acc_12.npy')[:, :-1, :]

    baseline_acc_15 = np.load('./treeMHN/plot/n15_N300_baseline_acc.npy')
    linear_acc_15 = np.load('./treeMHN/plot/n15_N300_our_acc_11.npy')[:, :-2, :]
    neural_acc_15 = np.load('./treeMHN/plot/n15_N300_our_acc_12.npy')[:, :-2, :]

    baseline_acc_20 = np.load('./treeMHN/plot/n20_N300_baseline_acc.npy')
    linear_acc_20 = np.load('./treeMHN/plot/n20_N300_our_acc_11.npy')[:, :-3, :]
    neural_acc_20 = np.load('./treeMHN/plot/n20_N300_our_acc_12.npy')[:, :-2, :]

    np.save('./sending/n10_TreeMHN.npy', baseline_acc_10)
    np.save('./sending/n15_TreeMHN.npy', baseline_acc_15)
    np.save('./sending/n20_TreeMHN.npy', baseline_acc_20)
    np.save('./sending/n10_LinearCloMu.npy', linear_acc_10)
    np.save('./sending/n15_LinearCloMu.npy', linear_acc_15)
    np.save('./sending/n20_LinearCloMu.npy', linear_acc_20)
    np.save('./sending/n10_NeuralCloMu.npy', neural_acc_10)
    np.save('./sending/n15_NeuralCloMu.npy', neural_acc_15)
    np.save('./sending/n20_NeuralCloMu.npy', neural_acc_20)
    quit()



    plt.plot(np.mean(neural_acc[:, :, 1], axis=0), np.mean(neural_acc[:, :, 0], axis=0))
    plt.plot(np.mean(linear_acc[:, :, 1], axis=0), np.mean(linear_acc[:, :, 0], axis=0))
    plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))

    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    plt.legend(['CloMu', 'Linear CloMu', 'TreeMHN'])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig('./images/TreeMHN_data_' + str(n))
    plt.show()
    #'''


    '''
    n = 15
    if n == 10:
        baseline_acc = np.load('./treeMHN/plot/n10_N300_baseline_acc.npy')
        linear_acc = np.load('./treeMHN/plot/n10_N300_our_acc_11.npy')[:, :-2, :]
        neural_acc = np.load('./treeMHN/plot/n10_N300_our_acc_12.npy')[:, :-1, :]

    elif n == 15:
        baseline_acc = np.load('./treeMHN/plot/n15_N300_baseline_acc.npy')
        linear_acc = np.load('./treeMHN/plot/n15_N300_our_acc_11.npy')[:, :-2, :]
        neural_acc = np.load('./treeMHN/plot/n15_N300_our_acc_12.npy')[:, :-2, :]

    elif n == 20:
        baseline_acc = np.load('./treeMHN/plot/n20_N300_baseline_acc.npy')
        linear_acc = np.load('./treeMHN/plot/n20_N300_our_acc_11.npy')[:, :-3, :]
        neural_acc = np.load('./treeMHN/plot/n20_N300_our_acc_12.npy')[:, :-2, :]



    plt.plot(np.mean(neural_acc[:, :, 1], axis=0), np.mean(neural_acc[:, :, 0], axis=0))
    plt.plot(np.mean(linear_acc[:, :, 1], axis=0), np.mean(linear_acc[:, :, 0], axis=0))
    plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))

    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    plt.legend(['CloMu', 'Linear CloMu', 'TreeMHN'])
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.savefig('./images/TreeMHN_data_' + str(n))
    plt.show()
    '''

    True

#plotMHN()
#quit()



def plotCausalSimulation():

    def doPrecisionRecall(x):

        x = x.astype(float)
        x[:, 2] = x[:, 0] / (x[:, 0] + x[:, 2])
        x[:, 3] = x[:, 0] / (x[:, 0] + x[:, 3])

        return x


    def makeLabel(lengthList, Labels):


        label1 = np.zeros( int(np.sum(np.array(lengthList))), dtype=int )

        posList = np.cumsum(np.array(lengthList))
        posList = np.concatenate(( np.zeros(1, dtype=int), posList ))

        for a in range(len(lengthList)):
            label1[posList[a]:posList[a+1]] = Labels[a]

        return label1



    import matplotlib as mpl
    import matplotlib.pyplot as plt

    x1 = np.load('./plotResult/cloMuCausal.npy')
    x2 = np.load('./plotResult/recapCausal.npy')
    x3 = np.load('./plotResult/revolverCausal.npy')
    x4 = np.load('./plotResult/geneAccordCausal.npy')


    x1 = doPrecisionRecall(x1)
    x2 = doPrecisionRecall(x2)
    x3 = doPrecisionRecall(x3)
    x4 = doPrecisionRecall(x4)

    x5 = np.zeros(x1.shape)
    x5[:, 2:] = 1

    #print (np.median(x1[:, 2]))
    #print (np.median(x4[:, 3]))
    #print (np.median(x4[:, 2]))
    #quit()

    np.save('./sending/AllMethodsCausal/CloMu.npy', x1[:, 2:])
    np.save('./sending/AllMethodsCausal/RECAP.npy', x2[:, 2:])
    np.save('./sending/AllMethodsCausal/REVOLVER.npy', x3[:, 2:])
    np.save('./sending/AllMethodsCausal/GeneAccord.npy', x4[:, 2:])
    np.save('./sending/AllMethodsCausal/TreeMHN.npy', x5[:, 2:])


    quit()


    if False:
        rand1 = (np.random.random(x1.shape) - 0.5) / 50
        rand2 = (np.random.random(x1.shape) - 0.5) / 50
        rand3 = (np.random.random(x1.shape) - 0.5) / 50
        rand4 = (np.random.random(x1.shape) - 0.5) / 50
        #x1, x2, x3, x4 = x1 + rand1, x2 + rand2, x3 + rand3, x4 + rand4



        plt.scatter(x1[:, 2], x1[:, 3])
        plt.scatter(x2[:, 2], x2[:, 3])
        plt.scatter(x3[:, 2], x3[:, 3])
        plt.scatter(x4[:, 2], x4[:, 3])
        plt.legend(['CloMu', 'RECAP', 'Revolver', 'GeneAccord'])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        #plt.xlabel('False Positives')
        #plt.ylabel('False Negatives')
        plt.show()


    #print (np.sum(x1, axis=0))
    #print (np.sum(x2, axis=0))
    #print (np.sum(x3, axis=0))
    #print (np.sum(x4, axis=0))
    #quit()

    #x = np.concatenate((x1, x2, x3, x4), axis=0)
    x = np.concatenate((x1, x2, x3, x4, x5), axis=0)

    lengthList = [x1.shape[0], x2.shape[0], x3.shape[0], x4.shape[0], x5.shape[0]]
    label1 = makeLabel(lengthList, [0, 1, 2, 3, 4])

    x = x[:, 1:]

    x[:, 0] = label1

    #x[:x1.shape[0], 0] = 0
    #x[x1.shape[0]:x1.shape[0] + x2.shape[0], 0] = 1
    #x[x1.shape[0] + x2.shape[0]:x1.shape[0] + x2.shape[0]+x3.shape[0], 0] = 2
    #x[x1.shape[0] + x2.shape[0]+x3.shape[0]:, 0] = 3

    x_0 = np.copy(x)[:, np.array([0, 2, 1])]
    x_1 = np.copy(x)
    x_0[:, 1] = 0
    x_1[:, 1] = 1
    x = np.concatenate((x_0, x_1), axis=0)

    x = x[:, np.array([1 , 2, 0])]



    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(x, columns = ['Number of Pathways','Accuracy','Method'])

    df['Method'][df['Method'] == 0] = 'CloMu'
    df['Method'][df['Method'] == 1] = 'RECAP'
    df['Method'][df['Method'] == 2] = 'REVOLVER'
    df['Method'][df['Method'] == 3] = 'geneAccord'
    df['Method'][df['Method'] == 4] = 'TreeMHN'


    df['Number of Pathways'][df['Number of Pathways'] == 0] = 'Precision'
    df['Number of Pathways'][df['Number of Pathways'] == 1] = 'Recall'
    #df['Number of Pathways'][df['Number of Pathways'] == 0] = 'False Positives'
    #df['Number of Pathways'][df['Number of Pathways'] == 1] = 'False Negatives'

    #methods = [0, 1]
    methods = ['CloMu', 'TreeMHN', 'RECAP', 'REVOLVER', 'geneAccord']

    sns.stripplot(data=df, x="Number of Pathways",
              y="Accuracy", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1,)
    sns.boxplot(data=df, x="Number of Pathways",
                y="Accuracy", hue="Method",
                hue_order=methods, showfliers=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
    plt.xlabel('')
    plt.show()

def plotSelectionSimulation():


    x1 = np.load('./plotData/cloMuSelect.npy')
    x2 = np.load('./plotData/recapSelect.npy')
    x3 = np.load('./plotData/revolverSelect.npy')

    #x1 = np.load('./plotResult/cloMuSelectPath.npy')
    #x2 = np.load('./plotResult/recapSelect.npy')
    #x3 = np.load('./plotResult/revolverSelect.npy')



    x_0 = np.concatenate((x1, x2, x3), axis=0)
    x = np.zeros((x_0.shape[0], 3))
    x[:, 0] = x_0
    x[:x1.shape[0], 1] = 0
    x[x1.shape[0]:x1.shape[0] + x2.shape[0], 1] = 1
    x[x1.shape[0] + x2.shape[0]:, 1] = 2


    x = x[:, np.array([2 , 0, 1])]


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame(x, columns = ['Number of Pathways','Tree Selection Accuracy','Method'])

    df['Method'][df['Method'] == 0] = 'CloMu'
    df['Method'][df['Method'] == 1] = 'RECAP'
    df['Method'][df['Method'] == 2] = 'REVOLVER'

    df['Number of Pathways'][df['Number of Pathways'] == 0] = ''

    #methods = [0, 1]
    methods = ['CloMu', 'RECAP', 'REVOLVER']

    my_colors = {'CloMu': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 'RECAP': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 'REVOLVER': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392)}
    #print (sns.color_palette("tab10"))
    #quit()

    sns.stripplot(data=df, x="Number of Pathways",
              y="Tree Selection Accuracy", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1, palette=my_colors)
    sns.boxplot(data=df, x="Number of Pathways",
                y="Tree Selection Accuracy", hue="Method",
                hue_order=methods, showfliers=False, palette=my_colors)
    handles, labels = plt.gca().get_legend_handles_labels()
    #plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
    plt.gca().legend([], frameon=False)
    #plt.gca().legend.remove()
    plt.xlabel('')
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.show()

def trainSimulationModels(name):
    import os

    #This function trains models on all of the RECAP data sets.

    #This finds the names of the files to be loaded in.
    arr = os.listdir('./data/' + name + '/simulations_input')

    for name2 in arr:

        name2 = name2[:-4]

        fileIn = './dataNew/p_' + name + '/simulations_input/' + name2 + '.txt.npy'
        fileSave = './dataNew/p_' + name + '/models/' + name2
        baselineSave = './dataNew/p_' + name + '/baselines/' + name2

        #Getting the number of mutations per patient from the file name.
        maxM = int(name.split('_')[1][1:])
        #maxM = 5
        #maxM = 7
        #maxM = 12
        #Processing the RECAP data into a form usable by my algorthm.
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, fileIn)

        #Defining the training set as the entire set.
        #This is because this data is all for training, and only the ground truth trees are left out for testing.
        N2 = int(np.max(sampleInverse)+1)
        trainSet = np.random.permutation(N2)

        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=fileSave, baselineSave=baselineSave, adjustProbability=True, trainSet=trainSet)

def evaluateSimulations(name):

    #This function evaluates the model trained on the RECAP data on the
    #RECAP task of predicting true trees from the set of possible trees.


    def makeTreeChoice(treeProbs, index, sampleInverse):

        #This calculates the highest liklyhood tree for each patient according to the model's probabilities.
        #It also calculates which trees are the only possible tree for some patient.


        treeChoice = np.zeros(index.shape[0]).astype(int)
        isRequired = np.zeros(index.shape[0]).astype(int)
        probabilityChangeLog = np.zeros(index.shape[0])

        for a in range(0, index.shape[0]):
            #start1 and end1 indicate the start and end of the section of treeProbs which are associated with
            #possible trees for this patient.
            start1 = index[a]
            if a == (index.shape[0] - 1):
                end1 = sampleInverse.shape[0]
            else:
                end1 = index[a+1]

            #argsLocal are the trees associated with this patient.
            argsLocal = np.arange(end1 - start1) + start1
            argsLocal = argsLocal.astype(int)
            #localTreeProb is the probability of each tree for this patient.
            localTreeProb = treeProbs[argsLocal]
            #treeMax is the index of the highest probability tree for this patient.
            treeMax = np.argmax(localTreeProb) + start1
            treeChoice[a] = treeMax

            if (np.sum(localTreeProb) - np.max(localTreeProb)) <= 0.05:
                #This is not currently used in this version of the code.
                #However, what it does is recording which trees represent the vast majority of the probability
                #in the remaining trees for some patient.
                isRequired[a] = 2
            if (np.sum(localTreeProb) - np.max(localTreeProb)) == 0:
                #This indicates that this tree is the only possible tree for this patient.
                #Therefore, this tree is required and can not be removed.
                isRequired[a] = 1
                probabilityChangeLog[a] = 1000
            else:

                sortedProb = np.sort(localTreeProb)
                #This determines how much lower probability the next best tree is for this patient.
                #This tells us how detrimental to the fit for this patient it is to remove this tree.
                #For example, if some patient fits two trees equally well, then removing either of them individually is
                #not very detrimental.
                probabilityChangeLog[a] = np.log( sortedProb[-1] + 1e-5 ) - np.log( sortedProb[-2] + 1e-5 )


        return treeChoice, isRequired, probabilityChangeLog

    def treeToString(choiceTrees):

        #This function converts trees to unique strings, which is useful for other tasks.
        #For instance, it allows to apply np.unique capabilities to trees.
        #The format used is also useful for going back to tree data.

        choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]
        choiceString = np.zeros(choiceNums.shape[0]).astype(str)
        for a in range(0, choiceNums.shape[0]):
            #choiceString[a] = str(choiceNums[a, 0]) + ':' + str(choiceNums[a, 1]) + ':' + str(choiceNums[a, 2]) + ':' + str(choiceNums[a, 3])
            choiceString1 = str(choiceNums[a, 0])
            for b in range(1, choiceNums.shape[1]):
                choiceString1 = choiceString1 + ':' + str(choiceNums[a, b])
            choiceString[a] = choiceString1

        return choiceString



    import os

    allSaveVals = []

    #doCluster means methods developed into the paper to use a small number of unique trees in the data set is used.
    #if doCluster is false, the model will then instead naively independently choose the best tree for each patient.
    doCluster = True


    clusterNums = [[], []]

    accuracies = []

    #This loads in the names of the data sets.
    arr = os.listdir('./dataNew/' + name + '/simulations_input')

    #This iterates through the RECAP data sets.
    for name2 in arr:

        print (name2)

        name2 = name2[:-4]


        if doCluster:
            trueCluster = int(name2.split('k')[1].split('_')[0])
            clusterNums[0].append(trueCluster)
            Ncluster = 1


        fileIn = './dataNew/p_' + name + '/simulations_input/' + name2 + '.txt.npy'
        fileSave = './dataNew/p_' + name + '/models/' + name2
        baselineSave = './dataNew/p_' + name + '/baselines/' + name2

        #This determines the number of mutations per patient based on the file name.
        if name == 'M5_m5':
            maxM = 5
        elif name == 'M12_m7':
            maxM = 7
        else:
            maxM = 12

        #This processes the data
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, fileIn)

        #This gives the index of the first tree for each patient.
        _, index = np.unique(sampleInverse, return_index=True)

        #This gives the probability of each tree according to the model.
        #if doCluster is true, this does do duplicate computations.
        treeProbs = np.load(baselineSave + '.npy')
        treeChoice = np.zeros(index.shape[0]).astype(int)

        #This code finds the maximum probability tree for each patient independently.
        for a in range(0, index.shape[0]):
            start1 = index[a]
            if a == (index.shape[0] - 1):
                end1 = sampleInverse.shape[0]
            else:
                end1 = index[a+1]
            argsLocal = np.arange(end1 - start1) + start1
            argsLocal = argsLocal.astype(int)
            localTreeProb = treeProbs[argsLocal]
            treeMax = np.argmax(localTreeProb) + start1
            treeChoice[a] = treeMax


        if doCluster:

            #This converts trees to unique strings
            treeString = treeToString(newTrees)
            treeProbs2 = np.copy(treeProbs)


            #This code finds the maximum probability tree for each patient independently.
            treeChoice, isRequired, probabilityChangeLog = makeTreeChoice(treeProbs2, index, sampleInverse)
            choiceTrees = newTrees[treeChoice]
            choiceString = treeToString(choiceTrees)
            #This updates the probability to zero for all trees not used.
            #Therefore, in future iterations, only the trees that were used in the first iteration can be used.
            treeProbs2[np.isin(treeString, choiceString) == False] = 0.0


            #This continous the iterations until an equilibrium is hit.
            #Note: iter = 1000 is set within the loop when an equilibrium condition is met.
            currentCluster = 100000
            iter = 0
            while (currentCluster > Ncluster) and (iter < 1000):

                #This determines the tree choices from the current probabilities.
                treeChoice, isRequired, probabilityChangeLog = makeTreeChoice(treeProbs2, index, sampleInverse)
                choiceTrees = newTrees[treeChoice]


                choiceString = treeToString(choiceTrees)

                probabilityChangeLog2 = np.zeros(probabilityChangeLog.shape)
                choiceStringUnique = np.unique(choiceString)
                for b in range(0, choiceStringUnique.shape[0]):
                    argsInString = np.argwhere(choiceString == choiceStringUnique[b])[:, 0]
                    #print (argsInString)
                    probabilityChangeLog2[argsInString] = np.sum(probabilityChangeLog[argsInString])


                #This finds the set of trees which are not required to be true for some patient.
                #A tree is required to be tree if it is the only remaining possible tree for some patient.
                argsDeletable = np.argwhere(np.isin(choiceString, choiceString[isRequired == 1]) == False)[:, 0]
                choiceString_deletable = choiceString[argsDeletable]
                probabilityChangeLog2_del = probabilityChangeLog2[argsDeletable]


                #If every tree is required, an equilibrium is reached and the iterations stop.
                if choiceString_deletable.shape[0] == 0:
                    iter = 1000
                else:


                    if np.min(probabilityChangeLog2_del) < 40:
                        #If some tree(s) is(are) practically deletable, then it finds the true which can be deleted while doing minimal damage
                        #to the quality of the fit. In other words, which tree can be removed as a possibility while still having a fairly likly tree
                        #being selected for every patient.
                        deleteString = choiceString_deletable[np.argmin(probabilityChangeLog2_del)]
                        #The probability of this deleted tree is then set to zero, so it is removed as a possibility.
                        treeProbs2[treeString == deleteString] = 0
                    else:
                        #If deleting any of the deletable trees would yeild a massive decrease in the probability of the fit,
                        #then there are no trees which could be practically deleted. Therefore, equilibrium is met and the
                        #iterations are done.
                        iter = 1000

                    iter += 1

                #This is the current number of unique trees predicted.
                currentCluster = np.unique(choiceString).shape[0]

            #This records the predicted number of clusters for this data set.
            clusterNums[1].append(currentCluster)


        #These are the predicted trees for all the patients.
        choiceTrees = newTrees[treeChoice]

        #This is the file of the tree tree solutions.
        solutionFile = './dataNew/p_' + name + '/simulations_solution/' + name2 + '.txt.npy'

        #The true tree solutions are then loaded in and processed.
        solutionTrees2, sampleInverse_, mutationCategory, treeLength_, uniqueMutation_, M_ = processTreeData(maxM, solutionFile)

        #There is a blank edge at the end of this format which is now removed.
        solutionTrees2 = solutionTrees2[:, :-1]
        choiceTrees = choiceTrees[:, :-1]


        #This converts the edges in the predicted trees and edges in the true trees to a numerical representation.
        choiceNums = (  choiceTrees[:, :, 0] * (maxM + 2) ) + choiceTrees[:, :, 1]
        solutionNums = (  solutionTrees2[:, :, 0] * (maxM + 2) ) + solutionTrees2[:, :, 1]

        #Strings representing the true trees and solution trees are also found.
        choiceString = treeToString(choiceTrees)
        solutionString = treeToString(solutionTrees2)


        #The edges are sorted so that two identical trees can not accidentally have a different
        #edge order.
        choiceNums = np.sort(choiceNums, axis=1)
        solutionNums = np.sort(solutionNums, axis=1)

        #The difference between the true trees and predicted trees is calculated.
        diff = np.sum(np.abs(choiceNums - solutionNums), axis=1).astype(int)
        argsValid = np.argwhere(diff == 0)[:, 0]
        argsBad = np.argwhere(diff != 0)[:, 0]

        #The accuracy is calculated, printed, and recorded.
        accuracy = argsValid.shape[0] / diff.shape[0]

        print ('accuracy: ' + str(accuracy))
        accuracies.append(accuracy)



        #The true trees and predicted trees are saved as string representations, incase other comparisons need to be made.
        valSave = np.zeros(202).astype(str)
        valSave[0] = name
        valSave[1] = name2
        valSave[2:2 + len(solutionString)] = solutionString
        valSave[102:102 + len(choiceString)] = choiceString

        allSaveVals.append(valSave)

        print (len(allSaveVals))
        print (name)
        np.save('./dataNew/allSave_' + name + '.npy', allSaveVals)

def savePredictedTrees(dataName):

    #This function saves the predicted trees and their probabilities for each patient in a data set.


    #This loads in the data
    if dataName == 'AML':
        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/manualCancer.npy')
        baseline = np.load('./Models/baseline_manual.npy')
        #treeData = np.load('./dataNew/manualCancer.npy', allow_pickle=True)
        mutationName = np.load('./data/categoryNames.npy')[:-2]
    elif dataName == 'breast':
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')
        baseline = np.load('./Models/baseline_breast.npy')
        #treeData = np.load('./dataNew/manualCancer.npy', allow_pickle=True)
        mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
    else:
        maxM = np.load('./Models/runInfo_' + dataName + '.npy')[0]
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, dataName)
        baseline = np.load('./Models/baseline_' + dataName + '.npy')
        #treeData = np.load('./dataNew/breastCancer.npy', allow_pickle=True)
        mutationName = np.load('./dataNew/mutationNames_' + dataName + '.npy')



    treeLength = treeLength.astype(int)

    mutationName[mutationName == 'ZZZZZZZZZZ'] = 'Root'



    #_, sampleInverse2 = np.unique(sampleInverse, return_inverse=True)
    _, startIndex = np.unique(sampleInverse, return_index=True)
    startIndex = np.concatenate((startIndex, np.array( [sampleInverse.shape[0]] )))

    trees_and_probs = []

    #This iterates through the patients
    for a in range(len(startIndex)-1):
        args1 = np.arange(startIndex[a+1] - startIndex[a]) + startIndex[a]
        baseline[args1] = baseline[args1] / np.sum(baseline[args1])

        probs1 = baseline[args1]
        trees = newTrees[args1][:, :treeLength[args1[0]]]
        trees = mutationName[trees]

        info1 = [sampleInverse[startIndex[a]], probs1, trees]

        trees_and_probs.append(copy.deepcopy(info1))

    trees_and_probs = np.array(trees_and_probs, dtype=object)
    np.savez_compressed('./dataNew/predictedTrees_' + dataName + '.npz', trees_and_probs)

def probPredictedTrees(modelName):
    if modelName == 'breast':

        import matplotlib.pyplot as plt

        baseline = np.load('./Models/baseline_breast.npy')
        mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]

        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')

        sizeFull = []
        sizeHalf = [[], [], []]

        #percentCutOff = 0.1

        _, sampleInverse = np.unique(sampleInverse, return_inverse=True)
        _, startIndex = np.unique(sampleInverse, return_index=True)
        startIndex = np.concatenate((startIndex, np.array( [sampleInverse.shape[0]] )))
        for a in range(len(startIndex)-1):
            args1 = np.arange(startIndex[a+1] - startIndex[a]) + startIndex[a]
            baseline[args1] = baseline[args1] / np.sum(baseline[args1])

            sizeFull.append(args1.shape[0])

            cumsum1 = np.cumsum(np.sort(baseline[args1]))
            sizeHalf[0].append(np.argwhere(cumsum1 > 0.1).shape[0])
            sizeHalf[1].append(np.argwhere(cumsum1 > 0.2).shape[0])
            sizeHalf[2].append(np.argwhere(cumsum1 > 0.5).shape[0])

        sizeFull = np.array(sizeFull)
        sizeHalf = np.array(sizeHalf).T

        sizeArgsort = np.argsort(-1 * sizeFull)

        sizeFull = sizeFull[sizeArgsort]
        sizeHalf = sizeHalf[sizeArgsort]

        smallNoise = np.random.random(sizeFull.shape[0])
        smallNoise = (smallNoise - np.mean(smallNoise)) * 0.01

        #plt.plot(sizeFull)
        #plt.plot(sizeHalf)
        plt.scatter(sizeFull + smallNoise, sizeHalf[:, 0] + smallNoise)
        #plt.plot(sizeFull, sizeHalf[:, 1])
        #plt.plot(sizeFull, sizeHalf[:, 2])
        plt.yscale('log')
        plt.xscale('log')
        plt.show()
        quit()



        import matplotlib.pyplot as plt
        plt.plot(baseline[:100])
        plt.plot(sampleInverse[1:101] - sampleInverse[:100])
        plt.show()

        print (sampleInverse.shape)
        print (baseline.shape)
        quit()


#probPredictedTrees('breast')
#quit()

def plotNumbrOfTrees():

    #simName = "Pathway"
    simName = 'Causal'
    #simName = 'I-c'

    if simName == "Pathway":
        Tset = [5]
        N = 30
        maxVal = 50
        saveName = './images/treeSizePathway.pdf'
    elif simName == "Causal":
        Tset = [4, 11, 12]
        N = 20
        maxVal = 30
        saveName = './images/treeSizeCausal.pdf'
    #elif simName == "I-c":
    #    Tset = [4, 11, 12]
    #    N = 20
    #    maxVal = 30
    #    #saveName = './images/.pdf'


    import matplotlib.pyplot as plt
    import os, sys, glob
    #import math
    #import numpy as np
    #import pandas as pd
    #%matplotlib inline
    #%config InlineBackend.figure_format = 'svg'
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')
    mpl.rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.4)


    countList = []

    for T in Tset:
        for a in range(N):

            #sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz')
            sampleInverse = loadnpz('./data/simulations/I-c/T_' + str(12) + '_R_' + str(a) + '_bulkSample.npz')
            _, counts = np.unique(sampleInverse, return_counts=True)

            countList = countList + list(counts)


    plt.hist(countList, bins=50, range=(0, maxVal))
    plt.gcf().tight_layout()
    plt.xlabel("number of possible trees")
    plt.ylabel('number of patients')
    #if
    #plt.savefig(saveName)

    plt.show()
    quit()

#plotNumbrOfTrees()
#quit()


def analyzeModel(modelName):

    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib.pyplot as plt

    if modelName == 'AML':
        model = torch.load('./Models/savedModel_manual.pt')
        mutationName = np.load('./data/categoryNames.npy')[:-2]
        M = 24
        latentMin = 0.1

    elif modelName == 'breast':
        model = torch.load('./Models/savedModel_breast.pt')
        mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
        M = 406
        #M = 365
        latentMin = 0.01

    else:

        model = torch.load('./Models/savedModel_' + dataName + '.pt')

        lastSize = 0
        for param in model.parameters():
            lastSize = param.shape[0]
        M = lastSize
        mutationName = np.load('./dataNew/mutationNames_' + modelName + '.npy')

        #Feel free to change this value.
        latentMin = 0.1





    #This creates a matrix representing all of the clones with only one mutation.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability weights and the predicted latent variables.
    pred, xNP = model(X)

    #This substracts the median from the latent representation, which makes it so the uniteresting mutations have a
    #value of zero.
    for a in range(0, 5):
        xNP[:, a] = xNP[:, a] - np.median(xNP[:, a])

    #This calculates the difference of the value of the latent parameter from the median
    #If latentSize is large, it means that mutation has at least some significant property.
    latentSize = np.max(np.abs(xNP), axis=1)



    #This finds the "interesting" mutations which have at least some significant property.
    argsInteresting = np.argwhere(latentSize > latentMin)[:, 0]
    np.save('./dataNew/interestingMutations_' + modelName + '.npy', argsInteresting)


    if False:
        #This is a optional analysis which groups the mutations based on there properties.
        #It is currently disabled but can be enable if one wants to cluster mutations by their properties.

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=8, random_state=0).fit(xNP[argsInteresting])
        labels = np.array(kmeans.labels_)

        plt.plot(xNP[argsInteresting][np.argsort(labels)])
        plt.plot((labels[np.argsort(labels)] / 4) - 1)
        plt.show()

        argsInteresting = argsInteresting[np.argsort(labels)]

    if True:
        #plt.plot(xNP[argsInteresting][np.argsort(xNP[argsInteresting, 0])])

        #This plots the latent parameters of the mutations
        plt.plot(xNP)
        plt.title("Mutation Properties")
        plt.xlabel("Mutation Number")
        plt.ylabel("Latent Variable Value")

        #This finds the mutations with substantial enough properties
        #that they should be annotated, and annotates them with the mutation name.
        argsHigh = np.argwhere(latentSize > 0.15)[:, 0]

        for i in argsHigh:
            name = mutationName[i]

            delt1 = np.max(xNP) / 100
            max1 = np.max(np.abs(xNP[i])) + (delt1 * 4)
            sign1 = np.sign(xNP[i][np.argmax(np.abs(xNP[i]))] )
            max1 = (max1  * sign1) - (delt1 * 3)

            plt.annotate(name, (i -  (M / 40), max1    ))
            #plt.annotate(name, (i -  (M / 20), np.max(xNP[i]) + (np.max(xNP) / 100)    ))


        #plt.savefig('./images/LatentPlot_1.png')
        plt.show()

    if False:

        #This plots the latent representations, but only for the interesting mutations with substaintial
        #latent representation values.
        #It is currently disabled but can be enabled.
        plt.plot(xNP[argsInteresting])
        plt.title("Mutation Properties")
        plt.xlabel("Mutation Number")
        plt.ylabel("Latent Variable Value")

        argsHigh = np.argwhere(latentSize[argsInteresting] > 0.1)[:, 0]

        for i in argsHigh:
            name = mutationName[argsInteresting[i]]

            delt1 = np.max(xNP) / 100
            max1 = np.max(np.abs(xNP[argsInteresting[i]])) + (delt1 * 4)
            sign1 = np.sign(xNP[argsInteresting[i]][np.argmax(np.abs(xNP[argsInteresting[i]]))] )
            max1 = (max1  * sign1) - (delt1 * 3)

            if i == 0:
                plt.annotate(name, (i, max1     ))
            else:
                plt.annotate(name, (i - 0.5, max1     ))
            #plt.annotate(name, (i - 0.5, np.max(xNP[argsInteresting[i]]) + (np.max(xNP) / 100)    ))


        plt.plot(xNP[np.argsort(np.sum(np.abs(xNP), axis=1) )])
        #plt.savefig('./images/SomeLatentPlot_1.png')
        plt.show()



    pred_np = pred.data.numpy()

    pred2 = pred.reshape((1, -1))

    #This calculates the relative probability of each mutation, for each clone representing a possible initial mutation.
    #The fitness of the clone is irrelevent to this probability.

    if True:
        prob = pred
    else:
        prob = torch.softmax(pred, dim=1)



    #This calculates the relative probability of each mutation clone pair. More fit clones will yeild higher probabilities.
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(prob.shape)

    prob_np = prob.data.numpy()
    prob2_np = prob2.data.numpy()

    #This calculates the total probability that a mutation will be added for each clone.
    #This is a measurement of the fitness of each clone.

    prob2_sum = np.sum(prob2_np, axis=1)


    #This calculates the mutations which have a high enough fitness that they should be annotated
    #with the mutation name in the plot.
    argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.5)[:, 0]

    if True:
        #This plots the relative fitness of all of the mutations in the data set.
        plt.plot(prob2_sum)
        plt.ylabel('Relative Fitness')
        plt.xlabel('Mutation Number')
        for i in argsHigh:
            name = mutationName[i]
            #plt.annotate(name, (i -  (M / 20), prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
            plt.annotate(name, (i , prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
        plt.savefig('./images/fitnessPlot_1.jpg')
        plt.show()

    #This calculates probabilities of new mutations given the  existing mutations, with the
    #mutations restricted to the set of interesting mutations.
    prob_np_inter = prob_np[argsInteresting][:, argsInteresting]

    #This adjusts the probability of each new mutation for the fact that in reality,
    #the already existing mutation can not be selected as the new mutation.
    #This gives more realistic information, but also removes the information of which mutations would tend to cause
    #mutations similar to itself to occur.
    prob_np_adj = np.copy(prob_np)
    prob_np_adj[np.arange(prob_np_adj.shape[0]), np.arange(prob_np_adj.shape[0])] = 0
    for a in range(len(prob_np_adj)):
        prob_np_adj[a] = prob_np_adj[a] / np.sum(prob_np_adj[a])

    prob_np_adj_inter = prob_np_adj[argsInteresting][:, argsInteresting]


    if False:

        #This is a plot of all of the causal relationships.
        #It is unused in this version of the code since it is hard to read when there are many mutations.
        #However, if your data set has very few mutations, feel free to enable it.
        plt.imshow(prob_np)
        #plt.imshow(pred_np)
        plt.ylabel('Existing Mutation')
        plt.xlabel("New Mutation")
        plt.colorbar()
        plt.savefig('./images/fullOccurancePlot_1.jpg')
        plt.show()





    if False:
        #This is a plot of the interesting causal relationship without labels.
        #Use this instead of the labeled one if there are too many interesting mutations in your data set to
        #effectively label.
        plt.imshow(prob_np_adj_inter)
        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()

        #plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()



    if True:
        #This is a plot of the causal relationship between all of the interesting mutations,
        #with the names of the mutations labeled.
        fig, ax = plt.subplots(1,1)

        plt.imshow(prob_np_inter)
        img = ax.imshow(prob_np_inter)
        #img = ax.imshow(pred.data.numpy()[argsInteresting][:, argsInteresting]) #TODO UNDO Jul 25 2022

        plt.xlabel("New Mutation")
        plt.ylabel('Existing Mutation')
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName[argsInteresting])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName[argsInteresting])


        plt.xticks(rotation = 90)
        #plt.savefig('./images/occurancePlot_1.jpg')
        plt.show()




    if False:

        #This is an analysis of which mutations have a positive or negative causal relationship.
        #This is used to quantifiably compare to the results of a different paper.
        #It is disabled by default since there is no reason to re-run the analysis, since it's entirely represented by
        #the numbers already in the paper.
        #Feel free to enable it in order to modify the could to do a similar comparison on other data sets.

        fig, ax = plt.subplots(1,1)

        prob_np_inter_adj = np.copy(prob_np_inter)

        #cutOff = 0.02
        cutOff = 0

        for a in range(prob_np_inter_adj.shape[1]):
            mean1 = np.mean(prob_np_inter_adj[:, a])

            #prob_np_inter_adj[:, a] = prob_np_inter_adj[:, a] - mean1
            prob_np_inter_adj[:, a] = (prob_np_inter_adj[:, a] / mean1)# - 1


        intername = mutationName[argsInteresting]

        #nameCheck = [['NRAS', 'PTPN11'], ['FLT3', 'NRAS'], ['NPM1', 'PTPN11'], ['KRAS', 'NRAS'], ['KRAS', 'PTPN11'], ['FLT3', 'KRAS'], ['FLT3', 'NPM1']]
        nameCheck = [['NRAS', 'PTPN11'], ['FLT3', 'NRAS'], ['NPM1', 'PTPN11'], ['KRAS', 'NRAS'], ['KRAS', 'PTPN11'], ['FLT3', 'KRAS'], ['FLT3', 'NPM1'], ['KRAS', 'NPM1'], ['FLT3', 'PTPN11'], ['NPM1', 'NRAS']]


        for a in range(len(nameCheck)):
            pair = nameCheck[a]

            name1 = np.argwhere(intername == pair[0])[0, 0]
            name2 = np.argwhere(intername == pair[1])[0, 0]

            print (prob_np_inter_adj[name1, name2], prob_np_inter_adj[name2, name1])


        prob_flat = prob_np_inter_adj.reshape((prob_np_inter_adj.size,))
        prob_argsort0 = np.argsort(prob_flat)
        prob_argsort = np.array([prob_argsort0 // prob_np_inter_adj.shape[0], prob_argsort0 % prob_np_inter_adj.shape[0]]).astype(int).T

        print (prob_flat[prob_argsort0][-15:-5])
        print (prob_flat[prob_argsort0][5:15])

        print (intername[prob_argsort[-15:-5, 0]][-1::-1])
        print (intername[prob_argsort[-15:-5, 1]][-1::-1])
        print (intername[prob_argsort[6:16, 0]])
        print (intername[prob_argsort[6:16, 1]])


#analyzeModel('AML')
#quit()



def newAnalyzeModel(modelName):

    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib.pyplot as plt



    import os, sys, glob
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

    sns.set_style('whitegrid')
    mpl.rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.4)

    #plt.gcf().tight_layout()

    #modelName = 'breast'
    modelName = 'AML'

    #modelName = 'temp'


    if modelName == 'AML':
        model = torch.load('./Models/realData/savedModel_AML.pt')
        #model = torch.load('./Models/savedModel_manual_oct12_8pm.pt')
        #model = torch.load('./Models/savedModel_manual_allTrain2.pt')
        #model = torch.load('./Models/savedModel_manual_PartialTrain.pt')
        #mutationName = np.load('./data/realData/AMLmutationNames.npy')[:-2]
        mutationName = np.load('./data/realData/categoryNames.npy')[:-2]

        #print (mutationName)
        #quit()
        M = 22
        #latentMin = 0.1


        #latentMin = 0.05
        latentMin = 0.02
        #latentMin = 0.005

        #print (mutationName.shape)
        #quit()
    elif modelName == 'breast':
        model = torch.load('./Models/realData/savedModel_breast.pt')
        #mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
        mutationName = np.load('./data/realData/breastCancermutationNames.npy')[:-2]
        M = 406
        #M = 365
        #latentMin = 0.01
        latentMin = 0.1
        #latentMin = 0.059


    elif modelName == 'temp':

        model = torch.load('./temp/model.pt')
        mutationName = np.load('./temp/mutationNames.npy')#[:-2]
        M = 22
        latentMin = 0.02

        #print (mutationName)
        #quit()



    else:

        model = torch.load('./Models/savedModel_' + dataName + '.pt')

        lastSize = 0
        for param in model.parameters():
            lastSize = param.shape[0]
        M = lastSize
        mutationName = np.load('./dataNew/mutationNames_' + modelName + '.npy')

        #Feel free to change this value.
        latentMin = 0.1





    #This creates a matrix representing all of the clones with only one mutation.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability weights and the predicted latent variables.
    pred, xNP = model(X)

    X_normal = torch.zeros((1, M))
    pred_normal, _ = model(X_normal)

    #pred_rel = pred.clone()
    #for a in range(pred_rel.shape[1]):
    #    pred_rel[:, a] = pred_rel[:, a] - pred_normal[0, a]
    #pred_rel = torch.softmax(pred_rel, axis=1)
    #pred_rel = pred_rel.data.numpy()
    #pred_rel = np.log(pred_rel)


    prob_normal = torch.softmax(pred_normal, axis=1)[0].data.numpy()
    pred_normal = pred_normal[0].data.numpy()


    #This substracts the median from the latent representation, which makes it so the uniteresting mutations have a
    #value of zero.
    for a in range(0, 5):
        xNP[:, a] = xNP[:, a] - np.median(xNP[:, a])

    #This calculates the difference of the value of the latent parameter from the median
    #If latentSize is large, it means that mutation has at least some significant property.
    latentSize = np.max(np.abs(xNP), axis=1)





    #This finds the "interesting" mutations which have at least some significant property.
    argsInteresting = np.argwhere(latentSize > latentMin)[:, 0]
    #np.save('./dataNew/interestingMutations_' + modelName + '.npy', argsInteresting)


    if False:

        #sns.set_context("notebook", font_scale=1.5)

        #plt.plot(xNP[argsInteresting][np.argsort(xNP[argsInteresting, 0])])

        #This plots the latent parameters of the mutations
        plt.plot(xNP)
        if modelName == 'AML':
            #plt.ylim(-1.6)
            plt.ylim(-1.8)

        #labelsize=20

        #plt.title("mutation properties")
        plt.xlabel("mutation", fontsize=19)
        plt.ylabel("latent variable value", fontsize=19)
        plt.legend(['comp.~1', 'comp.~2', 'comp.~3', 'comp.~4', 'comp.~5'], ncol=2, fontsize=18, loc='lower right')


        #This finds the mutations with substantial enough properties
        #that they should be annotated, and annotates them with the mutation name.
        #argsHigh = np.argwhere(latentSize > 0.15)[:, 0]

        argsHigh = np.argwhere(latentSize > latentMin)[:, 0]


        #print (argsHigh.shape)
        #quit()

        for i in argsHigh:
            name = mutationName[i]

            delt1 = np.max(xNP) / 100
            max0 = np.max(np.abs(xNP[i])) + (delt1 * 4)
            sign1 = np.sign(xNP[i][np.argmax(np.abs(xNP[i]))] )
            max1 = (max0  * sign1) - (delt1 * 3)

            name_italic = '\emph{' + name + '}'

            if name == 'NRAS':
                max1 = max1 + (delt1 * 4)

            ############plt.annotate(name, (i -  (M / 20), np.max(xNP[i]) + (np.max(xNP) / 100)    ))
            plt.annotate(name_italic, (i -  (M / 40), max1    ))




        plt.tight_layout()
        plt.savefig('./images/resized/LatentPlot_' + modelName + '.pdf')
        plt.show()

        quit()



    #pred_normal



    pred2 = pred.reshape((1, -1))

    #This calculates the relative probability of each mutation, for each clone representing a possible initial mutation.
    #The fitness of the clone is irrelevent to this probability.

    #if False:
    #    prob = pred
    #else:
    prob = torch.softmax(pred, dim=1)



    #This calculates the relative probability of each mutation clone pair. More fit clones will yeild higher probabilities.
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(prob.shape)

    prob_np = prob.data.numpy()
    prob2_np = prob2.data.numpy()



    #This calculates the total probability that a mutation will be added for each clone.
    #This is a measurement of the fitness of each clone.



    prob2_sum = np.sum(prob2_np, axis=1)
    #mutationNamePrint = ['NPM1', 'ASXL1', 'DNMT3A', 'NRAS', 'FLT3', 'IDH1', 'PTPN11', 'FLT3-ITD']
    #mutationNamePrint = np.array(mutationNamePrint)

    #ar1 = np.array([prob2_sum, mutationName]).T
    #ar1 = ar1[np.isin(mutationName, mutationNamePrint)]

    #np.save('./sending/proportion/OurFitness.npy', ar1)
    #print (ar1)
    #quit()

    #arg1 = np.argwhere(mutationName == 'FLT3')[0, 0]
    #arg2 = np.argwhere(mutationName == 'PIK3CA')[0, 0]
    #arg1 = np.argwhere(mutationName == 'GATA3')[0, 0]
    #arg2 = np.argwhere(mutationName == 'MAP3K1')[0, 0]

    #print (prob2_sum[arg1])

    #quit()

    #0.0394, 0.0546494, 0.052078

    #arg1 = np.argwhere(mutationName == 'NPM1')[0, 0]
    #arg2 = np.argwhere(mutationName == 'DNMT3A')[0, 0]
    #arg1 = np.argwhere(mutationName == 'FLT3')[0, 0]
    #print (prob2_sum[arg1])
    #print (prob2_sum[arg2])
    #print (np.median(prob2_sum))
    #quit()





    #This calculates the mutations which have a high enough fitness that they should be annotated
    #with the mutation name in the plot.
    #argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.5)[:, 0]
    argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.2)[:, 0]



    if modelName == 'AML':
        #argsGood = np.argwhere(np.isin(mutationName, mutationNamePrint))[:, 0]
        argsGood = np.argwhere(prob2_sum < np.median(prob2_sum) * 0.75)[:, 0]
        argsHigh = np.concatenate((argsHigh, argsGood))


    #print (np.sort(prob2_sum))
    #quit()

    #argsHigh = np.argwhere(latentSize > latentMin)[:, 0]

    if False:
        #print (argsHigh.shape)
        #quit()

        sns.set_context("notebook", font_scale=1.2)

        #This plots the relative fitness of all of the mutations in the data set.
        plt.plot(prob2_sum, c='r')#, yscale="log")
        plt.scatter( argsHigh, prob2_sum[argsHigh], c='r' )
        plt.ylabel('fitness', fontsize=19)
        plt.xlabel('mutation', fontsize=19)
        if modelName == 'AML':
            plt.yscale('log')

        # plt.gca().yaxis.set_major_locator(MultipleLocator(1))
                # ax.yaxis.set_major_locator(MultipleLocator(1))
        # plt.gca().set_yscale('log')
        for i in argsHigh:
            name = mutationName[i]
            name_italic = '\emph{' + name + '}'
            #################plt.annotate(name, (i -  (M / 20), prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
            plt.annotate(name_italic, (i , prob2_sum[i] + (np.max(prob2_sum) / 100)    ))


        #plt.yscale('log')
        plt.tight_layout()
        plt.savefig('./images/resized/fitnessPlot_' + modelName + '.pdf')
        plt.show()

        quit()



    #pred_np = pred.data.numpy()
    #print (pred_normal)
    #for a in range(pred_np.shape[1]):
    #    pred_np[:, a] = pred_np[:, a] - pred_normal[a]
    #    pred_np[:, a] = pred_np[:, a] - np.mean(pred_np[:, a])
    #print (pred_np)
    #quit()




    #This calculates probabilities of new mutations given the  existing mutations, with the
    #mutations restricted to the set of interesting mutations.
    #prob_np_inter = prob_np[argsInteresting][:, argsInteresting]
    #prob_np_inter = pred_np[argsInteresting][:, argsInteresting]
    #prob_np_inter = pred_rel[argsInteresting][:, argsInteresting]


    #print (prob_np_inter[-1])
    #print (prob_normal)

    #plt.imshow(prob_np_inter)
    #plt.show()

    #This adjusts the probability of each new mutation for the fact that in reality,
    #the already existing mutation can not be selected as the new mutation.
    #This gives more realistic information, but also removes the information of which mutations would tend to cause
    #mutations similar to itself to occur.
    prob_np_adj = np.copy(prob_np)
    #prob_np_adj[np.arange(prob_np_adj.shape[0]), np.arange(prob_np_adj.shape[0])] = 0
    for a in range(len(prob_np_adj)):
        #prob_np_adj[a] = prob_np_adj[a] / np.mean(prob_np_adj[a])
        #prob_np_adj[:, a] = prob_np_adj[:, a] / prob_normal[a] #np.mean(prob_np_adj[:, a])
        prob_np_adj[:, a] = prob_np_adj[:, a] / np.mean(prob_np_adj[:, a])


    prob_np_adj = np.log(prob_np_adj)

    #print (prob_np_adj.shape)
    #quit()

    #print (mutationName)

    #arg1 = np.argwhere(mutationName == 'ASXL1')[0, 0]
    #arg2 = np.argwhere(mutationName == 'NPM1')[0, 0]
    #arg3 = np.argwhere(mutationName == 'NRAS')[0, 0]
    #arg2 = np.argwhere(mutationName == 'FLT3-ITD')[0, 0]
    #arg3 = np.argwhere(mutationName == 'NPM1')[0, 0]
    #print (prob_np_adj[arg1, arg3])
    #print (prob_np_adj[arg2, arg3])
    #print (prob_np_adj[arg1, arg3])
    #print (prob_np_adj[arg3, arg1])
    #quit()



    argsHighElse = np.copy(argsHigh)
    argTP = np.argmax(prob2_sum)
    argsHighElse = argsHighElse[argsHighElse!=argTP]
    argsLowish = np.argwhere(prob2_sum <= np.median(prob2_sum) * 1.5)[:, 0]

    #print (np.median(prob_np_adj[argTP, argsHighElse]))
    #print (np.median(prob_np_adj[argTP, argsLowish]))
    #quit()





    if modelName == 'breast':
        reorder = np.array([4, 0, 1, 2, 3])
    elif modelName == 'AML':

        reorder = np.array([1, 0, 5, 6, 3, 7, 2, 4])
        #True
        #reorder = np.arange(argsInteresting.shape[0])

    elif modelName == 'temp':

        #reorder = np.array([1, 0, 5, 6, 3, 7, 2, 4])
        #True
        reorder = np.arange(argsInteresting.shape[0])







    prob_np_adj_inter = prob_np_adj[argsInteresting][:, argsInteresting]

    #prob_np_adj_inter = np.log(prob_np_adj_inter)







    #reorder = np.arange(prob_np_adj_inter.shape[0])

    if True: #mar 19 2023
        prob_np_adj_inter = prob_np_adj_inter[reorder]
        prob_np_adj_inter = prob_np_adj_inter[:, reorder]

    arange1 = np.arange(prob_np_adj_inter.shape[0])
    prob_np_adj_inter[arange1, arange1] = 0

    # [DNMT3A, ASXL1, NPM1, NRAS, GATA2, U2AF1] for luekemia


    if True:

        mutationName_italic = []
        for name in mutationName:
            name_italic = '\emph{' + name + '}'
            mutationName_italic.append(name_italic)
        mutationName_italic = np.array(mutationName_italic)


        vSize = np.max(np.abs(prob_np_adj_inter))
        vmin = vSize * -1
        vmax = vSize

        sns.set_context("notebook", font_scale=1.2)

        #This is a plot of the causal relationship between all of the interesting mutations,
        #with the names of the mutations labeled.
        fig, ax = plt.subplots(1,1)

        #from matplotlib.colors import DivergingNorm
        #norm = DivergingNorm(vmin=prob_np_adj_inter.min(), vcenter=0, vmax=prob_np_adj_inter.max())
        plt.imshow(prob_np_adj_inter, vmin=vmin, vmax=vmax, cmap='bwr')
        img = ax.imshow(prob_np_adj_inter, vmin=vmin, vmax=vmax, cmap='bwr')
        #img = ax.imshow(pred.data.numpy()[argsInteresting][:, argsInteresting]) #TODO UNDO Jul 25 2022
        # ax.set_xticks([], minor=True)
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        plt.grid(False)
        plt.xlabel("target mutation $t$", fontsize=19)
        plt.ylabel('source mutation $s$', fontsize=19)
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName_italic[argsInteresting][reorder])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName_italic[argsInteresting][reorder])
        #

        plt.xticks(rotation = 90)
        plt.tight_layout()
        #plt.savefig('./images/occurancePlot_' + modelName + '.pdf')
        plt.savefig('./images/resized/occurancePlot_' + modelName + '.pdf')
        plt.show()



#newAnalyzeModel("")
#quit()

################trainRealData('AML')#, trainPer=1.0) #around 150 rounds to converge for breast cancer
################quit()




def analyzeNonlinear(modelName):

    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib.pyplot as plt



    import os, sys, glob
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

    sns.set_style('whitegrid')
    mpl.rc('text', usetex=True)
    sns.set_context("notebook", font_scale=1.4)

    #plt.gcf().tight_layout()

    #modelName = 'breast'
    modelName = 'AML'

    #modelName = 'temp'


    if modelName == 'AML':
        model = torch.load('./Models/realData/savedModel_AML.pt')

        mutationName = np.load('./data/realData/categoryNames.npy')[:-2]
        #mutationName = np.load('./data/realData/AMLmutationNames.npy')[:-2]

        #print (mutationName)
        #quit()
        M = 22
        #latentMin = 0.1


        #latentMin = 0.05
        latentMin = 0.02
        #latentMin = 0.005

        #print (mutationName.shape)
        #quit()
    elif modelName == 'breast':
        model = torch.load('./Models/realData/savedModel_breast.pt')
        #mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
        mutationName = np.load('./data/realData/breastCancermutationNames.npy')[:-2]
        M = 406
        #M = 365
        #latentMin = 0.01
        latentMin = 0.1






    X2 = torch.zeros((M**2, M))
    X2[np.arange(M**2)  , np.arange(M**2)//M  ] = 1
    X2[np.arange(M**2)  , np.arange(M**2)%M  ] = 1

    #This gives the predicted probability of new mutations on the clones.
    pred2, _ = model(X2)
    pred2 = pred2.reshape((M, M, M))
    pred2 = pred2.data.numpy()

    #1, 2, 3, 12, 13


    #This creates a matrix representing all of the clones with only one mutation.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability weights and the predicted latent variables.
    pred, xNP = model(X)
    pred = pred.data.numpy()

    X_normal = torch.zeros((1, M))
    pred_normal, _ = model(X_normal)
    pred_normal = pred_normal.data.numpy()


    pred2 = pred2 - pred_normal
    pred = pred - pred_normal

    nonlinearPart = np.copy(pred2)

    #print (nonlinearPart.shape)
    #quit()


    for a in range(M):

        nonlinearPart[a]
        pred[a+np.zeros(1, dtype=int)]


        nonlinearPart[a] = nonlinearPart[a] - pred[a+np.zeros(1, dtype=int)]
        nonlinearPart[:, a] = nonlinearPart[:, a] - pred[a+np.zeros(1, dtype=int)]


    #'''
    print (mutationName)

    ar1 = nonlinearPart[0, 1]

    #print (mutationName[ np.abs(ar1 - np.median(ar1)) > 0.1 ])
    #print (mutationName[ np.abs(ar2 - np.median(ar2)) > 0.1 ])
    #'''
    #plt.plot(nonlinearPart[12, 13])
    #plt.plot(nonlinearPart[0, 1])
    #plt.show()
    #quit()

    #print (mutationName[12], mutationName[13])


    #quit()


    meanEffect = np.mean(nonlinearPart, axis=(0, 1))

    argsInteresting = np.argwhere(np.abs(meanEffect - np.median(meanEffect)) > 0.1 )[:, 0]

    for a0 in range(argsInteresting.shape[0]):
        a = argsInteresting[a0]
        name = mutationName[a]
        pos1 = a
        pos2 = meanEffect[a]

        plt.annotate(name, (pos1, pos2))


    plt.plot(meanEffect)
    plt.xlabel('mutation')
    plt.ylabel('nonlinear effect')
    plt.tight_layout()
    if modelName == 'breast':
        plt.savefig('./images/new1/nonlinear_breast.pdf')
    else:
        plt.savefig('./images/new1/nonlinear_AML.pdf')
    plt.show()
    quit()


    #sum1 = np.mean(nonlinearPart, axis=2)

    #sum1[np.arange(M), np.arange(M)] = np.median(sum1)


    plt.imshow(sum1)
    plt.show()





#analyzeNonlinear('breast')
#quit()




def analyzePathway():


    #This determines the predicted evolutionary pathways based on the model.
    #There is another function with very similar code which predicts evolutionary pathways specifically on
    #the data sets with simulated evolutionary pathways.
    #One difference between this functon and that function, is that this detects possible evolutionary pathways on data sets with weak ambigous evolutionary pathways,
    #and that function precisely finds the evolutionary pathways on data sets with strong true evolutionary pathways.



    #If makeLength2 is modified to True, it creates pathways of length 2 instead of pathways of length 1.
    #makeLength2 = True
    makeLength2 = False

    def pathwayTheoryProb(pathway, prob_assume):

        #This calculates the probability of an evolutionary pathway simply based
        #on the frequency of the different mutations that occur in the pathway,
        #completely ignoring the relationships between mutations.

        probLog = 0
        pathwayLength = len(pathway)

        if makeLength2:
            pathwayLength = 2

        for a in range(0, pathwayLength):
            pathwaySet = pathway[a]
            if len(pathwaySet) == 0:
                #probLog += -1000
                True
            else:
                prob = np.sum(prob_assume[np.array(pathwaySet)])
                probLog += np.log(prob)

        return probLog

    def pathwayRealProb(pathway, prob2_Adj_3D):

        #This calculates the probability of an evolutionary pathway
        #according to the model.

        subset = prob2_Adj_3D

        if min(min(len(pathway[0]), len(pathway[1])), len(pathway[2]) ) == 0:
            subset_sum = np.log(1e-50)
        else:

            subset = subset[np.array(pathway[0])]
            subset = subset[:, np.array(pathway[1])]

            if not makeLength2:
                subset = subset[:, :, np.array(pathway[2])]

            subset_max = np.max(subset)
            subset = subset - subset_max
            subset_sum = np.sum(np.exp(subset))
            subset_sum = np.log(subset_sum+1e-50)
            subset_sum = subset_sum + subset_max

        return subset_sum



    def evaluatePathway(pathway, prob2_Adj_3D, prob_assume, includeProb=False):

        #This determines how "good" or "valid" and evolutionary pathway is
        #based on its frequency of occuring and its frequency of occuring beyond the expected
        #probability from the frequency of mutations in the pathway.

        probTheory = pathwayTheoryProb(pathway, prob_assume)
        probReal = pathwayRealProb(pathway, prob2_Adj_3D)

        probDiff = probReal - probTheory


        #if probReal > np.log(0.1):
        #    probReal = np.log(0.1)

        if probReal > np.log(0.05): #Used for breast cancer data
            probReal = np.log(0.05)  #Used for breast cancer data

        #score = (probReal * 0.1) + probDiff


        #score = probDiff - (0.1 * (np.abs(probDiff) ** 2))

        #score = (probReal * 0.05) + probDiff
        #score = (probReal * 0.2) + probDiff #Used for breast cancer data

        #score = (probReal * 0.1) + probDiff

        score = (probReal * 0.2) + probDiff

        #score = (probReal * 0.05) + probDiff

        #score = (probReal * 0.2) + probDiff
        #score = (probReal * 0.2) + probDiff
        #score = probReal - (0.8 * probTheory)
        if includeProb:
            return score, probReal, probDiff
        else:
            return score



    def singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume):

        #This tries modifying a single mutation in the evolutionary pathway and then
        #scoring the results

        pathway2 = copy.deepcopy(pathway)
        set1 = pathway2[step]
        set1 = np.array(set1)

        if doAdd:
            set1 = np.concatenate((set1, np.array([position])))
            set1 = np.sort(set1)
        else:
            set1 = set1[set1 != position]

        #pathway2[step] = set1.astype(int)
        pathway2[step] = set1.astype(int)

        score= evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

        return score, pathway2

    def stepModifyPathway(doAdd, step, pathway, superPathway, prob2_Adj_3D, prob_assume):

        #This tries every possibile modification of a particular step in an evolutionary pathway

        M = prob_assume.shape[0]

        set1 = copy.deepcopy(pathway[step])
        if doAdd or (len(set1) > 1):
            set2 = copy.deepcopy(superPathway[step])
            set3 = set2[np.isin(set2, set1) != doAdd]

            pathways = []
            scores = []
            for position in set3:
                score, pathway2 = singleModifyPathway(doAdd, step, position, pathway, prob2_Adj_3D, prob_assume)

                pathways.append(copy.deepcopy(pathway2))
                scores.append(score)

            return pathways, scores

        else:

            return [], []


    def iterOptimizePathway2(doAddList, stepList, pathway, superPathway, prob2_Adj_3D, prob_assume):

        #This function does a single optimization step in optimizing an evolutionary pathway
        #It also allows for steps of the evolutionary pathway to be set to "all mutations" or
        #"only one mutation", which helps it break out of local minima.

        M = prob_assume.shape[0]

        #This scores the original pathway
        score = evaluatePathway(pathway, prob2_Adj_3D, prob_assume)
        pathway2 = copy.deepcopy(pathway)

        pathways2, scores2 = [pathway2], [score]

        #This iterates through the steps in the pathway
        for step in stepList:

            #This finds the mutations at this step in the current pathway
            set1 = copy.deepcopy(pathway[step])
            #This finds the mutations which are allowed to be used.
            #Currently this is set to the set of all mutations, but can be modified by the user.
            set2 = copy.deepcopy(superPathway[step])

            #This finds the mutations which are allowed to be added.
            set_add = set2[np.isin(set2, set1) != True]
            #This finds the mutations which are allowed to be removed.
            set_rem = set2[np.isin(set2, set1) != False]

            #This "-1 mutation" represents doing no modificiation.
            set_add = [-1] + list(set_add)
            set_rem = [-1] + list(set_rem)



            #This iterates through the mutations which can be added
            for pos_add in set_add:
                pathway2 = copy.deepcopy(pathway)

                #This modifies the pathway by adding the mutation pos_add and scores it.
                if pos_add >= 0:
                    score, pathway2 = singleModifyPathway(True, step, pos_add, pathway2, prob2_Adj_3D, prob_assume)

                #This records the new pathway and its score
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

            #This iterates through the mutations which can be removed
            for pos_rem in set_rem:
                pathway2 = copy.deepcopy(pathway)

                #This modifies the pathway by removing the mutation pos_rem and scores it.
                if pos_rem >= 0:
                    score, pathway2 = singleModifyPathway(False, step, pos_rem, pathway2, prob2_Adj_3D, prob_assume)

                #This records the modified pathway and its score.
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))

        #This part of the code tries setting a step in the pathway to the set of all mutations or the
        #set with only one mutation. This major modification to the pathway helps prevent the pathway optimization
        #from getting stuck in a local minima.
        for step in stepList:
            #The "-1 mutation" represents adding all mutations.
            set1 = [-1] + list(range(M))
            for pos_add in set1:

                pathway2 = copy.deepcopy(pathway)

                if pos_add == -1:
                    #If pos_add == -1, then it sets this step in the pathway to the set of all mutations
                    pathway2[step] = list(np.arange(M))
                else:
                    #It sets this step in the pathway to only being the mutation pos_add
                    pathway2[step] = [pos_add]

                #This scores the pathway created.
                score = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume)

                #This records the modified pathway and its score.
                pathways2.append(copy.deepcopy(pathway2))
                scores2.append(copy.copy(score))


        #This finds the modified pathway with the highest score.
        bestOne = np.argmax(np.array(scores2))
        bestPathway = pathways2[bestOne]
        bestScore = np.max(scores2)

        return bestPathway, bestScore


    def singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob_Adj_3D, prob_assume):

        #This function fully optimizes an evolutionary pathway.

        pathway2 = copy.deepcopy(pathway)
        bestScoreBefore = -10000
        notDone = True
        while notDone:
            #pathway2, bestScore = iterOptimizePathway(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)
            pathway2, bestScore = iterOptimizePathway2(doAddList, stepList, pathway2, superPathway, prob_Adj_3D, prob_assume)


            if bestScoreBefore == bestScore:
                notDone = False
            bestScoreBefore = bestScore

        #print (evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True))

        evalScores = evaluatePathway(pathway2, prob2_Adj_3D, prob_assume, includeProb=True)

        #print (evalScores)
        print (np.exp(evalScores[1]), np.exp(evalScores[2]))

        return pathway2



    def removePathFromProb(prob, pathway):

        #This removes the some pathway from the tensor of probabilities of different
        #evolutionary trajectories.
        #This makes it so the same pathway will not be found repeatedly,
        #since the probabilities of a pathway that has already been found is set to zero.
        #It also makes it unlikly for the method to find similar pathways repeatedly.

        set1, set2, set3 = np.array(pathway[0]), np.array(pathway[1]), np.array(pathway[2])
        M = prob.shape[0]
        modMask = np.zeros(prob.shape)
        modMask[np.arange(M)[np.isin(np.arange(M), set1) == False]] = 1
        modMask[:, np.arange(M)[np.isin(np.arange(M), set2) == False]] = 1

        if not makeLength2:
            modMask[:, :, np.arange(M)[np.isin(np.arange(M), set3) == False]] = 1

        prob[modMask == 0] = -1000

        return prob


    def doMax(ar1):

        #Calculating the maximum of an array while avoiding errors if
        #the array is empty.

        if ar1.size == 0:
            return 0
        else:
            return np.max(ar1)




    #Loading in the model for the breast cancer data set.
    model = torch.load('./Models/savedModel25.pt')
    #model = torch.load('./Models/savedModel26.pt')


    #mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
    #mutationName = np.load('./data/mutationNamesBreast.npy')[:-2]
    mutationName = np.load('./data/categoryNames.npy')[:-2]
    argsInteresting = np.load('./data/interestingMutations.npy')

    #X0 is one empty clone. It is used to get the probability of the initial mutation.
    #X1 is are clones with 1 mutation. It is used to get the probability of the second mutation.
    #X2 is clones with 2 mutations. It is used to get the probability of the third mutation.
    M = 24
    X0 = torch.zeros((1, M))
    X1 = torch.zeros((M, M))
    X2 = torch.zeros((M*M, M))

    arange0 = np.arange(M)
    arange1 = np.arange(M*M)

    X1[arange0, arange0] = 1
    X2[arange1, arange1 % M] = X2[arange1, arange1 % M] + 1
    X2[arange1, arange1 // M] = X2[arange1, arange1 // M] + 1

    #pred0, pred1, and pred2 give the probaility weights for the first second and third mutations.
    pred0, _ = model(X0)
    pred1, xLatent1 = model(X1)
    pred2, _ = model(X2)

    #This removes the possibility of an evolutionary trajectory having the same mutation multiple times.
    pred2 = pred2.reshape((M, M, M))
    pred1[arange0, arange0] = -1000
    pred2[arange0, arange0, :] = -1000
    pred2[:, arange0, arange0] = -1000
    pred2[arange0, :, arange0] = -1000
    pred2 = pred2.reshape((M * M, M))




    #This gives the probability for the first, second, and third mutations.
    prob0 = torch.softmax(pred0, dim=1)
    prob1 = torch.softmax(pred1, dim=1)
    prob2 = torch.softmax(pred2, dim=1)


    #This uses the probability for the first tree mutations to give the probability of
    #every trajectory of three consecutive mutations.
    prob0 = prob0.data.numpy()
    prob1 = prob1.data.numpy()
    prob2 = prob2.data.numpy()
    prob0_Adj = np.log(np.copy(prob0) + 1e-10)
    outputProb0 = np.log(prob0[0] + 1e-10)
    outputProb0 = outputProb0.repeat(M).reshape((M, M))
    prob1_Adj = outputProb0 + np.log(prob1 + 1e-10)



    outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
    prob2_Adj = outputProb1 + np.log(prob2 + 1e-10)

    if False:

        #This unused code is an alternative method of defining the probability of evolutionary trajectories based on the
        #expected rate in which they occur.
        #With the default method, the first mutation in a trajectory being more fit does not necesarily make the trajectory more likly.
        #However, with this method, the first mutation being more fit will mean the next mutations will occur more quickly and therefore
        #the trajectory occurs at a higher rate.

        prob0 = pred0.data.numpy() * -1
        prob1 = pred1.data.numpy() * -1
        prob2 = pred2.data.numpy() * -1
        prob0_Adj = np.copy(prob0)
        outputProb0 = prob0[0]
        outputProb0 = outputProb0.repeat(M).reshape((M, M))
        prob1_Adj = addFromLog([outputProb0, prob1])
        outputProb1 = prob1_Adj.repeat(M).reshape((M*M, M))
        prob2_Adj = addFromLog([outputProb1, prob2])
        prob2_Adj = prob2_Adj * -1



    prob2_Adj_3D = prob2_Adj.reshape((M, M, M))

    #This ensures the total probability of all trajectories is 1, despite certian
    #trajectories having their probability set to zero due to being impossible.
    prob2_Adj_3D = prob2_Adj_3D - np.log(np.sum(np.exp(prob2_Adj_3D)))



    if True:
        #This code combines all of the "boring" mutations with unimportant latent representations into
        #a single "generic" mutation.
        #Specifically, it modifies prob2_Adj_3D so that it now contains seperate value for only the interesting
        #mutations and a combined value for the generic mutations.


        argsBoring = np.arange(M)[np.isin(np.arange(M), argsInteresting) == False]
        for b in range(3):

            sizeBefore = list(prob2_Adj_3D.shape)

            sizeBefore[b] = argsInteresting.shape[0] + 1

            prob2_Adj_3D_new = np.zeros((sizeBefore[0], sizeBefore[1], sizeBefore[2]))

            if b == 0:
                prob2_Adj_3D_new[:-1, :, :] = np.copy(prob2_Adj_3D[argsInteresting, :, :])
                #max1 = np.max(prob2_Adj_3D[argsBoring])
                max1 = doMax(prob2_Adj_3D[argsBoring])
                prob2_Adj_3D_new[-1, :, :] = np.log(np.sum(np.exp(prob2_Adj_3D[argsBoring, :, :] - max1), axis=b)+1e-20) + max1
            if b == 1:
                prob2_Adj_3D_new[:, :-1, :] = np.copy(prob2_Adj_3D[:, argsInteresting, :])
                #max1 = np.max(prob2_Adj_3D[:, argsBoring])
                max1 = doMax(prob2_Adj_3D[:, argsBoring])
                prob2_Adj_3D_new[:, -1, :] = np.log(np.sum(np.exp(prob2_Adj_3D[:, argsBoring, :] - max1), axis=b)+1e-20) + max1
            if b == 2:
                prob2_Adj_3D_new[:, :, :-1] = np.copy(prob2_Adj_3D[:, :, argsInteresting])
                #max1 = np.max(prob2_Adj_3D[:, :, argsBoring])
                max1 = doMax(prob2_Adj_3D[:, :, argsBoring])
                prob2_Adj_3D_new[:, :, -1] = np.log(np.sum(np.exp(prob2_Adj_3D[:, :, argsBoring] - max1), axis=b)+1e-20) + max1

            prob2_Adj_3D = np.copy(prob2_Adj_3D_new)

        mutationName = np.concatenate((mutationName[argsInteresting], np.array(['Generic'])))


    #if argsBoring.size > 0:
    M = argsInteresting.shape[0] + 1
    #else:
    #    M = argsInteresting.shape[0]

    prob2_Adj = prob2_Adj_3D.reshape((M*M, M))


    #This first calculates the frequency at which each mutation occurs as the first mutation, second mutation, and third mutation.
    #If makeLength2 is true, then it only does the first and second mutations.
    prob2_sum0 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=0)
    prob2_sum1 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=1), axis=1)
    if not makeLength2:
        prob2_sum2 = np.sum(np.sum(np.exp(prob2_Adj_3D), axis=0), axis=1)
        #Now this averages those three calculations to give the general frequency at which each mutation occurs.
        prob2_Assume = (prob2_sum0 + prob2_sum1 + prob2_sum2) / 3
    else:
        #Now this averages those three calculations to give the general frequency at which each mutation occurs.
        prob2_Assume = (prob2_sum0 + prob2_sum1) / 2




    #This gives the initial pathway to start the optimization from, which is set to
    #the pathway of all mutations at all steps.
    pathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]
    #This limits the pathways to a subset of mutations at each step called a superPathway here.
    #However, in this case the superPathway is the superPathway of all mutations at all steps.
    superPathway = [np.arange(M-1), np.arange(M-1), np.arange(M-1)]


    prob2_Adj_3D_mod = np.copy(prob2_Adj_3D)


    #It iterates through 20 pathways since it is unlikly that there are more than 20 independent pathways.
    for a in range(0, 20):


        doAddList = [True, False]
        stepList = [0, 1, 2]

        #This finds the optimal pathway
        pathway2 = singleOptimizePathway(doAddList, stepList, pathway, superPathway, prob2_Adj_3D_mod, prob2_Assume)



        if True:
            #By default, the pathway itself is removed from the probabilities.
            prob2_Adj_3D_mod = removePathFromProb(prob2_Adj_3D_mod, pathway2)
        else:
            #As an alternative option, one can have all trajectories containing mutations inside any of the steps in the pathway
            #set to zero probability.
            pathway2_full = np.concatenate((pathway2[0], pathway2[1], pathway2[2])).astype(int)
            prob2_Adj_3D_mod[pathway2_full] = -1000
            prob2_Adj_3D_mod[:, pathway2_full] = -1000
            prob2_Adj_3D_mod[:, :, pathway2_full] = -1000


        #This prints the sets in the pathway found, and the size of the sets in the pathway
        print ("Pathway Found:")
        print (mutationName[pathway2[0]])
        print (mutationName[pathway2[1]])
        print (mutationName[pathway2[2]])
        print ("Sizes: ", len(pathway2[0]), len(pathway2[1]), len(pathway2[2]))



def findBreastPatientNamesSubset():


    patientNum = -1
    Pnames = []

    with open('./extra/breast_Razavi.txt') as f:
        lines = f.readlines()
        for a in range(len(lines)):
            line1 = lines[a]

            if 'P-' in line1:
                patientNum += 1
                Pname = 'P-' + line1.split('P-')[1][:-1]
                Pnames.append(Pname)

    file1 = './data/realData/breastCancer'
    maxM = 40
    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)

    _, index1 = np.unique(sampleInverse, return_index=True)
    miniTrees = newTrees[index1]
    treeLength = treeLength.astype(int)
    treeLength = treeLength[index1]

    nameAndLength = np.array(Pnames + Pnames).reshape(( len(Pnames), 2  ))
    nameAndLength[:, 0] = np.array(Pnames)
    nameAndLength[:, 1] = treeLength.astype(str)

    np.savez_compressed('./extra/nameAndLengthBreast.npz', nameAndLength)


#findBreastPatientNamesSubset()
#quit()




def checkBreastType():



    def reOrderTree(tree1):

        firstClone = np.argwhere(np.isin(tree1[:, 0], tree1[:, 1]) == False)[0, 0]

        order1 = np.array([]).astype(int)
        mutInclude = np.array([tree1[firstClone, 0]])

        #print ('')
        #print ('')
        #print ('')
        #rint ('')
        #print ('')
        #print ('')
        #print ('')

        loop1 = 0

        while order1.shape[0] < tree1.shape[0]:

            toInclude = np.argwhere( np.logical_and(  np.isin(tree1[:, 0], mutInclude  ), np.isin(tree1[:, 1],  mutInclude ) == False   ) )[:, 0]
            order1 = np.concatenate((order1, toInclude))
            mutInclude = np.concatenate((mutInclude,  tree1[toInclude, 1] ))

            #print (order1)
            #print ("A")
            #print (tree1[order1])
            #print (tree1)

            loop1 += 1
            if loop1 == 100:
                quit()

        tree1 = tree1[order1]

        assert np.unique(order1).shape[0] == order1.shape[0]

        return tree1


    def findClones(tree1, mutationCategory, Mcat):

        clones = np.zeros((tree1.shape[0]+1, Mcat))

        #print ("A")
        #print (tree1)
        #print (tree1)
        #print ("B")

        for a in range(0, tree1.shape[0]):

            #print (a)
            #print (tree1)
            #print (tree1[a])


            argBefore = np.argwhere(tree1[:, 1] == tree1[a, 0])
            if argBefore.shape[0] > 0:
                argBefore = argBefore[0, 0] + 1
                clones[a+1] = np.copy(clones[argBefore])

            clones[a+1, mutationCategory[ tree1[a, 1] ]] = 1

        return clones




    import matplotlib.pyplot as plt


    #print (np.unique(types, return_counts=True))


    modelFile = './Models/realData/savedModel_breast.pt'
    model = torch.load(modelFile)

    breastNames = np.load('./data/realData/breastCancermutationNames.npy')
    breastNames = breastNames[:-2]
    #print (breastNames.shape)

    #maxM = 10
    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/realData/breastCancer', fullDir=True)
    #_, index1 = np.unique(sampleInverse, return_index=True)
    #newTrees = newTrees[index1]

    nameAndLength = loadnpz('./extra/nameAndLengthBreast.npz')

    file1 = './data/realData/breastCancer'
    maxM = 9
    newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)

    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

    mutationCategory = mutationCategory[:-2]
    Mcat = mutationCategory.shape[0]

    _, index1 = np.unique(sampleInverse, return_index=True)
    miniTrees = newTrees[index1]
    treeLength = treeLength.astype(int)
    treeLength1 = treeLength[index1]

    treeLength2 = nameAndLength[:, 1].astype(int)

    patientNames = nameAndLength[treeLength2<=maxM, 0]



    data = np.loadtxt('./extra/BreastCancerTypes.csv', delimiter=',', dtype=str)

    data = data[np.isin(data[:, 0], patientNames)]
    types = data[:, 1]

    #print (np.unique(data[:, 1], return_counts=True))
    #quit()

    typeNames, types = np.unique(types, return_inverse=True)
    types_argsort = np.argsort(types)
    types = types[types_argsort]
    data = data[types_argsort]


    #latentRepAll = np.zeros((sampleInverse.shape[0], maxM, 5))
    latentRep = np.zeros((sampleInverse.shape[0], 5))

    for a in range(sampleInverse.shape[0]):

        treeSize = treeLength[a]
        tree1 = newTrees[a][:treeSize]

        tree1 = reOrderTree(tree1)

        clones = findClones(tree1, mutationCategory, Mcat)

        #print (clones.shape)
        #quit()

        clones = torch.tensor(clones).float()
        _, reps = model(clones)
        #reps = reps.data.numpy()

        latentRep[a] = np.mean(reps, axis=0)

    #print (np.max(latentRep))
    #quit()

    Npatient = np.unique(sampleInverse).shape[0]
    latentRepPatient = np.zeros(( Npatient, 5 ))
    for a in range(Npatient):
        args1 = np.argwhere(sampleInverse == a)[:, 0]

        latentRepPatient[a] = np.mean(latentRep[args1], axis=0)


    #print (data[:10, 0])
    #print (patientNames[:10])
    #quit()

    latentRepOrdered = np.zeros(( data.shape[0], 5 ))
    for a in range(data.shape[0]):
        arg1 = np.argwhere(patientNames == data[a, 0])[0, 0]
        latentRepOrdered[a] = latentRepPatient[arg1]



    #from sklearn.manifold import TSNE
    #embedding = TSNE(n_components=2, learning_rate='auto',
    #                init='random', perplexity=3).fit_transform(latentRepOrdered)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    embedding = pca.fit_transform(latentRepOrdered)




    #print (np.std(embedding[:, 0]))
    #print (np.std(embedding[:, 1]))
    #print (np.std(embedding[:, 2]))
    #quit()

    #embedding = latentRepOrdered


    plt.scatter(embedding[types < 2, 0], embedding[types < 2, 1], s=20, alpha=0.5)
    plt.scatter(embedding[types >= 2, 0], embedding[types >= 2, 1], s=20, alpha=0.5)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.tight_layout()
    plt.gcf().set_size_inches(8, 6)
    plt.legend(["HR+", "HR$-$"])
    plt.savefig('./images/new1/breastEmbedding_PCA_0_1.pdf')
    plt.show()

    plt.scatter(embedding[types < 2, 0], embedding[types < 2, 2], s=20, alpha=0.5)
    plt.scatter(embedding[types >= 2, 0], embedding[types >= 2, 2], s=20, alpha=0.5)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 3")
    plt.tight_layout()
    plt.gcf().set_size_inches(8, 6)
    plt.legend(["HR+", "HR$-$"])
    plt.savefig('./images/new1/breastEmbedding_PCA_0_2.pdf')
    plt.show()

    plt.scatter(embedding[types < 2, 1], embedding[types < 2, 2], s=20, alpha=0.5)
    plt.scatter(embedding[types >= 2, 1], embedding[types >= 2, 2], s=20, alpha=0.5)
    plt.xlabel("PCA component 2")
    plt.ylabel("PCA component 3")
    plt.tight_layout()
    plt.gcf().set_size_inches(8, 6)
    plt.legend(["HR+", "HR$-$"])
    plt.savefig('./images/new1/breastEmbedding_PCA_1_2.pdf')
    plt.show()


    quit()





#checkBreastType()
#quit()


def doPCAlatent():

    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib
    import matplotlib.pyplot as plt



    import os, sys, glob
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


    #plt.gcf().tight_layout()

    #modelName = 'AML'
    modelName = 'breast'

    if modelName == 'AML':
        model = torch.load('./Models/realData/savedModel_AML.pt')
        #mutationName = np.load('./data/realData/AMLmutationNames.npy')[:-2]
        mutationName = np.load('./data/realData/categoryNames.npy')[:-2]
        M = 22
        #latentMin = 0.1



        latentMin = 0.03


    elif modelName == 'breast':
        model = torch.load('./Models/realData/savedModel_breast.pt')
        mutationName = np.load('./data/realData/breastCancermutationNames.npy')[:-2]
        M = 406
        #M = 365
        #latentMin = 0.01
        latentMin = 0.1
        #latentMin = 0.05




    #This creates a matrix representing all of the clones with only one mutation.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability weights and the predicted latent variables.
    pred, xNP = model(X)

    X_normal = torch.zeros((1, M))
    pred_normal, _ = model(X_normal)


    #This substracts the median from the latent representation, which makes it so the uniteresting mutations have a
    #value of zero.
    for a in range(0, 5):
        xNP[:, a] = xNP[:, a] - np.median(xNP[:, a])

    #This calculates the difference of the value of the latent parameter from the median
    #If latentSize is large, it means that mutation has at least some significant property.
    latentSize = np.max(np.abs(xNP), axis=1)


    pred_shape = pred.shape
    pred2 = pred.reshape((1, -1))
    #This calculates the relative probability of each mutation clone pair. More fit clones will yeild higher probabilities.
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(pred_shape)
    prob2_np = prob2.data.numpy()
    prob2_sum = np.sum(prob2_np, axis=1)
    fitness = prob2_sum



    #This finds the "interesting" mutations which have at least some significant property.
    argsInteresting = np.argwhere(latentSize > latentMin)[:, 0]
    #np.save('./dataNew/interestingMutations_' + modelName + '.npy', argsInteresting)

    if modelName == 'breast':
        argC = np.argwhere(mutationName == 'CBFB')[:, 0]
        argsInteresting = np.concatenate((argsInteresting, argC), axis=0)





    #from sklearn.manifold import TSNE
    #embedding = TSNE(n_components=2, learning_rate='auto',
    #                init='random', perplexity=3).fit_transform(xNP)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    embedding = pca.fit_transform(xNP)


    #print (np.std(embedding[:, 0]))
    #print (np.std(embedding[:, 1]))
    #print (np.std(embedding[:, 2]))
    #quit()

    import scipy
    #print (scipy.stats.pearsonr(embedding[:, 0], fitness))
    #print (scipy.stats.pearsonr(embedding[:, 1], fitness))
    #print (scipy.stats.pearsonr(embedding[:, 2], fitness))
    #quit()


    axisNum = 2

    if axisNum == 0:
        embedding = embedding[:, np.array([0, 1])]
    if axisNum == 1:
        embedding = embedding[:, np.array([0, 2])]
    if axisNum == 2:
        embedding = embedding[:, np.array([1, 2])]




    #import umap.umap_ as umap
    #from umap import UMAP
    #from umap.umap_ import UMAP
    #reducer = UMAP()
    #embedding = reducer.fit_transform(xNP)

    #plt.plot(embedding)
    #plt.show()

    if True:

        from adjustText import adjust_text

        length1 = np.max(embedding[:, 0]) - np.min(embedding[:, 0])
        length2 = np.max(embedding[:, 1]) - np.min(embedding[:, 1])

        plt.scatter(embedding[:, 0], embedding[:, 1], c=fitness, cmap='viridis', norm=matplotlib.colors.LogNorm())

        if modelName == 'breast':
            plt.ylim( np.min(embedding[:, 1]) - (length2 * 0.05) , np.max(embedding[:, 1]) + (length2 * 0.1) )
        if modelName == 'AML':
            plt.xlim( np.min(embedding[:, 0]) - (length1 * 0.1) , np.max(embedding[:, 0]) + (length1 * 0.05) )
            plt.ylim( np.min(embedding[:, 1]) - (length2 * 0.1) , np.max(embedding[:, 1]) + (length2 * 0.1) )

        if modelName != 'AML':
            for a0 in range(argsInteresting.shape[0]):
                a = argsInteresting[a0]
                name1 = mutationName[a]

                doAnnotate = True

                if axisNum == 1:
                    if name1 == 'CBFB':
                        doAnnotate = False

                if doAnnotate:
                    name2 = mutationName[a]
                    name2 = '\emph{' + name2 + '}'
                    plt.annotate(name2, (embedding[a, 0] - (length1*0.045), embedding[a, 1] + (length2*0.02)    ))

        if modelName == 'AML':

            texts = []
            for a0 in range(argsInteresting.shape[0]):
                a = argsInteresting[a0]
                name1 = mutationName[a]

                pos1 = embedding[a, 0] - (length1*0.045)
                pos2 = embedding[a, 1] + (length2*0.02)

                doAnnotate = True

                if axisNum == 0:
                    if name1 in ['NPM1']:
                        pos1 = pos1 - (length1 * 0.03)
                    if name1 in ['FLT3', 'NRAS', 'U2AF1']: #'GATA2'
                        pos2 = pos2 - (length2 * 0.08)
                    if name1 in ['NRAS', 'U2AF1']: #'GATA2'
                        pos1 = pos1 + (length1 * 0.04)
                    if name1 in ['NRAS']:
                        pos1 = pos1 - (length1 * 0.03)
                    if name1 in ['FLT3', 'DNMT3A']:
                        pos1 = pos1 - (length1 * 0.04)
                    if name1 in ['GATA2', 'IDH2']:
                        pos1 = pos1 + (length1 * 0.04)
                    if name1 in ['DNMT3A']:
                        pos1 = pos1 - (length1 * 0.04)

                if axisNum == 1:
                    if name1 in ['FLT3', 'IDH2', 'U2AF1']:
                        doAnnotate = False

                if axisNum == 2:
                    if name1 in ['FLT3', 'IDH2', 'U2AF1']:
                        doAnnotate = False


                if doAnnotate:

                    name1 = '\emph{' + name1 + '}'

                    plt.annotate(name1, (pos1, pos2    ))
                    #texts.append(plt.text(embedding[a, 0], embedding[a, 1], mutationName[a]))


        if axisNum == 0:
            plt.xlabel('PCA component 1')
            plt.ylabel('PCA component 2')
            endName = '_1_2'
        if axisNum == 1:
            plt.xlabel('PCA component 1')
            plt.ylabel('PCA component 3')
            endName = '_1_3'
        if axisNum == 2:
            plt.xlabel('PCA component 2')
            plt.ylabel('PCA component 3')
            endName = '_2_3'

        #plt.colorbar()
        if modelName == 'AML':
            plt.colorbar().set_label('fitness', rotation=270)
        else:
            plt.colorbar().set_label('fitness', rotation=270, labelpad=20)

        #adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
        #adjust_text(texts)

        plt.tight_layout()
        plt.gcf().set_size_inches(8, 6)
        if True:
            if modelName == 'AML':
                plt.savefig('./images/new1/embeddedLatentAML' + endName + '.pdf')
            else:
                plt.savefig('./images/new1/embeddedLatentBreast' + endName + '.pdf')
        True
        plt.show()


    if False:

        #print (xNP.shape)

        distList = []
        for a in range(xNP.shape[0]):
            for b in range(xNP.shape[0]):
                if b > a:
                    dist1 = np.sum((xNP[a] - xNP[b]) ** 2) ** 0.5
                    distList.append(dist1)

        plt.hist(distList, bins=50)
        plt.yscale('log')
        plt.xlabel('Euclidean distance')
        plt.ylabel('number of mutation pairs')
        plt.gcf().set_size_inches(8, 6)
        if modelName == 'AML':
            plt.savefig('./images/euclidAML.pdf')
        else:
            plt.savefig('./images/euclidBreast.pdf')
        plt.show()




doPCAlatent()
quit()




def checkLateFitness():



    def reOrderTree(tree1):

        firstClone = np.argwhere(np.isin(tree1[:, 0], tree1[:, 1]) == False)[0, 0]

        order1 = np.array([]).astype(int)
        mutInclude = np.array([tree1[firstClone, 0]])

        #print ('')
        #print ('')
        #print ('')
        #rint ('')
        #print ('')
        #print ('')
        #print ('')

        loop1 = 0

        while order1.shape[0] < tree1.shape[0]:

            toInclude = np.argwhere( np.logical_and(  np.isin(tree1[:, 0], mutInclude  ), np.isin(tree1[:, 1],  mutInclude ) == False   ) )[:, 0]
            order1 = np.concatenate((order1, toInclude))
            mutInclude = np.concatenate((mutInclude,  tree1[toInclude, 1] ))

            #print (order1)
            #print ("A")
            #print (tree1[order1])
            #print (tree1)

            loop1 += 1
            if loop1 == 100:
                quit()

        tree1 = tree1[order1]

        assert np.unique(order1).shape[0] == order1.shape[0]

        return tree1


    def findClones(tree1, mutationCategory, Mcat):

        clones = np.zeros((tree1.shape[0]+1, Mcat))

        #print ("A")
        #print (tree1)
        #print (tree1)
        #print ("B")

        for a in range(0, tree1.shape[0]):

            #print (a)
            #print (tree1)
            #print (tree1[a])


            argBefore = np.argwhere(tree1[:, 1] == tree1[a, 0])
            if argBefore.shape[0] > 0:
                argBefore = argBefore[0, 0] + 1
                clones[a+1] = np.copy(clones[argBefore])

            clones[a+1, mutationCategory[ tree1[a, 1] ]] = 1

        return clones


    dataName = 'AML'
    #dataName = 'breast'


    if dataName == 'AML':
        file1 = './data/realData/AML'
        file2 = './extra/AML_treeSizes.npz'

        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
        mutationCategory = mutationCategory[:-2]
    elif dataName == 'breast':
        file1 = './data/realData/breastCancer'
        file2 = './extra/Breast_treeSizes.npz'

        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
        mutationCategory = mutationCategory[:-2]

        #print (mutationCategory.shape)
        #print (np.unique(mutationCategory).shape)
        #quit()


    #maxM = 9
    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './dataNew/breastCancer.npy')

    #_, index1 = np.unique(sampleInverse, return_index=True)
    #treeSizes = treeLength[index1]

    sampleInverse = sampleInverse.astype(int)
    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)
    _, count1 = np.unique(sampleInverse, return_counts=True)


    treeLength = treeLength.astype(int)


    maxM = newTrees.shape[1]
    M = int(np.max(newTrees+1)) - 2

    Mcat = np.unique(mutationCategory).shape[0]

    mutationEarlyList = np.zeros((Mcat, maxM))


    for a in range(newTrees.shape[0]):

        print (a)

        tree1 = newTrees[a]
        tree1 = tree1[:treeLength[a]]

        #print (tree1)
        #quit()

        tree1 = reOrderTree(tree1)

        clones1 = findClones(tree1, np.arange(M), M)
        clones1 = clones1[1:]
        cloneSum = np.sum(clones1, axis=1) - 1
        cloneSum = cloneSum.astype(int)

        #print (tree1)
        #print (cloneSum)
        #uit()

        mutationEarlyList[mutationCategory[tree1[:, 1]], cloneSum] += (1/count1[sampleInverse[a]])


    avgSum = np.copy(mutationEarlyList)
    avgSum = avgSum / np.sum(avgSum, axis=1).reshape((-1, 1))
    avgSum = avgSum * np.arange(maxM).reshape((1, maxM))
    avgSum = np.sum(avgSum, axis=1)

    _, index1 = np.unique(mutationCategory, return_index=True)

    uniqueMutation = uniqueMutation[index1]
    for a in range(uniqueMutation.shape[0]):
        uniqueMutation[a] = uniqueMutation[a].split('_')[0]
    #uniqueMutation = np.array(uniqueMutation)

    #print (uniqueMutation.shape)
    #print (Mcat)
    #quit()

    if dataName == 'AML':
        highFitness = ['ASXL1', 'DNMT3A', 'GATA2', 'IDH2', 'NPM1', 'NRAS', 'U2AF1'] #'FLT3'
    else:
        highFitness = ['ARID1A', 'CDH1', 'ESR1', 'GATA3', 'KMT2C', 'MAP3K1', 'PIK3CA', 'TP53']
    highFitness = np.array(highFitness)
    #avgSum2 = avgSum[np.isin(uniqueMutation, highFitness)]
    order1 = []
    for a in range(highFitness.shape[0]):
        order1.append(np.argwhere(uniqueMutation == highFitness[a] )[0, 0])
    order1 = np.array(order1)
    avgSum2 = avgSum[order1]
    avgSum3 = avgSum[np.isin(np.arange(avgSum.shape[0]), order1)==False]


    print (np.median(avgSum))

    print (np.array(highFitness[  avgSum2  >  np.median(avgSum) ]  ))
    print (np.array(avgSum2[  avgSum2  >  np.median(avgSum) ]  ))

    quit()


    print (avgSum2.shape)
    print (highFitness.shape)
    #quit()

    #plt.plot(avgSum[])

    print (np.mean(avgSum))
    print (np.mean(avgSum2))
    print (avgSum2)

    min1 = np.min(avgSum)
    max1 = np.max(avgSum)

    plt.hist(avgSum3, range=(min1, max1), alpha=0.5)
    plt.hist(avgSum2, range=(min1, max1), alpha=0.5)
    plt.xlabel('average number of existing mutations on clone')
    plt.ylabel('count')
    plt.legend(['low fitness mutations', 'high fitness mutations'])
    plt.gcf().set_size_inches(8, 6)
    if dataName == 'AML':
        if False:
            plt.savefig('./images/earlyFitAML.pdf')
    else:
        plt.yscale('log')
        if False:
            plt.savefig('./images/earlyFitBreast.pdf')
    plt.show()
    quit()




#checkLateFitness()
#quit()


def investigateRootPrevalance():





    #ar1 = [[2.8, 4.0, 0, 1], [4.0, 30.7, 1, 2], [30.7, 61.2, 2, 3], [62.1, 3.2, 3, 4]]
    #ar1 = [[3.7, 6.9, 0, 1], [6.9, 3.3]]


    #ar1 = [[2.8, 4.0, 0, 1], [4.0, 30.7, 1, 2], [30.7, 61.2, 2, 3], [62.1, 3.2, 3, 4]]
    '''
    ar1 = [[0, 0, 2.8], [0, 1, 4.0], [1, 2, 30.7], [2, 3, 62.1], [3, 4, 3.2]]
    ar1 = [[0, 0, 3.7], [0, 1, 6.9], [1, 2, 3.3], [2, 3, 15.9], [3, 4, 49.5], [4, 5, 20.6]]
    ar1 = [[0, 0, 8.4], [0, 1, 21.8], [1, 2, 69.8]]
    ar1 = [[0, 0, 2.6], [0, 1, 2.1], [1, 2, 4.2], [2, 3, 0.0], [2, 4, 0.7], [4, 5, 0.1], [5, 6, 4.4], [2, 7, 70.3], [7, 8, 17.2], [2, 9, 1.0], [9, 10, 0.1], []]
    '''

    isTerminal = True


    if isTerminal:
        #Terminal is FIRST. [Terminal, parent]
        ar = [[3.2, 62.1], [20.6, 49.5], [69.8, 21.8], [0, 0.1], [0, 0.1], [17.2, 70.3], [4.4, 0.1], [0.0, 4.2]]
        ar = ar + [[9.9, 25.5], [12.3, 25.5], [10.6, 30.7], [13.2, 13.3, 17.7, 23.2], [17.8, 14.8]]
        ar = ar + [[78.1, 13.3], [46.9, 34.4], [0.0, 1.7], [5.2, 8.6, 69.6, 14.9], [21.0, 26.3], [45.0, 29.2]]
        ar = ar + [[22.3, 15.1, 45.4], [29.1, 14.0], [7.5, 34.2], [22.0, 17.4, 6.6, 14.1, 6.0, 3.5, 18.6]]
        ar = ar + [[66.5, 23.7], [37.4, 16.5], [9.6, 36.5], [66.3, 19.9], [55.3, 27.3], [29.8, 45.6], [40.5, 16.4]]
        ar = ar + [[71.7, 16.3], [0.0, 0.0], [8.8, 12.1, 9.0, 7.7, 17.3], [15.0, 12.4, 9.0, 8.5], [34.0, 34.9]]
        ar = ar + [[32.0, 28.6], [1.6, 28.4], [4.3, 44.7], [15.2, 10.9, 8.9, 13.5], [9.3, 21.7], [62.0, 18.1]]
        ar = ar + [[13.5, 17.1, 42.4], [4.0, 46.4], [13.8, 14.5], [12.8, 19.5], [16.3, 13.9, 18.0]]
        ar = ar + [[66.1, 11.3, 8.3], [35.2, 23.0], [20.2, 17.9], [19.1, 18.4, 24.2], [21.4, 21.1, 32.4]]
        ar = ar + [[21.3, 26.3], [25.4, 27.0], [72.0, 17.1], [0.8, 0.5, 7.0], [3.1, 36.4], [21.9, 10.6, 1.7, 5.0, 5.4]]
        ar = ar + [[5.6, 44.8, 33.8], [5.9, 6.5, 27.6], [10.3, 4.8, 23.5], [17.8, 5.8, 6.1, 5.3, 4.9, 33.5]]
        ar = ar + [[7.6, 6.3, 8.1, 48.1], [82.6, 9.3], [7.9, 46.6], [6.5, 54.0], [13.4, 17.4], [67.9, 23.8]]
        ar = ar + [[64.9, 20.7], [49.9, 13.7, 36.4], [76.6, 14.4], [67.2, 20.5], [28.7, 17.4, 5.2, 3.8, 29.9]]
        ar = ar + [[14.2, 40.2], [5.6, 27.6], [24.9, 27.2], [42.1, 26.9], [23.5, 3.5], [32.8, 33.8]]
        ar = ar + [[45.2, 27.1], [17.6, 52.6], [10.1, 49.0], [5.6, 21.9], [15.3, 4.2], [12.9, 11.1]]
        ar = ar + [[40.6, 33.3, 1.4, 6.6, 5.1], [25.0, 22.3, 37.4], [4.7, 56.6], [62.9, 22.0], [34.0, 33.8]]
        ar = ar + [[2.2, 0.7, 38.7], [30.5, 6.6], [15.1, 3.4], [9.6, 50.2], [60.2, 25.6], [3.3, 10.7]]
        ar = ar + [[3.2, 42.6], [12.5, 75.8], [68.1, 22.4], [43.9, 50.8], [19.3, 15.5, 14.2, 13.5, 19.9]]
        ar = ar + [[39.7, 23.6], [18.5, 4.2], [14.0, 23.6], [8.0, 8.8, 7.5, 6.1, 44.0], [12.6, 6.9]]
        ar = ar + [[6.6, 80.9]]

    #quit()






    #quit()

    if not isTerminal:
        ar = []
        ar = ar + [[2.8, 4.0], [3.7, 6.9], [8.4, 21.8], [2.6, 2.1], [10.9, 25.5, 30.7], [14.8, 23.2, 17,8], [8.4, 8.6], [10.2, 8.4], [14.7, 14.9]]
        ar = ar + [[11.5, 15.6], [25.7, 29.2], [17.2, 45.4], [34.2, 15.2, 7.5], [11.7, 18.6], [9.7, 23.7], [36.5, 16.5, 9.6], [7.1, 6.6], [9.9, 17.4]]
        ar = ar + [[13.7, 10.9], [19.1, 24.0], [5.5, 12.0], [8.5, 17.3, 15.0, 12.4, 9.0], [30.9, 34.9], [5.9, 3.5], [17.8, 6.8], [20.5, 21.7]]
        ar = ar + [[9.0, 4.5], [5.3, 4.2], [6.6, 15.9], [23.6, 27.7], [19.4, 18.0], [9.3, 2.6], [25.1, 16.7], [24.2, 17.9, 19.1, 18.4]]
        ar = ar + [[25.1, 32.4], [27.0, 26.3, 25.4], [6.2, 4.6], [6.4, 1.7], [2.4, 15.8], [7.2, 12.6], [1.3, 5.3], [7.1, 2.0], [5.0, 3.1], [28.0, 17.4]]
        ar = ar + [[8.7, 17.4], [8.3, 23.8], [8.6, 5.8], [36.4, 49.9, 13.7], [4.5, 4.6]]
        ar = ar + [[12.3, 20.5], [6.9, 7.9], [26.1, 19.6], [27.2, 27.6, 24.6]]
        ar = ar + [[6.8, 24.2], [50.0, 18.4], [33.3, 33.8]]
        ar = ar + [[12.1, 15.6], [8.6, 6.0]]
        ar = ar + [[2.7, 12.0], [28.2, 11.1], [10.0, 2.9], [5.3, 15.3], [12.5, 6.3]]
        ar = ar + [[15.1, 22.0], [13.5, 16.2], [3.4, 2.7, 15.1], [15.3, 6.8], [25.8, 60.2, 10.7], [10.8, 6.5], [11.7, 75.8], [9.5, 22.4], [5.2, 50.8]]
        ar = ar + [[17.5, 19.9], [23.6, 39.7, 4.2, 14.0], [4.0, 2.2], [1.3, 1.4]]





    if isTerminal:
        rootSizes = []
        firstSizes = []
        ratios = []
        for a in range(len(ar)):
            rootSizes.append(ar[a][-1])
            for b in range(len(ar[a])-1):
                ratios.append(  np.log(ar[a][b]+ 1) - np.log(ar[a][-1] + 1)  )
                firstSizes.append(ar[a][b])

    else:
        rootSizes = []
        firstSizes = []
        ratios = []
        for a in range(len(ar)):
            rootSizes.append(ar[a][0])
            for b in range(1, len(ar[a])):
                ratios.append(  np.log(ar[a][b]+ 1) - np.log(ar[a][0] + 1)  )
                firstSizes.append(ar[a][b])

    ratios = np.array(ratios)
    mean1 = np.mean(ratios)
    sigma = np.mean((ratios - mean1)**2) / (ratios.shape[0] - 1)
    sigma = sigma ** 0.5

    print (mean1, '+-', sigma)

    #print (np.mean(np.array(ratios)))
    #plt.hist(ratios)
    #plt.show()

    min1 = 0
    max1 = max(firstSizes)


    ratioPlot = True


    if ratioPlot:


        plt.hist(ratios, bins=10)
        plt.xlabel('log prevalence ratio')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig('./images/new1/termLogPrevRatio.pdf')
        plt.show()
        quit()


    #print (np.min(rootSizes))
    #print (np.max(rootSizes))
    #quit()

    if isTerminal:
        plt.hist(rootSizes, alpha=0.5, range=(min1, max1))
        plt.hist(firstSizes, alpha=0.5, range=(min1, max1))
        plt.xlabel('prevalence')
        plt.ylabel('count')
        plt.legend(['clone before the terminal clone', 'terminal clone'])
        plt.gcf().set_size_inches(8, 6)
        plt.savefig('./images/new1/terminalPrev.pdf')
        #plt.savefig('./images/rootPrev.pdf')
        #plt.xscale('log')
        plt.show()
    else:
        plt.hist(rootSizes, alpha=0.5, range=(min1, max1))
        plt.hist(firstSizes, alpha=0.5, range=(min1, max1))
        plt.xlabel('prevalence')
        plt.ylabel('count')
        plt.legend(['normal clone', 'clone with one mutation'])
        plt.gcf().set_size_inches(8, 6)
        #plt.savefig('./images/new1/terminalPrev.pdf')
        plt.savefig('./images/new1/rootPrev.pdf')
        #plt.xscale('log')
        plt.show()

#investigateRootPrevalance()
#quit()


def compareTerminalityFitness():


    import scipy

    modelName = 'AML'
    #modelName = 'breast'


    if modelName == 'AML':
        #file1 = './data/realData/breastCancer'
        file1 = './data/realData/AML'
        maxM = 10
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
        M0 = M
    else:
        file1 = './data/realData/breastCancer'
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)
        M0 = M



    M = np.unique(mutationCategory).shape[0] - 2

    totals = np.zeros(M)
    terms = np.zeros(M)

    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

    _, counts1 = np.unique(sampleInverse, return_counts=True)

    for a in range(sampleInverse.shape[0]):
        weight1 = counts1[sampleInverse[a]]
        weight1 = 1.0 / float(weight1)

        length1 = int(treeLength[a])
        tree1 = newTrees[a, :length1]

        exist1 = tree1[:, 1]
        terminal1 = exist1[np.isin(exist1, tree1[:, 0]) == False]

        exist1 = mutationCategory[exist1]
        terminal1 = mutationCategory[terminal1]

        totals[exist1] = totals[exist1] + weight1
        terms[terminal1] = terms[terminal1] + weight1


    termRatio = terms / totals


    _, index1 = np.unique(mutationCategory[:-2], return_index=True)
    uniqueMutation = uniqueMutation[index1]



    #print (M)
    #quit()




    #modelName = 'breast'

    if modelName == 'AML':
        model = torch.load('./Models/realData/savedModel_AML.pt')
        mutationName = np.load('./data/realData/categoryNames.npy')[:-2]
        M = 22
    elif modelName == 'breast':
        model = torch.load('./Models/realData/savedModel_breast.pt')
        mutationName = np.load('./data/realData/breastCancermutationNames.npy')[:-2]
        M = 406
        latentMin = 0.1



    #This creates a matrix representing all of the clones with only one mutation.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability weights and the predicted latent variables.
    pred, xNP = model(X)
    shape1 = pred.shape

    pred2 = pred.reshape((1, -1))

    #This calculates the relative probability of each mutation clone pair. More fit clones will yeild higher probabilities.
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(shape1)


    prob2_np = prob2.data.numpy()

    prob2_sum = np.sum(prob2_np, axis=1)



    fit1 = prob2_sum


    #print (scipy.stats.pearsonr(fit1, termRatio))
    #print (scipy.stats.spearmanr(fit1[termRatio <= 0.5], termRatio[termRatio <= 0.5]))
    #quit()

    #plt.plot(fit1)
    #plt.plot(termRatio)
    #plt.plot(totals)

    if modelName == 'AML':
        #goodNames = ['ASXL1', 'DNMT3A', 'GATA2', 'IDH2', 'NPM1', 'NRAS', 'U2AF1', 'FLT3']
        goodNames = ['ASXL1', 'DNMT3A', 'GATA2', 'IDH2', 'NPM1', 'NRAS', 'FLT3']
    else:
        #goodNames = ['TP53', 'CDH1', 'GATA3', 'MAP3K1', 'PIK3CA', 'KMT2C', 'ESR1', 'ARID1A']
        goodNames = ['TP53', 'CDH1', 'GATA3', 'MAP3K1', 'PIK3CA', 'KMT2C']


    size1 = np.max(fit1) - np.min(fit1)


    plt.scatter(fit1, termRatio)

    for i in range(fit1.shape[0]):
        name = uniqueMutation[i]
        name = name.split('_')[0]
        #print (name)
        if name in goodNames:
            #print ("A")
            pos1 = fit1[i]
            pos2 = termRatio[i]
            #if name == 'ESR1':
            #    pos2 = pos2 * (1-0.03)
            if name == 'NPM1':
                pos1 = pos1 - (size1 * 0.08)
            if name == 'FLT3':
                pos1 = pos1 - (size1 * 0.04)
                #pos2 = pos2 + 0.01

            if modelName == 'breast':
                pos1 = pos1 - (size1 * 0.04)
                if name == 'TP53':
                    pos1 = pos1 - (size1 * 0.02)

            name = '\emph{' + name + '}'

            plt.annotate(name, (pos1 , pos2 + 0.02 ))





    #plt.xscale('log')
    #plt.annotate('hi', (0.2, 0.2))
    plt.xlabel("fitness")
    plt.ylabel('proportion terminal')
    plt.tight_layout()
    if modelName == 'AML':
        plt.savefig('./images/new1/terminalFitness_AML.pdf')
    else:
        plt.savefig('./images/new1/terminalFitness_breast.pdf')
    plt.show()



    #plt.plot(prob2_sum)
    #plt.show()

    quit()



#compareTerminalityFitness()
#quit()


def plotRuntimes():

    #times = [33785.17, 64729.83, 171946.17, 285694.37]

    times = ['1:13:12', '2:19:54', '6:02:03', '10:02:38']


    memory = [400142942, 1140313037, 2339702657, 4605256096]


    times2 = []
    for a in range(len(times)):
        time1 = times[a].split(':')
        time1 = float(time1[0]) + (float(time1[1]) / 60) + (float(time1[2]) / (60*60) )
        times2.append(time1)


    patients = [50, 100, 300, 500]

    times2 = np.array(times2)
    patients = np.array(patients)
    memory = np.array(memory)

    memory = memory / (1024 ** 3)

    times2 = times2 / 20 #instances

    #print (times2[-1] * 60)
    #print (times2[0]*60, ((times2[0] * 60) % 1) * 60)
    #print (times2[-1]*60, ((times2[-1] * 60) % 1) * 60)

    print (memory[0] * 1024)
    print (memory[-1])
    quit()

    plt.plot(patients, times2)
    plt.scatter(patients, times2)
    plt.xlabel('number of patients')
    plt.ylabel('runtime in hours')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/runtimes.pdf')
    plt.show()

    plt.plot(patients, memory)
    plt.scatter(patients, memory)
    plt.xlabel('number of patients')
    plt.ylabel('memory in GB')
    plt.gcf().set_size_inches(8, 6)
    plt.savefig('./images/memoryUsage.pdf')
    plt.show()


#plotRuntimes()
#quit()


def analyzeBreastSamples():


    if False:
        import pandas


        #df = pandas.read_excel('./extra/geneInfo.xlsx')

        sheet_to_df_map = pandas.read_excel('./extra/geneInfo.xlsx', sheet_name=None)


        ar = sheet_to_df_map['Somatic_Mutations'].to_numpy()
        ar = ar[1:, 1:]


        arg1 = np.argwhere(ar[0] == 'HGVSc')[0 ,0]
        arg2 = np.argwhere(ar[0] == 'Tumor_Sample_Barcode')[0, 0]



        ar1 = ar[1:, 0]
        ar2 = ar[1:, arg1]
        ar3 = ar[1:, arg2]


        args1 = np.argwhere(ar1 == 'PIK3CA')[:, 0]
        ar1 = ar1[args1]
        ar2 = ar2[args1]
        ar3 = ar3[args1]

        patientNames = []
        for a in range(ar3.shape[0]):
            name1 = ar3[a].split('-')[1]
            name1 = 'P-' + name1
            patientNames.append(name1)
        patientNames0 = np.array(patientNames)


        argsMutType1 = np.argwhere(np.isin(ar2, np.array(['c.1633G>A', 'c.1624G>A']) ))[:, 0]
        argsMutType2 = np.argwhere( np.isin(ar2, np.array(['c.1633G>A', 'c.1624G>A']) ) == False   )[:, 0]

        patientType1 = patientNames0[argsMutType1]
        patientType2 = patientNames0[argsMutType2]




        nameAndLength = loadnpz('./extra/nameAndLengthBreast.npz')



        file1 = './data/realData/breastCancer'
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, file1, fullDir=True)

        _, sampleInverse = np.unique(sampleInverse, return_inverse=True)


        _, index1 = np.unique(sampleInverse, return_index=True)
        miniTrees = newTrees[index1]


        treeLength2 = nameAndLength[:, 1].astype(int)
        patientNames = nameAndLength[treeLength2<=maxM, 0]


        argMut = np.argwhere(uniqueMutation == 'CDH1')[0, 0]

        boolCDH = np.zeros(miniTrees.shape[0])
        for a in range(boolCDH.shape[0]):
            if argMut in miniTrees[a]:
                boolCDH[a] = 1


        bool1 = boolCDH[np.isin(patientNames, patientType1)]
        bool2 = boolCDH[np.isin(patientNames, patientType2)]

        print (bool1.shape, bool2.shape)
        print (np.sum(bool1))
        print (np.sum(bool2))
        #print (uniqueMutation)
        quit()

        treeLength = treeLength.astype(int)
        treeLength1 = treeLength[index1]

        treeLength2 = nameAndLength[:, 1].astype(int)

        patientNames = nameAndLength[treeLength2<=maxM, 0]



        #c.1633G>A
        #c.1624G>A

        #print (ar2)
        quit()



        print (ar1[:10])
        print (ar2[:10])
        print (ar3[:10])
        quit()


        #np.loadtxt('./extra/1-s2.0-S1535610818303684-mmc2.xlsx')

        ar = df.to_numpy()
        ar = ar[1:, 1:]

        print (ar.shape)

        print (ar[:10, :10])

        quit()



    patientNum = -1
    Pnames = []
    treeNums = []

    with open('./extra/breast_Razavi.txt') as f:
        lines = f.readlines()
        for a in range(len(lines)):
            line1 = lines[a]

            #print (line1)
            #if a  == 500:
            #    quit()

            if 'P-' in line1:
                patientNum += 1
                Pname = 'P-' + line1.split('P-')[1][:-1]
                Pnames.append(Pname)

                treeNum = line1.split(' #trees ')[0]
                treeNum = int(treeNum)

                treeNums.append(treeNum)




    #print (np.unique(treeNums, return_counts=True))
    #quit()

    treeNums = np.array(treeNums)

    #print (treeNums.shape)
    #print (treeNums[treeNums == 1].shape)
    #quit()

    plt.hist(treeNums, bins=100, range=(0, 100))
    plt.show()
    quit()


    Pnames = np.array(Pnames)
    treeNums = np.array(treeNums)

    argsort1 = np.argsort(Pnames)
    Pnames = Pnames[argsort1]
    treeNums = treeNums[argsort1]


    df = pandas.read_excel('./extra/1-s2.0-S1535610818303684-mmc2.xlsx')

    #np.loadtxt('./extra/1-s2.0-S1535610818303684-mmc2.xlsx')

    ar = df.to_numpy()
    ar = ar[2:,1:-1]

    patientID = ar[:, 0]
    sampleID = ar[:, 1]

    #_, counts = np.unique(patientID, return_counts=True)
    #print (np.unique(counts, return_counts=True))

    argIn = np.argwhere(np.isin(patientID, Pnames))[:, 0]
    patientID = patientID[argIn]
    sampleID = sampleID[argIn]


    #_, counts = np.unique(patientID, return_counts=True)

    #print (np.unique(counts, return_counts=True))
    #quit()

    #print (patientID[:10])


    #plt.hist(treeNums[counts == 1], bins=100, range=(0, 100))
    #plt.hist(treeNums[counts == 2], bins=100, range=(0, 100))
    #plt.show()


    #print (counts.shape)
    #print (treeNums.shape)
    #quit()




    print (np.unique(patientID).shape)
    print (np.unique(sampleID).shape)

    patientID_unique = np.unique(patientID)

    _, counts = np.unique(patientID, return_counts=True)




    #for a in range(patientID.shape[0]):
    #    print (patientID[a])

    #print (patientID[:10])
    #print (sampleID[:10])



#analyzeBreastSamples()
#quit()


import sys

if __name__ == "__main__":

    #print (sys.argv)

    #print (sys.argv[1])

    if sys.argv[1] == 'custom':
        if sys.argv[3] == 'train':
            maxM = int(sys.argv[4])
            dataName = sys.argv[2]
            trainPer = float(sys.argv[5])
            trainRealData(dataName, maxM=maxM, trainPer=trainPer)

        if sys.argv[3] == 'plot':
            dataName = sys.argv[2]
            analyzeModel(dataName)

        if sys.argv[3] == 'predict':
            dataName = sys.argv[2]
            probPredictedTrees(dataName)




    if sys.argv[1] == 'test':

        if sys.argv[2] == 'causal':
            if sys.argv[3] == 'train':
                trainNewSimulations(4, 20)
            if sys.argv[3] == 'print':
                testOccurSimulations(4, 20)


        if sys.argv[2] == 'pathway':
            if sys.argv[3] == 'train':
                trainNewSimulations(1, 32)
            if sys.argv[3] == 'evaluate':
                savePathwaySimulationPredictions()
            if sys.argv[3] == 'print':
                testPathwaySimulation()

    if sys.argv[1] == 'recap':
        if sys.argv[2] == 'm5':
            name = 'M5_m5'
        if sys.argv[2] == 'm7':
            name = 'M12_m7'
        if sys.argv[2] == 'm12':
            name = 'M12_m12'

        if sys.argv[3] == 'train':
            trainSimulationModels(name)

        if sys.argv[3] == 'evaluate':
            evaluateSimulations('M5_m5')

        if sys.argv[3] == 'plot':
            if sys.argv[4] == 'cluster':
                doRECAPplot(name, doCluster=True)
            if sys.argv[4] == 'accuracy':
                doRECAPplot(name, doCluster=True)

    if sys.argv[1] == 'real':

        if sys.argv[2] == 'proportion':
            doProportionAnalysis()

        else:

            if sys.argv[3] == 'train':
                if sys.argv[2] == 'leukemia':
                    trainRealData('AML')
                if sys.argv[2] == 'breast':
                    trainRealData('breast')

            if sys.argv[3] == 'plot':
                if sys.argv[2] == 'leukemia':
                    analyzeModel('AML')
                if sys.argv[2] == 'breast':
                    analyzeModel('breast')

            if sys.argv[3] == 'predict':
                if sys.argv[2] == 'leukemia':
                    probPredictedTrees('AML')
                if sys.argv[2] == 'breast':
                    probPredictedTrees('breast')


    #Below are analyses from the paper which can be ran.
    #trainNewSimulations(1, 32)
    #savePathwaySimulationPredictions()
    #testPathwaySimulation()
    #trainRealData('AML')
    #analyzeModel('AML')
    #names1 = ['M5_m5', 'M12_m7', 'M12_m12']
    #trainSimulationModels('M5_m5')
    #evaluateSimulations('M5_m5')
    #doRECAPplot('M5_m5', doCluster=False)
    #trainSimulationModels('M12_m7')
    #evaluateSimulations('M12_m7')
    #doRECAPplot('M12_m7', doCluster=False)
    #trainSimulationModels('M12_m12')
    #evaluateSimulations('M12_m12')
    #doRECAPplot('M12_m12', doCluster=False)
