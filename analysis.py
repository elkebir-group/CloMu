
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


def loadnpz(name, allow_pickle=False):

    #This simple function more easily loads in compressed numpy files.

    if allow_pickle:
        data = np.load(name, allow_pickle=True)
    else:
        data = np.load(name)
    data = data.f.arr_0
    return data

class MutationModel(nn.Module):
    def __init__(self, M):
        super(MutationModel, self).__init__()

        self.M = M
        L = 5
        #L = 10




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

    T = 12

    #T = 1
    #dataSet = 9
    for dataSet in range(0, 20):
        print (dataSet)

        bulkTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkTrees.npz').astype(int)
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_bulkSample.npz').astype(int)
        treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_treeSizes.npz')
        #treeLength = np.zeros(int(np.max(sampleInverse))) + 5

        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(dataSet) + '_prob.npz')
        #print (probabilityMatrix)
        #quit()



        newFileLines = [  ['Patient_ID', 'Tree_ID', 'Node_ID', 'Mutation_ID', 'Parent_ID'] ]

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




        np.savetxt('./treeMHN/data/input/' + str(T) + '/' + str(dataSet) + '_trees.csv', newFileLines, delimiter=",", fmt='%s')

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


def doChoice(x):

    #This is a simple function that selects an option from a probability distribution given by x.


    x = np.cumsum(x, axis=1) #This makes the probability cummulative

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

def bulkFrequencyPossible(freqs, M):

    #This code determines the set of trees that are consistent with bulk frequency measurements.
    #It does this by determining whether or not a tree is constistent with the frequency measurement
    #for every possible tree of the correct length.

    clones = loadnpz('./data/allTrees/clones/' + str(M) + '.npz')

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

    trees = loadnpz('./data/allTrees/edges/' + str(M) + '.npz')
    trees = trees[argsGood[:, 1]][:, :, :2]

    #print (trees[0])
    #quit()

    return trees, sampleInverse

def simulationBulkFrequency(clones, M, S):

    #This function takes in clones, a the number of mutations, and the number
    #of samples, and (1) finds frequency measurements for those clones and
    #(2) calculates the set of possible trees for those measurements.

    #print (clones[0])
    clones_sum = np.sum(clones, axis=1)
    clones_argsort = np.argsort(clones_sum, axis=1)
    clones_argsort = clones_argsort[:, -1::-1]
    #print (clones_argsort[0])
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


    freqs = freqs.reshape((N, S, M))

    trees, sampleInverse = bulkFrequencyPossible(freqs, M) #This calculates the possible trees from the bulk frequency measurement.

    sampleInverse_reshape = sampleInverse.repeat(trees.shape[1]).repeat(trees.shape[2]).reshape(trees.shape)

    trees_copy = np.copy(trees)
    trees[trees == 100] = -1
    trees = clones_argsort[sampleInverse_reshape, trees]
    trees[trees_copy == 100] = 100

    #print (clones_argsort[0])
    #print (trees[0])
    #quit()

    return trees, sampleInverse

def multisizeBulkFrequency(clones, treeSizes, S):

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

        #print (clonesNow[0])
        #print (uniqueSizes[a])

        treesNow, sampleInverseNow = simulationBulkFrequency(clonesNow, uniqueSizes[a], S)

        #if uniqueSizes[a] == 6:
        #    print (clonesNow[0])
        #    print (treesNow[0])
        #    quit()
        #print (treesNow[0])
        #quit()

        treesNow2 = np.zeros((treesNow.shape[0], maxSize, 2))
        treesNow2[:] = 101
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





        #This computation determines which mutation types are present in the clone.
        clonesNow = np.matmul(clonesNow, mutationTypeMatrix_extended)
        clonesNow[:, K] = 1
        clonesNow[clonesNow > 1] = 1


        if useRequirement:
            #if "useRequirement" is true, this makes it so effects of mutations work by satisfying mutation requirements of effects,
            #rather than working addatively. In other words, having two mutations of the same effect won't do anything additional if one is satisfactory.
            #However, having two mutations that are required together won't do anything if they are not both together.

            clonesNow = np.matmul(clonesNow, reqMatrix)
            clonesNow[clonesNow < 1] = 0
            clonesNow[clonesNow >= 1] = 1





        #This calculates the probability of mutations of certian types
        #given which types of mutations already exist on the clone.
        clonesNow = np.matmul(clonesNow, probabilityMatrix)
        clonesNow[clonesNow > maxLogProb] = maxLogProb
        #This gives the probability of new mutations given the probability of mutation types,
        #and which mutations are of each mutation type.
        clonesNow = np.matmul(clonesNow, mutationTypeMatrix.T)


        if doNonClone:
            clonesNow = clonesNow + interfere



        clonesNow = clonesNow.reshape((N, treePos*M))

        clonesNow_max = np.max(clonesNow, axis=1)
        clonesNow_max = clonesNow_max.repeat(clonesNow.shape[1]).reshape(clonesNow.shape)
        clonesNow = clonesNow - clonesNow_max
        clonesNow = np.exp(clonesNow)

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

        #This determines which clone the mutation will be added to
        clonePoint = (choicePoint // M).astype(int)
        #This determnes which mutation will be added to the clone.
        mutPoint = (choicePoint % M).astype(int)

        #This adds to the tree the fact that mutation mutPoint is added to the clone clonePoint
        edges[:, a+1, 1] = np.copy(mutPoint)
        edges[:, a+1, 0] = np.copy(edges[np.arange(N), clonePoint, 1])

        #This adds the new clone to the list of clones.
        cloneSelect = np.copy(clones[np.arange(N), clonePoint])
        cloneSelect[np.arange(N), mutPoint] = 1
        clones[:, a+1] = np.copy(cloneSelect)

    #This remove the trivial edge associated with the clone with no mutations.
    edges = edges[:, 1:]

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

#treeSizes = loadnpz('./dataNew/specialSim/dataSets/T_' + str(12) + '_R_' + str(0) + '_treeSizes.npz')
#print (np.unique(treeSizes))
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


def trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False, regularizeFactor=0.02):


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

    nonLin = True

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
        iterNum = 1000 #Only for TreeMHN #Standard version sep 22 2022
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

def trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=False, trainSet=False, unknownRoot=False):


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


    model = MutationModel(M)
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


    for iter in range(0, 1000):#301): #3000


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






        probLog1_np = probLog1.data.numpy()
        probLog2_np = probLog2.data.numpy()

        #This adjusts the probabiltiy baseline for each tree.
        baseLine = baseLine * ((baseN - 1) / baseN)
        baseLine = baseLine + ((1 / baseN) * np.exp(probLog1_np - probLog2_np)   )

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
        regularization = regularization * 0.0002 #Best for breast cancer
        #regularization = regularization * 0.002 #Used for our occurance simulation as well

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
    if dataName == 'manual':
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
    if dataName == 'manual':
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_manual_PartialTrain_ex.pt', baselineSave='./Models/baseline_manual_Partial.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.015)#, regularizeFactor=0.005)#, regularizeFactor=0.01)
    elif dataName == 'breast':
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_breast_ex.pt', baselineSave='./Models/baseline_breast_ex.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
    else:
        #trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_' + dataName + '.pt', baselineSave='./Models/baseline_' + dataName + '.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True)
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./Models/savedModel_' + dataName + '.pt', baselineSave='./Models/baseline_' + dataName + '.npy',
                            adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.0005) #0.00005 #0.0005 #too small 0.00001
    #'''



#trainRealData('manual')
#quit()


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
                                        adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.0005 * 2)# * 0.2) #0.00005 #0.0005 #too small 0.00001

                    #2 for 10
                    #quit()

#trainMHNsim()
#quit()





def trainNewSimulations(T, N):

    #This function trains models on the new simulated data sets formed by this paper.


    for a in range(0, N):


        maxM = 5

        if T == 0:
            M = 10
        if T == 1:
            M = 20

        if T in [3, 4, 7, 10, 13, 14]:
            M = 10
            maxM = 7

        if T in [5, 6]:
            M = 20
            maxM = 7

        if T in [8, 11]:
            M = 10
            maxM = 7

        if T in [9, 12]:
            M = 15
            maxM = 7

        #Loading in the saved data sets
        newTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
        newTrees = newTrees.astype(int)
        sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
        mutationCategory = ''

        treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
        treeLength = treeLength[sampleInverse]



        _, sampleIndex = np.unique(sampleInverse, return_index=True)


        #Creating a training set test set split
        rng = np.random.RandomState(2)

        N2 = int(np.max(sampleInverse)+1)
        #trainSet = np.random.permutation(N2)
        trainSet = rng.permutation(N2)

        #trainSet = np.arange(N2)
        trainSet = trainSet[:N2//2]


        #Preparing the files to save the model and tree probabilities
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        baselineFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_baseline.pt'

        #Training the model
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelFile, baselineSave=baselineFile, adjustProbability=True, trainSet=trainSet, unknownRoot=True)


#trainNewSimulations(9, 20)
#trainNewSimulations(10, 20)
#trainNewSimulations(11, 20)
#trainNewSimulations(12, 20)
#trainNewSimulations(13, 20)
#trainNewSimulations(14, 20)
#quit()



def testOccurSimulations(T, N):

    import matplotlib.pyplot as plt

    #This tests the models ability to determine causal relationships in
    #the simulated data set of causal relationships

    #T = 4
    #T = 0
    #N = 100

    #N = 20

    M = 10

    if T in [9, 12]:
        M = 15

    errorList = []


    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    #for a in range(20, N):
    for a in range(0, N):

        #This matrix is the set of true probability of true causal relationships
        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_prob.npz')

        probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0

        #This matrix is a binary matrix of true causal relationships
        prob_true = np.zeros((M, M))
        prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])
        prob_true[prob_true > 0.01] = 1


        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
        model = torch.load(modelFile)

        #This prepares M clones, where the ith clone has only mutation i.
        X = torch.zeros((M, M))
        X[np.arange(M), np.arange(M)] = 1

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


        #import matplotlib.pyplot as plt
        #plt.imshow(probabilityMatrix)
        #plt.show()
        #plt.imshow(output_np)
        #plt.show()

        #import matplotlib.pyplot as plt

        #plt.imshow(probabilityMatrix)
        #plt.show()

        #quit()


        #This is currently disabled, but if enabled it allows the model to know
        #ahead of time exactly how many causal relationships there are.
        if False:

            output_np[np.arange(M), np.arange(M)] = -10
            output_np_flat = output_np.reshape((output_np.size,))

            num_high = int(np.sum(prob_true))

            cutOff = np.sort(output_np_flat)[-num_high]


            output_np_bool = np.copy(output_np) - cutOff
            output_np_bool[output_np_bool >= 0] = 1
            output_np_bool[output_np_bool < 0] = 0



        #This converts the output probabilities into a binary of predicted causal relationships
        #by using a cut off probability.
        if True:
            if T == 4:
                #output_np = output_np - (np.max(output_np) * 0.2) #0.4 #Before Oct 13 2021
                output_np = output_np - 1.1
                False
            elif T == 7:
                output_np = output_np - (np.max(output_np) * 0.35) #0.4 #Before Oct 13 2021
            else:
                output_np = output_np - (np.max(output_np) * 0.4)
            output_np[np.arange(M), np.arange(M)] = 0
            output_np_bool = np.copy(output_np)
            output_np_bool[output_np_bool > 0] = 1
            output_np_bool[output_np_bool < 0] = 0

        #plt.imshow(output_np_bool)
        #plt.show()
        #quit()


        categories_update = np.zeros((2, 2))

        #This uses the predicted causal relationships and real causal relationships to
        #determine the rate of true positives, false positives, true negatives and false negatives
        for b in range(output_np.shape[0]):
            for c in range(output_np.shape[1]):
                if b != c:
                    categories[int(prob_true[b, c]), int(output_np_bool[b, c])] += 1
                    categories_update[int(prob_true[b, c]), int(output_np_bool[b, c])] += 1

        errorList.append([categories_update[1, 1], categories_update[0, 0], categories_update[0, 1], categories_update[1, 0] ])


    #This prints the final information on the accuracy of the method.
    print ('True Positives: ' + str(categories[1, 1]))
    print ('True Negatives: ' + str(categories[0, 0]))
    print ('False Positives: ' + str(categories[0, 1]))
    print ('False Negatives: ' + str(categories[1, 0]))

    errorList = np.array(errorList).astype(int)
    if T == 4:
        np.save('./plotResult/cloMuCausal.npy', errorList)
        True

#testOccurSimulations(7, 20)
#testOccurSimulations(4, 20)
#testOccurSimulations(8, 20)
#testOccurSimulations(14, 1)
#quit()



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


#testOccurFitness(4, 20)
#quit()




def compareOccurSimulations(T, N):


    def simple_calculateErrors(cutOff, prob_true, theta, mask1):




        pred = theta[mask1 == 1].flatten()
        trueVal = prob_true[mask1 == 1].flatten()

        pred_bool = np.copy(pred)
        pred_bool[pred_bool < cutOff] = 0
        pred_bool[pred_bool >= cutOff] = 1

        TruePos = np.argwhere(np.logical_and(pred_bool == 1, trueVal == 1)).shape[0]
        predTrue = np.argwhere(pred_bool == 1).shape[0]
        realTrue = np.argwhere(trueVal == 1).shape[0]

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

    #Nsim = 20#20#4#20


    #T = 4
    #T = 0
    #N = 100

    #N = 20

    if T in [4, 8, 10, 11, 13, 14]:
        M = 10

    if T in [9, 12]:
        M = 15

    errorList = []






    categories = np.zeros((2, 2)).astype(int)
    #categories = np.array([[1600, 2], [0, 198]]).astype(int)

    cutOffs_MHN = [0, 1e-5, 0.0001, 0.001, 0.01, 0.1]
    cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    if T == 11:
        cutOffs_MHN = [0, 1e-5, 0.0001, 0.001]
        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    if T == 4:
        cutOffs_MHN = [0, 1e-5, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5]
        cutOffs = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8]

    if T == 14:
        cutOffs = [1.2, 1.5, 1.7, 2.0, 2.1, 2.2]


    baseline_acc = np.zeros((N, len(cutOffs_MHN), 2))
    our_acc = np.zeros((N, len(cutOffs), 2))

    #for a in range(20, N):
    for a in range(0, N):

        if not T in [13, 14]:
            theta = np.loadtxt('./TreeMHN/data/output/' + str(T) + '/' + str(a) + '_MHN_sim.csv', delimiter=",", dtype=str)
        else:
            theta = np.loadtxt('./TreeMHN/data/output/' + str(4) + '/' + str(a) + '_MHN_sim.csv', delimiter=",", dtype=str)
        theta = theta[1:].astype(float)
        theta = theta.T


        #This matrix is the set of true probability of true causal relationships
        probabilityMatrix = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_prob.npz')
        #probabilityMatrix[np.arange(probabilityMatrix.shape[1]), np.arange(probabilityMatrix.shape[1])] = 0
        #probabilityMatrix[probabilityMatrix > 0.01] = 1

        if T in [4, 7, 8, 9, 13, 14]:
            probabilityMatrix[probabilityMatrix > 0.01] = 1
            probabilityMatrix[probabilityMatrix < 0.01] = 0
        else:
            probabilityMatrix[probabilityMatrix < -0.01] = -1
            probabilityMatrix[probabilityMatrix > 0.01] = 1
            probabilityMatrix[np.abs(probabilityMatrix) < 0.01] = 0

        #print(probabilityMatrix)

        #quit()

        if T in [4, 10, 13, 14]:
            #This matrix is a binary matrix of true causal relationships
            prob_true = np.zeros((M, M))
            prob_true[:5, :5] = np.copy(probabilityMatrix[:5, :5])
            #prob_true[prob_true > 0.01] = 1
        elif T in [8, 11]:
            prob_true = probabilityMatrix[:5, :5]

            prob_true = prob_true[np.arange(M) // 2]
            prob_true = prob_true[:, np.arange(M) // 2]

        elif T in [9, 12]:
            prob_true = probabilityMatrix[:5, :5]

            prob_true = prob_true[np.arange(M) // 3]
            prob_true = prob_true[:, np.arange(M) // 3]

        #print (prob_true)



        #This loads in the model
        modelFile = './dataNew/specialSim/results/T_' + str(T) + '_R_' + str(a) + '_model.pt'
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



        #print (output_np)
        #plt.imshow(output_np)
        #plt.show()
        #quit()


        mask1 = np.ones(output_np.shape)
        mask1[np.arange(M), np.arange(M)] = 0

        #import matplotlib.pyplot as plt
        #plt.imshow(output_np)
        #plt.show()

        #output_np[output_np>0.01] = 1
        #output_np[output_np<-0.01] = -1

        #print (output_np[:3, :3])
        #print (prob_true[:3, 3])
        #print (np.sum(np.abs(output_np - prob_true)))
        #quit()


        #plt.imshow(probabilityMatrix)
        #plt.show()

        for cutOff0 in range(len(cutOffs_MHN)):

            cutOff = cutOffs_MHN[cutOff0]

            if T in [4, 13, 14]:
                precision1, recall1 = simple_calculateErrors(cutOff, prob_true, theta, mask1)
            else:
                #precision1, recall1 = calculateErrors(cutOff, prob_true, theta, mask1)
                precision1, recall1 = calculateErrors(cutOff, prob_true, theta, mask1)

            #print (precision1, recall1)

            baseline_acc[a, cutOff0, 0] = precision1
            baseline_acc[a, cutOff0, 1] = recall1

        for cutOff0 in range(len(cutOffs)):

            cutOff = cutOffs[cutOff0]

            if T in [4, 13, 14]:
                precision1, recall1 = simple_calculateErrors(cutOff, prob_true, output_np, mask1)
            else:
                precision1, recall1 = calculateErrors(cutOff, prob_true, output_np, mask1)

            print (precision1, recall1)

            our_acc[a, cutOff0, 0] = precision1
            our_acc[a, cutOff0, 1] = recall1



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


    plt.plot(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    #plt.plot(np.mean(baseline_acc[:, :, 1], axis=0), np.mean(baseline_acc[:, :, 0], axis=0))
    #plt.scatter(np.mean(our_acc[:, :, 1], axis=0), np.mean(our_acc[:, :, 0], axis=0))
    plt.xlabel("recall")
    plt.ylabel("precision")
    #plt.legend(['CloMu', 'TreeMHN'])
    plt.title(title1)
    #plt.savefig('./images/' + imgName )
    #plt.savefig('./images/betweenClone.pdf')
    plt.show()



#compareOccurSimulations(14, 20)
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



#singleTreePrediction(revolver=False)
#quit()




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

#name1 = "./treeMHN/treeMHNdata/CSV/n10_N100_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1/predNo_0_1.csv"
#name1 = "./treeMHN/treeMHNdata/CSV/n10_N100_s0.5_er0.5_MC50_M300_gamma1.00_norder4_cores125_subsampling200_threshold0.95_v1/predNo_0.csv"
#theta = np.loadtxt(name1, delimiter=",", dtype=str)
#N = int(theta.shape[1] ** 0.5)
#theta = theta.reshape((theta.shape[0], N, N))
#print (theta.shape)
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


#plotCausalSimulation()
#quit()


def plotSelectionSimulation():


    x1 = np.load('./plotResult/cloMuSelect.npy')
    x2 = np.load('./plotResult/recapSelect.npy')
    x3 = np.load('./plotResult/revolverSelect.npy')

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

    sns.stripplot(data=df, x="Number of Pathways",
              y="Tree Selection Accuracy", hue="Method",
              hue_order=methods,
              alpha=.4, dodge=True, linewidth=1, jitter=.1,)
    sns.boxplot(data=df, x="Number of Pathways",
                y="Tree Selection Accuracy", hue="Method",
                hue_order=methods, showfliers=False)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles[0:len(methods)], labels[0:len(methods)])
    plt.xlabel('')
    plt.show()

#plotSelectionSimulation()
#quit()


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
    if dataName == 'manual':
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

    simName = "Pathway"
    #simName = 'Causal'

    if simName == "Pathway":
        Tset = [5]
        N = 30
        maxVal = 50
        saveName = './images/treeSizePathway.pdf'
    else:
        Tset = [4, 11, 12]
        N = 20
        maxVal = 30
        saveName = './images/treeSizeCausal.pdf'


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

            sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz')
            _, counts = np.unique(sampleInverse, return_counts=True)

            countList = countList + list(counts)


    plt.hist(countList, bins=50, range=(0, maxVal))
    plt.gcf().tight_layout()
    plt.xlabel("number of possible trees")
    plt.ylabel('number of patients')
    #if
    plt.savefig(saveName)

    # plt.show()
    quit()

#plotNumbrOfTrees()
#quit()


def analyzeModel(modelName):

    #This function does an analysis of the model trained on a data set,
    #creating plots of fitness, causal relationships, and latent representations.

    print ("analyzeModel")

    import matplotlib.pyplot as plt

    if modelName == 'manual':
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


#analyzeModel("manual")
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




    if modelName == 'manual':
        model = torch.load('./Models/savedModel_manual_allTrain.pt')
        #model = torch.load('./Models/savedModel_manual_oct12_8pm.pt')
        #model = torch.load('./Models/savedModel_manual_allTrain2.pt')
        #model = torch.load('./Models/savedModel_manual_PartialTrain.pt')
        mutationName = np.load('./data/categoryNames.npy')[:-2]

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
        model = torch.load('./Models/savedModel_breast.pt')
        mutationName = np.load('./data/mutationNamesBreastLarge.npy')[:-2]
        M = 406
        #M = 365
        #latentMin = 0.01
        latentMin = 0.1

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
    np.save('./dataNew/interestingMutations_' + modelName + '.npy', argsInteresting)


    if False:
        #plt.plot(xNP[argsInteresting][np.argsort(xNP[argsInteresting, 0])])

        #This plots the latent parameters of the mutations
        plt.plot(xNP)
        if modelName == 'manual':
            plt.ylim(-1.6)

        #plt.title("mutation properties")
        plt.xlabel("mutation")
        plt.ylabel("latent variable value")
        plt.legend(['comp.~1', 'comp.~2', 'comp.~3', 'comp.~4', 'comp.~5'], ncol=2)


        #This finds the mutations with substantial enough properties
        #that they should be annotated, and annotates them with the mutation name.
        #argsHigh = np.argwhere(latentSize > 0.15)[:, 0]

        argsHigh = np.argwhere(latentSize > 0.02)[:, 0]


        #print (argsHigh.shape)
        #quit()

        for i in argsHigh:
            name = mutationName[i]

            delt1 = np.max(xNP) / 100
            max0 = np.max(np.abs(xNP[i])) + (delt1 * 4)
            sign1 = np.sign(xNP[i][np.argmax(np.abs(xNP[i]))] )
            max1 = (max0  * sign1) - (delt1 * 3)

            ############plt.annotate(name, (i -  (M / 20), np.max(xNP[i]) + (np.max(xNP) / 100)    ))
            plt.annotate(name, (i -  (M / 40), max1    ))




        plt.tight_layout()
        plt.savefig('./images/LatentPlot_' + modelName + '.pdf')
        plt.show()

        #quit()


    #
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
    mutationNamePrint = ['NPM1', 'ASXL1', 'DNMT3A', 'NRAS', 'FLT3', 'IDH1', 'PTPN11', 'FLT3-ITD']
    mutationNamePrint = np.array(mutationNamePrint)

    ar1 = np.array([prob2_sum, mutationName]).T
    ar1 = ar1[np.isin(mutationName, mutationNamePrint)]

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
    argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.5)[:, 0]
    #argsHigh = np.argwhere(prob2_sum > np.median(prob2_sum) * 1.2)[:, 0]



    if modelName == 'manual':
        #argsGood = np.argwhere(np.isin(mutationName, mutationNamePrint))[:, 0]
        argsGood = np.argwhere(prob2_sum < np.median(prob2_sum) * 0.75)[:, 0]
        argsHigh = np.concatenate((argsHigh, argsGood))


    #print (np.sort(prob2_sum))
    #quit()

    argsHigh = np.argwhere(latentSize > 0.02)[:, 0]

    if False:
        print (argsHigh.shape)
        #quit()

        #This plots the relative fitness of all of the mutations in the data set.
        plt.plot(prob2_sum, c='r')#, yscale="log")
        plt.scatter( argsHigh, prob2_sum[argsHigh], c='r' )
        plt.ylabel('fitness')
        plt.xlabel('mutation')
        if modelName == 'manual':
            plt.yscale('log')

        # plt.gca().yaxis.set_major_locator(MultipleLocator(1))
                # ax.yaxis.set_major_locator(MultipleLocator(1))
        # plt.gca().set_yscale('log')
        for i in argsHigh:
            name = mutationName[i]
            #################plt.annotate(name, (i -  (M / 20), prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
            plt.annotate(name, (i , prob2_sum[i] + (np.max(prob2_sum) / 100)    ))
        plt.tight_layout()
        plt.savefig('./images/fitnessPlot_' + modelName + '.pdf')
        plt.show()

        #quit()



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

    #print (mutationName)

    arg1 = np.argwhere(mutationName == 'ASXL1')[0, 0]
    arg2 = np.argwhere(mutationName == 'NPM1')[0, 0]
    arg3 = np.argwhere(mutationName == 'NRAS')[0, 0]
    #arg2 = np.argwhere(mutationName == 'FLT3-ITD')[0, 0]
    #arg3 = np.argwhere(mutationName == 'NPM1')[0, 0]
    print (prob_np_adj[arg1, arg3])
    print (prob_np_adj[arg2, arg3])
    #print (prob_np_adj[arg1, arg3])
    #print (prob_np_adj[arg3, arg1])
    quit()



    argsHighElse = np.copy(argsHigh)
    argTP = np.argmax(prob2_sum)
    argsHighElse = argsHighElse[argsHighElse!=argTP]
    argsLowish = np.argwhere(prob2_sum <= np.median(prob2_sum) * 1.5)[:, 0]

    #print (np.median(prob_np_adj[argTP, argsHighElse]))
    #print (np.median(prob_np_adj[argTP, argsLowish]))
    #quit()





    if modelName == 'breast':
        reorder = np.array([4, 0, 1, 2, 3])
    elif modelName == 'manual':

        reorder = np.array([1, 0, 5, 6, 3, 7, 2, 4])
        #True
        #reorder = np.arange(argsInteresting.shape[0])

    doBlank = False
    if doBlank:
        arange1 = np.arange(M)
        argExtra = np.argwhere(np.isin(arange1, argsInteresting) == False)[:, 0]
        #reorder = np.array([4, 0, 1, 2, 3])
        argsInteresting = np.concatenate((argsInteresting[reorder], argExtra))



    prob_np_adj_inter = prob_np_adj[argsInteresting][:, argsInteresting]

    #prob_np_adj_inter = np.log(prob_np_adj_inter)





    #else:
    #    reorder = np.arange(prob_np_adj_inter.shape[0])

    if doBlank:
        reorder = np.arange(prob_np_adj_inter.shape[0])


    #reorder = np.arange(prob_np_adj_inter.shape[0])


    prob_np_adj_inter = prob_np_adj_inter[reorder]
    prob_np_adj_inter = prob_np_adj_inter[:, reorder]

    arange1 = np.arange(prob_np_adj_inter.shape[0])
    prob_np_adj_inter[arange1, arange1] = 0

    # [DNMT3A, ASXL1, NPM1, NRAS, GATA2, U2AF1] for luekemia


    if False:

        vSize = np.max(np.abs(prob_np_adj_inter))
        vmin = vSize * -1
        vmax = vSize

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
        plt.xlabel("target mutation $t$")
        plt.ylabel('source mutation $s$')
        plt.colorbar()

        plt.xticks(rotation = 90)
        plt.tight_layout()
        plt.savefig('./images/allOccurancePlot_' + modelName + '.pdf')
        plt.show()

        quit()

    if True:

        vSize = np.max(np.abs(prob_np_adj_inter))
        vmin = vSize * -1
        vmax = vSize

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
        plt.xlabel("target mutation $t$")
        plt.ylabel('source mutation $s$')
        plt.colorbar()
        ax.set_yticks(np.arange(argsInteresting.shape[0]))
        ax.set_yticklabels(mutationName[argsInteresting][reorder])

        ax.set_xticks(np.arange(argsInteresting.shape[0]))
        ax.set_xticklabels(mutationName[argsInteresting][reorder])


        plt.xticks(rotation = 90)
        plt.tight_layout()
        plt.savefig('./images/occurancePlot_' + modelName + '.pdf')
        plt.show()



#newAnalyzeModel("manual")
#quit()

################trainRealData('manual')#, trainPer=1.0) #around 150 rounds to converge for breast cancer
################quit()


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
                    trainRealData('manual')
                if sys.argv[2] == 'breast':
                    trainRealData('breast')

            if sys.argv[3] == 'plot':
                if sys.argv[2] == 'leukemia':
                    analyzeModel('manual')
                if sys.argv[2] == 'breast':
                    analyzeModel('breast')

            if sys.argv[3] == 'predict':
                if sys.argv[2] == 'leukemia':
                    probPredictedTrees('manual')
                if sys.argv[2] == 'breast':
                    probPredictedTrees('breast')


    #Below are analyses from the paper which can be ran.
    #trainNewSimulations(1, 32)
    #savePathwaySimulationPredictions()
    #testPathwaySimulation()
    #trainRealData('manual')
    #analyzeModel('manual')
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
