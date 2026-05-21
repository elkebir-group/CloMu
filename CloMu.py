
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

def loadEither(name):

    if name[-1] == 'z':
        return loadnpz(name)
    else:
        return np.load(name)


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


def processTreeData(maxM, fileIn, mutationFile, infiniteSites=True, patientNames=''):

    #This processes the data from a form using lists of trees to
    #a form using numpy tensors.
    #It also processes the mutation names into mutation numbers.


    #print (fileIn)
    #quit()
    treeData = np.load(fileIn, allow_pickle=True)



    MVal = 100

    # sampleInverse can be seen as an array parallel to newTrees
    # its size is the muber of trees in the dataset with no more than maxM mutations
    # sampleInverse[i] is an integer representing the number of patient to which the tree newTrees[i] is associated
    # i.e., given the tree newTrees[i], the patient to which such a tree is associated is sampleInverse[i]
    sampleInverse = np.zeros(100000)
    treeLength = np.zeros(100000)
    # after the processing, newTrees will have size (total number of trees, maxM, 2)
    # indeed, it will contain all trees in the dataset with no more than maxM edges
    # newTrees[i] will contain all edges for tree i
    # newTrees[i][j] will contain the edge j in tree i, i.e. the pair of mutations (list of size 2)
    newTrees = np.zeros((100000, maxM, 2)).astype(str)
    lastName = 'ZZZZZZZZZZZZZZZZ'
    firstName = 'ZZZZZZZZZZ'
    # sampleInverse can be seen as an array parallel to newTree
    newTrees[:] = lastName

    count1 = 0

    # iterate over the patients: treeData[a] is the patient in position a in the dataset
    for a in range(0, len(treeData)):
        treeList = treeData[a]
        treeList = np.array(list(treeList))

        #print (treeList)
        #quit()

        # treeList.shape[1] is the number of edges of trees associated with patient a
        # remember that all trees associated with a given patient have the same length
        # we are checking whether the trees associated with a given patient have no more than maxM mutations
        if treeList.shape[1] <= maxM:

            # treeList.shape[0] is the number of trees for patient a
            size1 = treeList.shape[0]

            #print (treeList)

            # we are adding to newTrees all trees of patient a if there are less than maxM mutations
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
    # shape1 is (total number of trees, maxM, 2)
    shape1 = newTrees.shape
    # we flatten the array: it becomes a 1D array of mutations
    newTrees = newTrees.reshape((newTrees.size,))



    # uniqueMutation contains all mutations in newTrees exactly once.
    # newTrees contains, for each mutation in uniqueMutation, the index of such a mutation in the previous array newTrees
    uniqueMutation, newTrees = np.unique(newTrees, return_inverse=True)

    #print (infiniteSites)
    #quit()

    uniqueMutation2 = []
    for name in uniqueMutation:
        if infiniteSites:
            name1 = name
        else:
            name1 = name.split('_')[0]
        uniqueMutation2.append(name1)
    uniqueMutation2 = np.array(uniqueMutation2)

    # mutationCategory[i] = i
    # we represent each mutation in uniqueMutation2 with an integer
    uniqueMutation2, mutationCategory = np.unique(uniqueMutation2, return_inverse=True)

    # the mutation names are saved in a file and ZZZZ and ZZZZZZZZ are excluded
    np.save(mutationFile, uniqueMutation2[:-2])

    # M is the number of mutations in the dataset
    M = uniqueMutation.shape[0] - 2

    # now newTrees is a numpy array with shape (total number of trees, maxM, 2) and each mutation name
    # substituted by the index representing the mutation name
    newTrees = newTrees.reshape(shape1)

    if (lastName in uniqueMutation) and (lastName != uniqueMutation[-1]):
        print ("Error in Mutation Name")
        quit()

    if patientNames != '':
        np.save(patientNames, sampleInverse)

    # now sampleInverse[i] is the patient to which tree newTrees[i]
    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

    return newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M


def trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=True, trainSet=False, unknownRoot=False, regularizeFactor=0.02, iterations='default', verbose=False):


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

    nPrint = 1
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
    iterNum = 1000
    #iterNum = 20000
    if nonLin:
        #iterNum = 4000
        iterNum = 10000 #Modified Oct 12 2022
    else:
        iterNum = 1000 #Only for TreeMHN #Standard version sep 22 2022
    #iterNum = 2000
    #iterNum = 4000
    #iterNum = 5000


    if iterations != 'default':
        iterNum = iterations

    for iter in range(0, iterNum):#301): #3000
        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print ('iteration ' + str(iter) + ' of ' + str(iterNum))





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

            if verbose:
                print ("")
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

def trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=True, trainSet=False, unknownRoot=False, regularizeFactor=0.002, iterations='default', verbose=False):


    #This function trains a model to predict the probability of new mutations being added to clones,
    #and therefore the probability of phylogeny trees generated through this process.

    excludeSameMut = True

    doTrainSet = not (type(trainSet) == type(False))

    # total number of trees
    N1 = newTrees.shape[0]

    # total number of patients
    N2 = int(np.max(sampleInverse) + 1)

    #M2 = np.unique(mutationCategory).shape[0]

    #This calculates the test set patients, as well as the training set trees (trainSet2)
    if doTrainSet:
        #trainSet = np.argwhere(np.isin(sampleInverse, trainSet))[:, 0]

        # testSet contains the patients (and not trees) not included in the training set
        # they correspond to the last N2 - trainSize patients
        testSet = np.argwhere(np.isin(np.arange(N2), trainSet) == False)[:, 0]

        # trainSet2 contains all trees of patients included in the training set
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

    print ("The code runs for 1000 iterations.")
    #print ("If required, the code can be stopped early.")
    if verbose:
        print ("The user can stop the code at any time if the testing loss has ")
        print ("converged sufficiently close to the optimum for the user's applicaiton. ")

    iterMax = 1000
    if iterations != 'default':
        iterMax = iterations

    for iter in range(0, iterMax):#301): #3000


        #if True:
        #    baseN = (min(iter, 1000) + 10)

        doPrint = False
        if iter % nPrint == 0:
            doPrint = True

        if doPrint:
            print ('iteration ' + str(iter) + ' of ' + str(iterMax))


        # initialize the edges of the generated trees
        Edges = np.zeros((N1, maxM+1, 2))
        # the root of each tree is the clone M with no acquired mutation
        # M is an integer that is equal to the number of different mutations in the dataset
        Edges[:, 0, 1] = M
        # initialize the clones for each possible phylogeny tree
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
        if True:#adjustProbability:
            baseLineLog = np.log(baseLine)
            #baseLineLog = np.copy(baseLine)


            #print (probLog2[:10])

            #This is the reinforcement learning loss function, before some adjustments for sampling frequency
            loss_array = probLog1 / (torch.exp(probLog2.detach()) + 1e-10)

            loss_array = loss_array / maxM


            sampleUnique, sampleIndex = np.unique(sampleInverse, return_index=True)

            # HERE THEY NORMALIZE PROBABILITIES BY TAKING INTO ACCOUNT THE NUMBER OF TREES THAT EACH PATIENT
            # CONTAINS.
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





        #loss_array = torch.exp( probLog1 - probLog2.detach() )
        #loss_array = loss_array[trainSet2]


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
        regularization = regularization * regularizeFactor
        #regularization = regularization * 0.002 #Used for our occurance simulation as well

        #Adding regularization to the loss
        loss = loss + regularization

        #Printing information about training
        if doPrint:

            #print (baseLine)
            #quit()

            if verbose:
                print ("")
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



def trainModel(inputNameList, modelName, treeSelectionName, mutationName, patientNames='', inputFormat='simple', infiniteSites=True, trainSize='all', maxM=10, regularizeFactor='default', iterations='default', verbose=False):

    if inputFormat == 'raw':
        #maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, inputNameList[0], mutationName, infiniteSites=infiniteSites, patientNames=patientNames)
        #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/realData/AML.npy', './mutationName.npy')
        #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/realData/AML.npy', './mutationName.npy', infiniteSites=infiniteSites)

        #print (sampleInverse.shape)


    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/lungData/processed.npy')
    elif inputFormat == 'multi':

        #print ([inputNameList])

        newTrees = loadnpz(inputNameList[0])
        newTrees = newTrees.astype(int)
        sampleInverse = loadnpz(inputNameList[1]).astype(int)

        M = int(np.max(newTrees) - 1)
        #print (np.max(newTrees))
        #quit()

        #This loads the length of each tree
        treeLength = loadnpz(inputNameList[2])
        treeLength = treeLength[sampleInverse]

        mutationCategory = ''

    #quit()

    if trainSize == 'all':

        #N2 = np.unique(sampleInverse).shape[0]
        #trainSet = np.arange(N2)

        # the training set contains all patients
        trainSet = np.unique(sampleInverse).astype(int)


    elif trainSize == 'half':

        trainSet = np.unique(sampleInverse).astype(int)
        trainSize = trainSet.shape[0] // 2
        trainSet = trainSet[:trainSize]


    else:


        trainSet = np.unique(sampleInverse).astype(int)
        trainSet = trainSet[:trainSize]

        #This code creates a training set test set split using a random state.
        #rng = np.random.RandomState(2)

        #N2 = int(np.max(sampleInverse)+1)

        #trainSet = rng.permutation(N2)

    #print (newTrees.shape, sampleInverse.shape, mutationCategory.shape, treeLength.shape, uniqueMutation.shape)
    #quit()


    #trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave='./banana.pt', baselineSave='./banana.npy', adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=0.015)
    #quit()



    if infiniteSites:
        if regularizeFactor == 'default':
            regularizeFactor = 0.0002
        trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelName, baselineSave=treeSelectionName, trainSet=trainSet, regularizeFactor=regularizeFactor, iterations=iterations, verbose=verbose)
    else:
        if regularizeFactor == 'default':
            regularizeFactor = 0.015
        trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelName, baselineSave=treeSelectionName, adjustProbability=True, trainSet=trainSet, unknownRoot=True, regularizeFactor=regularizeFactor, iterations=iterations, verbose=verbose)
        #trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelName, baselineSave=treeSelectionName, trainSet=trainSet)



    #print (N2)
    #print (np.unique(sampleInverse).shape)
    #quit()
    #quit()


#newTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
#newTrees = newTrees.astype(int)
#sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
#mutationCategory = ''

#This loads the length of each tree
#treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
#treeLength = treeLength[sampleInverse]


#trainModel(['./data/realData/AML.npy'], './temp/model.pt', './temp/prob.npy', './temp/mutationNames.npy', patientNames='', inputFormat='raw', infiniteSites=False, trainSize='all')
#trainModel(['./data/simulations/I-a/T_4_R_0_bulkTrees.npz', './data/simulations/I-a/T_4_R_0_bulkSample.npz', './data/simulations/I-a/T_4_R_0_treeSizes.npz'], './temp/model.pt', './temp/prob.npy', './temp/mutationNames.npy', patientNames='', inputFormat='multi', infiniteSites=True, trainSize='all')



#import tracemalloc

#tracemalloc.start()



#trainModel(['./data/realData/AML.npy'], './temp/model2.pt', './temp/prob2.npy', './temp/mutationNames2.npy', patientNames='', inputFormat='raw', infiniteSites=False, trainSize='all', maxM=10)
#trainModel(['./data/realData/breastCancer.npy'], './temp/modelB.pt', './temp/probB.npy', './temp/mutationNamesB.npy', patientNames='', inputFormat='raw', infiniteSites=True, trainSize='all', maxM=9)


#print(tracemalloc.get_traced_memory())
#tracemalloc.stop()


#quit()

#prob = np.load('./temp/prob.npy')
#print (prob.shape)
#quit()

#import matplotlib.pyplot as plt
#data = np.load('./causality.npy')
#plt.imshow(data)
#plt.show()


def giveAbsoluteCausality(modelFile, saveFile, useCSV, nameFile):



    model = torch.load(modelFile)
    a = 0
    for param in model.parameters():
        if a == 3:
            M = param.shape[0]
        a += 1


    #This prepares M clones, where the ith clone has only mutation i.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability of new mutations on the clones.
    output, _ = model(X)


    #New
    X_normal = torch.zeros((1, M))
    output_normal, _ = model(X_normal)

    output_np = output.data.numpy()

    output_normal = output_normal.data.numpy()
    for b in range(output.shape[1]):
        output_np[:, b] = output_np[:, b] - output_normal[0, b]

    #import matplotlib.pyplot as plt
    #plt.imshow(output_np)
    #plt.show()
    #quit()

    if useCSV:



        sum1 = np.sum(output_np, axis=1)
        diff1 = np.abs(sum1 - np.median(sum1)) / np.median(sum1)
        argsort1 = np.argsort(diff1)[-1::-1]
        argsort1 = argsort1[diff1 > 0.1]

        output_np[np.arange(output_np.shape[0]), np.arange(output_np.shape[0])] = 0

        output_mod = np.empty( (argsort1.shape[0]+1, argsort1.shape[0]+1) , dtype="<U10")
        output_mod = output_mod.astype(str)


        output_mod[1:, 1:] = np.copy(output_np[argsort1][:, argsort1]).astype(str)
        names = np.load(nameFile)[:-2]
        output_mod[0, 1:] = np.copy(names[argsort1])
        output_mod[1:, 0] = np.copy(names[argsort1])

        np.savetxt(saveFile, output_mod, delimiter=",", fmt='%s')
    else:

        np.save(saveFile, output_np)


#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#modelFile = './model.pt'
#saveFile = './causality2.npy'
#output_np = giveAbsoluteCausality(modelFile, saveFile)
#quit()


def giveRelativeCausality(modelFile, saveFile, useCSV, nameFile, meanAdjust=False):



    model = torch.load(modelFile)
    a = 0
    for param in model.parameters():
        if a == 3:
            M = param.shape[0]
        a += 1


    #This prepares M clones, where the ith clone has only mutation i.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability of new mutations on the clones.
    pred, _ = model(X)


    X_normal = torch.zeros((1, M))
    pred_normal, _ = model(X_normal)



    prob_normal = torch.softmax(pred_normal, axis=1)[0].data.numpy()
    pred_normal = pred_normal[0].data.numpy()


    prob = torch.softmax(pred, dim=1)
    prob_np = prob.data.numpy()

    meanAdjust = True
    prob_np_adj = np.copy(prob_np)
    for a in range(len(prob_np_adj)):
        if meanAdjust:
            prob_np_adj[:, a] = prob_np_adj[:, a] / np.mean(prob_np_adj[:, a])
        else:
            prob_np_adj[:, a] = prob_np_adj[:, a] / prob_normal[a]


    prob_np_adj = np.log(prob_np_adj)


    if useCSV:

        effectSize = np.sum(np.abs(prob_np_adj), axis=1)
        argImpact = np.argwhere(effectSize > np.median(effectSize) * 1.1)[:, 0]
        argImpact = argImpact[np.argsort(effectSize[argImpact])[-1::-1]]



        output_mod = np.empty( (argImpact.shape[0]+1, argImpact.shape[0]+1) , dtype="<U10")
        output_mod = output_mod.astype(str)
        output_mod[1:, 1:] = np.copy(prob_np_adj[argImpact][:, argImpact]).astype(str)
        names = np.load(nameFile)[:-2]
        output_mod[0, 1:] = np.copy(names[argImpact])
        output_mod[1:, 0] = np.copy(names[argImpact])

        np.savetxt(saveFile, output_mod, delimiter=",", fmt='%s')
    else:

        np.save(saveFile, prob_np_adj)



#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#output_np = giveRelativeCausality(modelFile, saveFile)
#quit()


def giveLatentRepresentations(modelFile, saveFile, useCSV, nameFile):



    model = torch.load(modelFile)
    a = 0
    for param in model.parameters():
        if a == 3:
            M = param.shape[0]
        a += 1


    #This prepares M clones, where the ith clone has only mutation i.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability of new mutations on the clones.
    _, xNP = model(X)

    for a in range(xNP.shape[1]):
        xNP[:, a] = xNP[:, a] - np.median(xNP[:, a])


    if useCSV:

        names = np.load(nameFile)[:-2]

        import matplotlib.pyplot as plt



        diff1 = np.sum(np.abs(xNP), axis=1)
        argImpact = np.argwhere(diff1 > 0.1)[:, 0]
        argImpact = argImpact[np.argsort(diff1[argImpact])[-1::-1]]

        both = np.concatenate((  names.reshape((-1, 1)) , xNP  ), axis=1)
        both = both.astype(str)[argImpact]

        np.savetxt(saveFile, both, delimiter=",", fmt='%s')

    else:

        np.save(saveFile, xNP)


#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#giveLatentRepresentations(modelFile, saveFile)
#quit()


def giveFitness(modelFile, saveFile, useCSV, nameFile):



    model = torch.load(modelFile)
    a = 0
    for param in model.parameters():
        if a == 3:
            M = param.shape[0]
        a += 1


    #This prepares M clones, where the ith clone has only mutation i.
    X = torch.zeros((M, M))
    X[np.arange(M), np.arange(M)] = 1

    #This gives the predicted probability of new mutations on the clones.
    pred, _ = model(X)

    pred2 = pred.reshape((1, -1))
    prob2 = torch.softmax(pred2, dim=1)
    prob2 = prob2.reshape(pred.shape)
    prob2_np = prob2.data.numpy()

    prob2_sum = np.sum(prob2_np, axis=1)

    #import matplotlib.pyplot as plt
    #plt.plot(prob2_sum)
    #plt.show()

    #print ('hi', useCSV)

    if useCSV:

        names = np.load(nameFile)[:-2]

        diff1 = np.abs(prob2_sum - np.median(prob2_sum)) / np.median(prob2_sum)
        argImpact = np.argwhere(diff1 > 0.1)[:, 0]
        argImpact = argImpact[np.argsort(diff1[argImpact])[-1::-1]]

        both = np.array([names, prob2_sum])[:, argImpact]

        np.savetxt(saveFile, both, delimiter=",", fmt='%s')

    else:
        np.save(saveFile, prob2_sum)



#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#giveFitness(modelFile, saveFile)
#quit()

def giveTreeSelection(probFile, sampleFile, saveFile):

    prob = loadEither(probFile)
    sampleInverse = loadEither(sampleFile)

    predList = []
    unique1 = np.unique(sampleInverse)
    for a in range(len(unique1)):
        args1 = np.argwhere(sampleInverse == unique1[a])[:, 0]
        pred1 = np.argmax(prob[args1])
        predList.append(pred1)

    predList = np.array(predList)

    np.save(saveFile, predList)



def printFitness(saveFile, mutationNameFile):

    fitness = np.load(saveFile)
    names = np.load(mutationNameFile)

    median = np.median(fitness)

    highFit = np.argwhere(fitness > (median * 1.3))[:, 0]

    print ('high fitness mutations:')
    for a in highFit:
        print (names[a])


import sys

if __name__ == "__main__":

    #print (sys.argv)

    #print (sys.argv[1])

    #print (sys.argv[1])

    #newTrees = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkTrees.npz')
    #newTrees = newTrees.astype(int)
    #sampleInverse = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_bulkSample.npz').astype(int)
    #mutationCategory = ''

    #This loads the length of each tree
    #treeLength = loadnpz('./dataNew/specialSim/dataSets/T_' + str(T) + '_R_' + str(a) + '_treeSizes.npz')
    #treeLength = treeLength[sampleInverse]



    if sys.argv[1] == 'train':
        inputFormat = sys.argv[2]
        if inputFormat == 'raw':
            inputFiles = [sys.argv[3]]
            inNum = 4
        if inputFormat == 'multi':
            inputFiles = [sys.argv[3], sys.argv[4], sys.argv[5]]
            inNum = 6

        modelName = sys.argv[inNum]
        probName = sys.argv[inNum+1]
        mutationName = sys.argv[inNum+2]
        maxM = sys.argv[inNum+3]
        maxM = int(maxM)


        infiniteSites = True
        trainSize = 'all'
        regularizeFactor = 'default'
        iterations = 'default'
        inNum2 = inNum+4

        verbose = False

        if len(sys.argv) > inNum2:
            ar = sys.argv[inNum2:]

            if '-noInfiniteSites' in ar:
                infiniteSites = False

            if '-trainSize' in ar:
                arg1 = np.argwhere(np.array(ar) == '-trainSize')[0, 0]
                trainSize = int(ar[arg1+1])

            if '-regularization' in ar:
                arg1 = np.argwhere(np.array(ar) == '-regularization')[0, 0]
                regularizeFactor = float(ar[arg1+1])

            if '-iter' in ar:
                arg1 = np.argwhere(np.array(ar) == '-iter')[0, 0]
                iterations = int(ar[arg1+1])

            if '-verbose' in ar:
                verbose=True


        #print (inputFiles)
        #quit()

        #(patient number file) (infinite sites assumption) (training set size)
        trainModel(inputFiles, modelName, probName, mutationName, patientNames='', inputFormat=inputFormat, infiniteSites=infiniteSites, trainSize=trainSize, maxM=maxM, regularizeFactor=regularizeFactor, iterations=iterations, verbose=verbose)
        #trainModel(['./data/realData/breastCancer.npy'], './temp/model.pt', './temp/prob.npy', './temp/mutationNames.npy', patientNames='', inputFormat='raw', infiniteSites=True, trainSize='all')

    elif sys.argv[1] == 'predict':

        if sys.argv[2] == 'causality':

            if sys.argv[3] == 'absolute':

                modelFile = sys.argv[4]
                saveFile = sys.argv[5]
                useCSV = False
                if len(sys.argv) > 6:
                    if sys.argv[6] == '-csv':
                        useCSV = True
                nameFile = 'False'
                if len(sys.argv) > 7:
                    nameFile = sys.argv[7]

                giveAbsoluteCausality(modelFile, saveFile, useCSV, nameFile)

            if sys.argv[3] == 'relative':

                modelFile = sys.argv[4]
                saveFile = sys.argv[5]
                useCSV = False
                if len(sys.argv) > 6:
                    if sys.argv[6] == '-csv':
                        useCSV = True
                nameFile = 'False'
                if len(sys.argv) > 7:
                    nameFile = sys.argv[7]

                giveRelativeCausality(modelFile, saveFile, useCSV, nameFile)

        if sys.argv[2] == 'fitness':
            modelFile = sys.argv[3]
            saveFile = sys.argv[4]

            useCSV = False
            if len(sys.argv) > 5:
                if sys.argv[5] == '-csv':
                    useCSV = True
            nameFile = 'False'
            if len(sys.argv) > 6:
                nameFile = sys.argv[6]

            giveFitness(modelFile, saveFile, useCSV, nameFile)

        if sys.argv[2] == 'latent':
            modelFile = sys.argv[3]
            saveFile = sys.argv[4]


            useCSV = False
            if len(sys.argv) > 5:
                if sys.argv[5] == '-csv':
                    useCSV = True
            nameFile = 'False'
            if len(sys.argv) > 6:
                nameFile = sys.argv[6]


            giveLatentRepresentations(modelFile, saveFile, useCSV, nameFile)

        if sys.argv[2] == 'selection':
            probFile = sys.argv[3]
            sampleFile = sys.argv[4]
            saveFile = sys.argv[5]
            giveTreeSelection(probFile, sampleFile, saveFile)

    elif sys.argv[1] == 'show':

        if sys.argv[2] == 'fitness':
            saveFile = sys.argv[3]
            mutationNameFile = sys.argv[4]
            printFitness(saveFile, mutationNameFile)
