
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
        if infiniteSites:
            name1 = name
        else:
            name1 = name.split('_')[0]
        uniqueMutation2.append(name1)
    uniqueMutation2 = np.array(uniqueMutation2)
    uniqueMutation2, mutationCategory = np.unique(uniqueMutation2, return_inverse=True)

    np.save(mutationFile, uniqueMutation2[:-2])

    M = uniqueMutation2[:-2].shape[0]

    '''
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
    '''

    newTrees = newTrees.reshape(shape1)

    if (lastName in uniqueMutation) and (lastName != uniqueMutation[-1]):
        print ("Error in Mutation Name")
        quit()

    if patientNames != '':
        np.save(patientNames, sampleInverse)

    _, sampleInverse = np.unique(sampleInverse, return_inverse=True)

    return newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M


def trainGroupModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=True, trainSet=False, unknownRoot=False, regularizeFactor=0.02):


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

def trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=False, baselineSave=False, usePurity=False, adjustProbability=True, trainSet=False, unknownRoot=False):


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
        if True:#adjustProbability:
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


    #print (N2)
    #print (np.unique(sampleInverse).shape)
    ##quit()
    #quit()

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


def trainModel(inputNameList, modelName, treeSelectionName, mutationName, patientNames='', inputFormat='simple', infiniteSites=False, trainSize='all'):

    if inputFormat == 'raw':
        maxM = 9
        newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, inputNameList[0], mutationName, infiniteSites=infiniteSites, patientNames=patientNames)


    #newTrees, sampleInverse, mutationCategory, treeLength, uniqueMutation, M = processTreeData(maxM, './data/lungData/processed.npy')


    if trainSize == 'all':

        #N2 = np.unique(sampleInverse).shape[0]
        #trainSet = np.arange(N2)
        trainSet = np.unique(sampleInverse).astype(int)

    else:


        trainSet = np.unique(sampleInverse).astype(int)
        trainSet = trainSet[:trainSize]

        #This code creates a training set test set split using a random state.
        #rng = np.random.RandomState(2)

        #N2 = int(np.max(sampleInverse)+1)

        #trainSet = rng.permutation(N2)

    #print (newTrees.shape)
    #quit()

    trainModelTree(newTrees, sampleInverse, treeLength, mutationCategory, M, maxM, fileSave=modelName, baselineSave=treeSelectionName, trainSet=trainSet)



    #print (N2)
    #print (np.unique(sampleInverse).shape)
    #quit()
    #quit()


#trainModel(['./data/realData/breastCancer.npy'], './temp/model.pt', './temp/prob.npy', './temp/mutationNames.npy', patientNames='', inputFormat='raw', infiniteSites=True, trainSize='all')
#quit()

#prob = np.load('./temp/prob.npy')
#print (prob.shape)
#quit()


def giveAbsoluteCausality(modelFile, saveFile):



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

    np.save(saveFile, output_np)


#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#output_np = giveAbsoluteCausality(modelFile, saveFile)



def giveRelativeCausality(modelFile, saveFile, meanAdjust=False):



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


    prob_np_adj = np.copy(prob_np)
    for a in range(len(prob_np_adj)):
        if meanAdjust:
            prob_np_adj[:, a] = prob_np_adj[:, a] / np.mean(prob_np_adj[:, a])
        else:
            prob_np_adj[:, a] = prob_np_adj[:, a] / prob_normal[a]


    prob_np_adj = np.log(prob_np_adj)

    np.save(saveFile, prob_np_adj)



#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#output_np = giveRelativeCausality(modelFile, saveFile)
#quit()


def giveLatentRepresentations(modelFile, saveFile):



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

    np.save(saveFile, xNP)


#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#giveLatentRepresentations(modelFile, saveFile)
#quit()


def giveFitness(modelFile, saveFile):



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

    np.save(saveFile, prob2_sum)



#modelFile = './Models/simulations/I-a/T_' + str(4) + '_R_' + str(0) + '_model.pt'
#giveFitness(modelFile, saveFile)
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



import sys

if __name__ == "__main__":

    #print (sys.argv)

    #print (sys.argv[1])

    #print (sys.argv[1])

    if sys.argv[1] == 'train':
        inputFormat = sys.argv[2]
        if inputFormat == 'raw':
            inputFiles = [sys.argv[3]]
            inNum = 4

        modelName = sys.argv[inNum]
        probName = sys.argv[inNum+1]
        mutationName = sys.argv[inNum+2]

        #(patient number file) (infinite sites assumption) (training set size)
        trainModel(inputFiles, modelName, probName, mutationName, patientNames='', inputFormat=inputFormat, infiniteSites=True, trainSize='all')
        #trainModel(['./data/realData/breastCancer.npy'], './temp/model.pt', './temp/prob.npy', './temp/mutationNames.npy', patientNames='', inputFormat='raw', infiniteSites=True, trainSize='all')


    '''
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
    '''

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
