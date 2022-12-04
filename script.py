
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


from CloMu import *


#Prediction of causal relations on Simulation I-a
for R in range(20):
    modelFolder = './Models/simulations/I-a/'
    modelFile = modelFolder + 'T_4_R_' + str(R) + '_model.pt'
    saveFile = './results/simulations/I-a/absoluteCausality_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)

#Prediction of causal relations on Simulation I-b
for R in range(20):
    modelFolder = './Models/simulations/I-b/'
    modelFile = modelFolder + 'T_11_R_' + str(R) + '_model.pt'
    saveFile = './results/simulations/I-b/absoluteCausality_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)

#Prediction of causal relations on Simulation I-c
for R in range(20):
    modelFolder = './Models/simulations/I-c/'
    modelFile = modelFolder + 'T_12_R_' + str(R) + '_model.pt'
    saveFile = './results/simulations/I-c/absoluteCausality_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)

#Tree selection on Simulation I-a
for R in range(20):
    probFolder = './Models/simulations/I-a/'
    probFile = probFolder + 'T_4_R_' + str(R) + '_baseline.pt.npy'
    sampleFolder = './data/simulations/I-a/'
    sampleFile = sampleFolder + 'T_4_R_' + str(R) + '_bulkSample.npz'
    saveFile = './results/simulations/I-a/treeSelection_' + str(R) + '.npy'
    giveTreeSelection(probFile, sampleFile, saveFile)

#Prediction of Latent Representations on Simulation II
for R in range(30):
    modelFolder = './Models/simulations/II/'
    modelFile = modelFolder + 'T_1_R_' + str(R) + '_model.pt'
    saveFile = './results/simulations/II/latentRepresentations_' + str(R) + '.npy'
    giveLatentRepresentations(modelFile, saveFile)


#Analysis of AML data
modelFile = './Models/realData/savedModel_AML.pt'
saveFile = './results/realData/latentRepresentations_AML.npy'
giveLatentRepresentations(modelFile, saveFile)

modelFile = './Models/realData/savedModel_AML.pt'
saveFile = './results/realData/relativeCausality_AML.npy'
giveRelativeCausality(modelFile, saveFile)

modelFile = './Models/realData/savedModel_AML.pt'
saveFile = './results/realData/fitness_AML.npy'
giveFitness(modelFile, saveFile)

#Analysis of Breast Cancer data
modelFile = './Models/realData/savedModel_breast.pt'
saveFile = './results/realData/latentRepresentations_breast.npy'
giveLatentRepresentations(modelFile, saveFile)

modelFile = './Models/realData/savedModel_breast.pt'
saveFile = './results/realData/relativeCausality_breast.npy'
giveRelativeCausality(modelFile, saveFile)

modelFile = './Models/realData/savedModel_breast.pt'
saveFile = './results/realData/fitness_breast.npy'
giveFitness(modelFile, saveFile)



#Causal Prediction on TreeMHN simulations
for R in range(20):
    modelFolder = './Models/simulations/IV/n10_N300/'
    modelFile = modelFolder + str(R) + '_12.pt'
    saveFile = './results/simulations/IV/n10_N300/absoluteCausality_neural_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)

    modelFolder = './Models/simulations/IV/n15_N300/'
    modelFile = modelFolder + str(R) + '_12.pt'
    saveFile = './results/simulations/IV/n15_N300/absoluteCausality_neural_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)

    modelFolder = './Models/simulations/IV/n20_N300/'
    modelFile = modelFolder + str(R) + '_12.pt'
    saveFile = './results/simulations/IV/n20_N300/absoluteCausality_neural_' + str(R) + '.npy'
    giveAbsoluteCausality(modelFile, saveFile)
