# CloMu
CloMu is a neural network based software for cancer phylogeny analysis.

Dependencies: pytorch, numpy.

# How to use CloMu:

CloMu.py contains the CloMu software to be used on any data set. 

analysis.py contains specific analysis code used to write the CloMu paper. 


To train a model, run the code:

python CloMu.py train (input format) (input files) (model file) (tree probability file) (mutation name file) (maximum tree length) (optional arguements) 

As an example, one can run the below code:

python CloMu.py train raw ./data/realData/breastCancer.npy ./model.pt ./prob.npy ./mutationNames.npy 9

To be specific, "input format" can either be "raw" or "simple". If using raw, "input files" should be the name of the input file with a list of all trees for all patients. "model file" should be the location you want the model to be stored. "tree probability file" should be the location you want to put the predicted probability for each tree in the input data. "mutation name file" should be the name of the file where you want to store the ordered list of mutation names in your data set. "maximum tree length" should be the maximum tree size you want to analyze. Setting it below the length of the longest tree in your data will simply remove patients with a longer tree sizes. "optional arguements" are additional optional inputs you can add. 

One optional arguement is "-noInfiniteSites", to disable the infinite sites assumption. Below is an example of running the code with this.

python3 CloMu.py train raw ./data/realData/AML.npy ./temp/model.pt ./temp/prob.npy ./temp/mutationNames.npy 10 -noInfiniteSites










 


