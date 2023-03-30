A demo of running CloMu on the breast cancer dataset. 

First, either train the model or utilized our pretrained model. To train the model, run the command:
```bash
python CloMu.py train raw ./data/realData/breastCancer.npy ./model.pt ./prob.npy ./mutationNames.npy 9
```

To predict fitness values using the saved model, run the command:
```bash
python CloMu.py predict fitness ./model.pt ./fit.csv -csv ./mutationNames.npy
```
If you trained the model using the command provided, then to predict fitness values run the command:
```bash
python CloMu.py predict fitness ./Models/realData/savedModel_breast.pt ./fit.csv -csv ./data/realData/breastCancermutationNames.npy
```
