A demo of running CloMu and making predictions on real data. 

First, either train the model or utilized our pretrained model. To train the model on the breast cancer dataset, run the command:
```bash
python CloMu.py train raw ./data/realData/breastCancer.npy ./model.pt ./prob.npy ./mutationNames.npy 9
```

One can then predict fitness on the breast cancer dataset values using the pretrained model by runnning the command:
```bash
python CloMu.py predict fitness ./Models/realData/savedModel_breast.pt ./fit.csv -csv ./data/realData/breastCancermutationNames.npy
```
Alternatively, on the AML dataset, use the command:
```bash
python CloMu.py predict fitness ./Models/realData/savedModel_AML.pt ./fit.csv -csv ./data/realData/categoryNames.npy
```
If you trained the model using the command provided, then to predict fitness values run the command:
```bash
python CloMu.py predict fitness ./model.pt ./fit.csv -csv ./mutationNames.npy
```

To predict relative causality utilizing the pretrained model, run the command:
```bash
python CloMu.py predict causality relative ./Models/realData/savedModel_breast.pt ./causality.csv -csv ./data/realData/breastCancermutationNames.npy
```



