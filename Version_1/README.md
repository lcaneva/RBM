# RBM
Implementation of a Resticted Boltzmann Machine with ReLU hidden units for Master's thesis.

RBM.py contains the two classes BaseRBM and ReLU_RBM that actually implement the RBMs, while in main.py there is all the auxiliar code, from the construction of the dataset to the I/O interface.

In order to run the code, one needs to pass some important hyperparameters through command line while calling main.py,
e.g. `$ python main.py -nMB 2 -s 200 -lr 1 -lam 5e-3 -p 0`

where:
* nMB = number of mini-batches
* s = size of the training set
* lr = learning rate
* lam = weights cost
* p = bool to activate plots
