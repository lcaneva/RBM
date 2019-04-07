# Implementation of a Resticted Boltzmann Machine with ReLU hidden units for Master's thesis.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> ee69104... Update README
The folder Modules contains all the objects and functions used in the main script, subdivided by functionality and class. 
In particular:
- <code>AbstractRBM</code> contains the implementation of all the generic operations of RBMs (initialization, GibbsSampling,  fit, Annealed Importance Sampling for the approximation of the partition function, etc.) but it requires the specification of the updateVisible and updateHidden methods, allowing the arbitrary choice of the types of units constituting the layers of the machines. 
- The child classes <code>BaseRBM</code> and <code>ReLU RBM</code> redefines the two abstract methods and have a free energy function to monitor the overfitting during the learning phase. 
<<<<<<< HEAD
=======
The folder Modules contains all the objects and functions used in the main script, subdivided by functionality and class. In particular:
- <code>AbstractRBM</code> contains the implementation of all the generic operations of RBMs (initialization, GibbsSampling,  fit, Annealed Importance Sampling for the approximation of the partition function, etc.) but it requires the specification of the updateVisible and updateHidden methods, allowing the arbitrary choice of the types of units constituting the layers. 
- The child classes <code>BaseRBM</code> and <code>ReLU RBM</code> hence redefines these two methods and have a free energy function to monitor the overfitting during the learning phase. 
>>>>>>> d60fb6a... Update README.md
=======
>>>>>>> ee69104... Update README
- <code>Parameters.py</code> specifies all the hyperparameters of the model that are usually fixed, while the fine-tuned ones are obtained from command line.
- <code>Datasets.py</code> contains the functions needed for the generation of the Encoder problem instances or for reading the MNIST dataset. 
- The experimental analysis is performed through the <code>Analyzer</code> class, which takes also into account the I/O interface.

<<<<<<< HEAD
<<<<<<< HEAD
In order to run the code, one needs to specify the most important hyperparameters by command line while calling <code>main.py</code>.
E.g.
	      `$ python main.py -nMB 2 -s 200 -lr 1 -lam 5e-3 -p 0`

=======
The folder Modules contains all the objects and functions used in the main script, subdivided by functionality and class. In particular, Abstract RBM contains the implementation of all the generic operations of RBMs (initialization, GibbsSampling,  fit, Annealed Importance Sampling for the approximation of the partition function, etc.) but it requires the specification of the updateVisible and updateHidden methods, allowing the arbitrary choice of the types of units constituting the layers. 
The child classes BaseRBM and ReLU_RBM hence redefines these two methods and have a free energy function to monitor the overfitting during the learning phase. 
Parameters specifies all the hyperparameters of the model that are usually fixed, while the fine-tuned ones are obtained from command line.
Datasets contain the functions needed for the generation of the Encoder problem instances or for the reading of the MNIST dataset. 
The experimental analysis is performed through the Analyzer class, which takes also into account the I/O interface.

In order to run the code, one needs to pass the most hyperparameters through command line while calling main.py,
e.g. `$ python main.py -nMB 2 -s 200 -lr 1 -lam 5e-3 -p 0`
>>>>>>> ff9b5c9... Update modular version
=======
In order to run the code, one needs to pass the most important hyperparameters through command line while calling <code>main.py</code>, e.g. <br />
=======
In order to run the code, one needs to specify the most important hyperparameters by command line while calling <code>main.py</code>.
E.g.
>>>>>>> ee69104... Update README
	      `$ python main.py -nMB 2 -s 200 -lr 1 -lam 5e-3 -p 0`

>>>>>>> d60fb6a... Update README.md

where:
* nMB = number of mini-batches
* s = size of the training set
* lr = learning rate
* lam = weights cost
* p = bool to activate plots
