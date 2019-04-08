# Implementation of Resticted Boltzmann Machines with ReLU hidden units for Master's thesis.

The folder Modules contains all the objects and functions used in the main script, subdivided by functionality and class. In particular:
- <code>AbstractRBM</code> contains the implementation of all the generic operations of RBMs (initialization, GibbsSampling,  fit, Annealed Importance Sampling for the approximation of the partition function, etc.) but it requires the specification of the updateVisible and updateHidden methods, allowing the arbitrary choice of the types of units constituting the layers. 
- The child classes <code>BaseRBM</code> and <code>ReLU RBM</code> redefines the two abstract methods and have a free energy function to monitor the overfitting during the learning phase. 
- <code>Parameters.py</code> specifies all the hyperparameters of the model that are usually fixed, while the fine-tuned ones are obtained from command line.
- <code>Datasets.py</code> contains the functions needed for the generation of the Encoder problem instances or for reading the MNIST dataset. 
- The experimental analysis is performed through the <code>Analyzer</code> class, which takes also into account the I/O interface. In order to run the code, one needs to specify the most important hyperparameters by command line while calling <code>main.py</code>, e.g:
	      `$ python main.py -dt GEP -N 40 -M 20  -nMB 50 -s 5000 -e 200 -lr 0.05 -lam 5e-3 -t 0 -ph FM` <br>
where:
        - dt = type of dataset (Generalized Encoder Problem, Bars&Stripes, MNIST etc.)
        - N = number of visible units
        - M = number of hidden units
	- nMB = number of mini-batches
	- s = size of the training set
	- e = number of epochs for the training phase
	- lr = learning rate
	- lam = weights cost	
        - t = initial value of the biases of the hidden units
        - ph = prefix string to save the results
