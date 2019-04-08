import argparse

"""
Function to set and get all the needed hyperparameters.
-----------------------------------------------------------------------
"""
def getParameters():
    # Define an empty dictionary to store all the hyperparameters    
    pars = dict()

    # Define hyper-parameters that will be often fixed
    # Type of RBM to be used (False = BaseRBM, True = ReLU)
    pars['useReLU'] = True
    
    # Bool used to flip the bits of the examples
    pars['invert'] = False

    # Seed used for the dataset
    pars['seedTr'] = 10
    
    # Seed used for the machines
    pars['seedBM'] = None

    # Number of repetitions
    pars['nRuns'] = 30
    
    # Reduce noise sampling in Gibbs Sampling
    pars['useProb'] = True
    
    # Bool to deactivate the plots
    pars['plots'] = True
    
    # Learning algorithm
    pars['LA'] = "CD"

    # Steps Gibbs Sampling 
    pars['SGS'] = 1

    # Sampling period grahps of the learning phase
    pars['period'] = 25

    # Momentum coefficient
    pars['alpha'] = 0

    # Regularization parameter 
    pars['x'] = 3

    # Epsilon decaying coefficient
    pars['c_e'] = 0

    # Alpha growth coefficient
    pars['c_a'] = 0

    # Percentage of the dataset that will form the training set
    pars['ratioTest'] = 1.                
        


    # Define and add arguments to the argparse object
    ap = argparse.ArgumentParser()
    ap.add_argument("-dt", "--dataset", required=True, help="type of dataset to be used")
    ap.add_argument("-N", "--N", required=True, help="number of visible units")
    ap.add_argument("-M", "--M", required=True, help="number of hidden units")

    ap.add_argument("-e", "--epochs", required=True, help="maximum number of epochs during the learning phase")
    ap.add_argument("-nMB", "--numberMB",required=True, help="number of mini-batches to be used")
    ap.add_argument("-lr", "--epsilon",required=True, help="learning rate")
    ap.add_argument("-lam", "--lambda",required=True, help="weights cost")
    ap.add_argument("-s", "--sizeTrain",required=True, help="size training set")
    if pars['useReLU']:
        ap.add_argument("-t", "--theta_0",required=True, help="initial ReLU thresholds")
        ap.add_argument("-ph", "--phase",required=True, help="name to save the plots")

    # Create a dictionary with all the specified command line arguments
    args = vars(ap.parse_args())

    pars['dataset'] = args["dataset"]
    pars['N'] = int(args["N"])
    pars['M'] = int(args["M"])


    # Set types of dataset elements ('M'=Matrix, 'V'=Vector) and other properties
    if pars['dataset'] == 'MNIST' or pars['dataset'] == 'BES' or pars['dataset'] == 'CAL':        
        pars['typeDt'] = 'M'
    elif pars['dataset'] == 'GEP' or pars['dataset'] == 'SB':
        # Type of elements
        pars['typeDt'] = 'V'

        # Length of the categories
        pars['l'] = int(pars['N']/4)
                        
        # Set probabilities of noisy clamping, if using GEP
        if pars['dataset'] == 'GEP':
            
            if pars['l'] == 1:
                pars['p_01'] = 0.1
                pars['p_10'] = 0.0025
            elif pars['l'] == 10:
                pars['p_01'] = 0.01
                pars['p_10'] = 0.005
            elif pars['l'] == 20:
                pars['p_01'] = 0.005
                pars['p_10'] = 0.0025
            elif pars['l'] == 30:
                pars['p_01'] = 0.0025
                pars['p_10'] = 0.00125
            else:
                pars['p_01'] = 0.001
                pars['p_10'] = 0.0005
                

    # Get the hyperparameters tuned by command line
    pars['nMB'] = int(args["numberMB"])
    pars['sizeTrain'] = int(args["sizeTrain"])
    pars['nEpochs'] = int(args["epochs"])
    pars['epsilon'] = float(args["epsilon"])
    pars['lambda_x'] = float(args["lambda"])
    if pars['useReLU']:    
        pars['theta_0'] = float( args["theta_0"] )
        pars['phase'] = args["phase"]
    else:
        pars['theta_0'] = 0
        pars['phase'] = 'Base'
        

    return pars
