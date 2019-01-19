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

    # Dataset to be used
    pars['useMNIST'] = False
    if pars['useMNIST']:
        # Size of the machine    
        pars['N'] = 784
        pars['M'] = 400
    else:
        # Size of the machine    
        pars['N'] = 40
        pars['M'] = 20

        # Length of the categories
        pars['l'] = 10
        # Probabilities for noisy clamping
        if pars['N'] <= 5:
            pars['p_01'] = 0.15
            pars['p_10'] = 0.05
        elif pars['N'] <= 20:
            pars['p_01'] = 0.15
            pars['p_10'] = 0.02
        else:
            if pars['l'] == 1:
                pars['p_01'] = 0.1
                pars['p_10'] = 0.0025
            else:
                pars['p_01'] = 0.01
                pars['p_10'] = 0.005

        pars['invert'] = False


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

    # Seed for the dataset and the machine
    pars['seedTr'] = 0
    pars['seedBM'] = 10

    # Reduce noise sampling in Gibbs Sampling
    pars['useProb'] = True

    # Number of repetitions
    if pars['seedTr'] != None and pars['seedTr'] != None:
        pars['nRuns'] = 1
    else:
        pars['nRuns'] = 5


    # Define and add arguments to the argparse object
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", required=True, help="maximum number of epochs during the learning phase")
    ap.add_argument("-nMB", "--numberMB",required=True, help="number of mini-batches to be used")
    ap.add_argument("-lr", "--epsilon",required=True, help="learning rate")
    ap.add_argument("-lam", "--lambda",required=True, help="weights cost")
    ap.add_argument("-s", "--sizeTrain",required=True, help="size training set")
    ap.add_argument("-p", "--plots",required=True, help="show plots")


    # Create a dictionary with all the specified command line arguments
    args = vars(ap.parse_args())

    # Get the hyperparameters tuned by command line
    pars['nMB'] = int(args["numberMB"])
    pars['sizeTrain'] = int(args["sizeTrain"])
    pars['nEpochs'] = int(args["epochs"])
    pars['epsilon'] = float(args["epsilon"])
    pars['lambda_x'] = float(args["lambda"])
    pars['plots'] = (args["plots"] != "0")


    return pars
