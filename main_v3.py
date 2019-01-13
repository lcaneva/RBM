"""
Script to make the machine run

To-do list:

    - Analisi delle ricostruzioni tramite medie condizionate anziché tramite un singolo Gibbs Sampling
    - Modificare le chiamate in maniera tale da definire parameters

    - Determinazione della fase della macchina e comparazione con le performance di ricostruzione
    - Confrontare la macchina regolarizzata con quella di default
    - Controllare se la sparsità controlla \tilde{m}. Ripetere dopo aver cambiato dataset
    - Confrontare PCD, CD-1 e CD-10


    - Implementare una grid evaluation per la scelta degli iperparametri
    - Velocizzare ulteriormente gli update usando Theano?
    - Introdurre target sparsity?
    - Calcolo della pseudolikelihood?
    
"""

import os     
import sys    
import numpy as np
from sklearn.model_selection import train_test_split
from pudb import set_trace

curr_dir = os.getcwd()
sys.path.append( os.path.join( curr_dir, 'Modules') )
from Modules.BaseRBM import BaseRBM
from Modules.ReLU_RBM import ReLU_RBM
from Modules.Analyzer import Analyzer
import Modules.Parameters as pms
import Modules.Datasets as dt



############### Initialization
# Initialize numpy prints
np.set_printoptions( precision = 2, suppress= True, linewidth = 1000) 

# Store all hyperparameters in a dictionary
pars = pms.getParameters()


# Compute the size of the minibatches
sizeMB = int( pars['sizeTrain']/ pars['nMB'] )

############### Build or read the dataset
if pars['useMNIST']:         
        dataset = dt.readMNIST()
        
        # Resize the dataset
        dataset = dataset[0:int( (1+pars['ratioTest'])*pars['sizeTrain'])]
else:
    dataset, count_e = dt.buildDataset( pars['N'], pars['l'], pars['sizeTrain'], pars['ratioTest'],\
                                        pars['seedTr'], pars['p_01'], pars['p_10'] )

# Add a dimension to handle biases
dataset = np.insert( dataset, 0, 1, axis = 1)

# Use hold-out technique to avoid overfitting
X_train, X_test = train_test_split( dataset, test_size=pars['ratioTest']/(1.+pars['ratioTest']), random_state = pars['seedTr'])

g_init = np.sum( dataset, axis = 0 )/len(dataset)
    
###############  Learning phase 
# Define a matrix to store the results
fields = ['nC_last', 'nC_train','nC_test','RE_last', 'RE_train', 'RE_test', 'Sparsity',\
        'Theta_mean', 'g_mean', 'L_mean', 'I_mean', 'S_mean', 'm_tilde', 'm_mean', 'sqrt(r)',\
          'h_max', 'Delta', 'Temp', 'W^2' ] 
results = np.zeros( (pars['nRuns'], len(fields)) )

# Repeat the learning n times in order to obtain a more reliable statistics
for k in range( pars['nRuns'] ):

    ############### Define the RBM and make it learn
    if pars['useReLU']:       BM = ReLU_RBM( pars['N'], pars['M'], pars['seedBM'], g_init )
    else:                     BM = BaseRBM(  pars['N'], pars['M'], pars['seedBM'], g_init )
    
    ############### Learn the data model
    MRE_fit, nCorrect_fit, sparsity_fit, ovf = BM.fit( X_train, X_test, pars['LA'], pars['SGS'], pars['nMB'],\
        pars['nEpochs'], pars['epsilon'], pars['alpha'], pars['x'], pars['lambda_x'],pars['c_e'], pars['c_a'],\
        pars['period'], pars['plots'], pars['useProb'] )
    
    # Save the final model
    np.save( os.path.join(curr_dir, 'Results', 'final_weights.npy'), BM.W )
    # Create a list for the plots
    fit_res = [MRE_fit, nCorrect_fit, sparsity_fit, ovf[0,:], ovf[1,:] ]
    
    ############### Analyze the final weights
    analyzer = Analyzer( BM.N, BM.M )
    p, p_vec, T = analyzer.analyzeWeights( BM.W )
        
    ############### Analyze the perfomance of the RBM
    
    # Perfomance on the last mini-batch of the training set 
    print( "=========== Last MB training set results ===========" )
    res_last = analyzer.analyzePerfomance( X_train[-sizeMB:], BM )
    # Perfomance on the training set 
    print( "=========== Training set results ===========" )
    res_train = analyzer.analyzePerfomance( X_train, BM )
    # Perfomance on the test set
    print( "=========== Test set results ===========" )
    res_test = analyzer.analyzePerfomance( X_test, BM )

    print( "*********** Final weights ***********" )    
    print( "Thresholds:", BM.W[0] )
    print( "Max element of BM.W: {:.2f}".format( np.max( np.abs( BM.W.flatten() ) ) ) ) 
    print( "Sparsity: {:.2f}".format( p ) )

    # Concatenate the values that are independent on the type of set considered
    L =  np.append( res_train[3], res_test[3] ) 
    S =  np.append( res_train[4], res_test[4] )
    
    # Average measurements independent on the type of set considered
    m_tilde = (res_train[5]+res_test[5])/2
    m_nonmag = (res_train[6]+res_test[6])/2 
    r = (res_train[7]+res_test[7])/2
    h_max = (res_train[8]+res_test[8])/2
    h_nonmag_max = (res_train[9]+res_test[9])/2
    

    ############### Save the numerical results                
    # Write numerical results for the comparison of test and training data
    results[ k, : ] = np.array( [res_last[2], res_train[2], res_test[2], res_last[1], res_train[1], res_test[1],\
                                p, -np.mean( BM.W[0,1:] ), -np.mean( BM.W[1:,0] ),  np.mean(L), BM.M-np.mean(L)-np.mean(S), np.mean(S),\
                                m_tilde,  m_nonmag, np.sqrt( r ), h_max, h_max-h_nonmag_max,  T,\
                                np.linalg.norm( BM.W[1:,1:] )**2 ] )

    ############### Make plots    
    if pars['plots'] and k == 0:       
        analyzer.makePlots( pars, X_test, fit_res, L, S, BM ) 
        pars['plots'] = False
        
    input( "Continue?" )
        
        
######## File outputs    
analyzer.saveResults( pars, fields, results )
np.save( os.path.join(curr_dir, 'Results', 'results.npy'), results )
analyzer.saveDataset( X_train, X_test, res_train, res_test, BM )
