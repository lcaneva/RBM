"""
Script to make the machine run
"""

import os, sys, time
curr_dir = os.getcwd()
sys.path.append( os.path.join( curr_dir, 'Modules') )

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pudb import set_trace

from Modules.BaseRBM import BaseRBM
from Modules.ReLU_RBM import ReLU_RBM
from Modules.Analyzer import Analyzer
import Modules.Datasets as dt
import Modules.Parameters as Parameters


############### Initialization
# Initialize numpy prints
np.set_printoptions( precision = 2, suppress= True, linewidth = 1000) 

# Store all hyperparameters in a dictionary
pars = Parameters.getParameters()

# Print a recap of the selected hyperparameters
print( "========== Chosen parameters ==========" )
print( pars )
time.sleep(1.5)

# Compute the size of the minibatches
sizeMB = int( pars['sizeTrain']/ pars['nMB'] )
sizeTot = int( np.ceil( (1.0+pars['ratioTest'])*pars['sizeTrain'] ) )

############### Build or read the dataset
if pars['dataset'] == 'MNIST':         
        dataset = dt.readMNIST( sizeTot )
elif pars['dataset'] == 'CAL':
         X_train, X_test = dt.readCAL()
         dataset = np.vstack( (X_train, X_test) )
elif pars['dataset'] == 'GEP':
        dataset = dt.buildGEP( pars['N'], pars['l'], sizeTot, pars['seedTr'],\
                               pars['p_01'], pars['p_10'], pars['invert'] )
elif pars['dataset'] == 'SB':
        dataset = dt.buildSB( pars['N'], pars['l'], sizeTot, pars['seedTr'] ) 
elif pars['dataset'] == 'BES':
        dataset = dt.buildBES( pars['N'], sizeTot,  pars['seedTr'] )
            
# Add a dimension to handle biases
dataset = np.insert( dataset, 0, 1, axis = 1)

# Use hold-out technique to avoid overfitting
X_train, X_test = train_test_split( dataset, test_size=pars['ratioTest']/(1.+pars['ratioTest']), random_state = pars['seedTr'])

# Estimate mean activation of visible units from data
g_init = np.sum( dataset, axis = 0 )/len(dataset)

# Define the names of the numerical results that must be saved
fields_perf = ['nC_last','nC_train','nC_test','nC_cond_test', 'MRE_test', 'MRE_cond_test'] 
fields_weights = ['Sparsity','Theta_mean', 'g_mean']
fields_confs = ['L_mean', 'I_mean', 'S_mean', 'm_tilde', 'm_nmg', 'sqrt(r)', 'q_GS', 'Temp', 'W^2' ] 
    
l1 = len(fields_perf)
l2 = len(fields_weights)
l3 = len(fields_confs)

# Initialize a numpy array to store the numerical results
results = np.zeros( (pars['nRuns'], l1+l2+l3) )

###############  Learning phase 
# Repeat the learning n times in order to obtain a more reliable statistics
for k in range( pars['nRuns'] ):

    ############### Define the RBM and make it learn
    if pars['useReLU']:       BM = ReLU_RBM( pars['N'], pars['M'], pars['seedBM'], pars['theta_0'], g_init )
    else:                     BM = BaseRBM(  pars['N'], pars['M'], pars['seedBM'],  g_init )
    
    ############### Learn the data model        
    MRE, nCorrect, sparsity, ovf = BM.fit( X_train, X_test, pars['LA'], pars['SGS'], pars['nMB'],\
        pars['nEpochs'], pars['epsilon'], pars['alpha'], pars['x'], pars['lambda_x'],pars['c_e'], pars['c_a'],\
        pars['period'], pars['plots'], pars['useProb'] )
    
    # Save the final model
    np.save( os.path.join(curr_dir, 'Results', pars['phase'] + '_' + 'final_weights.npy'), BM.W )
    
    # Create a list for the plots
    fit_series = [MRE, nCorrect, sparsity, ovf[0,:], ovf[1,:] ]
    
    # Create an Analyzer instance
    analyzer = Analyzer( BM.N, BM.M, pars['typeDt'] )
    
    ############### Analyze the perfomance of the RBM
    # Perfomance on the last mini-batch of the training set 
    print( "=========== Last MB training set results ===========" )
    perf_last, __, __, __ = analyzer.analyzeStates( X_train[-sizeMB:], BM )

    # Perfomance on the training set 
    print( "=========== Training set results ===========" )
    perf_train, df_train, counts_train, __ = analyzer.analyzeStates( X_train, BM )

    # Perfomance on the test set
    print( "=========== Test set results ===========" )
    perf_test, df_test, counts_test, q = analyzer.analyzeStates( X_test, BM, cond=True, overlaps=int(len(X_test)/2) )

    q_vis_layer = np.mean( np.mean( BM.v, axis= 0 ) )

    ############### Analyze the final weights
    print( "========== Final weights ==========" )    
    p, p_2, p_vis, p_hid, T = analyzer.analyzeWeights( BM.W )    
    print( "Thresholds:", BM.W[0] )
    print( "Biases: ", BM.W[:,0 ] )
    print( "Max element of BM.W: {:.2f}".format(  np.max( np.abs( BM.W.flatten() ) ) ) )
    print( "Sparsity: {:.2f}".format( p ) )
    print( "Sparsity (full PR): {:.2f}".format( p_2 ) )
    
    ############### Save the numerical results                
    # Concatenate the two measurments dataframes 
    # (since the measures are independent of the type of set considered, i.e. training or test )
    df_tot = pd.concat( [df_train, df_test] )
    avgs   = df_tot.mean( axis = 0 )
    
    # Add column suffixes
    perf_last  = perf_last.add_suffix('_last')
    perf_train = perf_train.add_suffix('_train')
    perf_test  = perf_test.add_suffix('_test')

    # Concatenate the three performance dataframes
    perf = pd.concat( [perf_last, perf_train, perf_test], axis=1 )

    # Write numerical results for the comparison of test and training data    
    results[ k, :l1 ] = perf[fields_perf].values
    results[ k, l1:l1+l2] = np.array([p, -np.mean( BM.W[0,1:] ), -np.mean( BM.W[1:,0] )])
    results[ k, l1+l2:l1+l2+l3] = np.array([avgs['L'], BM.M-avgs['L']-avgs['S'], avgs['S'],avgs['m_t'], avgs['m_nmg'],\
        np.sqrt( avgs['r'] ), q_vis_layer, T, np.linalg.norm( BM.W[1:,1:] )**2 ] )

    ############### Make plots    
    if k == 0:

        # Save overlaps to allow the comparison
        np.save( os.path.join( curr_dir, 'Results', pars['phase'] + '_' + 'overlap.npy'), q )

        # Save fit results to allow the comparison
        np.save( os.path.join( curr_dir, 'Results', pars['phase'] + '_' + 'fit_series.npy'), fit_series )

        # Make plots
        analyzer.makePlots( pars, X_test, fit_series, df_tot['L'].values, df_tot['S'].values, q, BM ) 

        # Save dataset distribution
        analyzer.saveDataset( X_train, X_test, counts_train['corr'].values, counts_train['wrong'].values,BM, pars['phase'] + '_' +'training_set.csv' )
        analyzer.saveDataset( X_test, X_train, counts_test['corr'].values, counts_test['wrong'].values,BM, pars['phase'] + '_' +'test_set.csv' )

        pars['plots'] = False

        
######## File outputs    
fields = fields_perf + fields_weights + fields_confs
analyzer.saveResults( pars, fields, results )
np.save( os.path.join(curr_dir, 'Results', pars['phase'] + '_' + 'results.npy'), results )


