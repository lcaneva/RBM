"""
Script to make the machine run

To-do list:
    - Controllo dell'apprendimento attraverso la log-likelihood log P (v) anziché tramite energia libera. Utilizzo
    dell'algoritmo Annealed Importance Sampling per approssimare la funzione di partizione Z


    - Spostare i grafici fit() fuori, se possibile
    - Cambiare i segni di g_i e \theta_{\mu} (solo nella rappresentazione?); si potrebbe introdurre una reference W_t e una W
      della matrice estesa o non
    - Cercare CHANGE?

    - Implementare e confronto Persistent Contrastive Divergence


    - Controllare se la sparsità controlla \tilde{m}. Ripetere dopo aver cambiato dataset
    - Cercare dei valori buoni per epsilon nel caso 40-20
    - Reintrodurre alpha e vedere come varia la perfomance
    - Confrontare la macchina regolarizzata con quella di default
    - Determinazione della fase della macchina e comparazione con le performance di ricostruzione
    
    ----------------------------------------------------------------------------------------------------------    
    - Cambiare il nome della funzione GibbsSampling
    - Analisi delle ricostruzioni tramite medie condizionate anziché tramite un singolo Gibbs Sampling
    - Calcolare l'attivazione media delle hidden unit al variare delle epoche
    - Togliere i commenti inutili 
    - Introduzione target sparsity ?
    - Come si calcola la pseudolikelihood?
    - Introduzione Hamming distance ?

"""

from RBM_matrix import BaseRBM, ReLU_RBM
import datetime
import csv
from collections import Counter
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pudb import set_trace
import seaborn as sns; sns.set(); sns.set_style("ticks")    

############### Initialization
# Define hyper-parameters that will be often fixed
# Type of RBM to be used (False = BaseRBM, True = ReLU)
useReLU = True
# Size of the machine
N = 40
M = 20
# Length of the categories
l = 1
# Probabilities for noisy clamping
if N <= 5:
    p_01 = 0.15
    p_10 = 0.05
else:
    if l == 1:
        p_01 = 0.1
        p_10 = 0.0025
    else:
        p_01 = 0.1
        p_10 = 0.010


# Learning algorithm
LA = "CD"
# Steps Gibbs Sampling 
SGS = 1
# Sampling period during the learning
period = 50
# Momentum coefficient
alpha = 0
# Epsilon decaying coefficient
c_e = 2.0/3
# Alpha growth coefficient
c_a = 1.0 + 1.0/10
# Percentage of the dataset that will form the training set
ratioTrain = 0.5
# Seed for the dataset and the machine
seedTr = 0
seedBM = 0
# Number of repetitions
if seedTr != None and seedBM != None:
    n = 1
else:
    n = 5

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
nMB = int(args["numberMB"])
sizeTrain = int(args["sizeTrain"])
nEpochs = int(args["epochs"])
epsilon = float(args["epsilon"])
lambda_x = float(args["lambda"])
plots = (args["plots"] != "0")

# Compute the size of the minibatches, if it is unique
try: 
    if sizeTrain % nMB != 0:
        raise ValueError
    else:
        sizeMB = int( sizeTrain / nMB )
except:
    print('Error: sizeTrain must be a multiple of the number of minibatches')
    raise SystemExit

############### Build the dataset for the autoencoder
# Define the dataset of "environmental vectors", using the noisy clamping
# as specified in the article of Ackley et al 
def buildDataset( N, l, seedTr, p_01, p_10 ):
    
    # If seed is not None, fix the dataset
    np.random.seed( seedTr )
    
    # Build the different categories
    l_X = int(N/l)
    X = np.zeros( (l_X, N), dtype = float )
    for i in range( l_X ):
        X[i, i*l:(i+1)*l] = np.ones( l ) 

    # Compute the size of the dataset
    sizeTot = int( np.ceil( 1/ratioTrain*sizeTrain ) )
    
    # Obtain the dataset, applying noisy clamping
    dataset = np.zeros( (sizeTot,N), dtype = float )            
    maxIter = int( np.ceil( sizeTot/ l_X ) ) 
    for i in range( maxIter ):
        # Take a random permutation of the rows of the X matrix
        for j in np.random.permutation( l_X ):
        #for j in range( l_X ):
            if (i*l_X + j) >=sizeTot: continue
        
            x = np.copy( X[ j ] )

            # Noisy clamping
            y = np.random.uniform(low=0, high=1,size=N)
            for k in range( N ):
                if x[k] == 1 and y[k] <= p_01: x[k] = 0
                elif x[k] == 0 and y[k] <= p_10: x[k] = 1
 
            dataset[i*l_X+j] =  x            

    # Add a dimension to handle biases
    dataset = np.insert( dataset, 0, 1, axis = 1)
    
    return dataset


dataset = buildDataset( N, l, seedTr, p_01, p_10 )

# Use hold-out technique to avoid overfitting
X_train, X_test = train_test_split( dataset, test_size=(1-ratioTrain), random_state = seedTr)

if plots:
    plt.matshow( X_train[:,1:] ) 
    plt.show() 


###############  Learning phase 
# Define a matrix to store the results
fields = ['nC_last', 'nC_train','nC_test', 'MRE_last', 'MRE_train', 'MRE_test', 'p','T', 'L_mean', 'S_mean', 'm_tilde' ] 
results = np.zeros( (n, len(fields)) )

# Repeat the learning n times in order to obtain a more reliable statistics
for k in range( n ):

    ############### Define the RBM and make it learn
    if useReLU:
        BM = ReLU_RBM( N, M, seedBM )
    else:
        BM = BaseRBM( N, M, seedBM )

    W_init = np.copy(BM.W)
    
    # Iterate through the mini-batches
    for i in range( nMB ):
        print("***************Learning {}-th MB***************\n".format(i+1))
        BM.fit( X_train[i*sizeMB:(i+1)*sizeMB], X_test, LA, SGS, epsilon, alpha, lambda_x, nEpochs, c_e, c_a, period, plots )

    ############### Analyze the final weights 
    p, p_vec, T = BM.analyzeWeights()
        
    ############### Analyze the perfomance of the RBM
    def analyzePerfomance( X ):
        size = len(X)
        L_arr = np.zeros( size )
        S_arr = np.zeros( size )
        # Magnetizations matrix 
        magnet = np.empty( (size, M) )
        m_t = np.empty( size )
        
        # Describe the set according to the unique rows
        X_red, indices  = np.unique( X, axis = 0, return_inverse = True)
        # Arrays to store the errors and right reconstructions
        corr = np.zeros( len(X_red) )
        wrong = np.zeros( len(X_red) )
        
        # Iterate through the set, but taking into account the repetitions
        k = 0  
        MRE  = 0
        
        ## DEBUG
        #np.set_printoptions(precision=2 )#, threshold = 10)
        #print( BM.W ) 
        
        for i in indices:
            
            # Obtain the reconstruction given by the machine
            BM.GibbsSampling( v_init = X_red[i], SGS = 1 )

            # Compute its distance from the real visible state
            dist = np.linalg.norm( X_red[i] - BM.v )**2
            if dist == 0:
                corr[i] += 1 
            else:
                wrong[i] += 1 

            MRE += dist
            
            # Compute the number of silent and active units
            L_arr[k], S_arr[k] = BM.analyzeHiddenState( a = 3 )
            # Compute the magnetizations of the hidden units
            m_t[k], magnet[k] = BM.analyzeMagnetizations( X_red[i], L_arr[k] )
            
            ## DEBUG
            #print( "Visible example", X_red[i, 1:] ) 
            #print( "Hidden repr = ", BM.h[1:] )
            #print( "Hidden biases =", BM.W.T[1:,0]  )
            #print( "Hidden pre-act =", np.dot( BM.W.T[:,1:], X_red[i,1:] ) )
            #print( "magnetizations = ", magnet[k] ) 
            #print( "m^~ = ", m_t[k] )
            #print( "m_avg = ", np.mean( magnet[k] ) )
            #print( "L, S = ", L_arr[k], S_arr[k] )
            #input()
            k += 1
            
        # Compute mean reconstruction error
        MRE /= size
        # Compute percentage of correct reconstructions
        nCorrect = np.sum( corr )*1.0/size*100
        
        print( "Size set = ", size )
        print( "Correct reconstructions (%%) = %.2f " % nCorrect )
        print( "MRE = ", MRE )
        if MRE > 0:
            print( "RE averaged only on errors = ", MRE*size/np.sum( wrong ) )
        print( "L_avg = ", np.mean( L_arr ) )
        print( "m^~ = ", np.mean( m_t ) )
        print( "m_avg = ",  np.mean( magnet.flatten() )  )
        
        return MRE, nCorrect, L_arr, S_arr, corr, wrong, m_t, magnet
    
    # Perfomance on the last mini-batch of the training set 
    print( "Last MB training set results:" )
    MRE_last, nC_last, L_last, S_last, __, __, __, __ = analyzePerfomance( X_train[-sizeMB:] )
    # Perfomance on the training set 
    print( "Training set results:" )
    MRE_train, nC_train, L_train, S_train, corr_train, wrong_train, m_train, __  = analyzePerfomance( X_train )
    # Perfomance on the test set
    print( "Test set results" )
    MRE_test, nC_test, L_test, S_test, corr_test, wrong_test, m_test, __ = analyzePerfomance( X_test )

    # Concatenate the values of S and L, since they are independent on the type of set considered
    L = np.append( L_train, L_test )
    S = np.append( S_train, S_test )
    m_tilde = np.append( m_train, m_test )
    
    del L_train, L_test, S_train, S_test, m_train, m_test
    ############### Save the numerical results                
    # Write numerical results for the comparison of test and training data
    results[ k, : ] = np.array( [nC_last, nC_train, nC_test, MRE_last, MRE_train, MRE_test, \
                      p,T, np.mean( L ), np.mean(S), np.mean( m_tilde ) ] )
        
    ############### Make the plots
    # Define a function to make the histograms of the weights, if specified
    def plotWeights( weights ):
        f, axarr = plt.subplots(3)
        f.suptitle('Histograms of weights and biases')
        axarr[0].hist( weights[0,:], bins ="auto" )
        axarr[0].set_ylabel( "Visible biases" )
        axarr[1].hist( weights[:,0], bins ="auto" )
        axarr[1].set_ylabel( "Hidden biases" )
        axarr[2].hist( weights[1:,1:].flatten(), bins ="auto" )
        axarr[2].set_ylabel( "Weights" )
        plt.subplots_adjust(hspace=0.5)
        
        f, axarr = plt.subplots(1, 2 )#, figsize=(9, 3))
        f.suptitle('Histograms of weights and colormap')
        axarr[0].hist( weights.flatten(), bins= "auto" )
        axarr[1] = sns.heatmap( weights,  cmap="Greys_r", linewidths=.5, square=True)#, xticklabels = xticks, yticklabels = yticks)
        axarr[1].set_xlabel("Hidden units")
        axarr[1].set_ylabel("Visible units")
        axarr[1].spines['bottom'].set_visible(True)
        axarr[1].spines['top'].set_visible(True)
        axarr[1].spines['left'].set_visible('black')
        axarr[1].spines['right'].set_visible('black')
        axarr[1].set_aspect("equal")
        plt.subplots_adjust(wspace=0.5)
        plt.show() 

    if plots:
        # Compare the initial and final weights
        plotWeights( W_init ) 
        plotWeights( BM.W ) 

        # Make the histograms of L and S
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].hist(L)
        axarr[0].set_title("Hidden activities")
        axarr[0].set_ylabel("$\hat{L}$")
        axarr[1].hist(S)
        axarr[1].set_ylabel("$\hat{S}$")
        plt.show()
    
    np.set_printoptions( precision = 2) 
    print( BM.W )
    print( "Thresholds:", BM.W[0] )
    print( np.mean( BM.W[0] ) )
    print( "Max element:", np.max( np.abs( BM.W.flatten() ) ) ) 
    input()
######## File outputs    
with open('results.csv', 'a') as csvfile:            
    writer = csv.writer(csvfile, delimiter=' ')
    # Store hyperparameters
    writer.writerow( [ str(datetime.datetime.now()) ] )
    writer.writerow( ['ReLU_RBM?', 'N ', 'M', 'LA'] )
    writer.writerow( [useReLU, N, M, LA ])
    writer.writerow( ['seedBM ', 'seedTr', 'sizeTot', 'p_01', 'p_10' ])
    writer.writerow([seedBM,seedTr, len(dataset), p_01, p_10])
    writer.writerow([ 'alpha', 'c_e', 'c_a', 'SGS', 'Period'])
    writer.writerow([ alpha ,c_e, c_a, SGS, period])    
    writer.writerow( ['nEpochs', 'epsilon', 'lambda_x', 'sizeTrain','nMB' ] )
    writer.writerow( [nEpochs, epsilon, lambda_x,sizeTrain, nMB ] )
    # Store results and statistics
    writer.writerow( fields )
    writer.writerows(results)
    writer.writerow(['Averages'])
    writer.writerow(np.mean( results, axis=0 ) )
    writer.writerow(['Standard deviations'])
    writer.writerow(np.std(  results, axis=0 ) )
    writer.writerow([])

### Save the statistics of the dataset in a csv file, together with the perfomance of the last RBM      
# Create a string representation to get an overview of the statistics of the created sets    
def writePerformance( X, Y, corr, wrong, name ):
    X_red = np.unique( X, axis = 0 )
    Y_red = np.unique( Y, axis = 0 )
    X_str = [ np.array2string( X_red[i].astype(int),separator='-',max_line_width=10000 ) for i in range(len(X_red) ) ]  
    Y_str = [ np.array2string( Y_red[i].astype(int),separator='-',max_line_width=10000 ) for i in range(len(Y_red) ) ]  
    
    with open('dataset.csv', 'a') as csvfile:            
        writer = csv.writer(csvfile)
        writer.writerow( [name,'', 'Size=', np.sum( corr) + np.sum( wrong )] )
        writer.writerow( ["Example", "Unique?", "Right", "Wrong", "Total"] )
        for i in range( len(X_str) ):
            if X_str[i] in Y_str:
                uniqueness = False
            else:
                uniqueness = True
            writer.writerow( [ X_str[i], uniqueness,  corr[i], wrong[i], corr[i]+wrong[i]  ] )

writePerformance( X_train, X_test, corr_train, wrong_train, 'Training_set' )
writePerformance( X_test, X_train, corr_test, wrong_test, 'Test_set' )
