"""
Script to make the machine run

To-do list:
    - Ottimizzare il calcolo del numero di ricostruzioni corrette
    - Utilizzare reconstructionScore() e definire una funzione per stampare i vettori?
    - Analisi delle ricostruzioni tramite medie condizionate anziché tramite un singolo Gibbs Sampling

    - Determinazione della fase della macchina e comparazione con le performance di ricostruzione
    - Reintrodurre alpha e vedere come varia la perfomance
    - Confrontare la macchina regolarizzata con quella di default
    - Controllare se la sparsità controlla \tilde{m}. Ripetere dopo aver cambiato dataset
    - Confrontare PCD, CD-1 e CD-10

    - Modificare le chiamate in maniera tale da definire parameters
    - Implementare una grid evaluation per la scelta degli iperparametri
    - Velocizzare ulteriormente gli update usando Theano?
    - Unificare le funzioni per le due RBM e introdurre useReLU?
    - Introdurre target sparsity?
    - Calcolo della pseudolikelihood?
    
"""

from RBM import BaseRBM, ReLU_RBM
import datetime
import csv
import struct 
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pudb import set_trace
import seaborn as sns; sns.set(); sns.set_style("ticks")    


############### Initialization
# Initialize numpy prints
np.set_printoptions( precision = 2, suppress= True, linewidth = 1000) 
    
# Define hyper-parameters that will be often fixed
# Type of RBM to be used (False = BaseRBM, True = ReLU)
useReLU = True

# Size of the machine    
N = 40
M = 10

# Length of the categories
l = 5
# Probabilities for noisy clamping
if N <= 5:
    p_01 = 0.15
    p_10 = 0.05
elif N <= 20:
    p_01 = 0.15
    p_10 = 0.02
else:
    p_01 = 0.1
    p_10 = 0.0025

# Dataset to be used
useMNIST = False
if useMNIST:
    N = 784
    M = 400

# Learning algorithm
LA = "CD"
# Steps Gibbs Sampling 
SGS = 1
SGS_rec = 1
# Sampling period during the learning
period = 50
# Momentum coefficient
alpha = 0.2
# Epsilon decaying coefficient
c_e = 0.99
# Alpha growth coefficient
c_a = 1.005
# Percentage of the dataset that will form the training set
ratioTest = 1.
# Seed for the dataset and the machine
seedTr = 0
seedBM = None
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
    sizeTot = int( np.ceil( (1.0+ratioTest)*sizeTrain ) )
    
    # Obtain the dataset, applying noisy clamping
    dataset = np.zeros( (sizeTot,N), dtype = float )            
    maxIter = int( np.ceil( sizeTot/ l_X ) ) 
    ind = 0
    count_z = 0
    for i in range( maxIter ):
        # Take a random permutation of the rows of the X matrix
        for j in np.random.permutation( l_X ):
            if ind >=sizeTot: break
        
            x = np.copy( X[ j ] )

            # Noisy clamping
            y = np.random.uniform(low=0, high=1,size=N)
            for k in range( N ):
                if x[k] == 1 and y[k] <= p_01: x[k] = 0
                elif x[k] == 0 and y[k] <= p_10: x[k] = 1

            dataset[ind] =  x
            ind += 1
            
            if np.array_equal( x, np.zeros( N )):
                count_z += 1 
    
    print( count_z )
    return dataset

if useMNIST: 
    with open("train-images-idx3-ubyte", "rb") as fin:
        # Read first 16 bytes
        magic, num, rows, cols = struct.unpack(">IIII", fin.read(16))
        # Read pixels intensities
        img = np.fromfile(fin, dtype=np.uint8).reshape(num, rows, cols)
        
        dataset = np.zeros( (num, rows*cols), dtype = int ) 
        for i in range( num ):
            dataset[ i ] = img[ i, :, :].flatten() >= 128
            
        dataset = dataset[0:int( (1+ratioTest)*sizeTrain)]
else:
    dataset = buildDataset( N, l, seedTr, p_01, p_10 )
    #if plots:
        #plt.matshow( dataset[0:50] )
        #plt.xticks([]); plt.yticks([]) 
        #plt.show() 

# Add a dimension to handle biases
dataset = np.insert( dataset, 0, 1, axis = 1)

# Use hold-out technique to avoid overfitting
X_train, X_test = train_test_split( dataset, test_size=ratioTest/(1.+ratioTest), random_state = seedTr)

###############  Learning phase 
# Define a matrix to store the results
fields = ['nC_last', 'nC_train','nC_test','RE_last', 'RE_train', 'RE_test', 'Sparsity', 'Theta_mean', 'g_mean', 'L_mean', 'S_mean', 'm_tilde', 'm_mean', 'sqrt(r)', 'sqrt(r_mag)', 'Temp', 'W^2' ] 
results = np.zeros( (n, len(fields)) )

# Repeat the learning n times in order to obtain a more reliable statistics
for k in range( n ):

    ############### Define the RBM and make it learn
    if useReLU:
        BM = ReLU_RBM( N, M, seedBM )
    else:
        BM = BaseRBM( N, M, seedBM )

    W_init = np.copy(BM.W)
    
    ############### Learn the data model
    BM.fit( X_train, X_test, LA, SGS, nMB, nEpochs, epsilon, alpha, lambda_x, c_e, c_a, period, plots )
        
    ############### Analyze the final weights 
    p, p_vec, T = BM.analyzeWeights()
        
    ############### Analyze the perfomance of the RBM
    def analyzePerfomance( X, SGS = SGS_rec ):
        
        size = len(X)
        
        L_arr = np.zeros( size )
        S_arr = np.zeros( size )
        # Mean squared activations hidden units
        r = np.zeros( size )
        r_mag = np.zeros( size )

        # Magnetizations matrix 
        magnet = np.empty( (size, M) )
        m_t = np.empty( size )
        
        # Obtain the reconstruction score
        MRE, nCorrect = BM.reconstructionScore( X )
        
        # Describe the set according to the unique rows
        X_red, indices  = np.unique( X, axis = 0, return_inverse = True)
        
        # Arrays to store the errors and right reconstructions
        corr = np.zeros( len(X_red) )
        wrong = np.zeros( len(X_red) )
        
        # Iterate through the set, but taking into account the repetitions        
        for k,ind in enumerate(indices):
            
            # Compute its distance from the real visible state
            dist = np.linalg.norm( X_red[ind] - BM.v[k] )
            if dist == 0:
                corr[ind] += 1 
            else:
                wrong[ind] += 1 
            
            # Compute the number of silent and active units
            L_arr[k], S_arr[k] = BM.analyzeHiddenState( BM.h[k], a = 3 )
            
            # Compute the magnetizations of the hidden units
            m_t[k], magnet[k] = BM.analyzeMagnetizations( X_red[ind], L_arr[k] )
            
            # Sort the hidden state in ascending order
            BM.h[k] = np.sort( BM.h[k] )
            L = int( np.around( L_arr[k] ) )
            
            # Compute the mean squared activity of the two types of hidden units
            if L != BM.M and L > 0:
                r[k] = np.sum( np.power( BM.h[k, :-L], 2 ) )/(M-L)
                r_mag[k] = np.sum( np.power(BM.h[k,-L:],2) )/L
            elif L == 0:
                # No magnetized h.u., hence r_magnetized = 0
                r_mag[k] = 0
            elif L == BM.M:
                # No non-magnetized h.u.
                r[k] = 0

        
        # Compute mean r and r_mag
        r = np.mean( r )
        r_mag = np.mean( r_mag )
        
        print( "Size set = ", size )
        print( "Correct reconstructions (%%) = %.2f " % nCorrect )
        print( "MRE = {:.2f}".format( MRE ) )
        if MRE > 0:
            print( "RE averaged only on errors = {:.2f} ".format( MRE*size/np.sum( wrong ) ) )
        print( "L_mean = {:.2f}".format( np.mean( L_arr ) ) )
        print( "S_mean = {:.2f}".format( np.mean( S_arr ) ) )
        print( "r^(1/2) = {:.2f} ".format( np.sqrt( np.mean( r ) ) ) )
        print( "r_mag^(1/2) = {:.2f}".format( np.sqrt( np.mean( r_mag ) ) ) )
        print( "m^~ = {:.2f}".format( np.mean( m_t ) ) )
        print( "m_mean = {:.2f}".format(  np.mean( magnet.flatten() ) ) , "\n", )
                
        return MRE, nCorrect, L_arr, S_arr, m_t, magnet, r, r_mag, corr, wrong 
    
    # Perfomance on the last mini-batch of the training set 
    print( "=========== Last MB training set results ===========" )
    MRE_last, nC_last, L_last, S_last ,__ ,__, __, __, __, wrong_last = analyzePerfomance( X_train[-sizeMB:] )

    # Perfomance on the training set 
    print( "=========== Training set results ===========" )
    MRE_train, nC_train, L_train, S_train, m_train, mag_train, r_train, r_m_train, corr_train, wrong_train  = analyzePerfomance( X_train )
    # Perfomance on the test set
    print( "=========== Test set results ===========" )
    MRE_test, nC_test, L_test, S_test, m_test, mag_test, r_test, r_m_test, corr_test, wrong_test = analyzePerfomance( X_test )

    print( "*********** Final weights ***********" )
    W_0, W = np.copy( W_init), np.copy( BM.W )
    W_0[0,:], W[0,:] = -W_0[0,:], -W[0,:]
    W_0[:,0], W[:,0] = -W_0[:,0], -W[:,0]

    
    print( W[:5,:] )
    print( "\nThresholds:", W[0] )
    print( "Max element of BM.W:", np.max( np.abs( W.flatten() ) ) ) 
    print( "Sparsity: ", p )

    # Concatenate the values that are independent on the type of set considered
    L = np.append( L_train, L_test )
    S = np.append( S_train, S_test )
    m_tilde = np.append( m_train, m_test )
    mag = np.append( mag_train, mag_test ) 
    r = np.append( r_train, r_test )
    r_mag = np.append( r_m_train, r_m_test )
    
    # Compute the RE averaged only on the errors
    RE_last = MRE_last*sizeMB/np.sum(wrong_last)
    RE_train = MRE_train*len(X_train)/np.sum(wrong_train)
    RE_test  = MRE_test*len(X_test)/np.sum(wrong_test)
    del L_train, L_test, S_train, S_test, m_train, m_test, mag_test, mag_train, r_train, r_m_train, r_test, r_m_test

    ############### Save the numerical results                
    # Write numerical results for the comparison of test and training data
    results[ k, : ] = np.array( [nC_last, nC_train, nC_test, RE_last, RE_train, RE_test, p, -np.mean( BM.W[0] ), -np.mean( BM.W[:,0] ),
        np.mean( L ), np.mean(S), np.mean( m_tilde ), np.mean( mag.flatten() ), np.sqrt( np.mean(r) ), np.sqrt( np.mean(r_mag) ), T, np.linalg.norm( BM.W )**2 ] )
    
    
    ############### Make the plots
    # Define a function to make the histograms of the weights, if specified
    def plotWeights( W_0, W ):
        f, axarr = plt.subplots(3,2)
        f.suptitle('Histograms of weights and biases')
        axarr[0,0].hist( W_0[0,:], bins ="auto" )
        axarr[0,0].set_ylabel( "Hidden biases" )
        axarr[1,0].hist( W_0[:,0], bins ="auto" )
        axarr[1,0].set_ylabel( "Visible biases" )
        axarr[2,0].hist( W_0[1:,1:].flatten(), bins ="auto" )
        axarr[2,0].set_ylabel( "Weights" )

        plt.subplots_adjust(hspace=0.5, wspace= 0.5)
        
        axarr[0,1].hist( W[0,:], bins ="auto" )
        axarr[1,1].hist( W[:,0], bins ="auto" )
        axarr[2,1].hist( W[1:,1:].flatten(), bins ="auto" )
        
        plt.savefig('histograms_part.png')
        

        f, axarr = plt.subplots(1, 2)
        f.suptitle('Histograms of the weights')
        axarr[0].hist( W_0.flatten(), bins= "auto" )
        axarr[1].hist( W.flatten(), bins= "auto" )

        plt.savefig('histograms.png')

        #f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
        #g1 = sns.heatmap(W_0[1:,1:],cmap="coolwarm",cbar=False,ax=ax1, xticklabels=False, yticklabels=False)
        #g1.set_title('Initial weights')
        #g2 = sns.heatmap(W[1:,1:],cmap="coolwarm",cbar=False,ax=ax2, xticklabels=False, yticklabels=False)
        #g2.set_title('Final weights')
        #plt.subplots_adjust(wspace=0.5)
        #plt.savefig('colormaps.png')
        f, axarr = plt.subplots( M,1, sharex=True )
        for i in range( M ):
            axarr[i].imshow( np.around( W[:,i].reshape(1,N+1) ), cmap ="coolwarm" )
            axarr[i].set_yticks([]);axarr[i].set_xticks([])
            axarr[i].set_ylabel(i)
            axarr[i].set_adjustable('box-forced')

        plt.subplots_adjust(hspace=0.5)
        
        
        
        plt.show()

    if plots:        
        # Compare the initial and final weights
        plotWeights( W_0, W ) 

        # Make the histograms of L and S
        f, axarr = plt.subplots(2)
        axarr[0].hist(L)
        axarr[0].set_title("Hidden activities")
        axarr[0].set_ylabel("$\hat{L}$")
        plt.subplots_adjust(hspace=0.5)
        axarr[1].hist(S)
        axarr[1].set_ylabel("$\hat{S}$")
        plt.show()

    input( "Continue?" )
        
        
######## File outputs    
with open('results.csv', 'a') as csvfile:            
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    # Store hyperparameters
    writer.writerow( [ str(datetime.datetime.now()) ] )
    writer.writerow( ['ReLU_RBM', 'N', 'M', 'LA',  'SGS', 'sizeTot', 'p_01', 'p_10', 'alpha', 'c_e', 'c_a', 'Period', 'seedBM ', 'seedTr'] )
    writer.writerow( [useReLU, N, M, LA, SGS, len(dataset), p_01, p_10,alpha ,c_e, c_a, period,seedTr,seedBM] )
    writer.writerow( ['nEpochs', 'epsilon', 'lambda_x', 'sizeTrain','nMB' ] )
    writer.writerow( [nEpochs, epsilon, lambda_x,sizeTrain, nMB ] )
    writer.writerow( [])
    # Store results and statistics
    writer.writerow( fields )
    writer.writerows( results )
    writer.writerow([ ])
    tmp = np.vstack( (np.mean( results, axis=0 ), np.std(results,axis=0)) )
    tmp = np.hstack( ([[nEpochs, epsilon, lambda_x ],[0,0,0]], tmp) ) 
    fields_final = ['nEpochs', 'epsilon', 'lambda_x',] + fields  
    writer.writerow( fields_final )
    writer.writerows( tmp )
    writer.writerow([])

### Save the statistics of the dataset in a csv file, together with the perfomance of the last RBM      
# Create a string representation to get an overview of the statistics of the created sets    
def formatVectors( Z ):
    if Z.dtype == np.dtype('int64'):
        # Convert vectors into strings
        Z_str = [ np.array2string( Z[i],separator="",max_line_width=10000 ) for i in range(len(Z) ) ]  
        # Substitute zeros with underscore for representation purposes
        Z_str = [ Z_str[i].replace("0","_") for i in range(len(Z_str))] 
    else:
        # Convert vectors into strings
        Z_str = [ np.array2string( Z[i],separator="",max_line_width=10000, formatter={'all':lambda x: str(int(np.around(x)))+'|' if x > 0 else '_|'} ) for i in range(len(Z) ) ]  
    return Z_str


# Remove repetitions in the two sets
X_red  = np.unique( X_train, axis = 0 )
Y_red  = np.unique( X_test, axis = 0 )
X_rec, X_h = BM.GibbsSampling( v_init = X_red )
Y_rec, Y_h = BM.GibbsSampling( v_init = Y_red )

# Convert visible states into strings
X_red_str = formatVectors( X_red.astype(int) )
Y_red_str = formatVectors( Y_red.astype(int) )
# Convert reconstructions into string
X_rec_str = formatVectors( X_rec.astype(int) )
Y_rec_str = formatVectors( Y_rec.astype(int) )
X_h_str = formatVectors( X_h[:,1:] )
Y_h_str = formatVectors( Y_h[:,1:] )

with open('dataset.csv', 'w') as csvfile:            
    writer = csv.writer(csvfile)
    writer.writerow( ['Training_set'] )
    writer.writerow([ 'Size=' + str( np.sum( corr_train) + np.sum( wrong_train ) )] )
    writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
    for i in range( len(X_red_str) ):
        if X_red_str[i] in Y_red_str:
            uniqueness = False
        else:
            uniqueness = True
        writer.writerow( [ X_red_str[i], X_rec_str[i], X_h_str[i], uniqueness,corr_train[i]+wrong_train[i],  corr_train[i], wrong_train[i]  ] )

    writer.writerow( ['Test_set'] )
    writer.writerow([ 'Size=' + str( np.sum( corr_test) + np.sum( wrong_test ) )] )
    writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
    for i in range( len(Y_red_str) ):
        if Y_red_str[i] in X_red_str:
            uniqueness = False
        else:
            uniqueness = True
        writer.writerow( [ Y_red_str[i], Y_rec_str[i], Y_h_str[i], uniqueness, corr_test[i]+wrong_test[i],  corr_test[i], wrong_test[i]  ] )
