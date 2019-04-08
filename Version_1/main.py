"""
Script to make the machine run

To-do list:
    - Analisi delle ricostruzioni tramite medie condizionate anziché tramite un singolo Gibbs Sampling
    - Modificare le chiamate in maniera tale da definire parameters
    - Unificare le funzioni per le due RBM e introdurre useReLU?
    - Spostare le funzioni dal main a utils.py


    - Determinazione della fase della macchina e comparazione con le performance di ricostruzione
    - Confrontare la macchina regolarizzata con quella di default
    - Controllare se la sparsità controlla \tilde{m}. Ripetere dopo aver cambiato dataset
    - Confrontare PCD, CD-1 e CD-10


    - Implementare una grid evaluation per la scelta degli iperparametri
    - Velocizzare ulteriormente gli update usando Theano?
    - Introdurre target sparsity?
    - Calcolo della pseudolikelihood?
    
"""

from RBM import BaseRBM, ReLU_RBM
import datetime
import csv
import struct 
import numpy as np
import matplotlib.gridspec as gridspec
import argparse
from sklearn.model_selection import train_test_split
from pudb import set_trace

"""
Function to build the dataset for the autoencoder.
----------------------------------------------------
Define the dataset of "environmental vectors", using the noisy clamping as specified in the article of Ackley et al. 
"""
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
    count_e = 0
    for i in range( maxIter ):
        # Take a random permutation of the rows of the X matrix
        for j in np.random.permutation( l_X ):
            if ind >=sizeTot: break
        
            x = np.copy( X[ j ] )

            # Noisy clamping
            y = np.random.uniform(low=0, high=1,size=N)
            error = False
            for k in range( N ):
                if x[k] == 1 and y[k] <= p_01: 
                    x[k] = 0; error = True 
                elif x[k] == 0 and y[k] <= p_10: 
                    x[k] = 1; error =True 
            
            if error:
                count_e += 1 

            dataset[ind] =  x
            ind += 1
            
    print( "Number of errors: {}   {:.2f} %".format( count_e, count_e/len(dataset)*100 ) )
    return dataset, count_e


"""
Create a string representation to get an overview of the statistics of the created sets.
-------------------------------------------------------------------------------------------
"""
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

"""
Post-processing analysis function.
-------------------------------------
"""
def analyzePerfomance( X ):
    # Define useful parameters
    SGS_rec = 1 
    size = len(X)
    # Partecipation ratios
    L_arr = np.zeros( size )
    S_arr = np.zeros( size )
    # Mean squared activations hidden units
    r = np.zeros( size )
    h_max = np.zeros( size )
    h_nonmag_max = np.zeros( size )

    # Magnetizations matrix 
    m_nonmag = np.empty( size )
    m_t = np.empty( size )
    
    # Obtain the reconstruction score
    MRE, nCorrect = BM.reconstructionScore( X )
            
    # Describe the set according to the unique rows
    # ind_red = indices of the first occurrences of the unique rows
    # indices = indices that allow to reconstruct X from the unique rows
    X_red,  ind_red, indices  = np.unique( X, axis = 0, return_inverse = True, return_index=True)
    
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
        L_arr[k], S_arr[k], r[k], h_max[k], h_nonmag_max[k] = BM.analyzeHiddenState( BM.h[k], a = 3 )
        
        # Compute the magnetizations of the hidden units
        m_t[k], m_nonmag[k] = BM.analyzeMagnetizations( X_red[ind], L_arr[k] )
        

    # Print the 10-top reconstructions (arbitrarily chosen as the first occurrence of each unique row)  
    if N < 100:
        np.set_printoptions( linewidth = 1000, formatter={'all':lambda x: str(x) if x > 0 else '_'} )

        ind_ord = np.argsort( -corr )  
        X_top = X_red[ ind_ord ]
        ind_top = ind_red[ind_ord]
        if len(X_top) > 10: 
            nPrints = 10
        else:    
            nPrints = len(X_top)
        for k in range( nPrints ):
            print( X_top[k,1:].astype(int) )
            print( BM.v[ind_top[k],1:].astype(int), "\t{:.2f}".format( np.linalg.norm(X_top[k] - BM.v[ind_top[k]])**2), end="\t"  )                
            print( corr[ind_ord[k]], "\t", wrong[ind_ord[k]], end='\t' )
            print( '[', ''.join(formatVectors( BM.h[ind_top[k],1:] )), "]\n" )
        np.set_printoptions( precision = 2, suppress= True, linewidth = 1000)
        corr_top = np.sum( corr[ind_ord[:nPrints]] )
        wrong_top = np.sum( wrong[ind_ord[:nPrints]] )
        print( ' '*2*BM.N, "\t\t{}%\t{}%\n".format( corr_top/size*100, wrong_top/size*100 ) )
        
    if np.any( wrong ):
        RE = MRE*size/np.sum(wrong)
    else:
        RE = 0

    # Store the results and print them        
    res = [MRE, RE , nCorrect, L_arr, S_arr, np.mean( m_t ), np.mean(m_nonmag), np.mean( r), np.mean( h_max ), np.mean( h_nonmag_max), corr, wrong]
    
    L_mean = np.mean( res[3] )
    S_mean = np.mean( res[4] )

    print( "Size set = ", size )
    print( "MRE = {:.2f} ".format( res[0] ) )
    print( "REE  = {:.2f} ".format(  res[1] ) )
    print( "Correct reconstructions = {:.2f} %".format( res[2] ) )
    print( "L_mean = {:.2f}".format(  L_mean ) )
    print( "S_mean = {:.2f}".format(  S_mean ) )
    print( "I_mean = {:.2f}".format(  BM.M - L_mean - S_mean ) )
    print( "m_tilde = {:.2f}".format( res[5] ) ) 
    print( "m_nonmag = {:.2f}".format( res[6] ) )
    print( "sqrt(r) = {:.2f} ".format( res[7] ) )
    print( "delta = {:.2f}\n".format( res[8]-res[9] ) )
    
    return res

"""
Function to make the different plots necessary for the analysis.
-----------------------------------------------------------------
"""
def makePlots():
    import matplotlib.pyplot as plt
    import os
    curr_dir = os.getcwd()

    #import seaborn as sns; sns.set(); sns.set_style("ticks")    
    if not useMNIST:
        f = plt.figure()
        f.suptitle( "Percentage of errors: {:.2f} %".format( count_e/len(dataset)*100 ) )
        ax1 = plt.subplot2grid((1,1),(0,0))
        ax1.matshow( dataset[0:50,1:], cmap="Greys_r" )
        ax1.set_xticks([]); ax1.set_yticks([]) 
    
    # Display reconstruction error through epochs             
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot([i for i in range( nEpochs )], MRE_fit)
    axarr[0].set_ylabel('MRE')
    axarr[1].plot([i for i in range( nEpochs )], nCorrect_fit)
    axarr[1].set_ylabel('Correct %')
    axarr[1].set_xlabel('Epochs')
    plt.savefig( os.path.join(curr_dir, 'Plots', 'MRE.png' ) )
    
    # Monitor the sparsity
    plt.figure()
    plt.plot([period_ovf*i for i in range( len( sparsity_fit ) )], sparsity_fit )
    plt.ylabel('Sparsity')
    plt.xlabel('Epochs')
    plt.savefig( os.path.join(curr_dir, 'Plots', 'sparsity.png' ) )

    # Monitor the overfitting
    plt.figure()
    plt.plot([period_ovf*i for i in range( len( ovf[0] ) )], ovf[0,:], label="Training set" )
    plt.plot([period_ovf*i for i in range( len( ovf[1] ) )], ovf[1,:], label="Test set" )
    plt.ylabel('Average free-energy')
    plt.legend()
    plt.savefig( os.path.join(curr_dir, 'Plots', 'overfitting.png' ) )
    
    plt.show() 
    
    f = plt.figure()
    f.suptitle( 'Weights evolution' )
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[0, 2:4])
    ax3 = plt.subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1, 2:4])
    ax5 = plt.subplot(gs[2:, 0:2])
    ax6 = plt.subplot(gs[2:, 2:4])
    plt.subplots_adjust(hspace=0.75, wspace=0.65)

    ax1.hist( W_0[0,:], bins ="auto" )
    ax1.set_title("Initial weights", size=9 )
    ax1.set_ylabel( "Hidden biases", size=9 )
    ax2.hist( W[0,:], bins ="auto" )
    ax2.set_title("Final weights", size=9)
    
    ax3.hist( W_0[:,0], bins ="auto" )
    ax3.set_ylabel( "Visible biases",size=9 )
    ax4.hist( W[:,0], bins ="auto" )
    ax5.hist( W_0.flatten(), bins= "auto" )
    ax5.set_ylabel( 'Global view',size=9 )

    ax6.hist( W.flatten(), bins= "auto" )
    
    plt.savefig( os.path.join(curr_dir, 'Plots', 'histograms.png' ) )

    ##f,(ax1,ax2) = plt.subplots(1,2,sharey=True)
    ##g1 = sns.heatmap(W_0.T,cmap="coolwarm",cbar=False,ax=ax1, xticklabels=False, yticklabels=False)
    ##g1.set_title('Initial weights')
    #plt.figure()
    #g2 = sns.heatmap(W.T,cmap="Greys_r",cbar=True,  xticklabels=False, yticklabels=False)
    #g2.set_title('Final weights')
    
    #f, axarr = plt.subplots( M,2, sharex=True )
    fig, axes = plt.subplots(nrows=M, ncols=1)
    fig.suptitle( 'Learned features' )        
    vmin = min(W[1:,1:].flatten())
    vmax = max(W[1:,1:].flatten())
    for i, ax in enumerate(axes.flat):
        im = ax.matshow(  W[1:,i+1].reshape(1,N), cmap ="Greys_r", vmin=vmin,vmax=vmax )
        ax.set_yticks([])
        ax.set_xticks([])
        if (i+1)%5 == 0 or M < 10 or i==0 or i==M-1:
            ax.set_ylabel(i+1)
        ax.set_adjustable('box-forced')
        
    fig.colorbar(im, ax=axes.ravel().tolist())
    #f = plt.figure()
    #f.suptitle( 'Learned features' )        
    #gs = gridspec.GridSpec(M, 1)
    #for i in range( M ):
        #ax = plt.subplot( gs[i,0] )
        #im = ax.matshow(  W[1:,i+1].reshape(1,N), cmap ="coolwarm", vmin=min(W[1:,1:].flatten()),vmax=max(W[1:,1:].flatten()) )
        #ax.set_yticks([])
        #ax.set_xticks([])
        #if (i+1)%5 == 0 or M < 10 or i==0 or i==M-1:
            #ax.set_ylabel(i+1)
        #ax.set_adjustable('box-forced')
    plt.savefig( os.path.join(curr_dir, 'Plots', 'colormaps.png' ) )
    
    f = plt.figure()
    f.suptitle( 'Example of hidden state activations' )
    gs = gridspec.GridSpec(2, 1)
    gs.update(top=0.85, bottom=0.2, wspace=0.05) 
    ax = plt.subplot( gs[0:1,:] )
    ax.bar( np.arange( 1, M+1 ),  BM.h[0,1:] )

    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(top=0.5, bottom=0.1, hspace=0.1)
    ax = plt.subplot( gs1[0,:] )
    ax.matshow( X_test[0,1:].reshape(1,N),cmap ="Greys_r" )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_adjustable('box-forced')
    ax.set_ylabel( 'Input' )

    ax = plt.subplot( gs1[1,:] )
    ax.matshow( BM.W[1:, np.argmax(BM.h[0,1:])+1].reshape(1,N),cmap ="Greys_r" )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_adjustable('box-forced')
    ax.set_ylabel( 'Weights' )
    plt.savefig( os.path.join(curr_dir, 'Plots', 'hidden_example.png' ) )

    # Make the histograms of L and S
    f, axarr = plt.subplots(3)
    f.suptitle( 'Hidden states activities' )
    axarr[0].hist(L)
    axarr[0].set_ylabel("Magnetized")
    
    axarr[1].hist(M*np.ones(len(L))-L-S)
    axarr[1].set_ylabel("Non-magnetized")

    plt.subplots_adjust(hspace=0.5)
    axarr[2].hist(S)
    axarr[2].set_ylabel("Silent")

    plt.figtext(0.5, 0.925, '(Hyperparameters: N={}, M={}, $\epsilon={}$, $\lambda={}$)'.format(N,M,epsilon,lambda_x), wrap=True, ha='center', va='top',fontsize=8 )
    
    plt.savefig( os.path.join(curr_dir, 'Plots', 'hidden_activities.png' ) )
    plt.show()

"""
Function to save the obtained results in formatted CSV files.
-------------------------------------------
"""
def saveResults():
    with open('results.csv', 'a') as csvfile:            
        writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
        # Store hyperparameters
        writer.writerow( [ str(datetime.datetime.now()) ] )
        writer.writerow( ['ReLU_RBM', 'N', 'M', 'LA',  'SGS', 'sizeTot', 'p_01', 'p_10', 'alpha', 'c_e', 'c_a', 'Period', 'seedBM', 'seedTr'] )
        writer.writerow( [useReLU, N, M, LA, SGS, len(dataset), p_01, p_10,alpha ,c_e, c_a, period,seedTr,seedBM] )
        writer.writerow( ['nEpochs', 'epsilon', 'lambda_x', 'sizeTrain','nMB' ] )
        writer.writerow( [nEpochs, epsilon, lambda_x,sizeTrain, nMB ] )
        writer.writerow( [])
        # Store results and statistics
        writer.writerow( fields )
        writer.writerows( results )
        writer.writerow([ ])
        tmp = np.vstack( (np.mean( results, axis=0 ), np.std(results,axis=0)) )
        #tmp = np.hstack( ([[nEpochs, epsilon, lambda_x ],[0,0,0]], tmp) ) 
        #fields_final = ['nEpochs', 'epsilon', 'lambda_x',] + fields  
        #writer.writerow( fields_final )
        writer.writerows( tmp )
        writer.writerow([])

    ### Save the statistics of the dataset in a csv file, together with the perfomance of the last RBM
    if BM.N < 100:
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
            writer.writerow([ 'Size=' + str( np.sum( res_train[10] ) + np.sum( res_train[11] ) )] )
            writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
            for i in range( len(X_red_str) ):
                if X_red_str[i] in Y_red_str:
                    uniqueness = False
                else:
                    uniqueness = True
                writer.writerow( [ X_red_str[i], X_rec_str[i], X_h_str[i], uniqueness,res_train[10][i]+res_train[11][i],  res_train[10][i], res_train[11][i]  ] )

            writer.writerow( ['Test_set'] )
            writer.writerow([ 'Size=' + str( np.sum( res_test[10] ) + np.sum( res_test[11] ) )] )
            writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
            for i in range( len(Y_red_str) ):
                if Y_red_str[i] in X_red_str:
                    uniqueness = False
                else:
                    uniqueness = True
                writer.writerow( [ Y_red_str[i], Y_rec_str[i], Y_h_str[i], uniqueness, res_test[10][i]+res_test[11][i],  res_test[10][i], res_test[11][i]  ] )



############### Initialization
# Initialize numpy prints
np.set_printoptions( precision = 2, suppress= True, linewidth = 1000) 
    
# Define hyper-parameters that will be often fixed
# Type of RBM to be used (False = BaseRBM, True = ReLU)
useReLU = True

# Size of the machine    
N = 40
M = 20

# Length of the categories
l = 10
# Probabilities for noisy clamping
if N <= 5:
    p_01 = 0.15
    p_10 = 0.05
elif N <= 20:
    p_01 = 0.15
    p_10 = 0.02
else:
    if l == 1:
        p_01 = 0.1
        p_10 = 0.0025
    else:
        p_01 = 0.01
        p_10 = 0.005

# Dataset to be used
useMNIST = False
if useMNIST:
    N = 784
    M = 400

# Learning algorithm
LA = "CD"
# Steps Gibbs Sampling 
SGS = 1
# Sampling period during the learning
period = 50
# Momentum coefficient
alpha = 0.1
# Epsilon decaying coefficient
c_e = 0.98
# Alpha growth coefficient
c_a = 1.01
# Percentage of the dataset that will form the training set
ratioTest = 1.
# Seed for the dataset and the machine
seedTr = 0
seedBM = 1
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

# Compute the size of the minibatches
sizeMB = int( sizeTrain / nMB )

############### Build or read the dataset
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
    dataset, count_e = buildDataset( N, l, seedTr, p_01, p_10 )

# Add a dimension to handle biases
dataset = np.insert( dataset, 0, 1, axis = 1)

# Use hold-out technique to avoid overfitting
X_train, X_test = train_test_split( dataset, test_size=ratioTest/(1.+ratioTest), random_state = seedTr)

g_init = np.sum( dataset, axis = 0 )/len(dataset)
    
###############  Learning phase 
# Define a matrix to store the results
fields = ['nC_last', 'nC_train','nC_test','RE_last', 'RE_train', 'RE_test', 'Sparsity', 'Theta_mean', 'g_mean', 'L_mean', 'I_mean', 'S_mean', 'm_tilde', 'm_mean', 'sqrt(r)', 'h_max', 'Delta', 'Temp', 'W^2' ] 
results = np.zeros( (n, len(fields)) )

# Repeat the learning n times in order to obtain a more reliable statistics
for k in range( n ):

    ############### Define the RBM and make it learn
    if useReLU:
        BM = ReLU_RBM( N, M, seedBM, g_init )
    else:
        BM = BaseRBM( N, M, seedBM, g_init )

    W_init = np.copy(BM.W)
    
    ############### Learn the data model
    MRE_fit, nCorrect_fit, sparsity_fit, ovf, period_ovf = BM.fit( X_train, X_test, LA, SGS, nMB, nEpochs, epsilon, alpha, lambda_x, c_e, c_a, period, plots )
        
    ############### Analyze the final weights 
    p, p_vec, T = BM.analyzeWeights()
        
    ############### Analyze the perfomance of the RBM
    
    # Perfomance on the last mini-batch of the training set 
    print( "=========== Last MB training set results ===========" )
    res_last = analyzePerfomance( X_train[-sizeMB:] )
    # Perfomance on the training set 
    print( "=========== Training set results ===========" )
    res_train = analyzePerfomance( X_train )
    # Perfomance on the test set
    print( "=========== Test set results ===========" )
    res_test = analyzePerfomance( X_test )

    print( "*********** Final weights ***********" )
    W_0, W = np.copy( W_init ), np.copy( BM.W )
    W_0[0,:], W[0,:] = -W_0[0,:], -W[0,:]
    W_0[:,0], W[:,0] = -W_0[:,0], -W[:,0]
    
    print( "Thresholds:", W[0] )
    print( "Max element of BM.W: {:.2f}".format( np.max( np.abs( W.flatten() ) ) ) ) 
    print( "Sparsity: {:.2f}".format( p ) )

    # Concatenate the values that are independent on the type of set considered
    L =  np.append( res_train[3], res_test[3] ) 
    S =  np.append( res_train[4], res_test[4] )
    L_mean = np.mean( L )
    S_mean = np.mean( S )
    m_tilde = (res_train[5]+res_test[5])/2
    m_nonmag = (res_train[6]+res_test[6])/2 
    r = (res_train[7]+res_test[7])/2
    h_max = (res_train[8]+res_test[8])/2
    h_nonmag_max = (res_train[9]+res_test[9])/2
    

    ############### Save the numerical results                
    # Write numerical results for the comparison of test and training data
    results[ k, : ] = np.array( [res_last[2], res_train[2], res_test[2], res_last[1], res_train[1], res_test[1], p, -np.mean( BM.W[0] ), -np.mean( BM.W[:,0] ),  L_mean, BM.M-L_mean-S_mean, S_mean, m_tilde,  m_nonmag, np.sqrt( r ), h_max, h_max-h_nonmag_max,  T, np.linalg.norm( BM.W )**2 ] )
    
    
    ############### Make plots    
    if plots:       
        makePlots() 
        
    input( "Continue?" )
        
        
######## File outputs    
saveResults()
