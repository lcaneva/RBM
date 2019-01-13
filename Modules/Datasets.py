import os 
import struct 
import numpy as np

"""
Function to read the MNIST dataset.
----------------------------------------
"""
def readMNIST():
    with open(os.path.join(os.getcwd(), 'Data','train-images-idx3-ubyte'), 'rb') as fin:
        # Read first 16 bytes
        magic, num, rows, cols = struct.unpack(">IIII", fin.read(16))
        # Read pixels intensities
        img = np.fromfile(fin, dtype=np.uint8).reshape(num, rows, cols)
        
        dataset = np.zeros( (num, rows*cols), dtype = int ) 
        for i in range( num ):
            dataset[ i ] = img[ i, :, :].flatten() >= 128

        return dataset

"""
Function to build the dataset for the autoencoder.
----------------------------------------------------
Define the dataset of "environmental vectors", using the noisy clamping as specified in the article of Ackley et al. 
"""
def buildDataset( N, l, sizeTrain, ratioTest, seedTr, p_01, p_10 ):
    
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


