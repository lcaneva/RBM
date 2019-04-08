import os 
import numpy as np

"""
Function to read the CalTech 101 Silhouettes dataset.
--------------------------------------------------------
"""
def readCAL( ):
    import scipy.io 
    data_dict = scipy.io.loadmat( os.path.join(os.getcwd(), 'Data','caltech101_silhouettes_16_split1.mat') )    
    return data_dict['train_data'], data_dict['test_data'] 

"""
Function to read the MNIST dataset.
----------------------------------------
"""
def readMNIST( size ):
    import struct 
    with open(os.path.join(os.getcwd(), 'Data','train-images-idx3-ubyte'), 'rb') as fin:
        # Read first 16 bytes
        magic, num, rows, cols = struct.unpack(">IIII", fin.read(16))
        # Read pixels intensities
        img = np.fromfile(fin, dtype=np.uint8).reshape(num, rows, cols)
        
        dataset = np.zeros( (num, rows*cols), dtype = int ) 
        for i in range( num ):
            dataset[ i ] = img[ i, :, :].flatten() >= 128

        return dataset[:size]


"""
Function to build the dataset for the autoencoder.
---------------------------------------------------------------------
Define the dataset of "environmental vectors", using  noisy clamping 
as specified in the article of Ackley et al. 
"""
def buildGEP( N, l, size, seed, p_01, p_10, invert ):
    
    # If seed is not None, fix the dataset
    np.random.seed( seed )
    
    # Build the different categories
    l_X = int(N/l)
    X = np.zeros( (l_X, N), dtype = float )
    for i in range( l_X ):
        X[i, i*l:(i+1)*l] = np.ones( l ) 
            
    # Obtain the dataset, applying noisy clamping
    dataset = np.zeros( (size,N), dtype = float )            
    maxIter = int( np.ceil( size/ l_X ) ) 
    ind = 0
    count_e = 0
    for i in range( maxIter ):
        # Take a random permutation of the rows of the X matrix
        for j in np.random.permutation( l_X ):
            if ind >=size: break
        
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
            
    print( "Number of errors: {}   {:.2f} %".format( count_e, count_e/size*100 ) )
    if invert:
        dataset_2 = np.zeros( (size,N), dtype = float )            
        dataset_2[dataset == 0] = 1
        dataset = dataset_2 
        
    return dataset


"""
Function to generate the Bar and Stripes dataset.
---------------------------------------------------------------------
Construct the examples of the Bar and Stripes dataset, following the 
definition of MacKay (2003).
"""
def buildBES( N, size, seed ):
       
        # Fix the RNG
        np.random.seed( seed )
        
        # Compute the number of rows of each example
        D = int( np.sqrt( N ) )
        
        # Compute the size of the dataset
        data = np.zeros( (size, D*D) )
        
        #nB = 3
        # Generate the whole dataset
        for k in range( size ):
            example = np.zeros( (D, D) )
            
            # Sample the stripes
            y = np.random.rand( D )
            example[ y <= 0.5,: ] = 1 

            #y =  np.random.permutation( D )
            #example[y[1:nB+1],:] = 1 
                
            # Sample the rotation
            y = np.random.rand()
            if y <= 0.5:
                example = example.T 
            
            data[k] = example.flatten()
            
        return data


"""
Function to generate the Shifting bars dataset.
--------------------------------------------------------------------
Construct the examples in the Shifting bars dataset, following 
the definition of Melchior et al. (2016).
"""
def buildSB( N, l, size, seed ):
        
        # Fix the RNG
        np.random.seed( seed )
        
        data = np.zeros( (size, N) )
        # Iterate over the whole set
        for i in range( size ):
            # Extract at random the starting index of the bar
            ind = np.random.randint( low=0, high=N )
            
            # Generate the bar, using periodic BCs
            if ind <= N-l:
                data[i, ind: ind+l] = 1
            else:
                data[i, ind: ] = 1 
                data[i, 0:ind+l-N ] = 1
                        
        return data
