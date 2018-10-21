import numpy as np
import matplotlib.pyplot as plt
from pudb import set_trace
from scipy import special
import seaborn as sns

##################################################
# Class for Restricted Boltzmann Machines
# with binary logistic units
##################################################
class BaseRBM:
    
    """
    Constructor.
    -------------
    Input arguments:
        N, number of visible units
        M, number of hidden units
        seed, integer to initialize the RNG
        
    Define a RBM with N+1 visible units (the first of which is the bias unit)
    and M+1 hidden units. If instantiated as a child class, do not define the
    probabilities vectors used instead during the learning of BaseRBM.    
    """
    def __init__(self, N, M, seed = None):
            # Initialize the random state, if required
            np.random.seed(seed)
            
            # Number of visible units
            self.N = int( N )

            # Number of hidden units
            self.M = int( M )
            
            # Initialize visible layer
            self.v = np.empty( N+1, dtype = float )
            
            # Initialize hidden layer
            self.h = np.empty( M+1, dtype = float )
            
            # Initialize probabilities vectors
            self.vp = np.empty( N+1, dtype = float )
            self.hp = np.empty( M+1, dtype = float )
                        
            """
            Weight matrix.
            --------------------
            In order to be able to represent biases and thresholds of the units,
            the matrix is (N+1)x(M+1). 
            Generic element: w_{i\mu} 
            Each row, for i >= 1, corresponds to all the weights that connect the i-th
            visible unit to the hidden layer and to itself (bias, for \mu = 0).
            In particular:
                w_{i,0} = activation threshold of the i-th visible unit = g_i (cfr Monasson)
                w_{0,\mu} = activation threshold of the \mu-th hidden unit = g_\mu (if gaussian)
            The configurations will need a starting "1" to take account of these biases
            during the propagation of the signal between the layers.
            Notice that g_i is -\theta_i in the notation of Ackley and also g_{\mu} = -\theta{mu}. 
            Finally, the proper weights are initialized using He's method based on uniform distribution
            while the biases are set to null values.
            """            
            self.W = np.empty( (N+1, M+1), dtype = float  )
            
            # He's initialization
            r = np.sqrt(6/(N+M))
            self.W[1:,1:] = np.random.rand( N, M )*2*r - r 
            
            # Biases initialization
            self.W[0,:] = 0
            self.W[:,0] = 0 
            
            
    """
    Learning algorithm.
    -------------------
    Input arguments:
        X_train, input mini-batch as a matrix of size (sizeMB x N+1) 
        X_test, test set as a matrix of size (sizeTest x N+1 )
        LA, string that specifies the learning algorithm (Contrastive Divergence, Persistent CD, etc)
        Steps Gibbs Sampling, number of sampling steps in the Contrastive Divergence Markov Chain 
        nMB, number of mini-batches to split the training set
        nEpochs, maximum number of cycles through the training set X_train
        epsilon, learning rate
        alpha, coefficients for the momentum
        lambda_x, weights cost
        c_e, decay coefficient  of the learning rate
        c_a, growth coefficient of the momentum parameter
        period, frequency of update of alpha and epsilon
        plots, bool used to deactive plots when not needed
    
    Learn the weights according to the specified learning algorithm. 
    In order to be independent from the size of the mini-batches, the learning scale is scaled in the updating rule.
    """
    def fit( self, X_train, X_test, LA , SGS,  nMB, nEpochs, epsilon, alpha, lambda_x,  c_e, c_a, period, plots  ):
        # Initialize counter for the energies and sampling period
        counter = 0
        period_ovf = 10
        
        # Size of the input mini-batch
        sizeMB = int( len( X_train )/nMB )

        # Initialize arrays for the statistics
        MRE = np.zeros( nEpochs, dtype = float )
        nCorrect = np.zeros( nEpochs, dtype = float )
        p_arr = np.zeros( int(nEpochs/period_ovf), dtype = float )        
        ovf = np.zeros( (4,  int(nEpochs/period_ovf)), dtype = float )
        
        velocity = np.zeros( (self.N+1, self.M+1) )        
        # DEBUG
        W_up = np.zeros( (nEpochs,3*self.M), dtype=float )
        W_hist = np.zeros( (nEpochs,3*self.M), dtype=float ) 
        
        # Iterate through X_train nEpochs times
        for t in range( nEpochs ):
            for ind in np.random.permutation( nMB ):
                    
                # Create a view of the i-th mini-batch
                MB = X_train[ind*sizeMB:(ind+1)*sizeMB]
                
                # Compute the contribution to the weight updates deriving from the log-likelihood (heuristically) 
                if LA == 'CD':
                    W_updates = epsilon/sizeMB * self.CD( MB, SGS )
                    v_rec = self.v
                elif LA == 'PCD':
                    W_updates = epsilon/sizeMB * self.PCD( MB, SGS, num_chains = sizeMB  )
                    v_rec, __ = self.GibbsSampling( v_init = MB, SGS = SGS )
                    
                # Compute the regularization contribution
                W_updates -=  epsilon/sizeMB * lambda_x * self.regularize( x=2 )  
                                
                # Update the velocity
                velocity =  W_updates + alpha*velocity
                
                # Update the weights (one time per MB)
                self.W += velocity
                

            # Compute the energies and the sparsity of the model
            if plots:
                if t % period_ovf == 0:
                    ovf[:, counter] = self.monitorOverfitting( X_train, X_test )
                    p_arr[counter], __, T = self.analyzeWeights() 
                    counter += 1
                
                W_up[t,:] = W_updates[ 1:4, 1:].flatten()
                W_hist[t,:] = self.W[ 1:4, 1: ].flatten()


            # Compute and print statistics of the current epoch
            print("---------------Epoch {}--------------".format(t+1))
            MRE[t], nCorrect[t] = self.reconstructionScore( X_train )            

            print( "Mean Squared Error = ", MRE[t] )
            print( "Correct reconstructions (%%) = %.2f \n" % nCorrect[t] ) 
            
            # Increase alpha and decrease epsilon towards the end of learning
            if t > nEpochs*0.5:
                epsilon *= c_e 
            if t > nEpochs*0.5 and alpha < 0.9:
                    alpha *= c_a


        # Make plots for the current mini-batch, if required
        if plots:
            # Display what has been learned            
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot([i for i in range( nEpochs )], MRE)
            axarr[0].set_ylabel('MRE')
            axarr[1].plot([i for i in range( nEpochs )], nCorrect)
            axarr[1].set_ylabel('Correct %')
            axarr[1].set_xlabel('Epochs')

            # Monitor the sparsity
            plt.figure()
            plt.plot([period_ovf*i for i in range( len( p_arr ) )], p_arr )
            plt.ylabel('Sparsity')
            plt.xlabel('Epochs')

            # Monitor the overfitting
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].plot([period_ovf*i for i in range( len( ovf[0] ) )], ovf[0,:], label="Training set" )
            axarr[0].plot([period_ovf*i for i in range( len( ovf[1] ) )], ovf[1,:], label="Test set" )
            axarr[0].set_ylabel('Average log-likelihood')
            axarr[0].legend()
            axarr[1].plot([period_ovf*i for i in range( len( ovf[2] ) )], ovf[2,:], label="Training set" )
            axarr[1].plot([period_ovf*i for i in range( len( ovf[3] ) )], ovf[3,:], label="Test set" )
            axarr[1].set_ylabel('Average free energy')
            axarr[1].set_xlabel('Epochs')
            axarr[1].legend()
            plt.show()
            
            plt.figure()
            for j in range( W_hist.shape[1] ):
                plt.plot( [i for i in range( nEpochs )], W_hist[:,j] )

            plt.figure()
            for j in range( W_hist.shape[1] ):
                plt.plot( [i for i in range( nEpochs )], W_up[:,j] )
                
            plt.show()    

    """
    Regularization function.
    -----------------------------------
    Input argument: 
        x, parameter to adapt the regularization according to the level of coupling of the considered visible unit
           to the hidden layer. In particular, x = 1 yields L1-penalty, but can determine the disconnection of some
           hidden units, which make them useful and should be avoided.
    
    Return the contribution to the weights update given by the regularization defined in the Supplemental Material
    of Monasson's article at page 2.
    """
    def regularize( self, x ):
        # Determine the signs of the weights
        W_updates = np.zeros_like( self.W )
        W_updates[1:,1:] = np.sign( self.W[1:,1:] )
        # Iterate through the hidden units (excluding the bias one, as suggested by Hinton)
        for mu in range( 1, self.M+1 ):
            # Compute the contribution between square brackets, different for each h.u.
            tmp = np.sum( np.abs( self.W[1:,mu] ) )**(x-1)
            # Compute the correspondent column of the weights updates
            W_updates[1:,mu] *= tmp 
        
        return W_updates
    
    """
    Contrastive divergence function.
    -------------------------------------
    Input arguments: 
        v_example, training instance for the positive phase
    
    Return the weight matrix update to be used for Stochast Gradient Descent.
    USE ONLY THE PROBABILITIES FOR THE UPDATES.
    """
    def CD( self, v_example, SGS  ):
        # Positive phase 
        # Run the hidden update and 
        # GET THE PROBABILITIES (see Hinton ch. 3 )
        self.hp = self.updateHidden( v_example, mode = "Pr" ) 
        
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( v_example.T, self.hp )
        
        # Negative phase
        # Make the machine daydream, starting from the hidden state
        # obtained in the positive phase
        
        # Gibbs Sampling for the negative learning phase
        # Use only PROBABILITIES except for the last sampling
        for k in range( SGS-1 ):
            self.vp = self.updateVisible( self.hp, mode = "Pr" )
            self.hp = self.updateHidden(  self.vp, mode = "Pr" )

        # Get the total visible state
        self.vp, self.v = self.updateVisible( self.hp, mode = "Total" )
        # Obtain also the hidden state
        self.hp, self.h = self.updateHidden( self.vp, mode = "Total" )
        
        # Compute negative contribute for SGD
        # Again, obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.v.T, self.hp )
                        
        # Update the weights (and biases) 
        return  delta_pos - delta_neg 
        
    """
    Gibbs Sampling function.
    -----------------------
    Input arguments: 
        v_init, initial state (if present) of the machine in the visible layer
        h_init, initial hidden state of the machine, if present
        SGS, number of Steps of the Gibbs Sampling used in CD

    Create a MCMC either from a visible or a hidden state that makes the machine  daydream for SGS steps.
    """
    def GibbsSampling( self, v_init = np.empty(0), h_init = np.empty(0), SGS = 1 ):
        # Determine where the chain starts
        if v_init.size > 0:
            hp, h = self.updateHidden( v_init, mode = "Total" )
            vp = self.updateVisible( h, mode = "Pr" )
            
            for k in range( SGS-2 ):
                # Complete Gibbs Sampling starting from a visible state
                hp,h = self.updateHidden( vp )
                vp = self.updateVisible( h, mode = "Pr" )
            
            hp, h_samples = self.updateHidden( vp )
            vp, v_samples = self.updateVisible( h_samples )
        else:
            vp = self.updateVisible( h_init, mode = "Pr" )
            hp, h = self.updateHidden( vp, mode="Total" )

            for k in range( SGS-2 ):
                # Complete Gibbs Sampling starting from a hidden state
                vp = self.updateVisible( h, mode="Pr" )
                hp, h = self.updateHidden( vp )
            
            vp, v_samples = self.updateVisible( h )
            hp, h_samples = self.updateHidden( vp )

        return v_samples, h_samples
    
    """
    Computation of the hidden state from a visible one.
    ---------------------------------------------------
    Input arguments:
        x, input visible state or mini-batch
        mode, string, equal to "Total" or "Pr", to specify if only probabilities should be used.
        
    Update the vector of probabilities of activations of the hidden units, together with 
    the hidden state of the machine, if required. 
    """
    def updateHidden( self, x, mode = "Total" ):
        if x.ndim == 1:
            net = np.dot( self.W.T, x )            
            hp = special.expit( net )
            # Definition of bias unit that is always active
            hp[0] = 1
            if mode == "Total":
                y = np.random.rand( self.M+1 )
                # Redefine visible state
                h = np.zeros_like( hp ) 
                h[ hp >= y ] = 1
        else:
            net = np.dot( x, self.W )            
            hp = special.expit( net )
            # Definition of bias unit that is always active
            hp[:, 0] = 1
            if mode == "Total":
                y = np.random.rand( hp.shape[0], self.M+1 )
                # Redefine visible states
                h = np.zeros_like( hp ) 
                h[ hp >= y ] = 1
        
        if mode == "Total":
            return hp, h
        else:
            return hp
    
    """
    Computation of the visible state from a hidden one.
    ---------------------------------------------------
    Input arguments:
        x, input hidden state or mini-batch
        mode, string to specify if only probabilities should be used (mode = "Pr")
        
    Update the vector of probabilities of activations of the visible units, together with 
    the visible state of the machine, if required. 
    """
    def updateVisible( self, x, mode = "Total" ):
        if x.ndim == 1:
            net = np.dot( self.W, x )            
            vp = special.expit( net )
            # The bias unit is always active
            vp[0] = 1
            if mode == "Total":
                y = np.random.rand( self.N+1 )
                # Redefine visible state
                v = np.zeros_like( vp ) 
                v[ y <= vp ] = 1
        else:
            net = np.dot( x, self.W.T )            
            vp = special.expit( net )
            vp[:, 0] = 1
            if mode == "Total":
                y = np.random.rand( vp.shape[0], self.N+1 )
                v = np.zeros_like( vp ) 
                v[  y <= vp ] = 1

        if mode == "Total":
            return vp, v
        else:
            return vp

    """
    Compute free energy of a given set or example.
    -----------------------------------------------
    """
    def __freeEnergy(self, X):
        if X.ndim == 1:
            net = np.dot( self.W.T, X )
            tmp = np.log(1+np.exp(net))
            return  -np.dot( self.W.T[0], X) - np.sum( tmp )
        else:
            net = np.dot( X, self.W )
            energies = -np.dot( X, self.W[:,0]) - np.sum( np.log(1+np.exp(net)), axis = 1  )
            return np.sum( energies )        


    """
    Compute average free energies.
    ------------------------------
    Input arguments:
        X_train, training set (at least a subset)
        X_test,  test set

    As explained in Hinton's guide, use the comparison
    of the average free energies of the two sets of visibile instances 
    to monitor the overfitting.
    """
    def monitorOverfitting( self, X_train, X_test):            
        avg_train = self.__freeEnergy( X_train )/len(X_train)
        avg_test =  self.__freeEnergy( X_test )/len(X_test)
        label = 'Average free energy'        
        logl = 0
        return avg_train, avg_test, label

    """
    Compute the partecipation ratio PR_a
    ------------------------------------------
    Input: x, vector 
           a, coefficient of the partecipation ratio
    
    Function to estimate the number of non-zero components in a vector.
    If x has K nonzero and equal components, then PR = K for any a.    
    """
    def __PR( self, x, a ):
        if any( x != 0 ):        
            PR = np.sum( np.power( np.abs(x), a ) )**2
            PR /=  np.sum( np.power( np.abs(x), 2*a ) )
        else:
            PR = 0
        return PR
    
    def reconstructionScore( self, X, SGS = 1 ):
            # Obtain the reconstructions given by the machine
            self.v, self.h = self.GibbsSampling( v_init = X, SGS = SGS )
            
            # Compute their distance from the real visible states
            dist = np.linalg.norm( X-self.v, axis = 1 )
            MRE = np.sum( np.power( dist, 2 ) )
            nCorrect = np.sum( dist == 0 )

            # Print a small subset of the results
            if self.N < 100:
                np.set_printoptions( linewidth = 1000, formatter={'all':lambda x: str(x) if x > 0 else '_'} )
                indices = np.random.randint( len(X), size=5 ) 
                for i in indices:
                    print( X[i,1:].astype(int) )
                    print( self.v[i,1:].astype(int), \
                          "\t{:.2f}".format( np.linalg.norm(X[i] - self.v[i])**2), "\n"  )
                np.set_printoptions()

            
            return MRE/len(X), nCorrect*100/len(X)
    
    
    """
    Determine sparsity and effective temperature.
    ------------------------------------------------
    As did by Monasson, compute the degree of sparsity of each visible unit
    and the effective temperature of the machine.
    Since the important measure is the connectivity with the hidden layer,
    the self-threshold, i.e. the link with the bias unit, is discarded.
    """
    def analyzeWeights( self ):
        # Compute average sparsity
        p = 0
        for mu in range( 1, self.M +1 ):
            p += self.__PR( self.W[1:, mu], 2 )
        p /= (self.M * self.N )
        # Determine weight heterogeneities
        p_vector = np.zeros( self.N ) 
        # Compute normalization
        den = 1.0/self.M * np.linalg.norm( self.W[1:,1:], 'fro')**2
        for i in range( self.N ):
                p_vector[i] = 1.0/den * np.linalg.norm( self.W[i+1, 1:] )**2
        # Compute the effective temperature of the machine
        T = p/den
        
        return p, p_vector, T
        
    """
    Compute the number of magnetized and silent hidden units.
    -----------------------------------------
    Input: 
        a, power of the partecipation ratio to be used
        
    Use definitions of Monasson and Tubiana (pag. 6 SI).
    The default choice of a = 3 is related to the demonstration made for R-RBMs
    that shows that in this case \hat{L} converges to the true L, if all magnetizations are equal
    and in the thermodynamic limit.    
    """    
    def analyzeHiddenState( self, h, a = 3 ):
        
        # Partecipation ratio to get the number of (strongly, in the ReLU case) activated h.u.
        # Since the bias unit is always active (and it's not ReLU), it isn't considered
        L = self.__PR( h[1:], a ) 

        # Number of silent units
        S = np.sum( h[1:] == 0 )
    
        return L, S


    """
    Compute the magnetizations of the hidden units.
    -----------------------------------------
    Input: 
        x, visible instance
        L, number of strongly activated hidden units.

    Obtain a normalized measure of the superposition between the input x and the 
    different weight vectors w_{\mu}.
    Use again definitions of Monasson and Tubiana (pag. 6 SI).
    In particular, \tilde{m} is estimated through the mean of the top-L magnetizations.
    """    
    def analyzeMagnetizations( self, x, L ):
        # Cast the experimental value of L as an integer
        L = int( np.around( L ) )     
        
        # Create a vector for the magnetizations
        m = np.zeros( self.M  )
        for mu in range( self.M ):
            m[mu] = 2*np.dot( x[1:], self.W[1:, mu+1] ) - np.dot( np.ones(self.N), self.W[1:, mu+1] )
            m[mu] /= np.sum( np.abs( self.W[1:,mu+1] ) )
        
        # Compute \tilde{m}        
        if L > 0:
            net = np.dot( self.W.T[1:,1:], x[1:] )
            if np.sum( net != 0 ):
                ind = np.argsort( net ) 
                m_sorted = m[ind] 
            else:
                m_sorted= np.sort( m )
            m_t = 1.0/L * np.sum( m_sorted[-L:] )
        else:
            m_t = 0
            
        return m_t, m


##################################################
# Class for Restricted Boltzmann Machines
# with rectified linear hidden units
##################################################
class ReLU_RBM( BaseRBM ):
    def __init__( self, N, M, seed ):
        super().__init__( N,M,seed )
        del self.hp


    """
    Gibbs Sampling function.
    -----------------------
    Input arguments: 
        v_init, initial state (if present) of the machine in the visible layer
        h_init, initial hidden state of the machine, if present
        SGS, number of Steps of the Gibbs Sampling used in CD

    Create a MCMC either from a visible or a hidden state that makes the machine  daydream for SGS steps.
    """
    def GibbsSampling( self, v_init = np.empty(0), h_init = np.empty(0), SGS = 1 ):
        if SGS == 1:
            # Check if the chain should start from a visible state
            if v_init.size > 0:
                h_samples = self.updateHidden( v_init )
                vp, v_samples = self.updateVisible( h_samples )
            else:
                vp,v_samples = self.updateVisible( h_init )
                h_samples = self.updateHidden( vp )
        else:
            # Check if the chain should start from a visible state
            if v_init.size > 0:
                # Use the probabilities in the intermediate steps to reduce noise sampling
                h_samples = self.updateHidden( v_init )
                vp, v = self.updateVisible( h_samples )
                
                for k in range( SGS-2 ):
                    h_samples = self.updateHidden( vp )
                    vp, v = self.updateVisible( h_samples )
                
                h_samples = self.updateHidden( vp )
                vp, v_samples = self.updateVisible( h_samples )
            else:
                # Use the probabilities in the intermediate steps to reduce noise sampling
                vp,v = self.updateVisible( h_init )
                h_samples = self.updateHidden( vp )
                
                for k in range( SGS-2 ):
                    vp = self.updateVisible( h_samples, mode="Pr" )
                    h_samples = self.updateHidden( vp )
                
                vp, v_samples = self.updateVisible( h_samples )
                h_samples = self.updateHidden( vp )
                
        return v_samples, h_samples
    
    def __freeEnergy( self, v, beta = 1, W_ext = np.empty(0) ):
        if W_ext.size > 0:
            W = W_ext
        else:
            W = self.W
 
        en_g = beta * np.dot( v, W[:,0] )
        net = np.dot( W.T[1:,1:], v[1:] )
        # The different sign of the \theta term is due to the fact that I use a convention
        # different from Monasson, i.e. \theta = - \theta_{Monasson}
        # TO BE CHANGED?
        en_eff = 0.5*beta*np.linalg.norm( net + W[0,1:] )**2 
        arg_log = np.prod( np.sqrt( np.pi/(2*beta) )*special.erfc( np.sqrt( beta/2 )*(net+W[0,1:]) ) ) 

        en_eff += np.log( arg_log )
        
        return -en_g - en_eff


        
    def __phi( self, x ):
        # Add gaussian noise with null mean  and unit variance
        # to all preactivations of the hidden units 
        # given in input
        if x.ndim == 1:
            x = x + np.random.randn( self.M + 1 )
        else:
            x = x + np.random.randn( x.shape[0], self.M+1 )
        
        # Determine which hidden units are not active
        x[x<0] = 0
        
        return x

    """
    Computation of the hidden state from a visible one.
    ---------------------------------------------------
    Input arguments:
        x, input visible state
        W_ext, input weights (useful for AIS) 
        
    Update the hidden state of the machine. 
    """
    def updateHidden( self, x, W_ext = np.empty( 0 ) ):
        if W_ext.size > 0:
            W = W_ext
        else:
            W = self.W
            
        if x.ndim == 1:
            net = np.dot( W.T, x )
            h = self.__phi( net )
            
            # Definition of bias unit that is always active
            h[0] = 1
        else:
            net = np.dot( x, W )
            h = self.__phi( net )
            # Definition of bias unit that is always active
            h[:,0] = 1
    
        return h


    """
    Contrastive divergence.
    --------------------------------------------
    
    """
    def CD( self, v_example, SGS ):      
        
        # Positive phase 
        # Get the new hidden state
        self.h = self.updateHidden( v_example ) 
        
        #DEBUG
        h_pos = self.h 
        net_pos = np.dot( v_example, self.W )
        
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( v_example.T, self.h )
        
        # Negative phase
        # Make the machine daydream, starting from the hidden state
        # obtained in the positive phase
        # Gibbs Sampling for the negative learning phase
        for k in range( SGS-1 ):
            self.vp, self.v = super().updateVisible( self.h, mode = "Total" )
            self.h = self.updateHidden( self.vp )

        # Get the final visible state
        self.vp, self.v = super().updateVisible( self.h, mode = "Total" )
        # Obtain the correspondent hidden state
        self.h = self.updateHidden( self.vp  )

        # Compute negative contribute for SGD
        # Obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.vp.T, self.h )
        

        # Update the weights (and biases) 
        return  (delta_pos - delta_neg )

    """
    Persistent Contrastive Divergence.
    --------------------------------------------
    
    """
    def PCD( self, X, SGS, num_chains ):
        ###### Initialize the chains, if necessary
        if  not( hasattr( self, 'v_chains' ) ):
            # Generate bynary random states (see Hinton's guide pag. 4) 
            self.v_chains = np.random.randint( 2, size=(num_chains, self.N )  )
            # Add dimension for the biases
            self.v_chains = np.insert( self.v_chains, 0, 1, axis = 1 )
            # Compute the correspondent hidden states
            self.h_chains = self.updateHidden( self.v_chains ) 
        else:
            # Update the Markov chains
            self.v_chains, self.h_chains = self.GibbsSampling( h_init = self.h_chains, SGS=SGS )

        ###### Positive phase 
        # Compute the hidden states corresponding to the mini-batch X
        self.h = self.updateHidden( X ) 
        
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( X.T, self.h )
        
        ###### Negative phase
        # Compute negative contribute for SGD
        # Again, obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.v_chains.T, self.h_chains )
        
        
        # Return the update for the weights (and biases) 
        return  delta_pos - delta_neg 
    
    """
    Annealed Importance Sampling algorithm.
    -----------------------------------------
    Input arguments: 
        n_tot, total number of runs (that is M in the notation of Salakhutdinov)
        K, total number of samplings
        
    Get an approximation of the partition function of an RBM, using the AIS algorithm 
    defined in Salakhutdinov's article. In particular, it is considered a reference
    RBM, denoted as A, with null weight matrix and null ReLU-thresholds as the starting point
    of the annealing.
    """
    def AIS( self, K, n_tot ):
        # Define importance weights
        w = np.zeros( n_tot )
        
        # Define the weight matrix for the reference RBM 
        W_A = np.zeros( (self.N+1, self.M+1), dtype = float )
        W_A[:,0] = self.W[:,0] 
        
        
        # Compute Z for the reference RBM
        # Reference RBM with null weights and ReLU thresholds
        Z_A = np.sqrt(np.pi/2)**self.M * np.prod( 1 + np.exp( self.W[1:,0] ) )
        
        # Reference RBM with null weights but non-null ReLU thresholds
        #W_A[0,:] = self.W[0,:] 
        #term_hidden = 1.0
        #for mu in range( self.M ):
            #theta = W_A[0, mu+1]
            #term_hidden *= np.exp( -theta**2/2.0 )*np.sqrt( np.pi/2.0 )
            #if theta > 0:
                #term_hidden *= 1 + special.erf( theta/np.sqrt(2) )
            #elif theta < 0:
                #term_hidden *= special.erfc( -theta/np.sqrt(2) )
        #Z_A = np.prod( 1.0 + np.exp( self.W[1:,0] ) )*term_hidden
        
        

        # Define the marginalized distribution for the reference RBM
        p_A = special.expit( self.W[1:,0] )
        
        # Repeat n_tot times the annealing
        for i in range( n_tot ):
            # Generate the v's sequence
            for k in range( K ):
                # Define the inverse temperatures for the transition operator 
                # and the computation of w_i
                beta_curr = k*1.0/K
                beta_next = (k+1)*1.0/K

                if k == 0:
                    # Define the visible vector that spans the sequence v_1, ..., v_K
                    v_curr =  np.zeros( self.N, dtype= float )
                    # Add a dimension for the biases
                    v_curr = np.insert( v_curr, 0, 1, axis = 0)
        
                    # Sample v_1 through the marginalized of the reference RBM
                    y = np.random.rand( self.N ) 
                    v_curr[1:] = (y <= p_A)
                    
                    # Compute the starting contribution to w: p^*_1(v_1)/p^*_0(v_1)
                    w[i] =  -self.__freeEnergy( v_curr, 1.0-beta_next, W_A )-self.__freeEnergy( v_curr, beta_next )+ self.__freeEnergy( v_curr, 1.0-beta_curr, W_A ) 
            
                else:                        
                    # Compute h_A
                    h_A = self.updateHidden( (1-beta_curr) * v_curr, W_ext = W_A )
                    
                    # Compute h_B
                    h_B = self.updateHidden( beta_curr * v_curr )
                    
                    # Update v_curr through h_A, h_B 
                    net = (1-beta_curr) * np.dot( W_A, h_A ) + beta_curr * np.dot( self.W, h_B )            
                    vp = special.expit( net )
                    y = np.random.rand( self.N )
                    v_curr[1:] = (y <= vp[1:])                     
                
                    # Update the i-th importance weight
                    # Check if beta_next == 1.0
                    if k < K-1:
                        w[i] +=   -self.__freeEnergy( v_curr, 1.0-beta_next, W_A )-self.__freeEnergy( v_curr, beta_next ) + self.__freeEnergy( v_curr, 1.0-beta_curr, W_A ) + self.__freeEnergy( v_curr, beta_curr )
                    else:
                        w[i] +=  -self.__freeEnergy( v_curr, beta_next ) +self.__freeEnergy( v_curr, 1.0-beta_curr, W_A ) + self.__freeEnergy( v_curr, beta_curr ) 
        
            w[i] = np.exp( w[i] )
                
        Z_approx = np.sum( w )/n_tot * Z_A
        return Z_approx
        
        
    """
    Compute average log-likelihood.
    ------------------------------
    Input arguments:
        X_train, training set (at least a subset)
        X_test,  test set

    As explained in Monasson's article and in Hinton's guide, use the comparison of the average log-likelihood of the two sets of visibile instances  to monitor the overfitting.
    """
    def monitorOverfitting( self, X_train, X_test):            
        
        log_pstar = 0
        Z_approx = self.AIS( K=10, n_tot = 1 )
        for x in X_train:
            log_pstar -=  self.__freeEnergy( x )  
        avg_train = log_pstar/len(X_train)  - np.log(Z_approx)
        avg_train_2 = -log_pstar/len(X_train)
        
        log_pstar = 0
        for x in X_test:
            log_pstar -=  self.__freeEnergy( x )  
        avg_test = log_pstar/len(X_test)  - np.log(Z_approx)
        avg_test_2 = -log_pstar/len(X_test)

        return avg_train, avg_test, avg_train_2, avg_test_2
