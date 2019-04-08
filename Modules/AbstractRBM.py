from abc import ABC, abstractmethod
import numpy as np
from scipy import special
from Analyzer import Analyzer

##################################################
# Class for generic Restricted Boltzmann Machines
##################################################
class AbstractRBM(ABC):
    
    """
    Constructor.
    -------------
    Input arguments:
        N, number of visible units
        M, number of hidden units
        seed, integer to initialize the RNG
        p, array used to initialize the visible biases according to the statistics of the training examples 
        
    Define a RBM with N+1 visible units (the first of which is the bias unit)  and M+1 hidden units. 
    """
    def __init__(self, N, M, seed = None, p=np.empty(0)):
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
            for i in range(1,len(p)):
                if p[i] > 0:
                    self.W[i,0] = np.log( p[i]/(1.-p[i]) ) 
                else:
                    self.W[i,0] = 0
                
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
        x, regularization parameter
        lambda_x, weights cost
        c_e, decay coefficient  of the learning rate
        c_a, growth coefficient of the momentum parameter
        period_ovf, frequency of update of overfitting measures
        plots, bool used to deactive plots when not needed
        useProb, bool used in GibbsSampling
    
    Learn the weights according to the specified learning algorithm. 
    In order to be independent from the size of the mini-batches, the learning scale is scaled in the updating rule.
    """
    def fit( self, X_train, X_test, LA , SGS,  nMB, nEpochs, epsilon, alpha, x, lambda_x,  c_e, c_a, period_ovf, plots, useProb  ):

        # Initialize counter for the energies and sampling period
        counter = 0
        analyzer = Analyzer( self.N, self.M )
        
        # Define a validation set
        len_val = int(0.1*len(X_test))
        X_val = X_test[:len_val]
                
        # Standard size of the  mini-batches
        sizeMB = int( len( X_train )/nMB )
        
        # Initialize arrays for the statistics
        MRE = np.zeros( nEpochs, dtype = float )
        nCorrect = np.zeros( nEpochs, dtype = float )
        sparsity = np.zeros( int(nEpochs/period_ovf), dtype = float )        
        ovf = np.zeros( (2,  int(nEpochs/period_ovf)), dtype = float )
        
        bias_up = np.zeros( (nEpochs, nMB, self.M) )
        epsilon_0 = epsilon
        alpha_0 = alpha
        
        # Initialize velocity
        velocity = np.zeros( (self.N+1, self.M+1) )        
        
        # Iterate over X_train nEpochs times
        for t in range( nEpochs ):
            for ind in np.random.permutation( nMB ):
                if ind < nMB-1:
                    # Create a view of the i-th mini-batch
                    MB = X_train[ind*sizeMB:(ind+1)*sizeMB]
                else:
                    # Make a bigger mini-batch if len(X_train)%nMB != 0 
                    MB = X_train[ind*sizeMB:]

                # Compute weight updates deriving from log-likelihood (heuristically) maximization and the regularization term 
                if LA == 'CD':
                    W_updates = epsilon*(1./len(MB) *  self.CD( MB, SGS, useProb ) - lambda_x * self.regularize( x )  )
                elif LA == 'PCD':
                    W_updates = epsilon*( 1./len(MB) * self.PCD( MB, SGS, sizeMB, useProb ) - lambda_x * self.regularize( x )  )
                    
                # Update the velocity
                velocity =  W_updates + alpha*velocity

                #Update the weights (one time per MB)
                self.W += velocity
 
            # Compute and print statistics of the current epoch
            print("---------------Epoch {}--------------".format(t+1))
            self.GibbsSampling( X_val )
            MRE[t], nCorrect[t] = self.reconstructionScore( X_val, self.v )            
            
            print( "Mean Squared Error = ", MRE[t] )
            print( "Correct reconstructions (%%) = %.2f \n" % nCorrect[t] ) 

            # Update the arrays for the plots
            if plots and (t % period_ovf == 0):
                # Compute the energies and the sparsity of the model
                ovf[:, counter] = self.monitorOverfitting( X_train[:len_val], X_val )
                sparsity[counter],__, __, __, T = analyzer.analyzeWeights( self.W ) 
                counter += 1

            # DEBUG
            #MRE[t] = W_updates[0,1] 

            # Increase alpha and decrease epsilon while learning progresses
            epsilon = epsilon_0* np.exp(-c_e*(t+1.)/nEpochs)
            if alpha < 0.9: 
                alpha = alpha_0* np.exp(c_a*(t+1.)/nEpochs)

        # DEBUG
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range( self.M ):
            plt.plot( np.arange(nEpochs), np.sum( bias_up, axis = 1 )[:,i] )
        plt.figure()
        plt.plot( np.arange(nEpochs), np.sum( bias_up, axis = 1 )[:,0] )
        plt.plot( np.arange(nEpochs), [np.mean( np.sum( bias_up, axis = 1 )[10:,0] )]*nEpochs )
        plt.plot( np.arange(nEpochs), [0]*nEpochs )
        
        plt.show()
        # DEBUG
        #np.save('debug.npy', bias_up)
        return MRE, nCorrect, sparsity, ovf
        

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
        ## Determine the signs of the weights
        W_updates = np.zeros( (self.N+1, self.M+1) )
        W_updates[1:,1:] = np.sign( self.W[1:,1:] )
        
        # Use the weights to calibrate the update (one for each hidden unit)
        coeffs = np.power( np.sum( np.abs( self.W[1:, 1:] ), axis = 0 ), x-1 )
        W_updates[1:,1:] = np.multiply( W_updates[1:,1:], coeffs )
                
        ## Determine the signs of the weights
        #W_updates = np.zeros( (self.N+1, self.M+1) )
        #W_updates = np.sign( self.W )

        
        ## Use the weights to calibrate the update (one for each hidden unit)
        #coeffs = np.power( np.sum( np.abs( self.W ), axis = 0 ), x-1 )
        #W_updates = np.multiply( W_updates, coeffs )

        return W_updates
    
    """
    Contrastive divergence function.
    -------------------------------------
    Input arguments: 
        v_example, training mini-batch for the positive phase
        SGS, number of Gibbs Sampling steps
        useProb, exploit probabilities vectors to reduce noise sampling 
            in the negative phase
    
    Return the weight matrix update to be used with Stochast Gradient Descent.
    """
    def CD( self, v_example, SGS, useProb ):
        
        # Positive phase 
        # Get the new hidden state
        self.updateHidden( v_example ) 
                
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( v_example.T, self.h )
        
        # Negative phase
        # Make the machine daydream, starting from the hidden state
        # obtained in the positive phase
        self.GibbsSampling( h_init=self.h, SGS=SGS-1, useProb = useProb )
        
        # Get the final visible state
        self.updateVisible( self.h )
        # Obtain the correspondent hidden state
        self.updateHidden( self.vp )
        
        # Compute negative contribute for SGD
        # Obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.v.T, self.h )

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
    def GibbsSampling( self, v_init = np.empty(0), h_init = np.empty(0), SGS = 1, useProb = False ):
            if SGS <= 0: return -1
            # Check if the chain should start from a visible state
            if v_init.size > 0:
                # First full update
                self.updateHidden(  v_init )
                self.updateVisible( self.h )

                # Use the probabilities in the intermediate steps to reduce noise sampling  
                if useProb:
                    for k in range( SGS-1 ):
                        if k == SGS-1:
                            # Last full update
                            self.updateHidden(  self.vp )
                            self.updateVisible( self.h )
                        else:
                            self.updateHidden(  self.vp, useProb )
                            # Hidden units could not have the probability vector attribute
                            try:
                                self.updateVisible( self.hp, useProb )
                            except AttributeError:
                                self.updateVisible( self.h, useProb )
                else:
                    for k in range( SGS-1 ):
                        self.updateHidden(  self.v )
                        self.updateVisible( self.h )                    
            else:
                # First full update
                self.updateVisible( h_init )
                self.updateHidden(  self.v )

                # Use the probabilities in the intermediate steps to reduce noise sampling  
                if useProb:
                    for k in range( SGS-1 ):
                        if k == SGS-1:
                            # Last full update
                            self.updateVisible( self.hp )
                            self.updateHidden(  self.v )
                        else:
                            # Hidden units could not have the probability vector attribute
                            try:
                                self.updateVisible( self.hp, useProb )
                            except AttributeError:
                                self.updateVisible( self.h )
                            self.updateHidden(  self.vp, useProb )
                else:
                    for k in range( SGS-1 ):
                        self.updateVisible( self.h )
                        self.updateHidden(  self.v )
               
            return 0
    """
    Computation of the hidden state correspondent to an input visible one.
    ------------------------------------------------------------------------
    """
    @abstractmethod
    def updateHidden(self):
        pass


    """
    Computation of the visible state correspondent to an input hidden  one.
    -------------------------------------------------------------------------
    """
    @abstractmethod
    def updateVisible(self):
        pass

    """
    Compute average free energies or average log-likelihod.
    -----------------------------------------------------------
    Input arguments:
        X_train, training set (at least a subset)
        X_test,  test set

    As explained in Hinton's guide, use the comparison   of the average free energies of the two sets of visibile instances 
    to monitor the overfitting.
    # DEBUG
    As explained in Monasson's article and in Hinton's guide, use the comparison of the average log-likelihood of the two sets of visibile instances  to monitor the overfitting.
    """
    def monitorOverfitting( self, X_train, X_test):            
        ## Average free energies
        avg_train = self.freeEnergy( X_train, self.W )/len(X_train)
        avg_test =  self.freeEnergy( X_test,  self.W )/len(X_test)
        
        # Approximated log-likelihod
        #p = np.mean( X_train, axis = 0 )
        #Z_approx = self.AIS_2( K=10000, n_tot = 100, p=p[1:] )
        #Z_exact = self.partition()
        #print( Z_approx, Z_exact )
        #input()
        #avg_train = -self.freeEnergy( X_train, self.W )/len(X_train) - np.log(Z_approx)
        #avg_test =  -self.freeEnergy( X_test,  self.W )/len(X_test) - np.log(Z_approx)
        
        ## DEBUG
        #print( avg_train )
        #print( -self.freeEnergy( X_train, self.W )/len(X_train) )
        #print( np.log(Z_approx) )
        #input()
        return avg_train, avg_test

    """
    Compute the reconstruction performance.
    ---------------------------------------------------------
    Input: 
        X, set of examples
        v_rec, set of reconstructions
    
    Evaluate the mean reconstruction error over the set X and
    the number of correct reconstructions as a percentage.
    """
    def reconstructionScore( self, X, v_rec, SGS = 1 ):
            ## Obtain the reconstructions given by the machine
            #if not average:
                #self.GibbsSampling( v_init = X, SGS = SGS )
                #v_rec = self.v
            #else:
                #v_rec = self.findMaxima( X )
                
            # Compute their distance from the real visible states
            dist = np.linalg.norm( X-v_rec, axis = 1 )
            MRE = np.sum( np.power( dist, 2 ) )
            nCorrect = np.sum( dist == 0 )
            
            return MRE/len(X), nCorrect*100/len(X)


    """
    Persistent Contrastive Divergence.
    --------------------------------------------
    """
    def PCD( self, X, SGS, num_chains, useProb ):
        ###### Initialize the chains, if necessary
        if  not( hasattr( self, 'v_chains' ) ):
            # Generate bynary random states (see Hinton's guide pag. 4) 
            self.v_chains = np.random.randint( 2, size=(num_chains, self.N )  )
            # Add dimension for the biases
            self.v_chains = np.insert( self.v_chains, 0, 1, axis = 1 )
            # Compute the correspondent hidden states
            self.updateHidden( self.v_chains )
            self.h_chains = np.copy( self.h )
        else:
            # Update the Markov chains
            self.GibbsSampling( h_init = self.h_chains, SGS=SGS, useProb= useProb )
            self.v_chains = np.copy( self.v )
            self.h_chains = np.copy( self.h )
            
        ###### Positive phase 
        # Compute the hidden states corresponding to the mini-batch X
        self.updateHidden( X ) 
        
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
        
        ## Reference RBM with null weights but non-null ReLU thresholds
        ##W_A[0,:] = self.W[0,:] 
        ##term_hidden = 1.0
        ##for mu in range( self.M ):
            ##theta = W_A[0, mu+1]
            ##term_hidden *= np.exp( -theta**2/2.0 )*np.sqrt( np.pi/2.0 )
            ##if theta > 0:
                ##term_hidden *= 1 + special.erf( theta/np.sqrt(2) )
            ##elif theta < 0:
                ##term_hidden *= special.erfc( -theta/np.sqrt(2) )
        ##Z_A = np.prod( 1.0 + np.exp( self.W[1:,0] ) )*term_hidden
        

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
        
                    # Sample v_1 through the marginalized distribution of the reference RBM
                    y = np.random.rand( self.N ) 
                    v_curr[1:] = (y <= p_A)
                    
                    # Compute the starting contribution to w: p^*_1(v_1)/p^*_0(v_1)
                    w[i] =  -self.freeEnergy( v_curr, W_A, 1.0-beta_next )-self.freeEnergy( v_curr, self.W, beta_next )\
                            + self.freeEnergy( v_curr, W_A, 1.0-beta_curr ) 
            
                else:                        
                    # Compute h_A
                    self.updateHidden( (1-beta_curr) * v_curr, W = W_A )
                    net = (1-beta_curr) * np.dot( W_A, self.h )
                    
                    # Compute h_B
                    self.updateHidden( beta_curr * v_curr )
                    net +=  beta_curr * np.dot( self.W, self.h )
                    
                    # Update v_curr through h_A, h_B 
                    vp = special.expit( net )
                    y = np.random.rand( self.N )
                    v_curr[1:] = (y <= vp[1:])                     
                
                    # Update the i-th importance weight
                    # Check if beta_next == 1.0
                    if k < K-1:
                        w[i] +=   -self.freeEnergy( v_curr, W_A, 1.0-beta_next )-self.freeEnergy( v_curr, self.W, beta_next ) + self.freeEnergy( v_curr, W_A, 1.0-beta_curr ) + self.freeEnergy( v_curr, self.W, beta_curr )
                    else:
                        w[i] +=  -self.freeEnergy( v_curr,self.W, beta_next ) +self.freeEnergy( v_curr, W_A,1.0-beta_curr ) + self.freeEnergy( v_curr, self.W, beta_curr ) 
        
            w[i] = np.exp( w[i] )
                
        Z_approx = np.sum( w )/n_tot * Z_A
        return Z_approx
        
        
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
    def AIS_2( self, K, n_tot, p ):
        def __pstar( v, W_A, beta ):
            
            p_star = np.exp( (1-beta)*np.dot(W_A[1:,0], v[1:]) )* np.prod( 1+np.exp((1-beta)*np.dot(W_A.T, v ) ) )
            p_star *= np.exp( beta*np.dot(self.W[1:,0], v[1:]) ) * np.prod( 1+np.exp(beta*np.dot(self.W.T, v ) ) ) 
            
            return p_star
            
        # Define importance weights
        w = np.zeros( n_tot )
        
        # Define the weight matrix for the reference RBM 
        W_A = np.zeros( (self.N+1, self.M+1), dtype = float )
        W_A[1:,0] = np.log( np.divide( p, 1-p ) ) 
        
        # Compute Z for the reference RBM
        # Reference RBM with null weights and ReLU thresholds
        Z_A = np.power( np.sqrt(np.pi/2), self.M ) * np.prod( 1 + np.exp( W_A[1:,0] ) )

        # Define the marginalized distribution for the reference RBM
        p_A = special.expit( W_A[1:,0] )
        
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
                    w[i] =  __pstar( v_curr, W_A, beta_next ) /__pstar( v_curr, W_A, beta_curr ) 
                    
                else:                        
                    # Compute h_A
                    self.updateHidden( (1-beta_curr) * v_curr, W = W_A )
                    net = (1-beta_curr) * np.dot( W_A, self.h )
                    
                    # Compute h_B
                    self.updateHidden( beta_curr * v_curr )
                    net +=  beta_curr * np.dot( self.W, self.h )
                    
                    # Update v_curr through h_A, h_B 
                    vp = special.expit( net )
                    vp[0] = 1 
                    # Sample v_curr
                    y = np.random.rand( self.N+1  )
                    v_curr = (y <= vp )                     
                
                    # Update the i-th importance weight
                    w[i] *= __pstar( v_curr, W_A, beta_next )/__pstar(v_curr,W_A, beta_curr)
                    
            #w[i] = np.exp( w[i] )
                
        Z_approx = np.mean( w ) * Z_A
        return Z_approx


    def partition( self ):
        if self.N >= 20: 
            return
        else:
            Z = 0
            import itertools
            lst = list(itertools.product([0, 1], repeat=self.N))
            vecs = np.array( lst )
            vecs = np.insert( vecs, 0, 1, axis = 1 )
            for v in vecs:
                Z += np.exp( -self.freeEnergy( np.array(v), self.W ) )
            return Z
        
    
    
    def findMaxima( self, X, K = 25 ):
        if X.ndim > 1:
            # Initialize a vector to store the averaged reconstructions
            v_rec_avg = np.zeros( (len(X),self.N+1 ) )
            
            # Iterate over the input set
            for ind, v in enumerate(X):
                # Construct a mini-batch of identical copies of the same input vector
                V = np.tile( v, (K,1) )
                    
                # Get the correspondent hidden states
                self.updateHidden( V )
                
                # Compute the conditional average of the hidden states
                h_avg = np.mean( self.h, axis = 0 )
                
                # Return to the visible layer
                self.updateVisible( h_avg )
                
                # Save the result
                v_rec_avg[ ind ] = self.v 
        else:
           # Construct a mini-batch of identical copies of the same input vector
                V = np.tile( X, (K,1) )
                    
                # Get the correspondent hidden states
                self.updateHidden( V )
                
                # Compute the conditional average of the hidden states
                h_avg = np.mean( self.h, axis = 0 )
                
                # Return to the visible layer
                self.updateVisible( h_avg )
                
                # Save the result
                v_rec_avg = self.v 
                
        return v_rec_avg