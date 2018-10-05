import numpy as np
import matplotlib.pyplot as plt
from pudb import set_trace
from scipy import special


##################################################
# Class for Restricted Boltzmann Machines
# Use same interface of other ML methods in Scikit-learn
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
            self.vp = np.empty( N+1, dtype = float )
            
            # Initialize hidden layer
            self.h = np.empty( M+1, dtype = float )
            
            # Initialize probabilities vectors
            if not( isinstance( self, ReLU_RBM ) ):
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
            Notice that g_i is -\theta_i in the notation of Ackley. 
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
        epsilon, learning rate
        alpha, coefficients for the momentum
        nEpochs, maximum number of cycles through the mini-batch X_train
        c_e, decay coefficient  of the learning rate
        c_a, growth coefficient of the momentum parameter
        periods, frequency of update of alpha and epsilon and sampling of the energies
        plots, bool used to deactive plots when not needed
    
    Learn the weights according to the specified learning algorithm. 
    In order to be independent from the size of the mini-batches, the learning scale is scaled in the updating rule.
    """
    def fit( self, X_train, X_test, LA , SGS,  epsilon, alpha, lambda_x, nEpochs, c_e, c_a, period, plots  ):
        # Initialize counter for the energies and sampling period
        counter = 0
        period_ovf = 10
        # Size of the input mini-batch
        sizeMB = len( X_train )

        # Initialize arrays for the statistics
        MRE = np.zeros( nEpochs, dtype = float )
        nCorrect = np.zeros( nEpochs, dtype = float )
        p_arr = np.zeros( int(nEpochs/period_ovf), dtype = float )        
        ovf_train = np.zeros( int(nEpochs/period_ovf), dtype = float )
        ovf_test = np.zeros( int(nEpochs/period_ovf), dtype = float )        
        velocity = np.zeros( (self.N+1, self.M+1) )        
        
        # Iterate through X_train nEpochs times
        for t in range( nEpochs ):
            # Compute the contribution to the weight updates deriving from the log-likelihood (heuristically) 
            W_updates = epsilon/sizeMB * self.CD( X_train, SGS )
            # Compute the regularization contribution
            W_updates -=  epsilon/sizeMB * lambda_x * self.regularize( x=2 )  
            
            # Iterate through X_train
            for i in range( sizeMB ):
                # Compute reconstruction error
                sq_dist = np.linalg.norm(X_train[i] - self.v[i])**2 
                MRE[t] += sq_dist            
                # Check if it is the correct reconstruction
                if sq_dist == 0:
                    nCorrect[t] += 1
                
            # Update the velocity
            velocity =  W_updates + alpha*velocity
            
            # Update the weights (one time per epoch)
            self.W += velocity
            
            # Update the coefficients
            if t % period == 0:
                # Geometric decay of the learning rate
                epsilon *= c_e 

                # Increase alpha towards the end of learning
                if t > nEpochs*0.5: 
                    alpha *= c_a
                
            # Compute the energies and the sparsity
            if plots:
                if t % period_ovf == 0:
                    ovf_train[counter], ovf_test[counter] = self.monitorOverfitting( X_train, X_test )
                    p_arr[counter], aux, T = self.analyzeWeights() 
                    counter += 1

            # Print statistics of the current epoch
            MRE[t] /= sizeMB
            nCorrect[t] = nCorrect[t]*100.0/sizeMB
            
            print("---------------Epoch {}--------------".format(t+1))
            print( "Mean Squared Error = ", MRE[t] )
            print( "Correct reconstructions (%%) = %.2f \n" % nCorrect[t] ) 
        

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
            plt.show()            
            # Monitor the overfitting
            plt.figure()
            plt.plot([period_ovf*i for i in range( len( ovf_train ) )], ovf_train, label="Training set" )
            plt.plot([period_ovf*i for i in range( len( ovf_test ) )], ovf_test, label = "Test set")
            plt.legend()
            if not( isinstance( self, ReLU_RBM ) ):
                plt.ylabel('Average free energy')
            else:
                plt.ylabel('Log-likelihood')
            plt.xlabel('Epochs')
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
        self.updateHidden( v_example, mode = "Pr" ) 
        
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( v_example.T, self.hp )
        
        # Negative phase
        # Make the machine daydream, starting from the hidden state
        # obtained in the positive phase
        
        # Gibbs Sampling for the negative learning phase
        # Use only PROBABILITIES except for the last sampling
        for k in range( SGS ):
            if not k == SGS-1:
                self.updateVisible( self.hp, mode = "Pr" )
                self.updateHidden(  self.vp, mode = "Pr" )
            else:
                # Get the total visible state
                self.updateVisible( self.hp, mode = "Total" )
                # Obtain also the hidden state
                self.updateHidden( self.vp, mode = "Total" )
        
        # Compute negative contribute for SGD
        # Again, obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.vp.T, self.hp )
                        
        # Update the weights (and biases) 
        return  (delta_pos - delta_neg )
        
    """
    Gibbs Sampling function.
    -----------------------
    Input arguments: 
        v_init, initial state (if present) of the machine in the visible layer
        h_init, initial hidden state of the machine, if presente
        SGS, number of Steps of the Gibbs Sampling used in CD

    Create a MCMC either from a visible or a hidden state that makes the machine
    daydream.
    """
    def GibbsSampling( self, v_init = np.array([]), h_init = np.array( [] ), SGS = 1 ):
        for k in range( SGS ):
            if h_init.size == 0:
                # Complete Gibbs Sampling starting from a visible state
                self.updateHidden( v_init )
                self.updateVisible( self.h )
            else:
                # Complete Gibbs Sampling starting from a hidden state
                self.updateVisible( h_init )
                self.updateHidden( self.v )

    
    """
    Activation function.
    ------------------
    Input argument: 
        x, pre-activation of the considered unit
    
    Return the activation probability of an unit, given its input.
    Homogeneity is assumed, i.e. all the units are logistic in the current base class.
    """
    def __phi(self, x):
        return 1/(1+np.exp(-x))

    """
    Computation of the hidden state from a visible one.
    ---------------------------------------------------
    Input arguments:
        x, input visible state
        mode, string, equal to "Total" or "Pr", to specify if only probabilities should be used.
        
    Update the vector of probabilities of activations of the hidden units, together with 
    the hidden state of the machine, if required. 
    """
    def updateHidden( self, x, mode = "Total" ):
        if x.ndim == 1:
            net = np.dot( self.W.T, x )            
            self.hp = self.__phi( net )
            # Definition of bias unit that is always active
            self.hp[0] = 1
            if mode == "Total":
                y = np.random.rand( self.M+1 )
                # Redefine visible state
                self.h = np.zeros_like( self.hp ) 
                self.h[ self.hp >= y ] = 1
        else:
            net = np.dot( x, self.W )            
            self.hp = self.__phi( net )
            # Definition of bias unit that is always active
            self.hp[:, 0] = 1
            if mode == "Total":
                y = np.random.rand( self.hp.shape[0], self.M+1 )
                # Redefine visible states
                self.h = np.zeros_like( self.hp ) 
                self.h[ self.hp >= y ] = 1

    
    """
    Computation of the visible state from a hidden one.
    ---------------------------------------------------
    Input arguments:
        x, input hidden state
        mode, string to specify if only probabilities should be used (mode = "Pr")
        
    Update the vector of probabilities of activations of the visible units, together with 
    the visible state of the machine, if required. 
    """
    def updateVisible( self, x, mode = "Total" ):
        if x.ndim == 1:
            net = np.dot( self.W, x )            
            self.vp = self.__phi( net )
            # Definition of bias unit that is always active
            self.vp[0] = 1
            if mode == "Total":
                y = np.random.rand( self.N+1 )
                # Redefine visible state
                self.v = np.zeros_like( self.vp ) 
                self.v[ self.vp >= y ] = 1
        else:
            net = np.dot( x, self.W.T )            
            self.vp = self.__phi( net )
            # Definition of bias unit that is always active
            self.vp[:, 0] = 1
            if mode == "Total":
                y = np.random.rand( self.vp.shape[0], self.N+1 )
                # Redefine visible states
                self.v = np.zeros_like( self.vp ) 
                self.v[ self.vp >= y ] = 1

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
    of the average free energies of the two set of visibile instances 
    to monitor the overfitting.
    """
    def monitorOverfitting( self, X_train, X_test):
            
        # DEBUG
        #set_trace()
        if not( isinstance( self, ReLU_RBM ) ):            
            avg_train = self.__freeEnergy( X_train )/len(X_train)
            avg_test =  self.__freeEnergy( X_train )/len(X_test)
        else:
            logl = 0
            for x in X_train:
                logl +=   -self._ReLU_RBM__freeEnergy( x )  - np.log(self.AIS( K=100, n_tot = 5 ) ) 
            avg_train = logl/len(X_train)

            #en_train = 0
            #en_test  = 0
            #for i in range( len(X_train) ):
                #tmp =  self._ReLU_RBM__freeEnergy( X_train[i] )
                #en_train += tmp
            #for i in range( len(X_test) ):
                #en_test += self._ReLU_RBM__freeEnergy( X_test[i] )

            #avg_test =  self._ReLU_RBM__freeEnergy( X_test[0] )/len(X_test)
        
        return avg_train, avg_test

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
    def analyzeHiddenState( self, a = 3 ):
        
        # Partecipation ratio to get the number of (strongly, in the ReLU case) activated h.u.
        # Since the bias unit is always active (and it's not ReLU), it isn't considered
        L = self.__PR( self.h[1:], a ) 

        # Number of silent units
        S = np.sum( self.h[1:] == 0 )
    
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
            if np.sum( net != 0):
                ind = np.argsort( net ) 
                m_sorted = m[ind] 
                ## DEBUG
                #print( net, m )
                #print( "After ordering" )
                #print( net[ind], m_sorted )
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
    # TO BE CHANGED?
    def __energy( self, v, h ):
        # Compute the energy of the current state of the machine
        en_W = np.dot( np.dot( self.W[1:,1:], h[1:] ), v[1:] )
        # Bernoulli potential (binary visible units): v_i * g_i 
        en_g = np.dot( v, self.W[:,0] )
        # ReLU potential: 0.5*h_\mu^2 + h_\mu * \theta_\mu
        # The different sign of the \theta term is due to the fact that I use a convention
        # different from Monasson, i.e. \theta = - \theta_{Monasson}
        # TO BE CHANGED?
        en_t = 0.5*np.dot( h, h )  - np.dot( h, self.W[0] ) 
        return -en_W - en_g + en_t 
    
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
        en_eff = 0.5*beta*np.linalg.norm( net + W[0,1:] )**2 + np.log( np.sqrt( np.pi/(2*beta) ) )*self.M 
        for mu in range( self.M ):
            tmp = net[mu]+W[0,mu+1]
            with np.errstate(divide='raise'):
                try:
                    en_eff += np.log( 1. + np.sign(tmp)*special.erf( np.sqrt(beta/2)*np.abs( tmp ) ) )
                except FloatingPointError:
                    print(tmp,  np.sign(tmp)*special.erf( np.sqrt(beta/2)*np.abs( tmp ) ) )
                    input()
        return -en_g - en_eff

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
        # Compute Z for the reference RBM
        # TO BE CHANGED?
        #Z_A = 2**self.M * np.prod( 1 + np.exp( self.W[1:,0] ) )
        Z_A = np.sqrt(np.pi/2)**self.M * np.prod( 1 + np.exp( self.W[1:,0] ) )

        # Define the weight matrix for the reference RBM 
        W_A = np.zeros( (self.N+1, self.M), dtype = float )
        W_A = np.insert( W_A, 0, self.W[:,0], axis = 1 ) 
        
        # Define the marginalized distribution for the reference RBM
        p_A = 1.0/( 1 + np.exp(-self.W[1:,0]) )
        
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
                    v_curr = np.zeros( self.N, dtype= float )
                    # Add a dimension for the biases
                    v_curr = np.insert( v_curr, 0, 1, axis = 0)
        
                    # Sample v_1 through the marginalized of the reference RBM
                    y = np.random.rand( self.N ) 
                    v_curr[1:] = (y <= p_A)
                    
                    # Compute the starting contribution to w: p^*_1(v_1)/p^*_0(v_1)
                    w[i] = np.exp( -self.__freeEnergy( v_curr, 1.0-beta_next, W_A )-self.__freeEnergy( v_curr, beta_next ) )  
                    w[i] /= np.exp( -self.__freeEnergy( v_curr, 1.0-beta_curr, W_A ) )
            
                else:                        
                    # Compute h_A
                    self.updateHidden( (1-beta_curr) * v_curr, W_ext = W_A )
                    h_A = np.copy( self.h )
                    
                    # Compute h_B
                    self.updateHidden( beta_curr * v_curr )
                    h_B = np.copy( self.h )
                    
                    # Update v_curr through h_A, h_B 
                    net = (1-beta_curr) * np.dot( W_A, h_A ) + beta_curr * np.dot( self.W, h_B )            
                    vp = 1.0/(1 + np.exp( -net ) )
                    y = np.random.rand( self.N )
                    v_curr[1:] = (y <= vp[1:])                     
                
                    # Update the i-th importance weight
                    # Check if beta_next == 1.0
                    if k < K-1:
                        w[i] *=  np.exp( -self.__freeEnergy( v_curr, 1.0-beta_next, W_A )-self.__freeEnergy( v_curr, beta_next ) )
                    else:
                        w[i] *=  np.exp( -self.__freeEnergy( v_curr, beta_next ) )
                                        
                    #with np.errstate(divide='raise'):
                        #try:  
                            #w[i] /=  np.exp( -self.__freeEnergy( v_curr, 1.0-beta_curr, W_A )-self.__freeEnergy( v_curr, beta_curr ) )
                        #except:
                            #set_trace()

                
        Z_approx = np.sum( w )/n_tot * Z_A
        return Z_approx
        

        
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
            self.h = self.__phi( net )
            
            # Definition of bias unit that is always active
            self.h[0] = 1
        else:
            net = np.dot( x, W )
            self.h = self.__phi( net )
            # Definition of bias unit that is always active
            self.h[:,0] = 1
    
    """
    Contrastive divergence.
    --------------------------------------------
    
    """
    def CD( self, v_example, SGS ):        
        
        # Positive phase 
        # Get the new hidden state
        self.updateHidden( v_example ) 
        
        # Compute positive contribute for SGD
        # Get a matrix in R^{ N+1, M+1 } 
        delta_pos = np.dot( v_example.T, self.h )
        
        ### DEBUG
        #delta_pos_2 = np.zeros_like( delta_pos )
        #for i in range( len( v_example ) ):
            #delta_pos_2 += np.outer( v_example[i], self.h[i] )
            
        
        # Negative phase
        # Make the machine daydream, starting from the hidden state
        # obtained in the positive phase
        
        # Gibbs Sampling for the negative learning phase
        # Use only PROBABILITIES except for the last sampling
        for k in range( SGS ):
            if not k == SGS-1:
                super().updateVisible( self.h, mode = "Pr" )
                self.updateHidden(  self.vp )
            else:
                # Get the total visible state
                super().updateVisible( self.h, mode = "Total" )
                # Obtain also the hidden state
                self.updateHidden( self.vp  )
        
        # Compute negative contribute for SGD
        # Again, obtain a matrix in R^{ N+1, M+1 }
        delta_neg = np.dot( self.vp.T, self.h )
 
        # Update the weights (and biases) 
        return  (delta_pos - delta_neg )

    
        
