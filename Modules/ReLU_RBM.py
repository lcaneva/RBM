import numpy as np
from AbstractRBM import AbstractRBM
from scipy import special

##################################################
# Class for Restricted Boltzmann Machines
# with rectified linear hidden units
##################################################
class ReLU_RBM( AbstractRBM ):
    
    """
    Constructor.
    --------------------------------------------------------------------
    Use parent class Constructor, except for the  probabilities vectors
    of the hidden units that are not defined.
    """    
    def __init__( self, N, M, seed, theta_init, g_init=np.empty(0) ):
        # Initialize probabilities vectors
        self.vp = np.empty( N+1, dtype = float )                        
        super().__init__( N,M,seed, theta_init, g_init )


    """
    Computation of the visible state from a hidden one.
    ---------------------------------------------------------------------
    Input arguments:
        x, input hidden state or mini-batch
        useProb, bool to specify if only probabilities should be used
        
    Update the vector of probabilities of activations of the visible units, together with 
    the visible state of the machine, if required. 
    """
    def updateVisible( self, x, useProb = False ):
        # Determine if x is a vector or a matrix
        if x.ndim == 1:
            net = np.dot( self.W, x )            
            self.vp = special.expit( net )
            # Bias unit is always active
            self.vp[0] = 1
            if not useProb:
                y = np.random.rand( self.N+1 )
                self.v = np.zeros(  self.N+1 )
                self.v[ y <= self.vp ] = 1
        else:
            net = np.dot( x, self.W.T )            
            self.vp = special.expit( net )
            self.vp[:, 0] = 1
            if not useProb:
                l = self.vp.shape[0]
                y = np.random.rand( l, self.N+1 )
                self.v = np.zeros( (l, self.N+1) )
                self.v[  y <= self.vp ] = 1


    """
    Noisy ReLU activation function.
    -----------------------------------
    """
    def __phi( self, x ):
        # Add gaussian noise with null mean  and unit variance
        # to all preactivations of the hidden units given in input
        if x.ndim == 1:
            y = x +  np.random.randn( self.M + 1 )
        else:
            y = x + np.random.randn( x.shape[0], self.M+1 )
        
        # Determine which hidden units are not active
        y[y<0] = 0
        
        return y



    """
    Computation of the hidden state from a visible one.
    ---------------------------------------------------
    Input arguments:
        x, input visible state
        useProb, bool to avoid sampling (useful to maintain the interface of BaseRBMs)
        W, specific weight matrix
        
    Update the hidden state of the machine. 
    """
    def updateHidden( self, x, useProb = False, W = np.empty(0) ):
        # Determine which weights to use
        if W.size == 0:
            W = self.W
        
        # Update according to the number of dimensions of the input
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
    Compute free energy of an example or of a set of examples.
    --------------------------------------------------------------
    """
    def freeEnergy( self, X,  W, beta=1 ): 
        if X.ndim == 1:
            # Local visible fields energy
            en_g = beta*np.dot( X, W[:,0] )
            # Compute effective energy
            part_net = np.dot( W.T[1:,1:], X[1:] )
            net =  part_net + W[0,1:]
            arg_log = np.prod( np.sqrt( np.pi/(2*beta) )*special.erfc( np.sqrt( beta/2 )*net ) ) 
            en_eff = 0.5*np.linalg.norm( net )**2 +  np.log( arg_log )
            return -en_g - en_eff
        else:
            # Local fields energies (one for each example in X)
            en_g = np.dot( X, W[:,0] )            

            # Compute effective energy
            part_net = np.dot( X[:,1:], W[1:,1:] )   
            thresholds = np.tile( W[0,1:], (len(part_net),1) ) 

            # Use Hinton's convention and not Monasson's one
            net = part_net + thresholds
            arg_log = np.prod( np.sqrt( np.pi/(2*beta) )*special.erfc( np.sqrt( beta/2 )*net ), axis=1 ) 
            en_eff = 0.5*np.power( np.linalg.norm( net, axis=1 ), 2 )            
            en_eff += np.log( arg_log )
            
            return -np.sum(en_g) - np.sum(en_eff)


