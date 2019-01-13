import numpy as np
from scipy import special
from AbstractRBM import AbstractRBM

##################################################
# Class for Restricted Boltzmann Machines
# with binary logistic units
##################################################
class BaseRBM( AbstractRBM ):
    
    """
    Constructor.
    ----------------------------------------------
    Use parent class Constructor, except for the
    probabilities vectors of the hidden units 
    that are not defined.
    """    
    def __init__( self, N, M, seed, p=np.empty(0) ):
        # Initialize probabilities vectors
        self.vp = np.empty( N+1, dtype = float )
        self.hp = np.empty( M+1, dtype = float )
        super().__init__( N,M,seed, p )
        

    """
    Computation of the hidden state from a visible one.
    ---------------------------------------------------
    Input arguments:
        x, input visible state or mini-batch
        useProb, bool to specify whether only probabilities should be used
        
    Update the vector of probabilities of activations of the hidden units, together with 
    the hidden state of the machine, if required. 
    """
    def updateHidden( self, x, useProb = False, W = np.empty(0) ):
        # Determine which weights to use
        if W.size == 0:
            W = self.W
       
        if x.ndim == 1:
            net = np.dot( W.T, x )            
            self.hp = special.expit( net )
            # Definition of bias unit that is always active
            self.hp[0] = 1
            if not useProb:
                y = np.random.rand( self.M+1 )
                # Redefine hidden state
                self.h = np.zeros( self.M+1 )
                self.h[ y <= self.hp ] = 1
        else:
            net = np.dot( x, W )            
            self.hp = special.expit( net )
            # Definition of bias unit that is always active
            self.hp[:, 0] = 1
            if not useProb:
                y = np.random.rand( self.hp.shape[0], self.M+1 )
                # Redefine hidden states
                self.h = np.zeros( (self.hp.shape[0], self.M+1) )
                self.h[ y <= self.hp ] = 1
    
    """
    Computation of the visible state from a hidden one.
    ---------------------------------------------------
    Input arguments:
        x, input hidden state or mini-batch
        useProb, bool to specify if only probabilities should be used
        
    Update the vector of probabilities of activations of the visible units, together with 
    the visible state of the machine, if required. 
    """
    def updateVisible( self, x, useProb = False ):
        if x.ndim == 1:
            net = np.dot( self.W, x )
            self.vp = special.expit( net )
            # Bias unit is always active
            self.vp[0] = 1
            if not useProb:
                y = np.random.rand( self.N+1 )
                self.v = np.zeros( self.N+1 )
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
    Compute free energy of an example or of a set of examples.
    --------------------------------------------------------------
    """
    def freeEnergy(self, X, W, beta=1):
        if X.ndim == 1:
            net = np.dot( self.W.T, X )
            return  -beta*np.dot( self.W.T[0], X) - np.sum( np.log(1.+np.exp(beta*net)) )
        else:
            net = np.dot( X, self.W )
            energies = -beta*np.dot( X, self.W[:,0]) - np.sum( np.log(1+np.exp(beta*net)), axis = 1  )
            return np.sum( energies )        


