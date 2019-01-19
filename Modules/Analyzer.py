import numpy as np
import datetime
import csv
import os
import pandas as pd        


##################################################
# Class for analyzing Restricted Boltzmann Machines
##################################################
class Analyzer:
        
    """
    Constructor.
    --------------------
    """
    def __init__( self, N, M ):
        self.N = N
        self.M = M
        self.curr_dir = os.getcwd()
        
    """
    Compute the partecipation ratio PR_a.
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
    Create a string representation to get an overview of the statistics of the generated sets.
    -------------------------------------------------------------------------------------------
    """
    def __formatVectors(self, Z ):
        if Z.dtype == np.dtype('int64'):
            # Convert vectors into strings
            Z_str = [ np.array2string( Z[i],separator="",max_line_width=10000 ) for i in range(len(Z) ) ]  
            # Substitute zeros with underscore for representation purposes
            Z_str = [ Z_str[i].replace("0","_") for i in range(len(Z_str))] 
        else:
            # Convert vectors into strings
            Z_str = [ np.array2string( Z[i],separator="",max_line_width=10000,\
                formatter={'all':lambda x: str(int(np.around(x)))+'|' if x > 0 else '_|'} ) for i in range(len(Z) ) ]  
        return Z_str    
    
    
    """
    Determine sparsity and effective temperature.
    ------------------------------------------------
    Input:
        W, weights to analyze

    As did by Monasson, compute the degree of sparsity of each visible unit
    and the effective temperature of the machine.
    Since the important measure is the connectivity of each unit with the hidden
    layer,  the self-threshold, i.e. the link with the bias unit, is discarded.
    """
    def analyzeWeights( self, W ):
        ## Compute average sparsity
        p = 0
        for mu in range( 1, self.M +1 ):
            p += self.__PR( W[1:, mu], 2 )
        p /= (self.M * self.N )
        
        p_2 = 1.0/(self.M*self.N)*self.__PR( W[1:,1:].flatten(), 2 )

        # Determine weight heterogeneities
        p_vis = np.zeros( self.N ) 
        # Compute normalization
        den = 1.0/self.N * np.linalg.norm( W[1:,1:], 'fro')**2
        for i in range( self.N ):
                p_vis[i] = 1.0/den * np.linalg.norm( W[i+1, 1:] )**2
        
        p_hid = np.zeros( self.M )
        for mu in range( self.M ):
                p_hid[mu] = 1.0/den * np.linalg.norm( W[1:, mu+1] )**2
        
        # Compute the effective temperature of the machine
        T = p/den
        
        return p, p_2, p_vis, p_hid, T
        
    """
    Compute the number of magnetized and silent hidden units.
    ------------------------------------------------------------
    Input: 
        h, single hidden state
        a, power of the partecipation ratio to be used
        
    Use definitions of Monasson and Tubiana (pag. 6 SI).
    The default choice of a = 3 is related to the demonstration made for R-RBMs
    that shows that in this case \hat{L} converges to the true L, if all magnetizations are equal
    and in the thermodynamic limit.    
    """    
    def analyzeHiddenState( self, h, a = 3 ):
        # Partecipation ratio to get the number of strongly activated h.u.
        # Since the bias unit is always active (and it's not properly ReLU), 
        # it isn't considered
        L = self.__PR( h[1:], a ) 

        # Number of silent units
        S = np.sum( h[1:] == 0 )
    
        # Sort the hidden state in ascending order
        h_ord = np.sort( h[1:] )
        
        # Cast the experimental value of L and S as integers
        L_approx = int( np.around( L ) )
        S_approx = int( np.around( S ) )
            
        if L_approx != self.M and L_approx > 0:
            # Compute the mean squared activity of the two types of hidden units
            r = np.sum( np.power( h_ord[ :-L_approx], 2 ) )/(self.M-L_approx)
            # Get the maximum value taken by the intermediate category
            h_nmg_max =  h_ord[-L_approx-1]
        elif L_approx == 0:
            h_nmg_max = h_ord[-1]
            r = np.sum( np.power( h_ord, 2 ) )/self.M
        elif L_approx == self.M:
            r = 0
            h_nmg_max = 0
            
        # Get the maximum value taken by the magnetized category
        h_max = h_ord[-1]
    
        return L, S, r, h_max, h_nmg_max


    """
    Compute the magnetizations of the hidden units.
    -----------------------------------------
    Input: 
        x, visible instance
        L, number of strongly activated hidden units
        W, weights of the analyzed RBM

    Obtain a normalized measure of the overlap between the input x and the 
    different feature vectors w_{\mu}.
    Use again definitions of Monasson and Tubiana (pag. 6 SI).
    In particular, \tilde{m} is estimated through the mean of the top-L magnetizations.
    """    
    def analyzeMagnetizations( self, x, L, W ):
        # Cast the experimental value of L as an integer
        L = int( np.around( L ) )     
        
        # Compute the  vector of the "local" magnetizations        
        m = np.dot( W.T[1:,1:], 2*x[1:]-np.ones( self.N ) )
        den = np.sum( np.abs( W[1:,1:] ), axis = 0 )
        m = np.divide( m, den ) 
        
        # Compute \tilde{m}        
        if L > 0:
            # Compute hidden inputs
            h_input = np.dot( W.T[1:,1:], x[1:] )
            if np.sum( h_input != 0 ):
                # If there exists an input not null,
                # sort the magnetizations according
                # to the inputs
                ind = np.argsort( h_input ) 
                m_sorted = m[ind] 
            else:
                # Otherwise sort just the magnetizations
                m_sorted= np.sort( m )
                
            # Compute \tilde{m} according to the top inputs
            m_tilde = 1.0/L * np.sum( m_sorted[-L:] )
            
            # Compute the average magnetization of the weak activated units
            if L != self.M:
                m_nmg = 1.0/(self.M-L) * np.sum( m_sorted[:self.M-L] )
            else:
                m_nmg = 0        
        else:
            m_tilde = 0
            m_nmg = 1.0/self.M * np.sum( m )
            
        return m_tilde, m_nmg

    
    """
    Compute the overlap between the different hidden states.
    -------------------------------------------------------------------
    Input: 
        h, hidden configurations to analyze
        m, size of the set of configurations to analyze
    """
    def analyzeOverlap( self, h, m ):
        # Select the configurations in h according to m
        H = h[:m,1:]

        # Compute the unnormalized overlap matrix
        tmp = np.dot( H, H.T )

        # Compute the norm of each configuration
        norms = np.linalg.norm( H, axis = 1 ) 

        # Normalize the overlap matrix
        q = np.divide( tmp, np.outer( norms, norms ) )
        
        return q
       
       
       
    """
    Analyze the performance of the BM.
    ---------------------------------------
    Compute the reconstruction score (mean reconstruction
    error and percentage of correct reconstructions)
    through standard Gibbs Sampling or through Monasson's
    formulas for the conditional average of a reconstruction
    given an initial state.
    Input:
        X, set of examples
        BM, Restricted Boltzmann Machine
        cond_avg, bool to activate the search for local maxima
            of the free energy
    """
    def analyzePerformance( self, X,  BM, cond_avg = False ):
        if not cond_avg:
            BM.GibbsSampling( X )
            v_rec = BM.v
        else:
            v_rec = BM.findMaxima( X )

        return BM.reconstructionScore( X, v_rec )
        
    """
    Post-processing analysis function.
    -------------------------------------
    """
    def analyzeStates( self, X, BM, cond = False, overlap = False ):
        ######### Initializations
        # Get the size of the input set
        size = len(X)

        # Describe the set according to the unique rows
        #   ind_red: indices of the first occurrences of the unique rows
        #   indices: indices that allow to reconstruct X from the unique rows
        X_red,  ind_red, indices  = np.unique( X, axis = 0, return_inverse = True, return_index=True)        
        
        # Initialize a dictionary to store the results
        # together with 2 numpy arrays to store the series
        # 5 categories of measurements:
        #   - MRE, REE, nCorr    performance measures
        #   - L, S               partecipation ratios of hidden states 
        #                        (L = magnetized, S = silent)
        #   - m_t, m_nmg         magnetizations
        #   - r, h_max, h_nmg    activations of hidden units measures
        #   - corr, wrong        countings of reconstructions errors
        perf = { 'nC': 0, 'MRE': 0, 'REE': 0, 'MRE_cond': None, 'nC_cond': None }
        data =      np.zeros((size, 7))
        counts =    np.zeros((len(X_red), 2))
        

        ####### Reconstruction score
        if cond:
            perf['MRE_cond'],perf['nC_cond'] = self.analyzePerformance( X,BM, cond_avg = True )
        
        perf['MRE'], perf['nC'] = self.analyzePerformance( X, BM )
        
        ####### Other measures
        # Iterate through the entire set, but exploiting the repetitions for estimating corr and wrong        
        for k,ind in enumerate(indices):
            
            # Compute its distance from the real visible state
            dist = np.linalg.norm( X_red[ind] - BM.v[k] )
            if dist == 0:
                counts[ind,0] += 1 
            else:
                counts[ind,1] += 1 
            
            # Compute the number of silent and active units, and estimate their activations
            data[k,:5] = self.analyzeHiddenState( BM.h[k], a = 3 )
 
            # Compute the magnetizations of the hidden units
            data[k,5:]  = self.analyzeMagnetizations( X_red[ind], data[k,0], BM.W )

        # Compute the RE averaged only on the errors
        if np.any( counts[:,1] ):
            perf['REE'] = perf['MRE']*size/np.sum( counts[:,1] )
        else:
            perf['REE'] = 0
        
        # Analyze overlap hidden configurations related to the given set, if required 
        if overlap:
            q = self.analyzeOverlap( BM.h, m=500 )
        else:
            q = None

        ############### Store and print the results                         
        # Print the 10-top reconstructions (arbitrarily chosen as the first occurrence of such examples)  
        if BM.N < 100:
            # Sort in descendent order the number of correct reconstructions
            np.set_printoptions( linewidth = 1000, formatter={'all':lambda x: str(x) if x > 0 else '_'} )
            ind_ord = np.argsort( -counts[:,0] )  
            X_top = X_red[ ind_ord ]
            ind_top = ind_red[ind_ord]
            
            # Consider the case of big X sets with more than 10 unique examples 
            if len(X_red) > 10:    nPrints = 10
            else:                  nPrints = len(X_red)
            
            for k in range( nPrints ):
                # Print the input vectors
                print( X_top[k,1:].astype(int) )
                # Print the correspondent reconstructions
                print( BM.v[ind_top[k],1:].astype(int), end="\t\t"  )                
                # Print the correspondent "performance" for all their occurrences 
                print( counts[:,0][ind_ord[k]].astype(int), "\t", counts[:,1][ind_ord[k]].astype(int) )
                # Print the correspondent hidden vectors
                print( '[', ''.join(self.__formatVectors( BM.h[ind_top[k],1:] )), "]\n" )

            # Give the "performance" statistics of the top-10 w.r.t. the entire set X 
            np.set_printoptions( precision = 2, suppress= True, linewidth = 1000)
            corr_top = np.sum( counts[:,0][ind_ord[:nPrints]] )
            wrong_top = np.sum( counts[:,1][ind_ord[:nPrints]] )
            print( ' '*2*BM.N, "\t\t{}%\t{}%\n".format( corr_top/size*100, wrong_top/size*100 ) )
        
        # Create a dataframe of the measurements
        df = pd.DataFrame( data, columns=['L', 'S', 'r', 'h_max', 'h_nmg', 'm_t', 'm_nmg'] )
        counts = pd.DataFrame( counts, columns=['corr', 'wrong'] )
        perf_2 = pd.DataFrame( perf, index=[0] )
        
        # Compute mean values of the series
        means = df.mean(axis=0)

        print( "Size set = ", size )
        for key, value in perf.items():
            if value != None:
                print( key, ' = {:.2f}'.format( value ) ) 

        print( "L_mean = {:.2f}".format(  means['L'] ) )
        print( "S_mean = {:.2f}".format(  means['S'] ) )
        print( "I_mean = {:.2f}".format(  BM.M - means['L'] - means['S'] ) )
        print( "m_tilde = {:.2f}".format( means['m_t'] ) ) 
        print( "m_nonmag = {:.2f}".format( means['m_nmg'] ) )
        print( "sqrt(r) = {:.2f} ".format( means['r'] ) )
        print( "delta = {:.2f}\n".format( means['h_max']-means['h_nmg'] ) )
        
        return perf_2, df, counts, q

    """
    Function to make the different plots necessary for the analysis.
    -----------------------------------------------------------------
    """
    def makePlots( self, pams, X_test, fit_res, L, S,q, BM ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcl
        import matplotlib.cm as cm
        import matplotlib.gridspec as gridspec
        #import seaborn as sns; sns.set(); sns.set_style("ticks")    


        if not pams['useMNIST']:
            f = plt.figure()
            axprops = dict(xticks=[], yticks=[])
            num_bars  = 10
            height = 1./num_bars
            for i in range(10):
                ax = f.add_axes([0.25, 1-(i+1)/num_bars, 0.5, height], **axprops)
                ax.imshow( X_test[i,1:].reshape(1,BM.N), cmap="Greys_r")
            plt.savefig( os.path.join(self.curr_dir, 'Plots', 'dataset.png' ), bbox_inches='tight' )
        
        # Display reconstruction error over the epochs             
        f, axarr = plt.subplots(2, sharex=True)
        
        # Mean Reconstruction Error
        axarr[0].plot([i for i in range( pams['nEpochs'] )], fit_res[0])
        axarr[0].set_ylabel('MRE')
        
        # Percentage of correct reconstructions
        axarr[1].plot([i for i in range( pams['nEpochs'] )], fit_res[1])
        axarr[1].set_ylabel('Correct %')
        axarr[1].set_xlabel('Epochs')
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'MRE.png' ) )
        
        # Monitor the sparsity
        plt.figure()
        plt.plot([ pams['period']*i for i in range( len( fit_res[2] ) )], fit_res[2] )
        plt.ylabel('Sparsity')
        plt.xlabel('Epochs')
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'sparsity.png' ) )

        # Monitor the overfitting
        plt.figure()
        plt.plot([ pams['period']*i for i in range( len( fit_res[3] ) )], fit_res[3], label="Training set" )
        plt.plot([ pams['period']*i for i in range( len( fit_res[4] ) )], fit_res[4], label="Test set" )
        plt.ylabel('Average free-energy');  plt.legend()
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'overfitting.png' ) )        
        plt.show() 
        
        # Weights colormap
        vmax =  max( np.abs( BM.W[1:,1:].flatten() ) )  
        vmax2 =  max( np.abs( BM.W.flatten() ) ) 
        if not pams['useMNIST']:

            fig,ax  = plt.subplots(figsize=(8,8))
            plt.title( 'Learned features\n' )
            
            # Define an ad-hoc colorbar
            if vmax2 - vmax > 1:
                cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.5),\
                    np.arange(-vmax, vmax, 0.005), np.arange(vmax, vmax2, 0.5)  ) )
            else:
                cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.1),\
                    np.arange(-vmax, vmax, 0.005), np.arange(vmax, vmax2, 0.1)  ) )            
            cmap = cm.get_cmap('RdBu_r',lut=len(cmap_bounds)+1)
            norm = mcl.BoundaryNorm(cmap_bounds,cmap.N)    
            
            # Plot the colormap
            im = ax.matshow(  BM.W.T, cmap =cmap, norm=norm )
            ax.set_xticks(np.arange(0, BM.N+5, 5, dtype=int))
            ax.set_yticks(np.arange(0, BM.M+5, 5, dtype=int))
            ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False,
                            bottom=False, top=True, left=True, right=False)
            
            # Orientate the colorbar in the figure
            if BM.N > BM.M:   fig.colorbar(im, orientation='horizontal', pad =0.04, ticks=np.arange(-int(vmax), int(vmax)+1) )
            else:       fig.colorbar(im, orientation='vertical', pad =0.04, ticks=np.arange(-int(vmax), int(vmax)+1))

            plt.savefig( os.path.join(self.curr_dir, 'Plots', 'colormap.png' ),bbox_inches='tight' )
            
        # Hidden state example 
        f = plt.figure(figsize=(6,8))
        f.suptitle( 'Example of feature detection' )
        # Get the number of magnetized hidden units in the 
        # first example of the test set and cast it as an int 
        L_0 = int( np.around( L[len(X_test)] ) )
        # Cut L_0 for representation purposes
        if L_0 > 5: L_0 = 5
        
        # Sort the hidden state correspondent to the test example
        ind = np.argsort( BM.h[0,1:] )     
        print( "Indices sorted hidden units:", ind+ 1 )

        # Define a grid
        outer = gridspec.GridSpec(3, 1, height_ratios = [4, 1,1]) 
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[0])   
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1])
        if not pams['useMNIST']:
            gs3 = gridspec.GridSpecFromSubplotSpec(L_0, 1, subplot_spec = outer[2], hspace=0)
            shapes = (1, BM.N)
        else:
            gs3 = gridspec.GridSpecFromSubplotSpec(1, L_0, subplot_spec = outer[2], hspace=0)
            shapes = (28, 28)

            
        # Plot the activations
        ax = plt.subplot( gs1[0,:] )
        ax.bar( np.arange( 1, BM.M+1, dtype=np.int ),  BM.h[0,1:], align='center' )
        ax.set_xticks(np.arange( 0, BM.M+5,5, dtype=np.int )  )
        
        # Plot the activations
        ax = plt.subplot( gs1[1,:] )
        ax.bar( np.arange( 1, BM.M+1, dtype=np.int ),  np.dot( BM.W.T[1:,1:], X_test[0,1:]), align='center' )
        ax.set_xticks(np.arange( 0, BM.M+5,5, dtype=np.int )  )

        # Plot the test example
        ax = plt.subplot( gs2[0,:] )
            
        ax.matshow( X_test[0,1:].reshape( shapes ),cmap ='Greys_r' )
    
        ax.set_yticks([]);ax.set_xticks([]); ax.set_adjustable('box-forced')
        ax.set_title( 'Input' )
        
        # Plot the features vectors of the magnetized units
        for i in range( L_0 ):
            # Make nested gridspecs
            if not pams['useMNIST']:
                ax = plt.subplot( gs3[i,:] )
                if i == 0:
                    ax.set_title( 'Weights\nTop-L units', multialignment='center' )
            else:
                ax = plt.subplot( gs3[0,i] )
                if i == 2:
                    ax.set_title( 'Weights\nTop-L units', multialignment='center' )
                
            ax.matshow( BM.W[1:,ind[-i-1]+1].reshape( shapes ),cmap ="RdBu_r", vmin=-vmax, vmax=vmax )
            ax.set_yticks([]); ax.set_xticks([])
            ax.set_adjustable('box-forced')
        
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'hidden_example.png' ) )

        # Histogram of the number of activations
        mask = np.greater( BM.h, 0  )
        count = np.sum( mask, axis= 0 )
        
        plt.figure()
        plt.bar( np.arange( 1, BM.M+1, dtype=int ), count[1:], align='center' )
        
        if BM.M > 10:      plt.xticks(np.arange(0, BM.M+5, 5, dtype=np.int))
        else:           plt.xticks(np.arange(0, BM.M+1, 1, dtype=np.int))
        
        plt.title( 'Number of activations\n' )
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'n_activations.png' ) )

        
        # Make the histograms of L and S
        f, axarr = plt.subplots(1,3)
        f.suptitle( 'Hidden units classes distribution' )
        bins = np.arange( int( min(L) ), int( max(L) )+1, 0.5 )
        axarr[0].hist( L, bins=bins, align='left' )
        axarr[0].set_xlabel( "Magnetized" )
        
        I = BM.M*np.ones(len(L))-L-S
        bins = np.arange( int( min(I) ), int( max(I) )+1, 0.5 )
        axarr[1].hist( I, bins=bins, align='left' )
        axarr[1].set_xlabel("Weakly activated")

        plt.subplots_adjust(wspace=0.5)
        bins = np.arange( int( min(S) ), int( max(S) )+1, 1 )
        axarr[2].hist( S, bins=bins, align='left', rwidth=0.8 )
        axarr[2].set_xlabel("Silent")

        plt.figtext(0.5, 0.925, '(Hyperparameters: N={}, M={}, $\epsilon={}$,\
                    $\lambda={}$)'.format(BM.N,BM.M,pams['epsilon'],pams['lambda_x']), wrap=True, ha='center', va='top',fontsize=8 )
        
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'hidden_activities.png' ) )
        
        # Overlap between test hidden configurations, without self-overlap
        plt.figure()
        mask = np.ones(q.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        plt.hist( q[mask].flatten(), bins='auto' )
        plt.title('Hidden states overlap')
        plt.savefig( os.path.join(self.curr_dir, 'Plots', 'overlap.png' ) )
        plt.show()

    """
    Function to save the obtained results in CSV files.
    ------------------------------------------------------------------
    """
    def saveResults( self, pars, fields, results ):
        with open(os.path.join(self.curr_dir, 'Results','results.csv'), 'a') as csvfile:            
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            
            # Store hyperparameters
            writer.writerow([])
            writer.writerow( [ str(datetime.datetime.now()) ] )
            writer.writerow( ['ReLU_RBM', 'N', 'M', 'LA',  'SGS', 'p_01', 'p_10'] )
            writer.writerow( [pars['useReLU'], pars['N'], pars['M'], pars['LA'], pars['SGS'], pars['p_01'], pars['p_10']] )
            writer.writerow( ['SizeTot','x', 'c_e', 'c_a', 'Period', 'seedBM', 'seedTr'] )
            writer.writerow( [(1+pars['ratioTest'])*pars['sizeTrain'],pars['x'],pars['c_e'],\
                               pars['c_a'], pars['period'],pars['seedBM'],pars['seedTr']] )
            writer.writerow( ['nEpochs', 'epsilon', 'alpha', 'lambda_x', 'sizeTrain','nMB' ] )
            writer.writerow( [pars['nEpochs'],pars['epsilon'], pars['alpha'], pars['lambda_x'], pars['sizeTrain'], pars['nMB'] ] )
            writer.writerow( [])

            # Store results 
            writer.writerow(  fields )
            writer.writerows( results )            
            writer.writerow([])
            
            # Store statistics (mean and standard deviation of each column)
            tmp = np.vstack( (np.mean( results, axis=0 ), np.std(results,axis=0)) )
            writer.writerows( tmp )
            writer.writerow([])

    def saveDataset( self, X_train, X_test, corr_train, wrong_train, corr_test, wrong_test, BM ):
        # Save the statistics of the dataset in a CSV file, 
        # together with the perfomance of the last RBM
        if BM.N < 100:
            # Remove repetitions in the two sets
            X_red  = np.unique( X_train, axis = 0 )
            Y_red  = np.unique( X_test, axis = 0 )
            
            # Get reconstructions 
            BM.GibbsSampling( v_init = X_red )
            X_rec, X_h = np.copy( BM.v ), np.copy( BM.h )
            
            BM.GibbsSampling( v_init = Y_red )
            Y_rec, Y_h = np.copy( BM.v ), np.copy( BM.h )
            
            # Convert visible states into strings
            X_red_str = self.__formatVectors( X_red.astype(int) )
            Y_red_str = self.__formatVectors( Y_red.astype(int) )

            # Convert reconstructions into strings
            X_rec_str = self.__formatVectors( X_rec.astype(int) )
            Y_rec_str = self.__formatVectors( Y_rec.astype(int) )
            X_h_str = self.__formatVectors( X_h[:,1:] )
            Y_h_str = self.__formatVectors( Y_h[:,1:] )
            
            with open(os.path.join(self.curr_dir, 'Results','dataset.csv'), 'w') as csvfile:            
                writer = csv.writer(csvfile)
                writer.writerow( ['Training_set'] )
                writer.writerow([ 'Size=' + str( len(X_train) )  ] )
                writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
    
                # Write formatted reconstructions and check if they belong to the test set, correspondet hidden vectors and
                # the number of correct and wrong reconstructions, for each unique training example
                for i in range( len(X_red_str) ):
    
                    if X_red_str[i] in Y_red_str: uniqueness = False
                    else:                         uniqueness = True
                    
                    writer.writerow( [ X_red_str[i], X_rec_str[i], X_h_str[i], uniqueness,\
                        corr_train[i]+wrong_train[i],  corr_train[i], wrong_train[i]  ] )

                writer.writerow( ['Test_set'] )
                writer.writerow([ 'Size=' + str( len(X_test)  )] )
                writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )

                # Write formatted reconstructions and check if they belong to the training set, correspondet hidden vectors and
                # the number of correct and wrong reconstructions, for each unique test example
                for i in range( len(Y_red_str) ):
                    if Y_red_str[i] in X_red_str: uniqueness = False
                    else:                         uniqueness = True
                    
                    writer.writerow( [ Y_red_str[i], Y_rec_str[i], Y_h_str[i], uniqueness,\
                        corr_test[i]+wrong_test[i],  corr_test[i], wrong_test[i]  ] )

