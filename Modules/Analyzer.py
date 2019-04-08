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
    Input: 
        N,M =  sizes of the RBM
        typeDt = type of elements of the considered dataset
    """
    def __init__( self, N, M, typeDt='V' ):
        self.N = N
        self.M = M
        self.curr_dir = os.getcwd()
        self.typeDt = typeDt
        
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
    def __formatVector(self, Z ):
        if Z.dtype == np.dtype('int64'):
            # Convert vectors into strings
            Z_str = np.array2string( Z,separator="",max_line_width=10000 )  
        
            # Substitute zeros with underscore for representation purposes
            Z_str =  Z_str[1:-1].replace("0","_")

            # Determine the dimensions of the visible layer
            if self.typeDt == 'M':
                K = int( np.sqrt( len(Z) ) )
            else:
                K = len(Z)
                
            # Split in rows
            Z_str = [ Z_str[i:(i+K)] for i in range(0, len(Z_str), K ) ]
            
            return Z_str

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
            
            # Remove square brackets
            Z_str = [ Z_str[i][1:-1] for i in range( len(Z_str) )]

            if self.typeDt == 'M':
                K = int( np.sqrt( len(Z) ) )
            else:
                K = len(Z)

            for i in range( len( Z_str ) ):
                for j in range( int( (len( Z_str[i] ))/K )-1 ):
                    l = (j+1)*K+j
                    Z_str[i] = Z_str[i][:l] + '\n' + Z_str[i][l:]
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
    layer,  the self-threshold -the link with the bias unit- is discarded.
    """
    def analyzeWeights( self, W ):
        # Compute average sparsity
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
    The default choice of a = 3 is related to the demonstration made for R-RBMs that shows
    that in this case \hat{L} converges to the true L, if all magnetizations are equal
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
            
        # Take into account problematic cases
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
    different feature vectors w_{\mu}, using the definitions of Monasson and Tubiana
    (pag. 6 SI). In particular, \tilde{m} is estimated through the mean of the 
    top-L magnetizations.
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
        
        # Handle possible null configurations
        norms[ norms==0 ] = 1 

        # Normalize the overlap matrix
        q = np.divide( tmp, np.outer( norms, norms ) )
        
        return q
       
       
       
    """
    Analyze the performance of the BM.
    -----------------------------------------------------------------
    Compute the reconstruction score (mean reconstruction error 
    and percentage of correct reconstructions) through standard 
    Gibbs Sampling or through Monasson's formulas for the conditional
    average of a reconstruction given an initial state.
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
    ----------------------------------------------
    """
    def analyzeStates( self, X, BM, cond = False, overlaps = 0 ):
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
        # Zero-temperature MCMC
        if cond:
            perf['MRE_cond'],perf['nC_cond'] = self.analyzePerformance( X,BM, cond_avg = True )
        
        # Standard Gibbs Sampling
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
        if overlaps > 0:
            # Compute the whole overlap matrix
            q = self.analyzeOverlap( BM.h, m=overlaps )

            # Get only the upper triangular values
            mask = np.ones(q.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            mask = np.triu( mask )
            q = q[mask].flatten()
        else:
            q = None

        ############### Store and print the results                         
        # Print the 10-top reconstructions (arbitrarily chosen as the first occurrence of such examples)  
        if BM.N <= 100:
            # Sort in descendent order the number of correct reconstructions
            ind_ord = np.argsort( -counts[:,0] )  
            X_top = X_red[ ind_ord ]
            ind_top = ind_red[ind_ord]
            
            # Consider the case of big X sets with more than 10 unique examples 
            if len(X_red) > 10:    nPrints = 10
            else:                  nPrints = len(X_red)
            
            for k in range( nPrints ):
                # Print the input vectors and the related reconstruction
                s1 = self.__formatVector( X_top[k,1:].astype(int) )
                s2 = self.__formatVector( BM.v[ind_top[k],1:].astype(int) )
                for i in range( len( s1 )-1 ):
                    print( s1[i], '\t', s2[i] )
                print( s1[-1], '\t', s2[-1], end= '\t\t' )
                
                # Print the correspondent "performance" for all their occurrences 
                print( counts[:,0][ind_ord[k]].astype(int), "\t", counts[:,1][ind_ord[k]].astype(int) )

                # Print the correspondent hidden vectors
                print(''.join(self.__formatVectors( BM.h[ind_top[k],1:] )), "\n" )

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
        import seaborn as sns; sns.set(); sns.set_style("ticks")    

        ####### Fit properties
        # Percentage of correct reconstructions
        plt.figure()
        plt.plot([i for i in range( pams['nEpochs'] )], fit_res[0])
        plt.ylabel('MRE')
        plt.xlabel('Epochs')
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'MRE.png' ) )
        
        # Monitor the sparsity
        plt.figure()
        plt.plot([ pams['period']*i for i in range( len( fit_res[2] ) )], fit_res[2] )
        plt.ylabel('Sparsity')
        plt.xlabel('Epochs')
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'sparsity.png' ) )

        # Monitor the overfitting
        plt.figure()
        plt.plot([ pams['period']*i for i in range( len( fit_res[3] ) )], fit_res[3], label="Training set" )
        plt.plot([ pams['period']*i for i in range( len( fit_res[4] ) )], fit_res[4], label="Test set" )
        plt.ylabel('Average free-energy');  plt.legend()
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'overfitting.png' ) )        
        if pams['plots']:
            plt.show() 
        
        ############## Weights heatmap
        vmax =  max( np.abs( BM.W[1:,1:].flatten() ) )  
        vmax2 =  max( np.abs( BM.W.flatten() ) ) 

        # Define an ad-hoc colorbar
        if vmax2 - vmax > 1:
            cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.5),\
                np.arange(-vmax, vmax, 0.005), np.arange(vmax, vmax2, 0.5)  ) )
        else:
            cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.1),\
                np.arange(-vmax, vmax, 0.005), np.arange(vmax, vmax2, 0.1)  ) )            
        cmap = cm.get_cmap('RdBu_r',lut=len(cmap_bounds)+1)
        norm = mcl.BoundaryNorm(cmap_bounds,cmap.N)    

        if pams['typeDt']=='V':        

                fig,ax  = plt.subplots(figsize=(8,8))

                # Plot the colormap
                im = ax.matshow(  BM.W.T, cmap =cmap, norm=norm )
                if BM.N <= 90:
                    ax.set_xticks(np.arange(0, BM.N+5, 5, dtype=int))
                else:
                    ax.set_xticks(np.arange(0, BM.N+10, 10, dtype=int))                
                ax.set_yticks(np.arange(0, BM.M+5, 5, dtype=int))
                ax.set_xlabel(r'$i \; \longrightarrow$')
                ax.xaxis.set_label_position('top') 
                ax.set_ylabel(r'$\longleftarrow \; \mu$')
                ax.tick_params(labelbottom=False, labeltop=True, labelleft=True, labelright=False,
                                bottom=False, top=True, left=True, right=False)
                ax.set_title( 'Learned features $\{w_{i\mu}\}$\n\n\n' )                
            
                # Orientate the colorbar in the figure
                if BM.N > BM.M:   
                    fig.colorbar(im, orientation='horizontal', pad =0.04, ticks=np.arange(-int(vmax), int(vmax)+1) )
                else:       
                    fig.colorbar(im, orientation='vertical', pad =0.04, ticks=np.arange(-int(vmax), int(vmax)+1))

                plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'colormap.png' ),bbox_inches='tight' )
            
        ################ Example of feature detection 
        fig,ax  = plt.subplots(  )
        
        ind_ex = np.random.randint( 0, len(X_test) )
        
        # Get the number of magnetized hidden units in the first example of the test set
        L_0 = int( np.around( L[len(X_test)+ind_ex] ) )
        
        # Cut-off L_0 for representation purposes
        L_cut = 5
        if L_0 > L_cut: L_0 = L_cut
        
        # Sort the hidden state correspondent to the test example
        ind = np.argsort( BM.h[ind_ex,1:] )     

        # Determine a good set of xticks
        if BM.M < 10:   
            xticks_hid = np.insert( np.arange( 0, BM.M+1,1, dtype=np.int), 1,1 )[1:]
        elif BM.M < 100:    
            xticks_hid = np.insert( np.arange( 0, BM.M+5,5, dtype=np.int), 1,1 )[1:]
        else:
            xticks_hid = np.insert( np.arange( 0, BM.M+25,25, dtype=np.int), 1,1 )[1:]

        # Define a grid        
        if pams['typeDt']=='V':
            # Make outer gridspec
            outer = gridspec.GridSpec(4, 1, height_ratios=[1,3,2,1] )#, hspace=0.5,  left=0.1, right=0.9, top=0.975, bottom=0.025) 
            
            # Make nested gridspecs
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])
            gs2 = gridspec.GridSpecFromSubplotSpec(L_cut+1, 1, subplot_spec = outer[1])
            gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[2])
            gs4 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[3] )

            ax1 = plt.subplot(gs1[0])
            ax3 = plt.subplot(gs3[0])
            ax4 = plt.subplot(gs4[0])
            
            shapes = (1, BM.N)
        else:
            # Make outer gridspec
            outer = gridspec.GridSpec( 3, 1, height_ratios=[1,2,1] ) 
            
            # Make nested gridspecs
            gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec = outer[0])
            gs2 = gridspec.GridSpecFromSubplotSpec(1, L_0, subplot_spec = outer[1], wspace = .05)
            gs3 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[2])
            
            ax1 = plt.subplot(gs1[0,0])
            ax3 = plt.subplot(gs3[0])

            ax4 = plt.subplot(gs1[0,1])
            D = int( np.sqrt( BM.N ) )
            shapes = (D, D)
            
            vmax = max( np.abs( BM.W[1:,ind[-L_0:]+1].flatten() ) )

        # Plot the test example
        ax1.matshow( X_test[ind_ex,1:].reshape( shapes ),cmap ='Greys_r' )
        ax1.set_yticks([]);ax1.set_xticks([]); ax1.set_adjustable('box-forced')
        if pams['typeDt']=='V':
            ax1.set_title( 'Input' )
        else:
            ax1.set_ylabel('Input', rotation='horizontal', ha='right', va='center' )
            

        # Plot the features vectors of the magnetized units
        print( L_0 )

        for i, cell in enumerate(gs2):
            ax = plt.subplot(cell)
            
            if i < L_0:
                if pams['typeDt']=='V':
                    if i == 0:
                        ax.set_title( 'Receptive fields' )
                        ax.set_ylabel('$\mu$={}'.format(ind[-i-1]+1), rotation='horizontal', ha='right', va='center')
                    else:
                        ax.set_ylabel('{}'.format(ind[-i-1]+1), rotation='horizontal', ha='right', va='center')                    
                elif pams['typeDt']=='M':                
                    if i== 0:
                        ax.set_ylabel('Receptive\nfields', rotation='horizontal', ha='right', va='center' )
                        ax.set_title( '$\mu$={}'.format(ind[-i-1]+1) )
                    else:
                        ax.set_title( '{}'.format(ind[-i-1]+1) )
                                        
                im = ax.matshow( BM.W[1:,ind[-i-1]+1].reshape( shapes ),cmap ="RdBu_r", vmin=-vmax, vmax=vmax )
                ax.set_yticks([]); ax.set_xticks([]);   ax.set_adjustable('box-forced')
            elif i == L_0:
                ax_cb = ax
            else:
                ax.axis('off')

        if pams['typeDt']=='V':
            plt.colorbar(im,cax=ax_cb, orientation='horizontal')
        else:
            plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
            

        # Make the barplot
        ax3.set_title( 'Hidden state barplot' )
        ax3.bar( np.arange( 1, BM.M+1, dtype=np.int ),  BM.h[ind_ex,1:], align='center' )
        ax3.set_ylabel( 'Activation' )
        ax3.set_xticks( xticks_hid )
        ax3.annotate('$\mu$', xy=(1,0), xytext=(0, -13), ha='right', va='top',\
            xycoords='axes fraction', textcoords='offset points')      


        # Plot the reconstruction
        ax4.matshow( BM.v[ind_ex,1:].reshape( shapes ),cmap ='Greys_r' )
        ax4.set_yticks([]);ax4.set_xticks([]); ax4.set_adjustable('box-forced')
        if pams['typeDt']=='V':
            ax4.set_title( 'Reconstruction' )
        else:
            ax4.set_ylabel('Reconstruction', rotation='horizontal', ha='right', va='center' )
        
        outer.tight_layout(fig)
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'h_ex.png' ) )

        ############ Histogram of the number of activations
        mask = np.greater( BM.h, 0  )
        count = np.sum( mask, axis= 0 )
        
        plt.figure()
        plt.bar( np.arange( 1, BM.M+1, dtype=int ), count[1:], align='center' )
        plt.xticks( xticks_hid )
        plt.xlabel( '$\mu$' )
        plt.ylabel( 'Counts' )
        plt.title( 'Number of activations\n' )
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'n_activations.png' ) )

        ############## Make the histograms of L and S
        # Exploit the known types to determine the bins
        f, axarr = plt.subplots(1,3)
        f.suptitle( 'Hidden units classes distribution' )
        bins = np.arange( int( min(L) ), int( max(L) )+1, 0.5 )
        axarr[0].hist( L, bins=bins, align='left' )
        axarr[0].set_xlabel( "Magnetized" )
        axarr[0].set_ylabel( 'Counts' )
        
        I = BM.M*np.ones(len(L))-L-S
        bins = np.arange( int( min(I) ), int( max(I) )+1, 0.5 )
        axarr[1].hist( I, bins=bins, align='left' )
        axarr[1].set_xlabel("Weakly activated")

        plt.subplots_adjust(wspace=0.5)
        bins = np.arange( int( min(S) ), int( max(S) )+1, 1 )
        axarr[2].hist( S, bins=bins, align='left', rwidth=0.8 )
        axarr[2].set_xlabel("Silent")

        plt.figtext(0.5, 0.925, '(Hyperparameters: N={}, M={}, $\epsilon={}$, $\lambda={}$)'.format(\
            BM.N,BM.M,pams['epsilon'],pams['lambda_x']), wrap=True, ha='center', va='top',fontsize=8 )
        
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'hidden_activities.png' ) )
        
        ############## Overlap between test hidden configurations, without self-overlap
        plt.figure()
        plt.hist( q, bins='auto', density=True )
        plt.xlabel('q')
        plt.ylabel('Density')
        plt.title('Hidden states overlap')
        plt.savefig( os.path.join(self.curr_dir, 'Results', pams['phase'] + '_' + 'overlap.png' ) ) 
        if pams['plots']:
            plt.show() 
 
    """
    Function to save the obtained results in CSV files.
    ------------------------------------------------------------------
    """
    def saveResults( self, pars, fields, results ):
        with open(os.path.join(self.curr_dir, 'Results','log_results.csv'), 'a') as csvfile:            
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            
            # Save current time
            writer.writerow([])
            writer.writerow( [ str(datetime.datetime.now()) ] )

            # Save all hyperparameters
            keys = list( pars.keys() )
            values = list( pars.values() )
            K = 7
            n = int( len( keys )/K )
            
            for i in range( n ):
                writer.writerow( keys[i*K:(i+1)*K] )
                writer.writerow( values[i*K:(i+1)*K] )

            writer.writerow( keys[   n*K:  ] )
            writer.writerow( values[ n*K:  ] )
            writer.writerow( [])

            # Store results 
            writer.writerow(  fields )
            writer.writerows( results )            
            writer.writerow([])
            
            # Store statistics (mean and standard deviation of each column)
            tmp = np.vstack( (np.mean( results, axis=0 ), np.std(results,axis=0)) )
            writer.writerows( tmp )
            writer.writerow([])


    """
    Function to save the dataset and the corresponding examples of reconstructions.
    --------------------------------------------------------------------------------
    """
    def saveDataset( self, X, Y, corr, wrong, BM, nameFile ):
        # Save the statistics of the dataset in a CSV file, together with the perfomance of the RBM
        if BM.N <= 150:
            # Remove repetitions in the two sets
            X_red  = np.unique( X, axis = 0 )
            Y_red  = np.unique( Y, axis = 0 )
            
            # Get reconstructions 
            BM.GibbsSampling( v_init = X_red )
            X_rec, X_h = np.copy( BM.v ), np.copy( BM.h )
            
            # Convert visible states into strings
            X_red_str = self.__formatVectors( X_red[:,1:].astype(int) )
            Y_red_str = self.__formatVectors( Y_red[:,1:].astype(int) )

            # Convert reconstructions and hidden states into strings
            X_rec_str = self.__formatVectors( X_rec[:,1:].astype(int) )
            X_h_str = self.__formatVectors( X_h[:,1:] )

            # Construct a dataframe
            d = {'Examples': X_red_str, 'Reconstructions': X_rec_str, 'Hidden_states': X_h_str,\
                 'Total': corr+wrong, 'Right': corr, 'Wrong': wrong}
            df = pd.DataFrame( d, columns=['Examples', 'Reconstructions', 'Hidden_states', 'Total', 'Right', 'Wrong'] )
            
            # Determine which elements are not common between the two sets
            isUnique = np.ones( len( X_red_str), dtype=bool )
            for i in range( len(X_red_str) ):
                    if X_red_str[i] in Y_red_str: 
                        isUnique[i] = False

            df['Unique?'] = isUnique
            
            # Sort according to the total number of occurrences
            df = df.sort_values( ['Total'], ascending=[False] )
            
            # Save the dataframe
            df.to_csv( os.path.join(self.curr_dir, 'Results', nameFile ) )
