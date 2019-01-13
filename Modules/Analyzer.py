import numpy as np
import datetime
import csv
import os
        


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
        # Compute average sparsity
        p = 0
        for mu in range( 1, self.M +1 ):
            p += self.__PR( W[1:, mu], 2 )
        p /= (self.M * self.N )
        # Determine weight heterogeneities
        p_vector = np.zeros( self.N ) 
        # Compute normalization
        den = 1.0/self.M * np.linalg.norm( W[1:,1:], 'fro')**2
        for i in range( self.N ):
                p_vector[i] = 1.0/den * np.linalg.norm( W[i+1, 1:] )**2
        # Compute the effective temperature of the machine
        T = p/den
        
        return p, p_vector, T
        
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
        
        # Create a vector for the "local" magnetizations
        # (one for each hidden unit)
        m = np.zeros( self.M  )
        for mu in range( self.M ):
            m[mu] = 2*np.dot( x[1:], W[1:, mu+1] ) - np.dot( np.ones(self.N), W[1:, mu+1] )
            m[mu] /= np.sum( np.abs( W[1:,mu+1] ) )
        
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
    Post-processing analysis function.
    -------------------------------------
    """
    def analyzePerfomance( self, X, BM ):
        # Define size of the input set X
        size = len(X)

        # Partecipation ratios
        L_arr = np.zeros( size )
        S_arr = np.zeros( size )

        # Mean squared activations hidden units
        r = np.zeros( size )
        h_max = np.zeros( size )
        h_nmg_max = np.zeros( size )

        # Magnetizations matrix 
        m_nmg = np.empty( size )
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
            L_arr[k], S_arr[k], r[k], h_max[k], h_nmg_max[k] = self.analyzeHiddenState( BM.h[k], a = 3 )
            
            # Compute the magnetizations of the hidden units
            m_t[k], m_nmg[k] = self.analyzeMagnetizations( X_red[ind], L_arr[k], BM.W )
            
        # Print the 10-top reconstructions (arbitrarily chosen as the first occurrence of such examples)  
        if BM.N < 100:
            # Sort in descendent order the number of correct reconstructions
            np.set_printoptions( linewidth = 1000, formatter={'all':lambda x: str(x) if x > 0 else '_'} )
            ind_ord = np.argsort( -corr )  
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
                print( corr[ind_ord[k]].astype(int), "\t", wrong[ind_ord[k]].astype(int) )
                # Print the correspondent hidden vectors
                print( '[', ''.join(self.__formatVectors( BM.h[ind_top[k],1:] )), "]\n" )

            # Give the "performance" statistics of the top-10 w.r.t. the entire set X 
            np.set_printoptions( precision = 2, suppress= True, linewidth = 1000)
            corr_top = np.sum( corr[ind_ord[:nPrints]] )
            wrong_top = np.sum( wrong[ind_ord[:nPrints]] )
            print( ' '*2*BM.N, "\t\t{}%\t{}%\n".format( corr_top/size*100, wrong_top/size*100 ) )
        
        # Compute the RE averaged only on the errors
        if np.any( wrong ):
            RE = MRE*size/np.sum(wrong)
        else:
            RE = 0

        # Store and print the results         
        res = [MRE, RE , nCorrect, L_arr, S_arr, np.mean( m_t ), np.mean(m_nmg), np.mean( r), \
            np.mean( h_max ), np.mean( h_nmg_max), corr, wrong]
        
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
    def makePlots( self, pams, X_test, fit_res, L, S, BM ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcl
        import matplotlib.cm as cm
        import matplotlib.gridspec as gridspec
        #import seaborn as sns; sns.set(); sns.set_style("ticks")    


        if not pams['useMNIST']:
            #f = plt.figure()
            #f.suptitle( "Percentage of errors: {:.2f} %".format( count_e/len(dataset)*100 ) )
            #ax1 = plt.subplot2grid((1,1),(0,0))
            #ax1.matshow( dataset[0:50,1:], cmap="Greys_r",aspect='equal' )
            #ax1.set_xticks([]); ax1.set_yticks([])         
            
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
        fig,ax  = plt.subplots(figsize=(8,8))
        plt.title( 'Learned features\n' )
        
        # Define an ad-hoc colorbar
        vmax =  max( np.abs( BM.W[1:,1:].flatten() ) )  
        vmax2 =  max( np.abs( BM.W.flatten() ) ) 
        if vmax2 - vmax > 1:
            cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.5),\
                np.arange(-vmax, vmax, 0.01), np.arange(vmax, vmax2, 0.5)  ) )
        else:
            cmap_bounds = np.concatenate( (np.arange(-vmax2, -vmax, 0.1),\
                np.arange(-vmax, vmax, 0.01), np.arange(vmax, vmax2, 0.1)  ) )            
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
        outer = gridspec.GridSpec(3, 1, height_ratios = [2, 1,1]) 
        gs1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])   
        gs2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1])
        gs3 = gridspec.GridSpecFromSubplotSpec(L_0, 1, subplot_spec = outer[2], hspace=0)
        
        # Plot the activations
        ax = plt.subplot( gs1[0,:] )
        ax.bar( np.arange( 1, BM.M+1, dtype=np.int ),  BM.h[0,1:], align='center' )
        ax.set_xticks(np.arange( 0, BM.M+5,5, dtype=np.int )  )
        
        # Plot the test example
        ax = plt.subplot( gs2[0,:] )
        ax.matshow( X_test[0,1:].reshape(1,BM.N),cmap ='Greys_r' )
        ax.set_yticks([]);ax.set_xticks([]); ax.set_adjustable('box-forced')
        ax.set_title( 'Input' )
        
        # Plot the features vectors of the magnetized units
        for i in range( L_0 ):
            if i == 0:
                ax.set_title( 'Weights\nTop-L units', multialignment='center' )
            # Make nested gridspecs
            ax = plt.subplot( gs3[i,:] )
            ax.matshow( BM.W[1:,ind[-i-1]+1].reshape(1,BM.N),cmap ="RdBu_r", vmin=-vmax, vmax=vmax )
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
        plt.show()

    """
    Function to save the obtained results in CSV files.
    ------------------------------------------------------------------
    """
    def saveResults( self, pars, fields, results ):
        with open(os.path.join(self.curr_dir, 'Results','results.csv'), 'a') as csvfile:            
            writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
            
            # Store hyperparameters
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

    def saveDataset( self, X_train, X_test, res_train, res_test, BM ):
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
                writer.writerow([ 'Size=' + str( np.sum( res_train[10] ) + np.sum( res_train[11] ) )] )
                writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )
    
                # Write formatted reconstructions and check if they belong to the test set, correspondet hidden vectors and
                # the number of correct and wrong reconstructions, for each unique training example
                for i in range( len(X_red_str) ):
    
                    if X_red_str[i] in Y_red_str: uniqueness = False
                    else:                         uniqueness = True
                    
                    writer.writerow( [ X_red_str[i], X_rec_str[i], X_h_str[i], uniqueness,\
                        res_train[10][i]+res_train[11][i],  res_train[10][i], res_train[11][i]  ] )

                writer.writerow( ['Test_set'] )
                writer.writerow([ 'Size=' + str( np.sum( res_test[10] ) + np.sum( res_test[11] ) )] )
                writer.writerow( ["Example", "Sample", "Hidden_state", "Unique?", "Total", "Right", "Wrong"] )

                # Write formatted reconstructions and check if they belong to the training set, correspondet hidden vectors and
                # the number of correct and wrong reconstructions, for each unique test example
                for i in range( len(Y_red_str) ):
                    if Y_red_str[i] in X_red_str: uniqueness = False
                    else:                         uniqueness = True
                    
                    writer.writerow( [ Y_red_str[i], Y_rec_str[i], Y_h_str[i], uniqueness, res_test[10][i]+res_test[11][i],  res_test[10][i], res_test[11][i]  ] )

