"""Preprocessing pipeline"""
import os
import pandas as pd
import pytensor 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from biosc.functions import m2M, m2flux

# Set configuration
floatX = pytensor.config.floatX

# Uncertainty data imputation
def nearestNeighbor(df, value):
    idx = df['sigma'].notna()
    diff = np.abs(df['data'][idx]-value)
    idxmin = diff.idxmin()
    dfmin = df['sigma'].loc[idxmin]
    if isinstance(dfmin, pd.DataFrame):
        return np.diagonal(dfmin)
    else:
        return dfmin

class Preproccesing:
    """Data cleaning"""
    def __init__(self, filename : str, nStars : int = None, sortPho : bool = False):
        """Read dataset file and sample nStars from the generated dataframe
        
        Parameters
        ----------
        filename : str
            Dataset file name. It must be at data folder.
        nStars : int
            Number of stars to select from data. Default is None, therefore the full dataset is taken.
        sortPho : bool
            If nStars is given, sort sampled Photometry (ascending order) and take nStars (most brilliant). If False,
            just sample nStars at random.
        """

        # Read dataset
        self.dataset = pd.read_csv(os.path.join('./data/', filename))
        self.dataset.index = self.dataset['source_id']

        # Align to BTSettl model
        self.dataset = self.align2BTSettl()

        # Sample nStars from dataset
        if nStars is not None:
            if sortPho:
                # Most brilliant stars
                self.dataset = self.dataset.sort_values(by=['g', 'bp', 'rp', 'Jmag', 'Hmag', 'Kmag', 'gmag', 'rmag', 'imag', 'ymag', 'zmag'])
                self.dataset = self.dataset[:nStars]
            else:
                Li_data = self.get_Li()
                l = Li_data['idx'].astype(int).sum()
                if l > nStars:
                    # Sample stars having Lithium
                    self.dataset = self.dataset[Li_data['idx'] == True]
                    self.dataset = self.dataset[:nStars]
                else:
                    # Sample stars having Lithium
                    df_Li = self.dataset[Li_data['idx'] == True]
                    # Sample at random
                    df_ran = self.dataset[Li_data['idx'] == False].sample(n = nStars-l)
                    # Join both datasets
                    self.dataset = pd.concat([df_Li, df_ran])
            

    def align2BTSettl(self):
        """Filter out those stars having absolute magnitud lower than our BT-Settl model limits"""

        # Read BTSettl grid
        try:
            BTSettl = pd.read_csv('./data/BT-Settl_all_Myr_Gaia+2MASS+PanSTARRS.csv')
        except FileNotFoundError as error:
            raise FileNotFoundError("The BT-Settl model cannot be found. Please, provide a valid path.") from error
        else:
            # Get minimum absolute magnitud values from BT-Settl
            min_values = BTSettl[['G', 'G_BP', 'G_RP', 'J', 'H', 'K', 'g_p1', 'r_p1', 'i_p1', 'y_p1', 'z_p1']].min()
            # Get apparent magnitud and parallax from our dataset
            m = self.get_magnitude()['data']
            p = self.get_parallax()['data']
            # Compute the corresponding absolute magnitude
            M = m2M(m, p/1000)

            # Filter out stars
            idx = M-min_values.values > 0
            return self.dataset[idx.any(axis=1)]
        
    def get_parallax(self):
        """Returns parallax dataframe.

        Units:
            data  : [mas]
            sigma : [mas]
        """

        # Create dataframe
        parallax_data = pd.DataFrame({
            'data' : self.dataset['parallax'].astype(floatX),
            'sigma': self.dataset['parallax_error'].astype(floatX),
            'idx' : self.dataset['parallax'].notna()
        })
        return parallax_data
    
    def get_Li(self):
        """Returns Li dataframe.

        Units:
            data  : [1]
            sigma : [1]
        """

        # Create dataframe
        Li_data = pd.DataFrame({
            'data' : self.dataset['ALi'].astype(floatX),
            'sigma' : self.dataset['e_ALi'].astype(floatX),
            'idx' : self.dataset['ALi'].notna()
        })
        #TODO: Safety-Check every data has uncertainty

        return Li_data
    
    def get_magnitude(self, fillna : str = None):
        """Returns magnitude dataframe. 

        Units:
            data  : [1]
            sigma : [1]
        
        Parameters
        ----------
        fillna : str
            Option to fill missing values in uncertainty with pre-trainet neural network or the max value of each filter.

        Returns
        -------
        m_data : dict
            Dictionary with three Dataframes corresponding to apparent magnitud data, uncertainties
            and the index of those stars where the apparent magnitud is not missing (for each filter).
        """

        # A list containing every filter and its uncertainty
        m_names = ['g', 'g_error', 
           'bp', 'bp_error', 
           'rp', 'rp_error', 
           'Jmag', 'e_Jmag', 
           'Hmag', 'e_Hmag', 
           'Kmag', 'e_Kmag', 
           'gmag', 'e_gmag', 
           'rmag', 'e_rmag', 
           'imag', 'e_imag', 
           'ymag', 'e_ymag', 
           'zmag', 'e_zmag']
        
        # Create dataframes
        data = pd.DataFrame()
        sigma = pd.DataFrame()
        idx = pd.DataFrame()
        for k in range(len(m_names)//2):
            data[m_names[2*k]] = self.dataset[m_names[2*k]].astype(floatX)
            sigma[m_names[2*k]] = self.dataset[m_names[2*k+1]].astype(floatX)
            idx[m_names[2*k]] = data[m_names[2*k]].notna()

        # Fill missing uncertainty using max value of each filter
        if fillna == 'max':
            sigma = sigma.replace(0., np.NaN)
            values = {f : sigma[f].max() for f in sigma.columns}
            sigma = sigma.fillna(value=values)
        else: 
            if fillna is not None:
                raise AttributeError('Unknown uncertainty imputation')
        
        return {'data' : data, 'sigma' : sigma, 'idx' : idx}
    
    def explore(self, var : str = 'magnitude', filename : str = None):
        """Data Exploratory Analysis.
        
        Parameters
        ----------
        var : str
            Variable to explore.
        """

        if var == 'magnitude':
            # Uncertainty distribution
            m = self.get_magnitude(fillna='max')
            # Set data sources
            sources = {
                'GAIA' : ['g', 'bp', 'rp'],
                '2MASS' : ['Jmag', 'Hmag', 'Kmag'],
                'PAN-STARS' : ['gmag', 'rmag', 'imag', 'ymag', 'zmag']
            }
            fig = plt.figure(tight_layout=True, figsize=(15, 15))
            gs = gridspec.GridSpec(3, 3)
            # fig.suptitle('Uncertainty distribution')
            sigma = m['sigma']
            i = 0
            for key, source in sources.items():
                # Histogram
                ax = fig.add_subplot(gs[0, i])
                data = -np.log(sigma[source])
                data.plot.hist(bins=50, ax=ax)
                if i == 0:
                    ax.set_ylabel("Counts")
                else:
                    ax.set_ylabel("")
                ax.set_xlabel(r'$-\log(\sigma)$')
                ax.legend()
                ax.set_title(key)
                # Boxplot
                ax = fig.add_subplot(gs[1, i])
                ax.boxplot(sigma[source].dropna(), labels=source)
                ax.set_yscale('log')
                if i == 0:
                    ax.set_ylabel(r'$\sigma$ [1]')
                i +=1
            
            # Apparent magnitude vs uncertainty
            m_data = self.get_magnitude(fillna='max')
            ax = fig.add_subplot(gs[2, :])
            ax.set_yscale('log')
            for f in ['g', 'bp', 'rp', 'Jmag', 'Hmag', 'Kmag', 'gmag', 'rmag', 'imag', 'ymag', 'zmag']:
                ax.scatter(m_data['data'][f], m_data['sigma'][f], label = f, alpha=0.7)
            ax.set_title('Uncertainty as a function of apparent magnitude')
            ax.set_xlabel('m [1]')
            ax.set_ylabel(r'$\sigma$ [1]')
            ax.invert_xaxis()
            ax.legend()

            if filename is not None:
                plt.savefig(filename)

    
    def exploreNaN(self, var : str = 'magnitude'):
        """Study of missing values.
        
        Parameters
        ----------
        var : str
            Variable to study.
        """
        if var == 'magnitude':
            # Magnitude from dataset
            m = self.get_magnitude()
            # Set data sources
            sources = {
                'GAIA' : ['g', 'bp', 'rp'],
                '2MASS' : ['Jmag', 'Hmag', 'Kmag'],
                'PAN-STARS' : ['gmag', 'rmag', 'imag', 'ymag', 'zmag']
            }
            # Missing values
            fig, axs = plt.subplots(1, 3, sharey=True, figsize=(15, 5))
            fig.suptitle('Missing values')
            axs[0].set_ylabel('Count')
            i=0
            for key, source in sources.items():
                data = pd.DataFrame()
                sigma = pd.DataFrame()
                dps = pd.DataFrame()
                for f in source:
                    data[f] = m['data'][f].isna().astype(int)
                    sigma[f] = m['sigma'][f].isna().astype(int)
                    dps[f] = (data[f] + sigma[f])//2
                df = pd.DataFrame({'data' : data.sum(), 'sigma' : sigma.sum(), 'data+sigma' : dps.sum()})
                df[['data', 'sigma']].plot(kind='bar', stacked=True, ax=axs[i])
                df['data+sigma'].plot(kind='bar', ax=axs[i])
                axs[i].set_title(key)
                i+=1
    
    def get_flux(self):
        """Returns flux dataframe.

        Units:
            data  :
            sigma :
        """
        #TODO: Add units

        # Get magnitude data
        magnitude = self.get_magnitude()
        m_data = magnitude['data']
        m_sigma = magnitude['sigma']
        # Convert magnitude to flux
        data = m2flux(m_data)
        sigma = 0.4*np.log(10)*data*m_sigma
        return {'data' : data, 'sigma' : sigma, 'idx' : magnitude['idx']}