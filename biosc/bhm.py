"""Bayesian Hierarchical Model"""
import os
import warnings
import pytensor
import numpy as np
import pandas as pd
import pytensor.tensor as T
from pytensor.tensor import TensorVariable
import seaborn as sns
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
from scipy import stats
from typing import Union
from biosc.figures import *
from biosc.functions import *
import biosc.distributions as dists
from biosc.neuralnet import NeuralNetwork, Scaler

class BayesianModel:
    """Bayesian Hierarchical Model.

        - Priors:
            + age [Myr]:      Open cluster age
            + mass [Ms]:      Mass for each star
            + distance [pc]:  Distance to each star
        - Likelihoods:
            + parallax [mas]: Parallax for each star
            + photometry [1]: Photometry for each star. Group by filter:
                * GAIA/GAIA3.G      -> G
                * GAIA/GAIA3.Gbp    -> Gbp
                * GAIA/GAIA3.Grp    -> Grp
                * 2MASS/2MASS.J     -> J
                * 2MASS/2MASS.H     -> H
                * 2MASS/2MASS.Ks    -> K
                * PAN-STARRS/PS1.g  -> g
                * PAN-STARRS/PS1.r  -> r
                * PAN-STARRS/PS1.i  -> i
                * PAN-STARRS/PS1.y  -> y
                * PAN-STARRS/PS1.z  -> z 
            + Li [dex]: Lithium abundance for each star
    """
    
    def __init__(self, parallax_data : Union[pd.DataFrame,bool] = None, m_data : Union[dict,bool] = None, Li_data : Union[pd.DataFrame,bool] = None, nStars : int = None):
        """Input observed variables.
        
        Parameters
        ----------
        parallax_data : pd.Dataframe
            Dataframe where columns are data and sigma for parallax values and uncertainties, respectively.
        m_data : dict
            Dictionary where keys are data, sigma and idx. The associated values are dataframes where the columns 
            are the filters. The columns idx stands for not missing values.
        Li_data : pd.Dataframe
            Dataframe where columns data and sigma for Lithium abundance values and uncertainties, respectively.
        """

        # Save attributes
        self.parallax_data = parallax_data
        self.m_data = m_data
        self.Li_data = Li_data

        self.filters = ['g', 'bp', 'rp', 'Jmag', 'Hmag', 'Kmag', 'gmag', 'rmag', 'imag', 'ymag', 'zmag']

        # Set number of stars in the dataset
        if parallax_data is not None and not isinstance(parallax_data, bool):       
            self.nStars = len(parallax_data)
        elif m_data is not None and not isinstance(m_data, bool):
            self.nStars = len(m_data['data'])
        elif Li_data is not None and not isinstance(Li_data, bool):
            self.nStars = len(Li_data)
        elif nStars is not None:
            self.nStars = nStars
        else:
            raise AttributeError("Number of stars cannot be determined, please check input data")

            

    def compile(self, priors : dict, POPho : bool = False, POLi : bool = False):
        """Define model structure setting priors and likelihoods.
        
        Parameters
        ----------
        priors : dict
            Dictionary containing model priors: distribution and its associate parameters.
        POPho : bool
            Optional parameter to use our custom likelihood (PrunningOutliers) for photometry data.
        POLi : bool
            Optional parameter to use our custom likelihood (PrunningOutliers) for Lithium data.
        """

        # Create the model 
        with pm.Model() as model:
            ##########
            # Priors #
            ##########
            m_lower = 1e-2 # lower mass limit (in solar masses) given by BT-Settl
            m_upper = 1.5  # upper mass limit (in solar masses) given by BT-Settl
            mass = pm.Uniform('mass', lower=m_lower, upper=m_upper, shape = self.nStars)

            # Age
            if 'age' in priors.keys():
                if priors['age']['dist'] == 'normal':
                    age = pm.Normal('age', mu = priors['age']['mu'], sigma = priors['age']['sigma'])
                elif priors['age']['dist'] == 'uniform':
                    age = pm.Uniform('age', upper = priors['age']['upper'], lower = priors['age']['lower'])
                else: 
                    raise KeyError('Unknown age prior distribution')
            else:
                raise KeyError('Please, provide a prior for age')
            
            # Distance
            if 'distance' in priors.keys():
                if priors['distance']['dist'] == 'normal':
                    distance = pm.Normal('distance', mu = priors['distance']['mu'], sigma = priors['distance']['sigma'], shape = self.nStars)
                elif priors['distance']['dist'] == 'uniform':
                    distance = pm.Uniform('distance', upper = priors['distance']['upper'], lower = priors['distance']['lower'], shape = self.nStars)
                else: 
                    raise KeyError('Unknown distance prior distribution')
            elif self.m_data is None:
                pass
            else:
                raise KeyError('Please, provide a prior for distance')
            
            # Photometry
            if self.m_data is not None and not isinstance(self.m_data, bool):
                # Model uncertainty for flux
                sd_F = pm.HalfNormal('σ_F', sigma=1, shape=11)

                # PrunningOutliers parameters
                if POPho:
                    # Uniform prior on Pb, the fraction of bad points
                    Pb_F = pm.Uniform('Pb_F', 0, 1.0, shape=11)
                    # Uniform prior on Yb, the centroid of the outlier distribution
                    Yb_m = pm.Uniform('Yb_m', -10, 50, shape=11)
                    # Uniform prior on Vb, the variance of the outlier distribution
                    Vb_F = pm.Uniform('Vb_F', 0, 10, shape=11)
            
            # Lithium
            if self.Li_data is not None and not isinstance(self.Li_data, bool):
                # Model uncertainty for Lithium
                sd_Li = pm.HalfNormal('σ_Li', sigma=1)

                # PrunningOutliers parameters
                if POLi:
                    # Uniform prior on Pb, the fraction of bad points
                    Pb_Li = pm.Uniform('Pb_Li', 0, 1.0)
                    # Uniform prior on Yb, the centroid of the outlier distribution
                    Yb_Li = pm.Uniform('Yb_Li', -10, 10)
                    # Uniform prior on Vb, the variance of the outlier distribution
                    Vb_Li = pm.Uniform('Vb_Li', 0, 10)

            ###########################
            # Deterministic variables #
            ###########################
            if self.parallax_data is not None:
                parallax_true = pm.Deterministic('parallax*', distance2parallax(distance/1000)) # Units factor: pc -> kpc

            # Scaler pipeline
            scaler = Scaler()
            inputs = scaler.transform(age, mass)

            # Instantiate Pre-Trained Neural Network (BT-Settl)
            nnet = NeuralNetwork()

            # Neural Network predictions
            Li, Pho = nnet.predict(inputs.T)

            if self.Li_data is not None:
                Li_true = pm.Deterministic('Li*', Li[:, 0]*3.3) # [L_0] = 3.3 

            if self.m_data is not None:
                M = pm.Deterministic('M*', Pho)
                m_true = pm.Deterministic('m*', M2m(M, distance))

            ###############
            # Likelihoods #
            ###############
            if self.parallax_data is not None and not isinstance(self.parallax_data, bool):
                parallax_data = pm.ConstantData('parallax_data', self.parallax_data['data'])
                parallax_obs = pm.Normal('parallax', mu=parallax_true, sigma=self.parallax_data['sigma'], observed=parallax_data)

            # Photometry
            if self.m_data is not None and not isinstance(self.m_data, bool):
                # Compute "observed" flux
                flux_data = pm.ConstantData('flux_data', m2flux(self.m_data['data']).fillna(1))
                flux_sigma = 0.4*np.log(10)*flux_data*self.m_data['sigma']
                flux_idx = np.array(self.m_data['idx'].astype(int))

                # Compute true flux
                flux_true = m2flux(m_true)

                if POPho:
                    # Transform distribution centroid to flux
                    Yb_F = m2flux(Yb_m)

                    # Compute flux mixture likelihood
                    flux_likelihood = dists.MixtureLikelihood(
                        name='flux', 
                        size=(self.nStars, 11), 
                        mu=flux_true, 
                        sigma=flux_sigma, 
                        sd=sd_F, 
                        idx=flux_idx, 
                        Pb=Pb_F, 
                        Yb=Yb_F, 
                        Vb=Vb_F)
                else:
                    # Compute flux normal likelihood
                    flux_likelihood = dists.NormalLikelihood(
                        name='flux', 
                        size=(self.nStars, 11), 
                        mu=flux_true, 
                        sigma=flux_sigma, 
                        sd=sd_F, 
                        idx=flux_idx)
                
                # Include flux likelihood
                flux_obs = flux_likelihood.add(flux_data)

            
            # Lithium
            if self.Li_data is not None and not isinstance(self.Li_data, bool):
                Li_idx = np.array(self.Li_data['idx'])
                Li_data = pm.ConstantData('Li_data', self.Li_data['data'][Li_idx])
                Li_sigma = self.Li_data['sigma'][Li_idx].values
                if POLi:
                    Li_likelihood = dists.MixtureLikelihood(
                        name='Li', 
                        size=(Li_idx.sum(),), 
                        mu=Li_true[Li_idx], 
                        sigma=Li_sigma, 
                        sd=sd_Li, 
                        idx=np.ones(Li_idx.sum()), 
                        Pb=Pb_Li, 
                        Yb=Yb_Li, 
                        Vb=Vb_Li)
                else:
                    Li_likelihood = dists.NormalLikelihood(
                        name='Li', 
                        size=(Li_idx.sum(),), 
                        mu=Li_true[Li_idx], 
                        sigma=Li_sigma, 
                        sd=sd_Li, 
                        idx=np.ones(Li_idx.sum()))

                # Include Lithium likelihood
                Li_obs = Li_likelihood.add(Li_data)

        # Save model as attributes
        self._model = model
        self.priors = priors

    def summary(self):
        """Show model summary"""

        try:
            return pm.model_to_graphviz(self._model)
        except AttributeError as error:
            raise AttributeError("Bayesian Model cannot be found, please compile the model.") from error

    def sample(self, draws : int = 1000, step = 'NUTS', **kwards):
        """Sample from posterior distribution using NUTS.
        
        Parameters
        ----------
        draws : int
            The number of samples to draw. Defaults to 1000.
        step : str
            Samplers available: NUTS (default), Metropolis, HamiltonianMC
        """

        try:
            with self._model:
                # Select step method
                if step == 'NUTS':
                    s = pm.NUTS()
                elif step == 'Metropolis':
                    s = pm.Metropolis()
                elif step == 'HamiltonianMC':
                    s = pm.HamiltonianMC()
                else:
                    raise KeyError("Unknown step method")

                # Sample
                self.trace = pm.sample(draws=draws, return_inferencedata=True, step=s, **kwards)
        except AttributeError as error:
            raise AttributeError("Bayesian Model cannot be found, please compile the model.") from error
        else:           
            # Save trace in inference data
            if hasattr(self, 'idata'):
                self.idata.extend(self.trace)
            else: 
                self.idata = self.trace

    def sample_prior_predictive(self, samples : int = 500, return_inferencedata : bool = False):
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        samples : int
            Number of samples from the prior predictive to generate. Defaults to 500.
        """

        try:
            with self._model:
                self.prior_predictive = pm.sample_prior_predictive(samples=samples)
        except AttributeError as error:
            raise AttributeError("Bayesian Model cannot be found, please compile the model first.") from error
        else:
            # Save prior predictive in inference data
            if hasattr(self, 'idata'):
                self.idata.extend(self.prior_predictive)
            else: 
                self.idata = self.prior_predictive
            
            # Return prior predictive
            if return_inferencedata:
                return self.prior_predictive

        
    def sample_posterior_predictive(self):
        """Generate posterior predictive samples from our model given a trace."""

        try:
            with self._model:
                self.posterior_predictive = pm.sample_posterior_predictive(self.idata)
        except AttributeError as error:
            raise AttributeError("Bayesian Model cannot be found, please compile the model.") from error
        else:
            # Save prior predictive in inference data
            self.idata.extend(self.posterior_predictive)

    def save(self, filename : str, dir : str = "idata") -> None:
        """Save model inference data to netcdf.
        
        Parameters
        ----------
        filename : str
            Name of the file with nc extension.
        dir : str
            Directory where will be saved the inference data.
        """
        self.idata.to_netcdf(os.path.join(dir, filename))

    def load(self, filename : str, dir : str = "idata"):
        """Load model inference data from netcdf.

        Parameters
        ----------
        filename : str
            Name of the file with nc extension.
        dir : str
            Directory where the inference data is stored.
        """
        self.idata = az.from_netcdf(os.path.join(dir, filename))
        
    def plot_trace(self, var_names : list = ['age', 'distance']):
        """Plot inference data trace.
        
        Parameters
        ----------
        var_names : list
            Model variables to be plotted.
        """
        _ = pm.plot_trace(self.idata, var_names = var_names)

    def generate_data(self, mode : str = 'dist') -> dict:
        """Generate synthetic data.
        
        Parameters
        ----------
        mode : str
            Generate data from constant value or distribution. Options are: 'dist' or 'cte'.

        Outputs
        -------
        data : dict
            A dictionary with generated dataframes.
        """

        # Safety check
        if mode not in ['dist', 'cte']:
            raise KeyError("Please introduce a valid mode: 'dist' or 'cte'")

        def selectSamples(prior, idx = None):
            """Uniformly select samples from prior distribution.
            
            Parameters
            ----------
            prior : data-array
                Prior predictive data.
            idx : array
                Option to provide indexes.
                
            Outputs
            -------
            x : array
                Selected samples from distribution
            """
            nSamples = prior.shape[0]
            nStars = prior.shape[1]
            x = np.zeros(prior.shape[1:])

            # Check if idx is provided
            if idx is not None: 
                for i in range(nStars): x[i] = prior[idx[i], i]
                return x
            else:
                idx = np.random.randint(0, nSamples, nStars)
                for i in range(nStars): x[i] = prior[idx[i], i]
                return x, idx
        
        # Generate data
        data = dict()

        # Photometry
        if self.m_data is not None:
            # Take the always the first sample
            if mode == 'cte':
                m = self.prior_predictive.prior['m*'][0, 0, :, :]
            # Uniformly select samples from dist
            elif mode == 'dist':
                m, idx = selectSamples(self.prior_predictive.prior['m*'][0])

            m_df = pd.DataFrame(m, columns=self.filters)
            m_data = {'data' : m_df}
            m_data['idx'] = m_data['data'].notna()
            data['m_data'] = m_data

        # Parallax
        if self.parallax_data is not None:
            if mode == 'cte':
                p = self.prior_predictive.prior['parallax*'][0, 0, :]
            elif mode == 'dist':
                p = selectSamples(self.prior_predictive.prior['parallax*'][0], idx)
            idxp = ~ np.isnan(p)
            parallax_data = pd.DataFrame({
                'data' : p,
                'idx' : idxp
            })
            data['parallax_data'] = parallax_data
        
        # Lithium
        if self.Li_data is not None:
            if mode == 'cte':
                Li = self.prior_predictive.prior['Li*'][0, 0, :]
            elif mode == 'dist':
                Li = selectSamples(self.prior_predictive.prior['Li*'][0], idx)
            idxLi = ~ np.isnan(Li)
            Li_data = pd.DataFrame({
                'data' : Li,
                'idx' : idxLi
            })
            data['Li_data'] = Li_data

        # Age that generate the data
        if mode == 'cte':
            data['age'] = self.prior_predictive.prior['age'][0, 0]
        elif mode == 'dist':
            data['age'] = self.prior_predictive.prior['age'][0, idx]

        return data
    
    def plot_posterior(self, kind, data, fig, ax, **kwargs):
        """Plot posterior distributions for each chain."""

        if kind == 'age':
            try:
                # Plot generative model age distribution
                sns.kdeplot(data['age'], ax=ax, fill=True, color = 'orange', label='Generative model')
            except:
                # Real-world data
                pass

            # Z-score threshold
            threshold = 1.5

            # Compute mean
            mu = self.idata.posterior['age'].mean(axis=1)
            c = -1
            l = False
            for x in mu:
                # Chain
                c += 1

                # Compute Z-score
                z_score = (x-mu.mean())/mu.std()

                # Clip chains with z-score < threshold
                if z_score < threshold:
                    if l:
                        sns.kdeplot(self.idata.posterior['age'][c], ax=ax, fill=True, color = 'blue')
                    else:
                        sns.kdeplot(self.idata.posterior['age'][c], ax=ax, fill=True, color = 'blue', label=f"chains")
                        l = True

        elif kind == 'CMDiagram':
            # Colors for diagram axis
            x, y = kwargs['x'], kwargs['y']

            def createX(M, x):
                M[x] = M[x.split('-')[0]] - M[x.split('-')[-1]]
                return M

            # Prior
            try:
                lp = False
                for s in range(self.idata.prior['M*'].shape[1]):
                    M_prior = pd.DataFrame(self.idata.prior['M*'][0, s].to_numpy(), columns = self.filters)
                    M_prior = createX(M_prior, x)
                    if lp:
                        sns.scatterplot(data=M_prior, x=x, y = y, ax=ax, color = 'gray', alpha = 0.25, zorder=1)
                    else:
                        sns.scatterplot(data=M_prior, x=x, y = y, ax=ax, color = 'gray', alpha = 0.25, zorder=1, label = 'prior')
                        lp = True
            except:
                warnings.warn("Inference data does not contain prior samples.")
                
            
            # Posterior
            lpo = False
            for s in range(self.idata.posterior['M*'].shape[1]):
                M_post = pd.DataFrame(self.idata.posterior['M*'][0, s].to_numpy(), columns = self.filters)
                M_post = createX(M_post, x)
                if lpo:
                    sns.lineplot(data=M_post, x=x, y = y, ax=ax, color = 'blue', alpha = 0.25, zorder=10)
                else:
                    sns.lineplot(data=M_post, x=x, y = y, ax=ax, color = 'blue', alpha = 0.25, zorder=10, label = 'posterior')
                    lpo = True

            # Posterior predictive
            try:
                m = flux2m(self.idata.posterior_predictive['flux'].to_numpy())
                p = self.idata.posterior_predictive['parallax'].to_numpy()
                def computeM(m, p):
                    for i in range(11):
                            parallax_v = [p for _ in range(11)]
                            parallax = np.stack(parallax_v, axis=3)
                    return m + 5*(np.log10(parallax/1000)+1)
                lppc = False

                for s in range(self.idata.posterior_predictive['flux'].shape[1]):
                    M_ppc = pd.DataFrame(computeM(m, p)[0, s], columns = self.filters)
                    M_ppc = createX(M_ppc, x)
                    if lppc:
                        sns.scatterplot(data=M_ppc, x=x, y = y, ax=ax, color = 'green', alpha = 0.25, zorder=10)
                    else:
                        sns.scatterplot(data=M_ppc, x=x, y = y, ax=ax, color = 'green', alpha = 0.25, zorder=10, label = 'ppc')
                        lppc = True
            except:
                warnings.warn("Inference data does not contain posterior predictive samples.")

            # Observation
            m_data, parallax_data = data['m_data'], data['parallax_data']
            plot_CMDiagram(m_data, parallax_data, y = y, x = x, fig=fig, ax=ax, facecolor = 'orange', edgecolor = 'black', zorder=100, label='observed')
    
    def plot_QQ(self, var_name : str, fig, ax):
        """Quantile-Quantile plot."""

        if var_name in ['parallax', 'Li']:
            # Store observed and simulated data
            if var_name == 'parallax':
                q_obs= self.parallax_data['data'].values
            elif var_name == 'Li':
                q_obs= self.Li_data['data'].values
            q_ppc = self.idata.posterior_predictive[var_name][0].mean(axis=0)

            # Compute percentiles
            per_ppc = [np.nanpercentile(q_ppc, i) for i in range(100)]
            per_prior = [np.nanpercentile(q_obs, i) for i in range(100)]

            # Plot percentiles
            sns.scatterplot(x = per_ppc, y = per_prior, ax= ax)

            # Plot identity line
            x = np.linspace(np.min(per_prior), np.max(per_prior), 1000)
            sns.lineplot(x = x, y = x, color = 'black', linestyle = 'dashed')

            # Include reference points
            idx = [1, 5, 50, 80, 95, 99]
            sns.scatterplot(x = np.array(per_ppc)[idx], y = np.array(per_prior)[idx], ax= ax, color = 'black')
            for i in idx:
                ax.annotate(f"{i} %", (per_ppc[i], per_prior[i]), textcoords="offset points", xytext=(10,10), ha='center')
                        

    def plot_CMDiagram(self, prior_predictive : bool = True, posterior_predictive : bool = False):
        """DEPRECATED"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})
        axs[0].invert_yaxis()
        if self.priors['age']['dist'] == 'normal':
            fig.suptitle(
                r'$p(age|I)$ ~ $\mathcal{{N}}(\mu = {mu}, \sigma = {sigma})$ Myr'.format(
                mu=self.priors['age']['mu'], 
                sigma=self.priors['age']['sigma']))
        elif self.priors['age']['dist'] == 'uniform':
                        fig.suptitle(
                r'$p(age|I)$ ~ $\mathcal{{U}}(a = {a}, b = {b})$ Myr'.format(
                a=self.priors['age']['lower'], 
                b=self.priors['age']['upper']))
        
        # CM Diagram
        axs[0].set_xlabel(r'$G$ - $G_{rp}$')
        axs[0].set_ylabel(r'$G$')
        axs[0].set_title('Color-Magnitude Diagram')
        axs[1].set_xlabel(r'$G$ - $G_{rp}$')
        axs[1].set_ylabel(r'[Li]')
        if prior_predictive:
            G_prior = self.prior_predictive['M*'][0, :, 0]
            Grp_prior = self.prior_predictive['M*'][0, :, 2]
            if self.Li_data is not None or self.Li_data:
                Li_prior = self.prior_predictive['Li*'].mean(axis=0)
                size = (Li_prior - Li_prior.min())/(Li_prior.max() - Li_prior.min())*100+50
                axs[1].scatter(G_prior-Grp_prior, Li_prior, alpha=.7, s=50)
                axs[1].set_title('Lithium abundance')
            # TODO: ALi Units
            else:
                axs[1].set_title('Model without Lithium')
                size = 100
            axs[0].scatter(G_prior-Grp_prior, G_prior, marker='*', s = size, c = 'b', zorder=10, label = 'Prior Predictive Samples')
            G_prior_all = self.prior_predictive['M*'][:, :, 0]
            Grp_prior_all = self.prior_predictive['M*'][:, :, 2]
            for i in range(G_prior_all.shape[0]):
                x = G_prior_all[i, :]-Grp_prior_all[i, :]
                idx = np.argsort(x)
                y = G_prior_all[i, :]
                axs[0].plot(x[idx], y[idx], alpha=.3, c='c')
            axs[0].legend()
        


