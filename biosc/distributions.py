"""Custom distributions module"""
import pymc as pm
import numpy as np
import scipy.stats as st
from pytensor.tensor import TensorVariable

class MixtureLikelihood:
    def __init__(self, name : str, size : tuple, **dist_params) -> None:
        """Mixture likelihood helper class. David Hogg Method <- https://arxiv.org/abs/1008.4686
        
        Parameters
        ----------
        name : str
            Variable name.
        size : tuple
            Desired size of the random draw.
        dist_params : dict
            Mixture likelihood parameters.
        """
        self.name = name
        self.size = size
        self.mu = dist_params["mu"]
        self.sigma = dist_params["sigma"] 
        self.sd = dist_params["sd"] 
        self.idx = dist_params["idx"] 
        self.Pb = dist_params["Pb"] 
        self.Yb = dist_params["Yb"] 
        self.Vb = dist_params["Vb"]
    
    @staticmethod
    def logp(
            value : TensorVariable, 
            mu : TensorVariable, 
            sigma : TensorVariable, 
            sd : TensorVariable,
            idx : np.ndarray,
            Pb : TensorVariable,
            Yb : TensorVariable,
            Vb : TensorVariable,
        ) -> TensorVariable:
        """Custom log-likelihood"""
        # Compute inlier component
        ic = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma+sd), value) * idx
        # Compute outlier component
        oc = pm.logp(pm.Normal.dist(mu=Yb, sigma=sigma + Vb), value) * idx
        return ((1 - Pb) * ic).sum() + (Pb * oc).sum()
        
    @staticmethod    
    def rng_fn(
            mu : np.ndarray | float,
            sigma : np.ndarray | float,
            sd : np.ndarray | float,
            idx : np.ndarray,
            Pb : np.ndarray | float,
            Yb : np.ndarray | float,
            Vb : np.ndarray | float,
            rng : np.random.RandomState = None,
            size : tuple = None,
        ) -> np.ndarray | float:
        """Custom random method"""
        ic = st.norm.rvs(loc=mu, scale=sigma+sd, size = size, random_state=rng)
        oc = st.norm.rvs(loc=Yb, scale=sigma+Vb, size = size, random_state=rng)
        return (1-Pb)*ic + Pb*oc

    def add(self, data : np.ndarray) -> pm.CustomDist:
        """Include a CustomDist instance in the context model.
        
        Parameters
        ----------
        data : array-like
            Observed values.
        """
        # Mixture likelihood
        likelihood = pm.CustomDist(
            self.name,
            self.mu, self.sigma, self.sd, self.idx, self.Pb, self.Yb, self.Vb,
            logp = self.logp,
            observed = data,
            random = self.rng_fn,
            size = self.size,
            class_name = 'Mixture'
        )
        return likelihood

class NormalLikelihood:
    def __init__(self, name : str, size : tuple, **dist_params) -> None:
        """Normal likelihood helper class.
        
        Parameters
        ----------
        name : str
            Variable name.
        size : tuple
            Desired size of the random draw.
        dist_params : dict
            Normal likelihood parameters.
        """
        self.name = name
        self.size = size
        self.mu = dist_params["mu"]
        self.sigma = dist_params["sigma"] 
        self.sd = dist_params["sd"] 
        self.idx = dist_params["idx"] 

    @staticmethod
    def logp(
            value : TensorVariable, 
            mu : TensorVariable, 
            sigma : TensorVariable, 
            sd : TensorVariable,
            idx : np.ndarray
        ) -> TensorVariable:
        """Custom log-likelihood"""
        # Compute inlier component
        logp = pm.logp(pm.Normal.dist(mu=mu, sigma=sigma+sd), value) * idx
        return logp.sum()
        
    @staticmethod    
    def rng_fn(
            mu : np.ndarray | float,
            sigma : np.ndarray | float,
            sd : np.ndarray | float,
            idx : np.ndarray,
            rng : np.random.RandomState = None,
            size : tuple = None,
        ) -> np.ndarray | float:
        """Custom random method"""
        return st.norm.rvs(loc=mu, scale=sigma+sd, size = size, random_state=rng)

    def add(self, data : np.ndarray) -> pm.CustomDist:
        """Include a CustomDist instance in the context model.
        
        Parameters
        ----------
        data : array-like
            Observed values.
        """
        # Mixture likelihood
        likelihood = pm.CustomDist(
            self.name,
            self.mu, self.sigma, self.sd, self.idx,
            logp = self.logp,
            observed = data,
            random = self.rng_fn,
            size = self.size,
            class_name = 'Normal'
        )
        return likelihood