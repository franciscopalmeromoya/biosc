# Bayesian Inference on Stellar Clusters (biosc)
[![License: GNU GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python Version](https://img.shields.io/badge/python-%3E%3D3.9-blue.svg)](https://www.python.org/downloads/release)

## Overview
The resulting model combines existent photometric, parallax, and chemical abundance of Lithium data sets of stars belonging to stellar open clusters to infer its age distribution through modern and robust artificial intelligence methods. A Neural Network is trained given a grid of pre-calculated BT-Settl models to interpolate the spectral energy distributions of stars, working as a black-box interpolator in the model. The Bayesian hierarchical model not only facilitates simultaneous inference of star-level parameters but also offers an elegant framework for effectively pooling open cluster information and propagating uncertainty. Markov Chain Monte Carlo techniques allow us to sample the posterior distribution using the Hamiltonian Monte Carlo algorithm.

## Usage
To use the Bayesian model, follow these steps:
1. ***Data Preprocessing***: Use the ``preprocessing`` module to prepare your data, including parallax, photometric, and lithium measurements.
```python
from biosc.preprocessing import Preprocessing

# Example: Load and preprocess data
prep = Preprocessing('Pleiades_GDR3+2MASS+PanSTARRS1+EW_Li.csv', sortPho=False)
parallax_data = prep.get_parallax()
Li_data = prep.get_Li()
m_data = prep.get_magnitude(fillna='max')
```
2. ***Model Configuration and Compilation***: Create an instance of the BayesianModel class and configure the model with priors and optional parameters.
```python
from biosc.bhm import BayesianModel

# Example: Configure and compile the model
priors = {
    'age': {'dist': 'uniform', 'lower': 0, 'upper': 200},
    'distance': {'dist': 'normal', 'mu': 135, 'sigma': 20}
}

model = BayesianModel(parallax_data, m_data, Li_data)
model.compile(priors, POPho=False, POLi=True)
```
3. ***Sampling***: Run the sampling process to obtain posterior distributions.
```python
# Example: Sample from the posterior distribution
model.sample(chains=4)
```
4. ***Posterior Predictive Sampling (Optional)***: Generate samples from the posterior predictive distribution.
```python
# Example: Sample from the posterior predictive distribution
model.sample_posterior_predictive()
```
5. ***Saving Inference Data***: Save the inference data to a NetCDF file.
```python
# Example: Save the inference data
model.save("output_filename.nc")
```
6. ***Visualization***: Plot posterior distributions, trace plots, and other diagnostic plots.
```python
# Example: Plot posterior distributions
model.plot_trace(var_names=['age', 'distance'])

# Example: Plot posterior predictive checks
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
az.plot_ppc(model3.idata, var_names = 'parallax', ax = axs[0])
az.plot_ppc(model3.idata, var_names = 'flux', ax = axs[1])
az.plot_ppc(model3.idata, var_names = 'Li', ax = axs[2])
plt.show()

# Example: Plot QQ for parallax
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_ylabel('observed parallax [mas]')
ax.set_xlabel('parallax [mas]')
model3.plot_QQ('parallax', fig, ax)
plt.show()
```

## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

