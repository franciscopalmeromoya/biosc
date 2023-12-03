"""Module with Useful funtions to visualize data"""
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from biosc.functions import m2M

def plot_CMDiagram(m_data : dict, parallax_data : pd.DataFrame, y : str, x : str, fig, ax, **kwards):
    """Create CM Diagram given filters"""

    # Compute absolute magnitude
    M_data = m2M(m_data['data'].iloc[:, :11], parallax_data['data']/1000)
    M_data[x] = M_data[x.split('-')[0]] - M_data[x.split('-')[-1]]
    if 'out' in m_data['data'].columns:
        M_data['out'] = m_data['data']['out']

    # Make plot
    sns.scatterplot(data=M_data, x=x, y = y, ax=ax, **kwards)

def joinplotSigma(m_data, m_prior, f):
    """Compare uncertainty"""

    # Create a copy
    prior = copy.deepcopy(m_prior)

    # Summary dataframe for uncertainty
    prior['sigma']['source'] = 'synthetic'
    m_data['sigma']['source'] = 'observed'
    df_sigma = pd.concat([m_data['sigma'], prior['sigma']], ignore_index=True)
    columns = ['e_' + f for f in df_sigma.columns[:-1]] + ['source']
    df_sigma.columns = columns

    # Summary dataframe for data
    df_data = pd.concat([m_data['data'], prior['data']], ignore_index=True)

    # Summary full dataframe
    df_sum = pd.concat([df_data, df_sigma], axis=1)
    sns.jointplot(df_sum, x = f, y = 'e_' + f, hue='source', palette='Set2')