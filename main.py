from biosc.preprocessing import Preproccesing
from biosc.bhm import BayesianModel
import os
import logging

def configure_logging():
    # Create a logs folder if it doesn't exist
    log_folder = 'logs'
    os.makedirs(log_folder, exist_ok=True)

    # Configure logging
    log_file = os.path.join(log_folder, 'logfile.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    configure_logging()

    # Data preprocessing
    prep = Preproccesing('Pleiades_GDR3+2MASS+PanSTARRS1+EW_Li.csv', sortPho=False)
    parallax_data = prep.get_parallax()
    Li_data = prep.get_Li()
    m_data = prep.get_magnitude(fillna='max')
    # Priors
    priors = {
        'age' : {
            'dist' : 'uniform',
            'lower' : 0,
            'upper' : 200
        },
        'distance' : {
            'dist' : 'normal',
            'mu' : 135,
            'sigma' : 20
        }
    }
    logging.info(repr(priors))

    # Model configuration 3
    logging.info('Model3')
    model3 = BayesianModel(parallax_data, m_data, Li_data)
    model3.compile(priors, POPho = False, POLi=True)
    logging.info('Model3: Starting sampling ...')
    model3.sample(chains=4)
    logging.info('Model3: Sampling completed.')
    logging.info('Model3: Starting sampling posterior predictive ...')
    model3.sample_posterior_predictive()
    logging.info('Model3: Sampling completed.')
    logging.info('Model3: Saving inference data ...')
    model3.save("uniformConf3.nc")
    logging.info('Model3 processing completed.')

    # Model configuration 4
    logging.info('Model4')
    model4 = BayesianModel(parallax_data, m_data, Li_data)
    model4.compile(priors, POPho = True, POLi=True)
    logging.info('Model4: Starting sampling ...')
    model4.sample(chains=4)
    logging.info('Model4: Sampling completed.')
    logging.info('Model4: Starting sampling posterior predictive ...')
    model4.sample_posterior_predictive()
    logging.info('Model4: Sampling completed.')
    logging.info('Model4: Saving inference data ...')
    model4.save("uniformConf4.nc")
    logging.info('Model4 processing completed.')

if __name__ == "__main__":
    main()