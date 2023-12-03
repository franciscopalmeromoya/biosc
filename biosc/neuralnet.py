"""Pre-Trained Neural Network as BT-Settl model interpolator. PyTensor implementation."""
import os
from pickle import load
import numpy as np
import pymc as pm
import pytensor.tensor as T
import warnings

def relu(x, alpha=0):
    """
    Compute the element-wise rectified linear activation function.

    Parameters
    ----------
    x : symbolic tensor
        Tensor to compute the activation function for.
    alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).

    Returns
    -------
    symbolic tensor
        Element-wise rectifier applied to `x`.

    Notes
    -----
    This is numerically equivalent to ``T.switch(x > 0, x, alpha * x)``
    (or ``T.maximum(x, alpha * x)`` for ``alpha < 1``), but uses a faster
    formulation or an optimized Op, so we encourage to use this function.

    """
    if alpha == 0:
        return 0.5 * (x + abs(x))
    else:
        # We can't use 0.5 and 1 for one and half.  as if alpha is a
        # numpy dtype, they will be considered as float64, so would
        # cause upcast to float64.
        alpha = T.as_tensor_variable(alpha)
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * x + f2 * abs(x)

class Scaler:
    """PyTensor implementation of scaler pipeline.
    1. PowerTransform (Box-Cox) <- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html
    2. MinMaxScaler <- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    """
    def __init__(self, folder : str = './neuralnet/'):
        warnings.filterwarnings("ignore")
        # Load scalers from Pre-Trained Neural Network folder.
        try:
            with open(os.path.join(folder, 'scalers.pkl'),'rb') as file:
                scalers = load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError("The file containing scalers cannot be found. Please, provide a valid path") from error
        else:
            self.BoxCox = scalers[0]
            self.MinMax = scalers[1]
    
    def transform(self, age, mass):
        """Transform age and mass into scaled inputs.

        Parameters
        ----------
        age : Tensor (pytensor)
            Open cluster age. Units: [Myr] 
        mass : Tensor (pytensor)
            Mass for each star. Units: [Ms]
        
        Returns
        -------
        inputs : Tensor (pytensor)
            2-dimensional tensor after transformations.
        """

        age = T.as_tensor_variable(age)
        mass = T.as_tensor_variable(mass)

        age_v = T.tile(age, mass.type.shape)
        inputs = T.stack([age_v, mass])

        # BoxCox transformation
        lamb = np.array(self.BoxCox.lambdas_).reshape(-1, 1)
        x = T.switch(T.neq(lamb, 0), (inputs ** lamb - 1) / lamb, T.log(inputs))
        # MinMax transformation
        min_val = self.MinMax.data_min_.reshape(-1, 1)
        max_val = self.MinMax.data_max_.reshape(-1, 1)
        return (x - min_val) / (max_val - min_val)

class NeuralNetwork:
    """Pre-Trained Neural Network weights and compute the predictions.
    Implementation in PyTensor.
    """
    def __init__(self, folder : str = './neuralnet/'):
        """Load optimal weights from Pre-Trained Neural Network folder. 
        
        Parameters
        ----------
        folder : str
            Folder containing neural network weights.
        """
        
        # Load weights
        try:
            with open(os.path.join(folder, 'weights.pkl'), 'rb') as file:
                weights = load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError("The file containing optimal weights cannot be found. Please, provide a valid path") from error
        else:
            self.W1 = weights[0]
            self.b1 = weights[1]
            self.W2 = weights[2]
            self.b2 = weights[3]
            self.W3 = weights[4]
            self.b3 = weights[5]
            self.WLi = weights[6]
            self.bLi = weights[7]
            self.WPho = weights[8]
            self.bPho = weights[9]

    def predict(self, inputs):
        """Neural Network predictions as interpolation function for BTSettl model.
        PyTensor implementation.

        Parameters
        ---------- 
        inputs : Tensor (pytensor)     
            Scaled age and mass as 2-dimensional tensor. 
            - age = inputs[0]
            - mass = inputs[1]

        Returns
        -------
        Li : Tensor (pytensor)
            Lithium abundance for each star. Units: [1]
        Pho : Tensor (pytensor)
            Photometry (apparent magnitude) for each star. Units: [1]
        """

        # Forward pass
        Z1 = T.dot(inputs, self.W1) + self.b1
        A1 = relu(Z1)
        Z2 = T.dot(A1, self.W2) + self.b2
        A2 = relu(Z2)
        Z3 = T.dot(A2, self.W3) + self.b3
        A3 = relu(Z3)
        Li = pm.math.sigmoid(T.dot(A3, self.WLi) + self.bLi)
        Pho = T.dot(A3, self.WPho) + self.bPho

        return Li, Pho
        