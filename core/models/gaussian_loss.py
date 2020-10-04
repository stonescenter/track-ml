import tensorflow as tf
import tensorflow_probability as tfp
import keras.backend as K
import numpy as np

def gaussian_layer(layer):

    dim_layer = len(layer.get_shape())
    param1, param2 = tf.unstack(layer, num=dim_layer, axis=-1)
    param1 = tf.expand_dims(param1, -1)
    param2 = tf.expand_dims(param2, -1)

    #apply softplus to bound between -1 and 1
    mu = tf.keras.activations.linear(param1)
    #mean =  tf.keras.activations.softsign(a)
    var = tf.keras.activations.elu(param2) + 1
    #b = tf.nn.elu(a) + 1

    out_tensor = tf.concat((mu, var), axis = dim_layer-1)

    return out_tensor

def gaussian_layer_2d(layer):

    dim_layer = len(layer.get_shape())
    param1, param2 = tf.unstack(layer, num=dim_layer, axis=-1)
    param1 = tf.expand_dims(param1, -1)
    param2 = tf.expand_dims(param2, -1)

    mu = tf.keras.activations.linear(param1)
    #mean =  tf.keras.activations.softsign(a)
    var = tf.keras.activations.elu(param2) + 1
    
    out_tensor = tf.concat((mu, var), axis = dim_layer-1)

    return out_tensor

def gaussian_loss(y_true, y_pred):

  par1, par2 = tf.unstack(y_pred, num=2, axis=-1)
  mu = tf.expand_dims(par1, -1)
  var = tf.expand_dims(par2, -1)
  dist = tfp.distributions.Normal(loc=mu, scale=var)
  #dist = tfp.distributions.Normal(loc=mu, scale=sigma)

  return tf.reduce_mean(-dist.log_prob(y_true))


def gaussian_nll(ytrue, ypreds):
    """Keras implmementation of multivariate Gaussian negative loglikelihood loss function. 
    This implementation implies diagonal covariance matrix.
    
    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, n_dims]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, n_dims*2]
        predicted mu and logsigma values (e.g. by your neural network)
        
    Returns
    -------
    neg_log_likelihood: float
        negative loglikelihood averaged over samples
        
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam') 
    
    """
    
    n_dims = int(int(ypreds.shape[1])/2)
    
    mu = ypreds[:, 0:n_dims]
    logsigma = ypreds[:, n_dims:]
    #dims = ypreds.shape[1]
    #mu = ypreds[:,0:1]
    #logsigma = ypreds[:, 1:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)