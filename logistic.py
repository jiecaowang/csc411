""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    data_with_bias_column = np.c_[data, np.ones(data.shape[0])]
    y = sigmoid(np.dot(data_with_bias_column, weights))
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    ce = -1 * np.sum(np.multiply(targets, np.log(y)) + np.multiply(1.0 - targets, np.log(1.0 - y)))
    frac_correct = float(np.sum(np.multiply(targets, np.floor(y+0.5))) + np.sum(np.multiply(1.0 - targets, np.round(1.0 - y))))/targets.shape[0]
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    y = logistic_predict(weights, data)
    data_with_bias_column = np.c_[data, np.ones(data.shape[0])]
    z = np.dot(data_with_bias_column, weights)
    f = np.sum(np.log(1.0 + np.exp(-1.0*z))) + np.sum(np.multiply((1.0 - targets), z))
    y_minus_t_column = y - targets
    y_minus_t_row = y_minus_t_column.T
    df_bias = np.sum(y_minus_t_column)
    df = np.c_[np.dot(y_minus_t_row, data), np.array([df_bias])].T
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # y = logistic_predict(weights, data)
    # weights_with_no_bias = np.delete(weights, [weights.shape[0]-1], 0)
    # data_with_bias_column = np.c_[data, np.ones(data.shape[0])]
    # z = np.dot(data_with_bias_column, weights)
    # f = np.sum(np.log(1.0 + np.exp(-1.0*z))) + np.sum(np.multiply((1.0 - targets), z)) + lamb * 0.5 * np.sum(np.multiply(weights_with_no_bias, weights_with_no_bias))

    # y_minus_t_column = y - targets
    # y_minus_t_row = y_minus_t_column.T
    # df_bias = np.sum(y_minus_t_column)
    # df = np.c_[np.dot(y_minus_t_row, data), np.array([df_bias])].T
    # lamb = hyperparameters['weight_regularization']
    # weights_with_no_bias = np.delete(weights, [weights.shape[0]-1], 0)
    # f += np.sum(lamb * 0.5 * np.multiply(weights_with_no_bias, weights_with_no_bias))
    # df_weights = df[:df.shape[0] - 1,:]
    # df_bias = df[df.shape[0] - 1:df.shape[0],:]
    # df_weights += np.sum(lamb * weights_with_no_bias)
    # df_bias += lamb * np.sum(weights[weights.shape[0] - 1]) # this is just one single number in the sum
    # df = np.r_[df_weights, df_bias]

    y = logistic_predict(weights, data)

    data_with_bias_column = np.c_[data, np.ones(data.shape[0])]
    z = np.dot(data_with_bias_column, weights)
    p_y_zeros = np.multiply(sigmoid(z), np.exp(-z))

    lamb = hyperparameters['weight_regularization']
    f = np.sum(np.log(1.0 + np.exp(-z)) + np.multiply(z, 1.0 - targets)) + lamb * 0.5 * np.sum(np.square(weights[:-1]))

    v = (1.0 - targets - p_y_zeros)
    df = (data_with_bias_column.T.dot(v).T + lamb * np.append(weights[:-1], 0)).T
    return f, df, y
