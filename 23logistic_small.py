import numpy as np
from check_grad import check_grad
from plot_digits import *
from utils import *
from logistic import *
import matplotlib.pyplot as plt

def weights_initializer(M):
    weights = np.random.randn(M+1, 1)
    weights *= 0.001
    return weights

def run_logistic_regression():
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    lambda_list = [0.001, 0.01, 0.1, 1.0]
    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                'learning_rate': 0.0002,
                'num_iterations': 1000
             }

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    weights = weights_initializer(M)
        

    # Verify that your logistic_pen function produces the right gradient.
    # diff should be very close to 0.
    

    y_cross_entropy_train = []
    y_cross_entropy_valid = []
    x_lambda = []

    for lamb in lambda_list:
        hyperparameters['weight_regularization'] = lamb
        run_check_grad(hyperparameters)
        sum_cross_entropy_train = 0
        sum_cross_entropy_valid = 0
        sum_frac_correct_train = 0
        sum_frac_correct_valid = 0
        for i in range(10):
            weights = weights_initializer(M)
            # initalize vars that needs to be averaged later
            cross_entropy_train = None
            cross_entropy_valid = None
            frac_correct_train = None
            frac_correct_valid = None
            # Begin learning with gradient descent
            for t in xrange(hyperparameters['num_iterations']):

                # TODO: you may need to modify this loop to create plots, etc.

                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                
                # Evaluate the prediction.
                cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

                if np.isnan(f) or np.isinf(f):
                    raise ValueError("nan/inf error")

                # update parameters
                weights = weights - hyperparameters['learning_rate'] * df / N

                # Make a prediction on the valid_inputs.
                predictions_valid = logistic_predict(weights, valid_inputs)

                # Evaluate the prediction.
                cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)

            sum_cross_entropy_train += cross_entropy_train
            sum_cross_entropy_valid += cross_entropy_valid
            sum_frac_correct_train += frac_correct_train
            sum_frac_correct_valid += frac_correct_valid

        avg_cross_entropy_train = sum_cross_entropy_train/10.0
        avg_cross_entropy_valid = sum_cross_entropy_valid/10.0
        avg_frac_correct_train = sum_frac_correct_train/10.0
        avg_frac_correct_valid = sum_frac_correct_valid/10.0

        # only print at the end some stats
        stat_msg = "LAMBDA:{:3f}   ITERATION:{:4d}  AVG TRAIN CE:{:.6f}  "
        stat_msg += "AVG TRAIN FRAC:{:2.2f}  AVG VALID CE:{:.6f} AVG VALID FRAC:{:2.2f}"
        print stat_msg.format(lamb,
                              hyperparameters['num_iterations'],
                              float(avg_cross_entropy_train),
                              float(avg_frac_correct_train*100),
                              float(avg_cross_entropy_valid),
                              float(avg_frac_correct_valid*100))
        
        y_cross_entropy_train.append(avg_cross_entropy_train)
        y_cross_entropy_valid.append(avg_cross_entropy_valid)
        x_lambda.append(lamb)
        
    plt.plot(x_lambda, y_cross_entropy_train)
    plt.plot(x_lambda, y_cross_entropy_valid)
    plt.legend(['train', 'valid'], loc='upper right')
    plt.xlabel('lambda')
    plt.ylabel('cross entropy')
    plt.title('train small')
    plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic_pen function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.round(np.random.rand(num_examples, 1), 0)

    diff = check_grad(logistic_pen,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print "diff =", diff

if __name__ == '__main__':
    run_logistic_regression()
