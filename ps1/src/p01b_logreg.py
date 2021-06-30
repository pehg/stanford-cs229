import numpy as np
import util
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import os
from linear_model import LinearModel

# to compare our model's accuracy with sklearn model
from sklearn import linear_model as skl_linearmodel
import sklearn

sns.set()
internet = 1

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_train2, y_train2 = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    
    # 1. First step after reading a lot, is to normalize/scale the data. This is an important step to get rid of
    #    computing problems (like getting inf when computing the result of cost function)
    # 2. Start looking at the training data by plotting box plots to determine the best scaling technique:
    #    https://benalexkeen.com/feature-scaling-with-scikit-learn/
    #    https://en.wikipedia.org/wiki/Feature_scaling
    # 3. From the plot we can see that feature 2 has a lot of outliers. This means that it would be better to use
    #    the Robust scaler, but let's give a try the regular Min Max scaler
    # 4. NOTE: but after scaling the estimator doesn't make a good job unless we scale the new data set. Need to
    #          investigate if this is done every time or if the parameters are scaled (like inverting the process
    #          of scaling the training data)


    # sns.boxplot(data=x_train)
    # plt.title('Training data')
    # plt.show()

    # x_train = sklearn.preprocessing.MinMaxScaler().fit_transform(x_train)
    # x_eval = sklearn.preprocessing.MinMaxScaler().fit_transform(x_eval)

    examples = -1
    # Train a logistic regression classifier
    clf = LogisticRegression(alpha=0.002, n_iters=20000, eps=1e-5)
    clf_scikit = skl_linearmodel.LogisticRegression()
    clf_scikit.fit_intercept = False

    if examples == -1:
        print('\nx_train.shape: ', x_train.shape)
        print('y_train.shape: ', y_train.shape)

        clf.fit(x_train, y_train)
        clf_scikit.fit(x_train, y_train)
    else:
        clf.fit(x_train[::examples, :], y_train[::examples])
        clf_scikit.fit(x_train[::examples, :], y_train[::examples])

    print('\nResults')
    print("clf.theta_gradient:   ", clf.theta_gradient)
    # print("clf.theta_newton: ", clf.theta_newton)
    print("clf_sikit.theta:  ", clf_scikit.coef_[0])

    print("clf.J_gradient:   ", clf.J_gradient)
    # print("clf.J_newton: ", clf.J_newton)

    ax1 = plt.subplot(131)
    sns.lineplot(range(len(clf.J_gradient_hist)), clf.J_gradient_hist)
    ax1.set_title('Loss function - gradient descent 1')

    ax2 = plt.subplot(132)
    sns.lineplot(range(len(clf.J_gradient2_hist)), clf.J_gradient2_hist)
    ax2.set_title('Loss function - gradient descent 2')

    ax3 = plt.subplot(133)
    sns.lineplot(range(len(clf.J_newton_hist)), clf.J_newton_hist)
    ax3.set_title('Loss function - newton')

    plt.show()

    # sns.lineplot(range(len(clf.J_newton_hist)), clf.J_newton_hist)
    # plt.show()

    # Plot decision boundary on top of the training set
    fig = plt.figure()
    util.myplot(x_train, y_train, clf.theta_gradient, fig=fig, label='gradient')
    util.myplot(x_train, y_train, clf.theta_newton, fig=fig, label='newton')
    util.myplot(x_train, y_train, clf.theta_gradient2, fig=fig, label='gradient2')
    util.myplot(x_train, y_train, clf_scikit.coef_[0], fig=fig, label='sklearn')
    plt.legend()
    plt.show()

    # Plot decision boundary on top of the validation set
    fig = plt.figure()
    util.myplot(x_eval, y_eval, clf.theta_gradient, fig=fig, label='gradient')
    util.myplot(x_eval, y_eval, clf.theta_newton, fig=fig, label='newton')
    util.myplot(x_eval, y_eval, clf.theta_gradient2, fig=fig, label='gradient2')
    util.myplot(x_eval, y_eval, clf_scikit.coef_[0], fig=fig, label='sklearn')
    plt.legend()
    plt.show()


    #
    # Evaluate model
    #
    print('\n--------------------------')
    print(' Accurcy of models (training data)')
    print('--------------------------')

    accuracy = (y_train == clf.predict(x_train, method='grad')).mean()
    print('\nAccuracy (grad): ', accuracy)

    accuracy = (y_train == clf.predict(x_train, method='newton')).mean()
    print('\nAccuracy (newton): ', accuracy)

    accuracy = (y_train == clf.predict(x_train, method='grad2')).mean()
    print('\nAccuracy (grad2): ', accuracy)

    print('\n--------------------------')
    print(' Accurcy of models (evaluation data)')
    print('--------------------------')
    accuracy = (y_eval == clf.predict(x_eval, method='grad')).mean()
    print('\nAccuracy (grad): ', accuracy)

    accuracy = (y_eval == clf.predict(x_eval, method='newton')).mean()
    print('\nAccuracy (newton): ', accuracy)

    accuracy = (y_eval == clf.predict(x_eval, method='grad2')).mean()
    print('\nAccuracy (grad2): ', accuracy)

    # Use np.savetxt to save predictions on eval set to pred_path
    # y_pred = clf.predict(x_eval)
    #
    # if not os.path.exists('./output'):
    #     os.makedirs('./output')
    # np.savetxt(pred_path, y_pred)

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, alpha=0.01, n_iters=100, eps=1e-5):
            super().__init__(step_size=alpha, max_iter=n_iters, eps=eps)
            self.alpha = alpha
            self.n_iters = n_iters
            self.J_gradient = 0
            self.J_newton = 0
            self.theta_gradient = None
            self.theta_newton = None
            self.theta_gradient2 = None
            self.J_gradient_hist = None
            self.J_newton_hist = None
            self.J_gradient2_hist = None

    def fit(self, X, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """

        # *** START CODE HERE ***
        def _h(theta, X):
            """vectorized implementation of h_theta(x)
            Args:
                theta: Shape (n,)
                x: Shape (m,n)

            Returns:
                numpy array of shape (m,)
            """
            # The hypothesis h_theta is the sigmoid function
            z = X @ theta
            sigmoid = 1 / (1 + np.exp(-z))
            return sigmoid

        # ---------------------------------------------------------------------------------------------------------

        def _gradient(X, y, theta):
            """
            Args:
                x: numpy array of shape (m,n)
                y: numpy array of shape (m,)
                theta: numpy array of shape (n,)

            Returns:
                numpy.array of shape (m,n) - Gradient of the sigmoid function for the given arguments

            """
            # The gradient is equivalent to make the following operation:
            #   ( (y-h).T dot (x) ).T
            #   (1 x m) dot (m x n) = transpose(1 x n)
            # But in numpy there is no row vector, all vectors are column vectors, but this operation of transposing
            # and multiplying is autmatically done with numpy.matmul() if the first argument is 1-D and second arg is
            # N-D.
            # NOTE: x1 @ x2 is a shorthand for numpy.matmul(x1, x2)

            if internet:
                errors = _h(theta, X) - y
                grad = (errors @ X) / y.size
            else:
                errors = y - _h(theta, X)
                grad = errors @ X
                # grad = (errors @ X) / y.size

            return grad

        # ---------------------------------------------------------------------------------------------------------

        def _hessian(X, theta):
            """
            Args:
                x: numpy array of shape (m,n)
                theta: numpy array of shape (n,)

            Returns:
                numpy.array - Hessian matrix for the sigmoid function and the given arguments
            """
            h_theta = _h(theta, X)
            b = np.multiply(h_theta, 1 - h_theta)

            B = np.diag(b)

            H = X.T @ (B @ X)

            # print('h_theta.shape: ', h_theta.shape)
            # print('b.shape: ', b.shape)
            # print('B.shape: ', B.shape)
            # print('H.shape: ', H.shape)

            return H

        # ---------------------------------------------------------------------------------------------------------

        # Operations
        def _log_likelihood(X, y, theta):
            """Compute the log(likelihood) of the parameters given x, y and theta
            Args:
                x (np.array): shape (m,n)
                y (np.array): shape (m,)
                theta (np.array): shape (n,)

            Returns:
                log(Likelihood) of the parameters
            """
            if internet:
                loglike = -((y * np.log(_h(theta, X))) + ((1 - y) * np.log(1 - _h(theta, X)))).mean()  # internet
            else:
                positive = np.dot(y[y == 1], np.log(_h(theta, X[y == 1, :])))
                negative = np.dot((1 - y[y == 0]), np.log(1 - _h(theta, X[y == 0, :])))
                loglike = (positive + negative) / y.size

            return loglike

        # -------------------------------------------------------------------------------------------------------------

        self.theta = np.zeros(X.shape[1])
        self.theta_gradient = np.zeros(X.shape[1])
        self.theta_newton = np.zeros(X.shape[1])
        self.theta_gradient2 = np.zeros(X.shape[1])
        self.J_gradient_hist = []
        self.J_newton_hist = []
        self.J_gradient2_hist = []

        stop = 0
        iter = 0
        while not stop and iter < self.n_iters:
            # print('\nIteration ', stop_counter)
            # Gradient ascent update rule

            if internet:
                self.theta_gradient -= (self.alpha * _gradient(X, y, self.theta_gradient))  # internet
            else:
                self.theta_gradient += (self.alpha * _gradient(X, y, self.theta_gradient))

            # ---------------------------------
            #  Just to compare if it's basically the same regarding the signs (course and internet version)
            # ---------------------------------
            self.theta_gradient2 += (self.alpha * -_gradient(X, y, self.theta_gradient2))
            # loglike2 = (np.dot(y, np.log(_h(theta, X))) + np.dot((1 - y), np.log(1 - _h(theta, X)))) / y.size
            loglike2 = ((y * np.log(_h(self.theta_gradient2, X))) + ((1 - y) * np.log(1 - _h(self.theta_gradient2, X)))).mean()
            self.J_gradient2_hist.append(loglike2)
            # ---------------------------------

            J_tmp = _log_likelihood(X, y, self.theta_gradient)

            if iter > 0 and np.fabs(J_tmp - self.J_gradient) < self.eps:
                print('  gradient iterations: ', iter)
                stop = 1

            self.J_gradient = J_tmp
            self.J_gradient_hist.append(self.J_gradient)

            iter += 1

        iter = 0
        stop = 0
        # while iter < self.n_iters:
        while not stop and iter < self.n_iters:
            # print('\nIteration ', stop_counter)
            # Newton's method update rule
            self.theta_newton -= (np.linalg.pinv(_hessian(X, self.theta_newton)) @ _gradient(X, y, self.theta_newton))
            J_tmp = _log_likelihood(X, y, self.theta_newton)

            if iter > 0 and np.fabs(J_tmp - self.J_newton) < self.eps:
                print('  newton iterations: ', iter)
                stop = 1

            self.J_newton = J_tmp
            self.J_newton_hist.append(self.J_newton)
            iter += 1

        # *** END CODE HERE ***

    # -----------------------------------------------------------------------------------------------------------------

    def predict(self, X, method='grad'):
        """Make a prediction given new inputs x.

        Args:
            method:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        # The hypothesis h_theta is the sigmoid function

        if method == 'grad':
            theta = self.theta_gradient
        elif method == 'newton':
            theta = self.theta_newton
        elif method == 'grad2':
            theta = self.theta_gradient2

        z = X @ theta
        sigmoid = 1 / (1 + np.exp(-z))

        # Need to set the threshold after we have the values of the sigmoid function
        res = sigmoid >= 0.5
        # print('pred.shape: ', sigmoid.shape)

        return res
        # *** END CODE HERE ***
    # -----------------------------------------------------------------------------------------------------------------
