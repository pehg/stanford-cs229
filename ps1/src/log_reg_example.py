# Example of implementation of logistic regression algorithm found in:
#   https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
#
# The code can be found in github:
#   https://github.com/martinpella/logistic-reg
#

from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import util

# import Iris data
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

# import Iris data
X2, y2 = datasets.make_moons(n_samples=800, shuffle=False, noise=0.24, random_state=9)
print(type(X2))
print(type(y2))
# X2 = moons.data[:, :2]
# y2 = (moons.target != 0) * 1


train_path = '../data/ds1_train.csv'
eval_path='../data/ds1_valid.csv'
x_train, y_train = util.load_dataset(train_path, add_intercept=False)
x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)


def my_plot(X, y, fig=None, display=True):
    if fig is None:
        plt.figure()

    sns.scatterplot(x=X[y == 0, 0], y=X[y == 0, 1], label='0')
    sns.scatterplot(x=X[y == 1, 0], y=X[y == 1, 1], label='1')

    if display:
        plt.legend()
        plt.show()


class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = None
        self.J = 0
        self.J_hist = None

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _sigmoid(z):
        res = 1 / (1 + np.exp(-z))
        return res

    @staticmethod
    def _J(h, y):
        loss = (-(y * np.log(h)) - (1 - y) * np.log(1 - h)).mean()
        return loss

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept(X)

        self.theta = np.zeros(X.shape[1])
        self.J_hist = []

        iter = 0
        while iter < self.n_iters:
            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            grad = np.dot(X.T, (h - y)) / y.size

            self.theta -= self.lr * grad

            z = np.dot(X, self.theta)
            h = self._sigmoid(z)
            J = self._J(h, y)
            self.J_hist.append(J)

            if (self.verbose == True and iter % 10000 == 0):
                print(f'loss: {J} \t')

            iter += 1

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)

        return self._sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        return self.predict_prob(X).round()

    def plot_cost_func(self, title=None):
        sns.lineplot(x=range(len(self.J_hist)), y=self.J_hist, label='J(\{theta})')

        if title is None:
            title = 'Cost function over each iteration'
        plt.title(title)

        plt.show()

    def plot_boundary(self, X, y, theta=None, correction=1.0):
        # plt.figure(figsize=(10, 6))
        plt.figure()
        sns.scatterplot(X[y == 0][:, 0], X[y == 0][:, 1], label='0')
        sns.scatterplot(X[y == 1][:, 0], X[y == 1][:, 1], label='1')
        plt.legend()

        if theta is None:
            x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
            x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),

            xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

            grid = np.c_[xx1.ravel(), xx2.ravel()]
            probs = self.predict_prob(grid).reshape(xx1.shape)
            plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
        else:
            x1 = np.arange(min(X[:, -2]), max(X[:, -2]), 0.01)
            x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
            # plt.plot(x1, x2, c='red', linewidth=2)
            plt.plot(x1, x2, linewidth=1)

        plt.show()


# --------------------------------------------------
#  model - Using iris dataset
# --------------------------------------------------
# model = LogisticRegression(lr=0.1, n_iters=3000)
# model.fit(X, y)
# preds = model.predict(X)
# print('model - accuracy: ', np.mean(preds == y))
# print('Parameters: ', model.theta)
# model.plot_cost_func()
# model.plot_boundary(X, y)

# --------------------------------------------------
#  model2 - Using moons dataset
# --------------------------------------------------
# model2 = LogisticRegression(lr=0.1, n_iters=5000)
# model2.fit(X2, y2)
# preds2 = model.predict(X2)
# print('model2 - accuracy: ', np.mean(preds2 == y2))
# print('Parameters: ', model2.theta)
# model2.plot_cost_func()
# model2.plot_boundary(X2, y2)


# --------------------------------------------------
#  model3 - Using CS229's problem 1b dataset
# --------------------------------------------------
model3 = LogisticRegression(lr=0.001, n_iters=5000)
model3.fit(x_train, y_train)
preds3 = model3.predict(x_train)
print('\nmodel3 - Accuracy: ', np.mean(preds3 == y_train))
print('model3 - Parameters: ', model3.theta)
model3.plot_cost_func()
model3.plot_boundary(x_train, y_train)
model3.plot_boundary(x_train, y_train, theta=model3.theta)


# using eval data set
model3.plot_boundary(x_eval, y_eval)
model3.plot_boundary(x_eval, y_eval, theta=model3.theta)