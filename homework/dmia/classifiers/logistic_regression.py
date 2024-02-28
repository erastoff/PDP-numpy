import numpy as np
from scipy import sparse


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            self.w = np.random.randn(dim) * 0.01

        self.loss_history = []
        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)

            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w -= learning_rate * gradW
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return self

    def predict_proba(self, X, append_bias=False):
        if append_bias:
            X = LogisticRegression.append_biases(X)

        scores = X.dot(self.w)
        probs = 1 / (1 + np.exp(-scores))
        y_proba = np.vstack((1 - probs, probs)).T

        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = np.argmax(y_proba, axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        dw = np.zeros_like(self.w)
        num_train = X_batch.shape[0]

        scores = X_batch.dot(self.w)
        probs = 1 / (1 + np.exp(-scores))
        loss = (
            -np.sum(y_batch * np.log(probs) + (1 - y_batch) * np.log(1 - probs))
            / num_train
        )

        dw = X_batch.T.dot(probs - y_batch) / num_train

        # Add regularization to the loss and gradient
        loss += 0.5 * reg * np.sum(self.w[:-1] ** 2)  # Exclude bias term
        dw[:-1] += reg * self.w[:-1]  # Exclude bias term

        return loss, dw

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
