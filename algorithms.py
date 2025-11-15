import numpy as np


# PCA

class PCA_Model:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov = np.cov(X_centered, rowvar=False)

        eig_vals, eig_vecs = np.linalg.eigh(cov)

        idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, idx]

        self.components = eig_vecs[:, :self.n_components]

    def transform(self, X):
        return (X - self.mean).dot(self.components)


# KNN 

class KNN_Model:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        probs = []

        for i in range(X.shape[0]):
            distances = np.sum((self.X_train - X[i])**2, axis=1)

            idx = np.argsort(distances)[:self.k]

            neighbor_labels = self.y_train[idx]

            prob = np.zeros(10)
            for lab in neighbor_labels:
                prob[int(lab)] += 1

            prob /= self.k
            probs.append(prob)

        return np.array(probs)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
    
#soft max 

class LogisticRegressionModel:
    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def _softmax(self, Z):
        eZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return eZ / np.sum(eZ, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(self.epochs):
            Z = X.dot(self.W) + self.b
            A = self._softmax(Z)

            dW = (1 / n_samples) * X.T.dot(A - y_onehot)
            db = (1 / n_samples) * np.sum(A - y_onehot, axis=0, keepdims=True)

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict_proba(self, X):
        Z = X.dot(self.W) + self.b
        return self._softmax(Z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

#stacked model
class StackedModel:
    def __init__(self, knn_model, logistic_model):
        self.knn = knn_model
        self.logistic = logistic_model

    def fit(self, X, y):
        self.knn.fit(X, y)
        knn_feats = self.knn.predict_proba(X)
        self.logistic.fit(knn_feats, y)

    def predict(self, X):
        knn_feats = self.knn.predict_proba(X)
        return self.logistic.predict(knn_feats)

    def predict_proba(self, X):
        knn_feats = self.knn.predict_proba(X)
        return self.logistic.predict_proba(knn_feats)
