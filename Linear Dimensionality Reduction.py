
import numpy as np
import pickle as pkl

class LDA:
    def __init__(self,k):
        self.n_components = k
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        X: (n,d,d) array consisting of input features
        y: (n,) array consisting of labels
        return: Linear Discriminant np.array of size (d*d,k)
        """
        # TODO
        self.linear_discriminants=np.zeros((len(X[0]*len(X[0])),self.n_components)) # Modify as required 
        d_2 = X.shape[1]*X.shape[2]
        n = X.shape[0]
        X = X.reshape(n, -1)
        mean = np.mean(X, axis=0)
        Sb = np.zeros((d_2, d_2))
        Sw = np.zeros((d_2, d_2))
        class_items = np.unique(y)

        for i in class_items:
            X_i = X[y == i]
            mean_i = np.mean(X_i, axis=0)
            Sw += (X_i - mean_i).T.dot(X_i - mean_i)
            mean_diff = (mean_i - mean).reshape(-1, 1)
            Sb += X_i.shape[0] * mean_diff.dot(mean_diff.T)

        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))

        eigs = []
        for i in range(len(eigenvalues)):
            eigs.append((np.abs(eigenvalues[i]), eigenvectors[:,i]))
        eigs.sort(key=lambda x: x[0], reverse=True)

        linear_discriminants = []
        for i in range(0, self.n_components):
            linear_discriminants.append(eigs[i][1].reshape(d_2, 1))
        self.linear_discriminants = np.hstack(linear_discriminants)
        return self.linear_discriminants                # Modify as required        
        #END TODO 
    
    def transform(self, X, w):
        """
        w:Linear Discriminant array of size (d*d,1)
        return: np-array of the projected features of size (n,k)
        """
        # TODO
        projected = np.dot(X.reshape(len(X), -1), w)     # Modify as required
        return projected                   # Modify as required
        # END TODO
if __name__ == '__main__':
    mnist_data = 'mnist.pkl'
    with open (mnist_data,'rb') as f:
        data = pkl.load(f)
    X = data[0]
    y = data[1]
    k = 10
    lda = LDA(k)
    w = lda.fit(X,y)
    XLda = lda.transform(X,w)