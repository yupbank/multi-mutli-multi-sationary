from numpy import *
import copy
from sklearn.linear_model import LogisticRegression, SGDClassifier
import numpy as np

class CC() :
    '''
        Classifier Chain
        ----------------
    '''

    h = None
    L = -1

    def __init__(self, L=-1, h=LogisticRegression()):
        ''' note: L option to be deprecated here ! '''
        self.hop = h

    def fit(self, X, Y):
        N, self.L = Y.shape
        N, L = Y.shape
        N, D = X.shape

        self.h = [ copy.deepcopy(self.hop) for j in range(L)]
        XY = zeros((N, D + L-1))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:L-1]
        for j in range(self.L):
            self.h[j].fit(XY[:,0:D+j], Y[:,j])
        return self

#    def fitOLD(self, X, Y):
#        N, self.L = Y.shape
#        self.h = [ copy.deepcopy(self.hop) for j in range(self.L)]
#        for j in range(self.L):
#            if j>0:
#                X = column_stack([X, Y[:,j-1]])
#            self.h[j].fit(X, Y[:,j])
#        return self

    def predict(self, X):
        '''
            return predictions for X
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict(X)
        return Y

    def predict_proba(self, X):
        '''
            return confidence predictions for X
            NOTE: for multi-label (binary) data only at the moment.
                ( WARNING: may give index-out-of-bounds error if uni- or multi-target (of > 2 values) data is used in training )
        '''
        N,D = X.shape
        Y = zeros((N,self.L))
        for j in range(self.L):
            if j>0:
                X = column_stack([X, Y[:,j-1]])
            Y[:,j] = self.h[j].predict_proba(X)[:,1]
        return Y


class RCC(CC):
    '''
        Random Classifier Chain
        -----------------------
        Chain will be in a random order.
    '''

    chain = None

    def fit(self, X, Y):
        N,L = Y.shape
        self.chain = arange(L)
        random.shuffle(self.chain)
        return CC.fit(self, X, Y[:,self.chain])

    def predict(self, X):
        '''
            return predictions for X
        '''
        Y = CC.predict(self,X)
        return Y[:,argsort(self.chain)]

class FCC(CC):

    def fit(self, X, Y, learn=True):
        N, self.L = Y.shape
        N, D = X.shape

        self.h = [ copy.deepcopy(self.hop) for j in range(self.L)]
        XY = zeros((N, D + self.L))
        XY[:,0:D] = X
        XY[:,D:] = Y[:,0:self.L]
        for j in range(self.L):
            tmp_XY = np.copy(XY)
            if learn:
                tmp_XY[:, D+j] = 0
            self.h[j].fit(tmp_XY, Y[:,j])

    def predict(self, X, round_iter):
        '''
            return predictions for X
        '''
        N, D = X.shape
        Y = zeros((N, self.L))

        for j in range(self.L):
            t_X = column_stack([X, Y])
            Y[:,j] = self.h[j].predict(t_X)
        for i in xrange(round_iter):
            for j in range(self.L):
                t_X = column_stack([X, Y])
                Y[:,j] = self.h[j].predict(t_X)

        return Y

def demo():
    #from molearn.core.tools import make_XOR_dataset
    import sys
    sys.path.append( '../core' )
    from tools import make_XOR_dataset

    X,Y = make_XOR_dataset()
    N,L = Y.shape

    cc = RCC(L, SGDClassifier(n_iter=100))
    cc.fit(X, Y)
    # test it
    print(cc.predict(X))
    print("vs")
    print(Y)

if __name__ == '__main__':
    demo()

