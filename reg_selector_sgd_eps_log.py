import numpy as np
from sklearn.feature_selection import RFECV
from sklearn import linear_model

def main():
    xtrain=np.load('data/x_train.npy')
    ytrainreg=np.load('data/loss.npy')
    xtrain=xtrain[ytrainreg>0]
    ytrainreg=ytrainreg[ytrainreg>0]
    reg1=linear_model.SGDRegressor(loss='epsilon_insensitive',random_state=0,n_iter=5)
    selector1=RFECV(estimator=reg1,scoring='mean_squared_error',verbose=10)
    selector1.fit(xtrain,np.log(ytrainreg)) #training on the log of the loss
    print "sel1, optimal number of features:", selector1.n_features_
    np.save('features/reg_sel_sgd_eps.npy', selector1.support_)
    
if __name__=="__main__":
    main()