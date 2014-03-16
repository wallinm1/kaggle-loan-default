from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import ShuffleSplit
import numpy as np
from sklearn.metrics import mean_absolute_error

def main():
    xtrain=np.load('data/x_train.npy')
    ytrainreg=np.load('data/loss.npy')
    xtrain=xtrain[ytrainreg>0,:]
    ytrainreg=ytrainreg[ytrainreg>0]
    
    #training on half of the train set to save time
    ss1=ShuffleSplit(np.shape(ytrainreg)[0],1, test_size=0.5, random_state=42)
    for train_idx, test_idx in ss1:
        xtrain=xtrain[test_idx,:]
        ytrainreg=ytrainreg[test_idx]
    
    #train-test split
    ss=ShuffleSplit(np.shape(ytrainreg)[0],n_iter=1,test_size=0.3, random_state=42)
    for train_idx, test_idx in ss:
        xtest=xtrain[test_idx,:]
        ytestreg=ytrainreg[test_idx]
        xtrain=xtrain[train_idx,:]
        ytrainreg=ytrainreg[train_idx]
    
    
    reg_init=GradientBoostingRegressor(loss='lad',min_samples_leaf=5,
                                   n_estimators=100,random_state=42,verbose=10)
    reg_init.fit(xtrain,np.log(ytrainreg)) #training on the log of the loss
    feat_imp=reg_init.feature_importances_
    sorted_fi=feat_imp[np.argsort(feat_imp)[::-1]] #descending
    reg=GradientBoostingRegressor(loss='lad',min_samples_leaf=5,
                                   n_estimators=100,random_state=42)
    feats_tot=np.shape(xtrain)[1]
    mae_best=10000
    for feats in range(1,feats_tot+1):
        threshold_idx=min(len(sorted_fi),feats)
        threshold=sorted_fi[threshold_idx]
        select=(feat_imp>threshold)
        reg.fit(xtrain[:,select],np.log(ytrainreg)) #training on the log of the loss
        tmp_loss=reg.predict(xtest[:,select])
        tmp_loss=np.exp(tmp_loss) #training was done on log of loss, hence the exp
        tmp_loss=np.abs(tmp_loss)
        tmp_loss[tmp_loss>100]=100
        mae=mean_absolute_error(ytestreg,tmp_loss)
        if mae<mae_best:
            mae_best=mae
            np.save('features/reg_sel_lad.npy',select)
        print feats,mae
        if feats>150:
            break
    print "mae_best:", mae_best
    
if __name__=="__main__":
    main()