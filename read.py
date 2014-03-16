import pandas as pd
import numpy as np
import  scipy.stats as stats
from sklearn import preprocessing

def main():
    dat=pd.read_table('data/train_v2.csv',sep=',')
    print "reading done, train"
    loss=np.asarray(dat.loss)
    dat=dat.drop(['loss','id'],1)
    dat['new1']=dat['f528']-dat['f527'] #golden feature 1
    dat['new2']=dat['f528']-dat['f274'] #golden feature 2
    dat=np.asarray(dat.values, dtype=float)
    col_med = stats.nanmedian(dat,axis=0)
    print "calculated medians, train"
    inds = np.where(np.isnan(dat))
    dat[inds]=np.take(col_med,inds[1])
    print "median imputation done, train"
    scaler=preprocessing.Scaler().fit(dat)
    dat=scaler.transform(dat)
    print "scaling done, train"
    labels=(loss>0).astype(int)
    np.save('data/x_train.npy',dat)
    np.save('data/y_train.npy',labels)
    np.save('data/loss.npy',loss)
    print "trainset done"
    
    dat=pd.read_table('data/test_v2.csv',sep=',')
    print "reading done, test"
    ids=np.asarray(dat.id)
    dat=dat.drop(['id'],1)
    dat['new1']=dat['f528']-dat['f527'] #golden feature 1
    dat['new2']=dat['f528']-dat['f274'] #golden feature 2
    dat=np.asarray(dat.values,dtype=float)
    col_med=stats.nanmedian(dat,axis=0)
    print "calculated medians, test"
    inds=np.where(np.isnan(dat))
    dat[inds]=np.take(col_med,inds[1])
    print "imputation done, test"
    dat=scaler.transform(dat)
    print "scaling done, test"
    np.save('data/x_test.npy',dat)
    np.save('data/ids.npy',ids)
    print "testset done"
    
if __name__=="__main__":
    main()