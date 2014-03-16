import pandas as pd
import numpy as np
from sklearn.externals import joblib

def main():
    #based on models trained by train.py
    xtest=np.load('x_test.npy')
    #classification features
    sel_clf_feats=np.load('features/clf_sel.npy')
    #regression features
    sel_reg1=np.load('features/reg_sel_sgd_eps.npy')
    sel_reg2=np.load('features/reg_sel_quant.npy')
    sel_reg3=np.load('features/reg_sel_lad.npy') 
    
    feats_mat=np.vstack((sel_reg1,sel_reg2,sel_reg3))
    regs_unique=5
    feat_indic=np.hstack((0*np.ones(regs_unique),1*np.ones(regs_unique),
                          2*np.ones(regs_unique))) #maps regressors to features
                          
    clf=joblib.load('models/clf.pkl')
    rows=np.shape(xtest)[0]
    
    class_preds=clf.predict(xtest[:,sel_clf_feats])
    print "class_preds done"
    
    reg1=joblib.load('models/reg1.pkl')
    reg2=joblib.load('models/reg2.pkl')
    reg3=joblib.load('models/reg3.pkl')
    reg4=joblib.load('models/reg4.pkl')
    reg5=joblib.load('models/reg5.pkl')
    reg6=joblib.load('models/reg6.pkl')
    reg7=joblib.load('models/reg7.pkl')
    reg8=joblib.load('models/reg8.pkl')
    reg9=joblib.load('models/reg9.pkl')
    reg10=joblib.load('models/reg10.pkl')
    reg11=joblib.load('models/reg11.pkl')
    reg12=joblib.load('models/reg12.pkl')
    reg13=joblib.load('models/reg13.pkl')
    reg14=joblib.load('models/reg14.pkl')
    reg15=joblib.load('models/reg15.pkl')
    
    regs=[reg1,reg2,reg3,reg4,reg5,reg6,reg7,reg8,reg9,reg10,reg11,reg12,
          reg13,reg14,reg15]#,reg16,reg17,reg18,reg19,reg20]  
    n_regs=len(regs)  
    
    reg_ens1=joblib.load('models/reg_ens1.pkl')
    reg_ens2=joblib.load('models/reg_ens2.pkl')
    reg_ens3=joblib.load('models/reg_ens3.pkl')
    reg_ens4=joblib.load('models/reg_ens4.pkl')
    reg_ens5=joblib.load('models/reg_ens5.pkl')
    reg_ens6=joblib.load('models/reg_ens6.pkl')
    
    reg_ens=[reg_ens1,reg_ens2,reg_ens3,reg_ens4,reg_ens5,reg_ens6]
    n_reg_ens=len(reg_ens) 
    
    test_mat=np.zeros((rows,n_regs))
    
    print "predicting regression values for test set"    
    j=0
    i=1
    for reg in regs:
        feats=feats_mat[(feat_indic[j]),:]
        print "predicting for reg",i, "no of features", np.sum(feats) 
        tmp_preds=reg.predict(xtest[:,feats])
        tmp_preds=np.exp(tmp_preds) #training was done on log of loss, hence the exp
        tmp_preds=np.abs(tmp_preds)
        tmp_preds[tmp_preds>100]=100
        test_mat[:,j]=tmp_preds
        j+=1
        i+=1    
        
    ens_mat=np.zeros((rows,n_reg_ens))
    j=0
    i=1
    print "predicting ensembles"
    for reg in reg_ens:
        print "predicting for reg_ens",i
        tmp_preds=reg.predict(test_mat)
        tmp_preds=np.abs(tmp_preds)
        tmp_preds[tmp_preds>100]=100
        ens_mat[:,j]=tmp_preds
        j+=1
        i+=1
    
    #multiply ensemble loss predictions by class predictions
    ens_losses=np.multiply(ens_mat,class_preds[:,np.newaxis])
    
    #best ensemblers on the basis of cv
    good_ens=np.mean(ens_losses[:,(0,2)],1)
    
    ids=np.load('data/ids.npy')
    
    predsdf=pd.DataFrame({'id':ids.astype(int),'loss':good_ens})
    predsdf.to_csv('outputs/loss_preds.csv', index_label=False, index=False)
    np.save('outputs/loss_preds.npy',good_ens)
    
if __name__=="__main__":
    main()