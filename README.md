kaggle-loan-default
===================

Scripts for Kaggle Loan Default prediction contest

A .csv-file of predictions can be generated as follows:

1. Download and extract the data from http://www.kaggle.com/c/loan-default-prediction/data. Place the files train_v2.csv and test_v2.csv in the data-directory.

2. Run the read.py script. This generates a few .npy-files in the data-directory. This does not take long but can be quite RAM-intensive.

3. Run the feature selector scripts clf_selector.py, reg_selector_lad_log.py, reg_selector_quant_log.py and reg_selector_sgd_eps_log.py in any order. These scripts generate the feature selector vectors in .npy-format in the features directory.

4. Run read.py. This generates the necessary models as .pkl-files in the models-directory. It also displays some diagnostic information on cross-validated prediction performance.

5. Run pred.py to generate the output files in the output-directory.
