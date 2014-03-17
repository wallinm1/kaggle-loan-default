kaggle-loan-default
===================

Scripts for Kaggle Loan Default prediction contest

The .csv-files of predictions can be generated as follows:

1. Download and extract the data from http://www.kaggle.com/c/loan-default-prediction/data. Place the files train_v2.csv and test_v2.csv in the data-directory.

2. Run the read.py script. This generates a few .npy-files in the data-directory. This does not take long but is very RAM-intensive.

3. Run the feature selector scripts clf_selector.py, reg_selector_lad_log.py, reg_selector_quant_log.py and reg_selector_sgd_eps_log.py in any order. These scripts generate the feature selector vectors in .npy-format in the features directory. For the GBM-selector scripts in this repo, the maximum feature numbers to look for have been set quite low (26 for clf-features and 151 for regression-features) to reduce running time. In practice, the scripts were run for quite a bit longer and then interrupted when it became clear that the prediction error was no longer going to improve.

4. Run read.py. This generates the necessary models as .pkl-files in the models-directory. It also displays some diagnostic information on cross-validated prediction performance.

5. Run pred.py to generate the output files in the output-directory. The output file best_ens_preds.csv scores 0.46747 on the private leaderboard and the file mean_ens_preds.csv scores 0.46670. These are not exact replications of my pre-deadline submissions, as I forgot to set the random_state for the classifiers in the reg_selector_quant_log.py-file, but they are very close.

Hardware:

Most scripts were run on a laptop with 4GBs of RAM and a i5-2410M CPU @ 2.30 GHz. Reading the read.py-script did, however, completely clog up my RAM already when imputing the training set. Hence, I had to run read.py on the servers of my university. At worst, the RAM usage went over 10GB. The rest of the scripts can be run on 4GBs of RAM. With this RAM limitation, I was struggling to keep both training and test sets in memory, and hence I divided the code into several scripts and pickled the models, feature-vectors and data matrices between the different scripts. 

Running times:

Despite being RAM-intensive, the read.py scripts runs very quickly (~5 mins). The GBM-feature selector scripts (clf_selector.py, reg_selector_lad_log.py, reg_selector_quant_log.py) run in roughly one hour each on the aforementioned laptop. The SGD-feature selector (reg_selector_sgd_eps_log.py) runs in about 30 mins. The running time for train.py is between one and two hours and the pred.py-script runs in about 15 mins.
