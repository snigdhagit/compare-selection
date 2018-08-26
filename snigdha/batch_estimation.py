from copy import copy
from argparse import Namespace

import numpy as np, pandas as pd

from compare_fdr import main as main_fdr
from compare_intervals import main as main_intervals
from compare_estimators import main as main_estimators


base_opts = Namespace(all_methods=False, 
                      all_methods_noR=False, 
                      concat=False, 
                      cor_thresh=0.5, 
                      csvfile='estimators.csv', 
                      htmlfile='estimators.html', 
                      instance='bestsubset_instance', 
                      level=0.2, 
                      list_instances=False, 
                      list_methods=False, 
                      methods=['selective_MLE_half_1se', 'randomized_LASSO_half_1se', 'randomized_relaxed_LASSO_half_1se',
                               'LASSO_1se', 'relaxed_LASSO_1se'],
                      n=500, 
                      nsim=50,
                      p=100,
                      rho=[0.35],
                      s=5, 
                      snr=[0.4, 0.6], 
                      use_BH=True, 
                      verbose=True, 
                      wide_only=False)


# estimator results

estimator_opts = copy(base_opts)
estimator_opts.csvfile = 'estimation.csv'
estimator_opts.use_BH = False
main_estimators(estimator_opts)

estimation = pd.read_csv('estimation_summary.csv')

