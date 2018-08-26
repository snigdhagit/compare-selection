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
                      methods=['lee_1se', 'liu_1se', 'randomized_lasso_half_1se'],
                      n=500,
                      nsim=50,
                      p=100,
                      rho=[0.35],
                      s=5,
                      snr=[0.4, 0.6],
                      use_BH=True,
                      verbose=True,
                      wide_only=False)

# BH results

BH_opts = copy(base_opts)
BH_opts.csvfile = 'fdr_BH.csv'
BH_opts.use_BH = True
main_fdr(BH_opts)

# marginal screening results

marginal_opts = copy(base_opts)
marginal_opts.csvfile = 'fdr_marginal.csv'
marginal_opts.use_BH = False
main_fdr(marginal_opts)

# estimator results

estimator_opts = copy(base_opts)
estimator_opts.csvfile = 'estimation.csv'
estimator_opts.use_BH = False
main_estimators(estimator_opts)

# interval results

interval_opts = copy(base_opts)
interval_opts.csvfile = 'intervals.csv'
interval_opts.use_BH = False
main_intervals(interval_opts)

intervals = pd.read_csv('intervals_summary.csv')
estimation = pd.read_csv('estimation_summary.csv')
marginal = pd.read_csv('fdr_marginal_summary.csv')
BH = pd.read_csv('fdr_BH_summary.csv')

half1 = pd.merge(marginal, BH, left_on=['snr', 'class_name'], right_on=['snr', 'class_name'],
                 suffixes=(' (marginal)', ' (BH)'))
half2 = pd.merge(estimation, intervals, left_on=['snr', 'class_name'], right_on=['snr', 'class_name'],
                 suffixes=(' (estimation)', ' (intervals)'))
full = pd.merge(half1, half2, left_on=['snr', 'class_name'], right_on=['snr', 'class_name'])

full.to_csv('full_data.csv', index=False)

columns_for_plots = pd.DataFrame({'median_length':full['Median Length'],
                                  'marginal_power':full['Full Model Power (marginal)'],
                                  'BH_power':full['Full Model Power (BH)'],
                                  'marginal_FDP':full['Full Model FDR (marginal)'],
                                  'BH_FDP':full['Full Model FDR (BH)'],
                                  'coverage':full['Coverage'],
                                  'snr':full['snr'],
                                  'class_name':full['class_name']})
columns_for_plots.to_csv('plotting_data.csv', index=False)
