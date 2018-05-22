import os
from copy import copy
from itertools import product
import time

import numpy as np
import pandas as pd

from instances import data_instances
from utils import gaussian_setup
from gaussian_methods import methods
from liu_setup import liu_null

# import knockoff_phenom # more instances

def compare(instance, 
            nsim=50, 
            q=0.2,
            methods=[], 
            verbose=False,
            htmlfile=None,
            csvfile=None):
    
    results = [[] for m in methods]
    
    run_CV = False

    # find all columns needed for output

    colnames = []
    for method in methods:
        M = method(np.random.standard_normal((10,5)), np.random.standard_normal(10), np.nan, np.nan, np.nan, np.nan)
        colnames += M.trait_names()
    colnames = sorted(np.unique(colnames))

    def get_col(method, colname):
        if colname in method.trait_names():
            return getattr(method, colname)

    def get_cols(method):
        return [get_col(method, colname) for colname in colnames]

    for i in range(nsim):

        X, Y, beta = instance.generate()
        l_theory = np.sqrt(instance.n) * instance.penalty # no standardization
        true_active = np.nonzero(beta)[0]

        def summary(result):
            result = np.atleast_2d(result)

            return [result.shape[0],
                    np.mean(result[:,0]), 
                    np.std(result[:,0]) / np.sqrt(result.shape[0]), 
                    np.mean(result[:,1]), 
                    np.mean(result[:,2]), 
                    np.std(result[:,2]) / np.sqrt(result.shape[0]),
                    np.mean(result[:,3]),
                    np.mean(result[:,4])]

        method_instances = []
        for method, result in zip(methods, results):
            if verbose:
                print('method:', method)
            toc = time.time()
            M = method(X.copy(), Y.copy(), l_theory.copy(), np.nan, np.nan, np.nan)
            method_instances.append(M)
            M.q = q
            selected, active = M.select()
            tic = time.time()
            if len(active) > 0: # did we make any discoveries?
                TD = instance.discoveries(selected, true_active)
                FD = len(selected) - TD
                FDP = FD / max(TD + 1. * FD, 1.)
            else:
                TD = FD = FDP = 0
            if len(true_active) > 0: # null case
                result.append((TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)))
            else:
                result.append((0, FD, FDP, tic-toc, len(active)))
            if i > 0:
                df = pd.DataFrame([get_cols(m) + summary(r) for m, r in zip(method_instances, results)], 
                                  columns=colnames + ['Replicates', 'Full model power', 'SD(Full model power)', 'False discoveries', 'Full model FDR', 'SD(Full model FDR)', 'Time', 'Active'])

                if verbose:
                    print(df[['Replicates', 'Full model power', 'Time']])

                if htmlfile is not None:
                    f = open(htmlfile, 'w')
                    f.write(df.to_html(index=False) + '\n')
                    f.write(instance.params.to_html())
                    f.close()

                if csvfile is not None:

                    df_cp = copy(df)
                    param = instance.params
                    for col in param.columns:
                        df_cp[col] = param[col][0] 
                    df_cp['distance_tol'] = instance.distance_tol
                    f = open(csvfile, 'w')
                    f.write(df_cp.to_csv(index=False) + '\n')
                    f.close()

    big_df = copy(df)
    param = instance.params
    for col in param.columns:
        big_df[col] = param[col][0] 
    big_df['distance_tol'] = instance.distance_tol
    return big_df, results


def main(opts, clean=False):

    _instance = data_instances[opts.instance](n=20, p=10)
    instance = data_instances[opts.instance](**dict([(n, getattr(opts, n)) for n in _instance.trait_names() if hasattr(opts, n)]))

    _methods = [methods['randomized_lasso_half'],
                methods['liu_theory'],
                methods['lee_theory']]

    results_df, results = compare(instance,
                                  nsim=opts.nsim,
                                  methods=_methods,
                                  verbose=opts.verbose,
                                  htmlfile=opts.htmlfile,
                                  csvfile=opts.csvfile)

    f = open(opts.csvfile, 'w')
    f.write('# parsed arguments: ' + str(opts) + '\n') # comment line indicating arguments used
    f.write(results_df.to_csv(index=False) + '\n')
    f.close()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Compare different LASSO methods in terms of full model FDR and Power.

Try:
    python compare.py --instance indep_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html --csvfile indep.csv
''')
    parser.add_argument('--instance',
                        default='liu_null',
                        dest='instance', help='Which instance to generate data from -- only one choice. To see choices run --list_instances.')
    parser.add_argument('--list_instances',
                        dest='list_instances', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--nsample', default=100, type=int,
                        dest='n',
                        help='number of data points, n (default 100)')
    parser.add_argument('--nfeature', default=50, type=int,
                        dest='p',
                        help='the number of features, p (default 50)')
    parser.add_argument('--nsignal', default=5, type=int,
                        dest='s',
                        help='the number of nonzero coefs, s (default 5)')
    parser.add_argument('--q', default=0.2, type=float,
                        help='target for FDR (default 0.2)')
    parser.add_argument('--nsim', default=100, type=int,
                        help='How many repetitions?')
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--htmlfile', help='HTML file to store results for one (signal, rho). When looping over (signal, rho) this HTML file tracks the current progress.',
                        dest='htmlfile')
    parser.add_argument('--csvfile', help='CSV file to store results looped over (signal, rho).',
                        dest='csvfile')

    opts = parser.parse_args()

    results = main(opts)

