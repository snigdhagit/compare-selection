import os
from copy import copy
from itertools import product
import time

import numpy as np
import pandas as pd

from instances import data_instances
from utils import gaussian_setup
from gaussian_methods import methods

# import knockoff_phenom # more instances

def compare(instance, 
            statistic,
            summary,
            nsim=50, 
            methods=[], 
            verbose=False,
            htmlfile=None,
            method_setup=True,
            csvfile=None,
            q=0.2):
    
    results = [[] for m in methods]
    
    run_CV = np.any([getattr(m, 'need_CV') for m in methods])

    for method in methods:
        if method_setup:
            method.setup(instance.feature_cov)
        method.q = q

    method_params, class_names = get_method_params(methods)

    for i in range(nsim):

        X, Y, beta = instance.generate()
        l_min, l_1se, l_theory, sigma_reid = gaussian_setup(X.copy(), Y.copy(), run_CV=run_CV)

        for method, result in zip(methods, results):
            if verbose:
                print('method:', method)
            result.append(statistic(method, instance, X.copy(), Y.copy(), beta.copy(), l_theory.copy(), l_min, l_1se, sigma_reid))

            if i > 0:
                df = summarize(method_params, summary, results)
                for p in instance.params.columns:
                    df[p] = instance.params[p][0]
                df['class name'] = class_names
                if verbose:
                    print(df[summary(results[0]).columns])

                if htmlfile is not None:
                    f = open(htmlfile, 'w')
                    f.write(df.to_html() + '\n')
                    f.write(instance.params.to_html())
                    f.close()

                if csvfile is not None:
                    f = open(csvfile, 'w')
                    f.write(df.to_csv(index_label=True) + '\n')
                    f.close()

    return df, results

def get_method_params(methods):

    # find all columns needed for output

    colnames = []
    for method in methods:
        M = method(np.random.standard_normal((10,5)), np.random.standard_normal(10), 1., 1., 1., 1.)
        colnames += M.trait_names()
    colnames = sorted(np.unique(colnames))

    def get_col(method, colname):
        if colname in method.trait_names():
            return getattr(method, colname)

    def get_params(method):
        return [get_col(method, colname) for colname in colnames]

    method_params = []
    for method in methods:
        M = method(np.random.standard_normal((10,5)), np.random.standard_normal(10), 1., 1., 1., 1.)
        method_params.append(get_params(M))
    method_params = pd.DataFrame(method_params, columns=colnames)

    return method_params, [m.__name__ for m in methods]

def summarize(method_params, summary, results):
    results_df = pd.concat([summary(r) for r in results])
    results_df.index = method_params
    results_df.index.name = str(tuple(method_params.columns))
    return results_df

# Specific to  FDR comparison

def FDR_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):
    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    selected, active = M.select()
    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
        TD = instance.discoveries(selected, true_active)
        FD = len(selected) - TD
        FDP = FD / max(TD + 1. * FD, 1.)
        return TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)

def FDR_summary(result):
    result = np.atleast_2d(result)

    return pd.DataFrame([[result.shape[0],
                          np.mean(result[:,0]), 
                          np.std(result[:,0]) / np.sqrt(result.shape[0]), 
                          np.mean(result[:,1]), 
                          np.mean(result[:,2]), 
                          np.std(result[:,2]) / np.sqrt(result.shape[0]),
                          np.mean(result[:,3]),
                          np.mean(result[:,4])]],
                        columns=['Replicates', 
                                 'Full model power', 
                                 'SD(Full model power)', 
                                 'False discoveries', 
                                 'Full model FDR', 
                                 'SD(Full model FDR)', 
                                 'Time', 
                                 'Active'])

def main(opts, clean=False):

    if opts.list_instances:
        print('Instances:\n')
        print(sorted(data_instances.keys()))
    if opts.list_methods:
        print('Methods:\n')
        print(sorted(methods.keys()))
    if opts.list_instances or opts.list_methods:
        return

    if opts.signal_strength is not None:  # looping over signal strengths
        signal_vals = np.atleast_1d(opts.signal_strength)
    else:
        signal_vals = [None]

    new_opts = copy(opts)
    prev_rho = np.nan

    csvfiles = []
    results_dict = {}

    if opts.all_methods_noR: # noR takes precedence if both are used
        new_opts.methods = sorted([n for n, m in methods.items() if not m.selectiveR_method])
    elif opts.all_methods:
        new_opts.methods = sorted(methods.keys())
    
    if opts.wide_only: # only allow methods that are ok if p>n
        new_opts.methods = [m for m in new_opts.methods if m.wide_OK]

    for rho, signal in product(np.atleast_1d(opts.rho),
                               signal_vals):

        # try to save some time on setup of knockoffs

        method_setup = rho != prev_rho 
        prev_rho = rho

        new_opts.signal_strength = signal
        new_opts.rho = rho

        try:
            _methods = [methods[n] for n in new_opts.methods]
        except KeyError: # list the methods and quit
            print("Method not found. Valid methods:")
            print(sorted(methods.keys()))
            return
        try:
            _instance = data_instances[new_opts.instance]
        except KeyError: # list the methods and quit
            print("Data generating mechanism not found. Valid mechanisms:")
            print(sorted(data_instances.keys()))
            return
            
        _instance = _instance() # default instance to find trait names
        instance = data_instances[new_opts.instance](**dict([(n, getattr(new_opts, n)) for n in _instance.trait_names() if hasattr(new_opts, n)]))

        if signal is not None: # here is where signal_fac can be ignored
            instance.signal = new_opts.signal_strength

        if opts.csvfile is not None:
            new_opts.csvfile = (os.path.splitext(opts.csvfile)[0] + 
                       "_signal%0.1f_rho%0.2f.csv" % (new_opts.signal_strength,
                                                      new_opts.rho))
        csvfiles.append(new_opts.csvfile)

        results_df, results = compare(instance,
                                      FDR_statistic,
                                      FDR_summary,
                                      nsim=new_opts.nsim,
                                      methods=_methods,
                                      verbose=new_opts.verbose,
                                      htmlfile=new_opts.htmlfile,
                                      method_setup=method_setup,
                                      csvfile=new_opts.csvfile)
        results_dict[(rho, signal)] = results
        if opts.csvfile is not None:

            f = open(new_opts.csvfile, 'w')
            f.write('# parsed arguments: ' + str(new_opts) + '\n') # comment line indicating arguments used
            f.write(results_df.to_csv() + '\n')
            f.close()

            dfs = [pd.read_csv(f, comment='#') for f in csvfiles]
            df = pd.concat(dfs)
            df.to_csv(opts.csvfile, index=False)

    if opts.clean:
        [os.remove(f) for f in csvfiles]

    return results_dict


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Compare different LASSO methods in terms of full model FDR and Power.

Try:
    python compare.py --instance indep_instance --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html --csvfile indep.csv
''')
    parser.add_argument('--instance',
                        default='AR_instance',
                        dest='instance', help='Which instance to generate data from -- only one choice. To see choices run --list_instances.')
    parser.add_argument('--list_instances',
                        dest='list_instances', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--nsample', default=800, type=int,
                        dest='n',
                        help='number of data points, n (default 800)')
    parser.add_argument('--nfeature', default=300, type=int,
                        dest='p',
                        help='the number of features, p (default 300)')
    parser.add_argument('--nsignal', default=20, type=int,
                        dest='s',
                        help='the number of nonzero coefs, s (default 20)')
    parser.add_argument('--signal', type=float, nargs='+',
                        dest='signal_strength',
                        help='signal strength to override instance default (default value: None) -- signals are all of this magnitude, randomly placed with random signs')
    parser.add_argument('--signal_fac', default=1.2, type=float,
                        help='Scale applied to theoretical lambda to get signal size. Ignored if --signal is used.')
    parser.add_argument('--rho', nargs='+', type=float,
                        default=0.,
                        dest='rho',
                        help='Value of AR(1), equicor or mixed param.')
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
    parser.add_argument('--clean', help='Remove individual CSV files after termination?',
                        default=False)
    parser.add_argument('--all_methods', help='Run all methods.',
                        default=False,
                        action='store_true')
    parser.add_argument('--all_methods_noR', help='Run all methods except the R selectiveInference methods. Takes precendence over --all_methods when both used.',
                        default=False,
                        action='store_true')
    parser.add_argument('--wide_only', help='Require methods that are OK for wide -- silently ignore other methods.',
                        default=False,
                        action='store_true')

    parser.add_argument('--cor_thresh', help='Correlation threshold for determining true or false discovery',
                        default=0.5,
                        type=float)

    opts = parser.parse_args()

    results = main(opts)

