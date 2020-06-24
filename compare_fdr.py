import os, hashlib
from copy import copy
from itertools import product
import time

import numpy as np
import pandas as pd

from instances import data_instances
from utils import gaussian_setup, summarize
from statistics import (BH_statistic, 
                        BH_summary,
                        marginal_statistic,
                        marginal_summary)
from gaussian_methods import methods
import posi, knockoffs # register these methods

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
            concat=False,
            level=0.2,
            use_BH=True):
    
    results = []
    
    run_CV = np.any([getattr(m, 'need_CV') for m in methods])

    for method in methods:
        if method_setup:
            method.setup(instance.feature_cov, instance)
        if use_BH:
            method.q = level
        else:
            method.alpha = level

    method_params, class_names, method_names = get_method_params(methods)

    if csvfile is not None and os.path.exists(csvfile):
        results_existing = pd.read_csv(csvfile)
    else:
        results_existing = None

    for i in range(nsim):

        X, Y, beta = instance.generate()

        # make a hash representing same data

        instance_hash = hashlib.md5()
        instance_hash.update(X.tobytes())
        instance_hash.update(Y.tobytes())
        instance_hash.update(beta.tobytes())
        instance_id = instance_hash.hexdigest()

        l_min, l_1se, l_theory, sigma_reid = gaussian_setup(X.copy(), Y.copy(), run_CV=run_CV)

        for method, method_name, class_name, idx in zip(methods, 
                                                        method_names,
                                                        class_names,
                                                        range(len(methods))):
            if verbose:
                print('method:', method)

            M, result_df = statistic(method, 
                                     instance, 
                                     X.copy(), 
                                     Y.copy(), 
                                     beta.copy(), 
                                     l_theory.copy(), 
                                     copy(l_min), 
                                     copy(l_1se), 
                                     sigma_reid)

            if result_df is not None:
                result_df['instance_id'] = instance_id
                result_df['method_param'] = str(method_params.loc[idx])
                result_df['model_target'] = M.model_target
                result_df['method_name'] = method_name
                result_df['class_name'] = class_name
                results.append(result_df)
            else:
                print('Result was empty.')

            if i > 0 and len(results) > 0:

                results_df = pd.concat(results)

                for p in instance.params.columns:
                    results_df[p] = instance.params[p][0]

                if csvfile is not None:
                    if results_existing is not None:
                        results_full = pd.concat([results_existing, results_df])
                    else:
                        results_full = results_df
                    f = open(csvfile, 'w')
                    f.write(results_full.to_csv(index_label=False) + '\n')
                    f.close()

                summary_df = summarize('method_param',
                                       results_full,
                                       summary)

                for p in instance.params.columns:
                    summary_df[p] = instance.params[p][0]

                if htmlfile is not None:
                    f = open(htmlfile, 'w')
                    f.write(summary_df.to_html() + '\n')
                    f.write(instance.params.to_html())
                    f.close()

                    # also write a summary CSV

                    f = open(csvfile.replace('.csv', '_summary.csv'), 'w')
                    f.write(summary_df.to_csv(index_label=False) + '\n')
                    f.close()

                    # also write a summary html

                    f = open(csvfile.replace('.csv', '_summary.html'), 'w')
                    f.write(summary_df.to_html() + '\n')
                    f.close()

        if i > 0 and len(np.unique(results_full['instance_id'])) >= nsim:
            break

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

    method_names = []
    method_params = []
    for method in methods:
        M = method(np.random.standard_normal((10,5)), np.random.standard_normal(10), 1., 1., 1., 1.)
        method_params.append(get_params(M))
        method_names.append(M.method_name)

    method_params = pd.DataFrame(method_params, columns=colnames)

    return method_params, [m.__name__ for m in methods], method_names

def main(opts):

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
            invalid_methods = [n for n in new_opts.methods if n not in methods.keys()]
            print("Methods %s not found. Valid methods:" % str(invalid_methods))
            print(sorted(methods.keys()))
            return
        try:
            _instance = data_instances[new_opts.instance]
        except KeyError: # list the methods and quit
            print("Data generating mechanism %s not found. Valid mechanisms:" % new_opts.instance)
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

        if new_opts.use_BH:
            statistic, summary = BH_statistic, BH_summary
        else:
            statistic, summary = marginal_statistic, marginal_summary

        compare(instance,
                statistic,
                summary,
                nsim=new_opts.nsim,
                methods=_methods,
                verbose=new_opts.verbose,
                htmlfile=new_opts.htmlfile,
                method_setup=method_setup,
                csvfile=new_opts.csvfile,
                use_BH=new_opts.use_BH,
                level=new_opts.level)

    # concat all csvfiles

    if opts.csvfile is not None:
        all_results = pd.concat([pd.read_csv(f) for f in csvfiles])
        all_results.to_csv(opts.csvfile)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Compare different LASSO methods in terms of full model FDR and Power.

Try:
    python compare_fdr.py --instance AR_instance --rho 0.3 --nsample 100 --nfeature 50 --nsignal 10 --methods lee_theory liu_theory --htmlfile indep.html --csvfile indep.csv --signal 2.0
''')
    parser.add_argument('--instance',
                        default='AR_instance',
                        dest='instance', help='Which instance to generate data from -- only one choice. To see choices run --list_instances.')
    parser.add_argument('--list_instances',
                        dest='list_instances', action='store_true')
    parser.add_argument('--methods', nargs='+', help='Which methods to use -- choose many. To see choices run --list_methods.', dest='methods')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true')
    parser.add_argument('--concat',
                        dest='concat', action='store_true', default=False)
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
    parser.add_argument('--level', default=0.2, type=float,
                        help='target for FDR or type I error if using marginal screening (default 0.2)')
    parser.add_argument('--use_BH', dest='use_BH', action='store_true',
                        help='use BH on pvalues as selection rule', default=True)
    parser.add_argument('--use_marginal', dest='use_BH', action='store_false',
                        help='use marginal screening of pvalues rather than BH')
    parser.add_argument('--nsim', default=100, type=int,
                        help='How many repetitions?')
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose')
    parser.add_argument('--htmlfile', help='HTML file to store results for one (signal, rho). When looping over (signal, rho) this HTML file tracks the current progress.',
                        dest='htmlfile')
    parser.add_argument('--csvfile', help='CSV file to store results looped over (signal, rho). Serves as a file base for individual (signal, rho) pairs.',
                        dest='csvfile')
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

