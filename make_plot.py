import os, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gaussian_methods import methods

default_methods = [methods[n].method_name for n in ['knockoffs_sigma',
                                                    'knockoffs_sigma_equi',
                                                    'randomized_lasso_half_1se',
                                                    'lee_aggressive']]

def make_plot(results,
              axis,
              method_labels,
              metric,
              xlabel=None,
              ylim=None,
              ax=None):

    fig = plt.gcf()
    if ax is None:
        ax = fig.gca()

    for i, method in enumerate(results):
        df_ = method[1]
        print(method)
        print(df_)
        if len(df_) > 0:
            ax.plot(df_[axis], df_[metric], 'o--', label=method_labels[i])
            print(method)
    if ylim:
        ax.set_ylim(ylim)
    xlabel = xlabel or axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)

    return ax, fig

def signal_plot(rho_results, rho, outbase, methods, method_labels):

    fig = plt.figure(num=1, figsize=(12, 5))
    fig.clf()
    rho_results = [(k[1:], v) for k, v in rho_results if k[2] == rho and k[1] in methods]

    ax = plt.subplot(121)
    ax, fig = make_plot(rho_results, 
                        'signal',
                        method_labels,
                        ylim=(0,1), 
                        ax=ax, 
                        metric='Full model power')

    ax = plt.subplot(122)
    ax, fig = make_plot(rho_results, 
                        'signal',
                        method_labels,
                        ylim=(0,1), 
                        ax=ax,
                        metric='Full model FDR', )
    ax.plot([3.5,5], [0.2, 0.2], 'k--', linewidth=2)

    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(outbase + '_rho%0.2f.pdf' % rho)
    fig.savefig(outbase + '_rho%0.2f.png' % rho)

    df = pd.concat([v for _, v in rho_results])
    df.to_csv(outbase + '_rho%0.2f.csv' % rho)

def rho_plot(signal_results, signal, outbase, methods, method_labels):

    signal_results = [(k[1:], v) for k, v in signal_results if k[2] == signal and k[1] in methods]

    fig = plt.figure(num=1, figsize=(10, 5))
    fig.clf()
    ax = plt.subplot(121)
    ax, fig = make_plot(signal_results, 
                        'signal',
                        method_labels,
                        ylim=(0,1), 
                        ax=ax,
                        metric='Full model FDR')

    ax = plt.subplot(122)
    ax, fig = make_plot(signal_results, 
                        'signal',
                        method_labels,
                        ylim=(0,1), 
                        ax=ax,
                        metric='Full model power')

    ax.plot([0, 0.75], [0.2, 0.2], 'k--', linewidth=2)
    ax.legend(loc='upper right', fontsize='xx-small')
    fig.savefig(outbase + '_signal%0.1f.pdf' % signal)
    fig.savefig(outbase + '_signal%0.1f.png' % signal)

def extract_results(df):

    results = [v for v in df.groupby([df.index, 'class name'])]
    rho_results = [v for v in df.groupby([df.index, 'class name', 'rho'])]
    signal_results = [v for v in df.groupby([df.index, 'class name', 'signal'])]

    return results, rho_results, signal_results
    
def main(opts):

    csvfiles = glob.glob(opts.csvbase + '*signal*rho*csv')
    df = pd.concat([pd.read_csv(f, comment='#') for f in csvfiles])

    results, rho_results, signal_results = extract_results(df)

    methods = opts.methods or np.unique(df['class name'])
    method_labels = opts.method_labels or methods
    if len(method_labels) != len(methods):
        raise ValueError('each method should have a name')

    outbase = opts.outbase or opts.csvbase

    df['method_label'] = df['class name']

    # plot with rho on x axis
    g_rho = sns.FacetGrid(df, col="signal", hue='method_label', sharex=True, sharey=True, col_wrap=2, size=5)
    
    def power_plot(rho, power, color='r', label='foo'):
        ax = plt.gca()
        ax.plot(rho, power, 'o--', color=color, label=label)

    g_rho.map(power_plot, 'rho', 'Full model power')

    # plot with signal on x axis
    g_signal = sns.FacetGrid(df, col="rho", hue='method_label', sharex=True, sharey=True, col_wrap=2, size=5)
    g_signal.map(power_plot, 'signal', 'Full model power')

    return df

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Make plots for

Try:
    python make_plot.py --methods lee_theory liu_theory --csvbase indep.csv
''')
    parser.add_argument('--csvbase', help='Basename of csvfile. To see choices run --list_methods.', dest='csvbase')
    parser.add_argument('--methods', nargs='+',
                        dest='methods', 
                        help='Names of methods in plot (i.e. class name). Defaults to all methods.')
    parser.add_argument('--method_labels', nargs='+',
                        dest='method_labels', 
                        help='Labels for methods in plot. Defaults to all methods.')
    parser.add_argument('--list_methods',
                        dest='list_methods', action='store_true', help='Methods in the csvfile.')
    parser.add_argument('--signal', type=float, 
                        nargs='+',
                        dest='signal',
                        help='Make a plot rho on x-axis for fixed signal')
    parser.add_argument('--rho', nargs='+', type=float,
                        dest='rho',
                        help='Make a plot with signal on x-axis for fixed rho')
    parser.add_argument('--outbase', help='Begginning of name of pdf / png files where results are plotted. Defaults to the base of csvfile.')

    opts = parser.parse_args()

    df = main(opts)



