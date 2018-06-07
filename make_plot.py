import os, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import summarize
from statistics import FDR_summary, estimator_summary, interval_summary

def feature_plot(param, power, color='r', label='foo', ylim=None, horiz=None):
    ax = plt.gca()
    ax.plot(param, power, 'o--', color=color, label=label)
    ax.set_xticks(sorted(np.unique(param)))
    if ylim is not None:
        old_ylim = ax.get_ylim()
        if ylim[1] is None:
            ylim = (ylim[0], old_ylim[1])
        if ylim[0] is None:
            ylim = (old_ylim[0], ylim[1])
        ax.set_ylim(ylim)
    if horiz is not None:
        ax.plot(ax.get_xticks(), horiz * np.ones(len(ax.get_xticks())), 'k--')

def plot(df,
         fixed,
         param,
         feature,
         outbase,
         methods=None,
         q=0.2,
         level=0.95):

    # results, rho_results, signal_results = extract_results(df)

    methods = methods or np.unique(df['class_name'])

    df['Method'] = df['method_name']
    # plot with rho on x axis
    g_plot = sns.FacetGrid(df, col=fixed, hue='Method', sharex=True, sharey=True, col_wrap=2, size=5, legend_out=False)
    
    if feature == 'Full model power':
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0,1))
    elif feature == 'Full model FDR':
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0,0.3), horiz=q)
    elif feature in ['Mean length', 'Median length', 'Risk']:
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0, df[feature].max() + 0.1 * np.std(df[feature])))
    elif feature == 'Coverage':
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0.5, 1), horiz=level)
    else:
        raise ValueError("don't know how to plot '%s'" % param)
    rendered_plot.add_legend()
    rendered_plot.savefig(outbase + '.pdf')
    rendered_plot.savefig(outbase + '.png')

    return df

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser(
        description='''
Make plots for

Try:
    python make_plot.py --methods lee_theory liu_theory --csvfile indep.csv
''')
    parser.add_argument('--methods', nargs='+',
                        dest='methods', 
                        help='Names of methods in plot (i.e. class name). Defaults to all methods.')
    parser.add_argument('--param', 
                        dest='param',
                        default='rho',
                        help='Make a plot with param on x-axis for varying fixed')
    parser.add_argument('--fixed', 
                        dest='fixed',
                        default='signal',
                        help='Which value if fixed for each facet')
    parser.add_argument('--feature', 
                        dest='feature',
                        default='power',
                        help='Variable for y-axis')
    parser.add_argument('--target_q', help='FDR target', dest='target_q', default=0.2)
    parser.add_argument('--target_level', help='Confidence level target', dest='target_level', default=0.95)
    parser.add_argument('--csvfile', help='csvfile.', dest='csvfile')
    parser.add_argument('--csvbase', help='csvfile.', dest='csvbase')
    parser.add_argument('--outbase', help='Base of name of pdf file where results are plotted.')

    opts = parser.parse_args()

    if opts.csvbase is not None:
        full_df = pd.concat([pd.read_csv(f) for f in glob.glob(opts.csvbase + '*signal*csv')])
        full_df.to_csv(opts.csvbase + '.csv')
        csvfile = opts.csvbase + '.csv'
    else:
        csvfile = opts.csvfile

    if opts.param == opts.fixed:
        raise ValueError('one should be rho, the other signal')

    df = pd.read_csv(csvfile)
    
    if opts.feature in ['power', 'fdr']:
        summary = FDR_summary
    elif opts.feature == 'risk':
        summary = estimator_summary
    elif opts.feature in ['coverage', 'mean_length', 'median_length']:
        summary = interval_summary
    else:
        raise ValueError("don't know how to summarize '%s'" % opts.feature)

    summary_df = summarize(['method_param',
                            opts.param,
                            opts.fixed],
                           df,
                           summary)

    plot(summary_df,
         opts.fixed,
         opts.param,
         {'power':'Full model power', 
          'fdr': 'Full model FDR',
          'risk': 'Risk',
          'coverage': 'Coverage',
          'mean_length': 'Mean length',
          'median_length': 'Median length'}[opts.feature],
         opts.outbase,
         methods=opts.methods,
         q=opts.target_q,
         level=opts.target_level)



