import os, glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import summarize
from statistics import BH_summary, estimator_summary, interval_summary

palette = {'Randomized LASSO':'k',
           'Liu':'r',
           'Lee':'g',
           'Lee CV':'tab:brown',
           'Knockoffs':'b',
           'POSI':'y',
           'Data splitting':'tab:orange',
           'SqrtLASSO':'tab:purple'
           }

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
    df = df.loc[df['class_name'].isin(methods)]

    df['Conditional Power'] = df['Full Model Power'] / df['Selection Quality']
    df['Method'] = df['method_name']
    if 'lee_CV' in methods:
        df.loc[df['class_name'] == 'lee_CV', 'Method'] = 'Lee CV'

    # plot with rho on x axis
    g_plot = sns.FacetGrid(df, col=fixed, hue='Method', sharex=True, sharey=True, col_wrap=2, size=5, legend_out=False, palette=palette)
    
    if feature in ['Full Model Power', 'Selection Quality', 'Conditional Power']:
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0,1))
    elif feature == 'Full Model FDR':
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0,0.3), horiz=q)
    elif feature in ['Mean Length', 'Median Length', 'Risk', 'Mean Naive Length', 'Median Strong Length']:
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0, df[feature].max() + 0.1 * np.std(df[feature])))
    elif feature == 'Coverage':
        rendered_plot = g_plot.map(feature_plot, param, feature, ylim=(0.5, 1), horiz=level)
    else:
        raise ValueError("don't know how to plot '%s'" % feature)
    rendered_plot.add_legend()
    rendered_plot.savefig(outbase + '.pdf')
    rendered_plot.savefig(outbase + '.png', dpi=200)

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
    parser.add_argument('--target_q', help='FDR target', dest='target_q', default=0.2) # if using marginal screening no real
                                                                                       # q value for here
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
    
    if opts.feature in ['power', 'fdr', 'selection_quality', 'conditional_power']:
        summary = BH_summary # same as marginal_summary
    elif opts.feature == 'risk':
        summary = estimator_summary
    elif opts.feature in ['coverage', 'mean_length', 'median_length', 'naive_length', 'median_strong_length']:
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
         {'power':'Full Model Power', 
          'selection_quality':'Selection Quality',
          'fdr': 'Full Model FDR',
          'risk': 'Risk',
          'coverage': 'Coverage',
          'mean_length': 'Mean Length',
          'naive_length': 'Mean Naive Length',
          'median_strong_length': 'Median Strong Length',
          'conditional_power': 'Conditional Power',
          'median_length': 'Median Length'}[opts.feature],
         opts.outbase,
         methods=opts.methods,
         q=opts.target_q,
         level=opts.target_level)



