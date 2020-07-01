from __future__ import division

import numpy as np, pandas as pd, time
from utils import BHfilter

def interval_statistic(method,
                       instance,
                       X,
                       Y,
                       beta,
                       l_theory,
                       l_min,
                       l_1se,
                       sigma_reid,
                       M=None):

    if M is None:
        toc = time.time()
        M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    else:
        toc = np.inf
    try:
        active, lower, upper, pvalues = M.generate_intervals()
    except AttributeError:
        return M, None 

    if len(active) > 0:
        naive_lower, naive_upper = M.naive_intervals(active)[1:]
        naive_pvalues = M.naive_pvalues(active)[1]
    else:
        naive_lower, naive_upper, naive_pvalues = None, None, None
    target = M.get_target(active, beta) # for now limited to Gaussian methods
    full_target = M.full_target(active, beta)
    tic = time.time()

    if len(active) > 0:
        alpha = 1 - M.confidence
        fdp = (pvalues[full_target == 0] < alpha).sum() / pvalues.shape[0]
        value = pd.DataFrame({'active_variable':active,
                              'lower_confidence':lower,
                              'upper_confidence':upper,
                              'target':target,
                              'full_target':full_target,
                              'fdp':fdp * np.ones_like(pvalues)})
        if naive_lower is not None:
            value['naive_lower_confidence'] = naive_lower
            value['naive_upper_confidence'] = naive_upper
            value['naive_pvalue'] = naive_pvalues
        if np.isfinite(toc):
            value['Time'] = tic-toc
        value['pvalue'] = pvalues
        return M, value
    else:
        return M, None

def interval_summary(result):

    length = result['upper_confidence'] - result['lower_confidence']
    if 'naive_lower_confidence' in result.columns:
        naive_length = result['naive_upper_confidence'] - result['naive_lower_confidence']
    else:
        naive_length = np.ones_like(length) * np.nan

    def coverage_(result):
        return np.mean(np.asarray(result['lower_confidence'] <= result['target']) *
                       np.asarray(result['upper_confidence'] >= result['target']))
        
    def naive_coverage_(result):
        return np.mean(np.asarray(result['naive_lower_confidence'] <= result['target']) *
                       np.asarray(result['naive_upper_confidence'] >= result['target']))
        
    instances = result.groupby('instance_id')
    len_cover = np.array([(len(g.index), coverage_(g)) for _, g in instances])

    instances = result.groupby('instance_id')
    naive_cover = np.array([(len(g.index), naive_coverage_(g)) for _, g in instances])
    naive_coverage = np.mean(naive_cover, 0)[1]
    active_vars, mean_coverage = np.mean(len_cover, 0)
    sd_coverage = np.std(len_cover[:,1])

    # XXX we should group by instances before averaging and computing SD

    value = pd.DataFrame([[len(np.unique(result['instance_id'])),
                           mean_coverage,
                           sd_coverage,
                           np.median(length),
                           np.mean(length),
                           np.mean(naive_length),
                           np.median(naive_length),
                           naive_coverage,
                           active_vars,
                           np.mean(result['Time']),
                           result['model_target'].values[0]]],
                         columns=['Replicates',
                                  'Coverage',
                                  'SD(Coverage)',
                                  'Median Length',
                                  'Mean Length',
                                  'Mean Naive Length',
                                  'Median Naive Length',
                                  'Naive Coverage',
                                  'Active',
                                  'Time',
                                  'Model'])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def estimator_statistic(method,
                        instance,
                        X,
                        Y,
                        beta,
                        l_theory,
                        l_min,
                        l_1se,
                        sigma_reid,
                        M=None):

    if M is None:
        toc = time.time()
        M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    else:
        toc = np.inf
        
    try:
        active, point_estimate = M.point_estimator()
    except AttributeError:
        return M, None  # cannot make point estimator

    if len(active) > 0:
        naive_estimate = M.naive_estimator(active)[1]
    else:
        naive_estimate = np.zeros_like(point_estimate)

    tic = time.time()

    S = instance.feature_cov

    full_risk = np.sum((beta - point_estimate) * S.dot(beta - point_estimate)) / beta[active].shape
    naive_full_risk = np.sum((beta - naive_estimate) * S.dot(beta - naive_estimate)) / beta[active].shape

    # partial risk -- only active coordinates

    target = M.get_target(active, beta) # for now limited to Gaussian methods

    S_active = S[active][:,active]
    delta = target - point_estimate[active]
    partial_risk = np.sum(delta * S_active.dot(delta)) / delta.shape[0]
    naive_delta = target - naive_estimate[active]
    naive_partial_risk = np.sum(naive_delta * S_active.dot(naive_delta)) / delta.shape[0]

    if np.linalg.norm(target) > 0:
        partial_relative_risk = partial_risk / max(np.sum(target * S_active.dot(target)), 1)
        naive_partial_relative_risk = naive_partial_risk / max(np.sum(target * S_active.dot(target)), 1)

    # relative risk

    relative_risk = full_risk / (np.sum(beta * S.dot(beta)) * beta.shape[0])
    naive_relative_risk = naive_full_risk / np.sum(beta * S.dot(beta))

    bias = np.mean(point_estimate - beta)
    naive_bias = np.mean(naive_estimate - beta)

    value = pd.DataFrame({'Full Risk':[full_risk],
                          'Naive Full Risk':[naive_full_risk],
                          'Partial Risk':[partial_risk],
                          'Partial Relative Risk':[partial_relative_risk],
                          'Naive Partial Relative Risk':[naive_partial_relative_risk],
                          'Naive Partial Risk':[naive_partial_risk],
                          'Relative Risk':[relative_risk],
                          'Naive Relative Risk':[naive_relative_risk],
                          'Bias':[bias],
                          'Naive Bias':[naive_bias],
                          })

    if np.isfinite(toc):
        value['Time'] = tic-toc
    value['Active'] = len(active)

    return M, value

def estimator_summary(result):

    nresult = result['Full Risk'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.median(result['Full Risk']),
                           np.std(result['Full Risk']),
                           np.median(result['Naive Full Risk']),
                           np.std(result['Naive Full Risk']),
                           np.median(result['Partial Risk']),
                           np.std(result['Partial Risk']),
                           np.median(result['Naive Partial Risk']),
                           np.std(result['Naive Partial Risk']),
                           np.median(result['Relative Risk']),
                           np.std(result['Relative Risk']),
                           np.median(result['Naive Relative Risk']),
                           np.std(result['Naive Relative Risk']),
                           np.median(result['Bias']),
                           np.std(result['Bias']),
                           np.median(result['Naive Bias']),
                           np.std(result['Naive Bias']),
                           np.mean(result['Time']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates',
                                  'Median(Full Risk)',
                                  'SD(Full Risk)',
                                  'Median(Naive Full Risk)',
                                  'SD(Naive Full Risk)',
                                  'Median(Partial Risk)',
                                  'SD(Partial Risk)',
                                  'Median(Naive Partial Risk)',
                                  'SD(Naive Partial Risk)',
                                  'Median(Relative Risk)',
                                  'SD(Relative Risk)',
                                  'Median(Naive Relative Risk)',
                                  'SD(Naive Relative Risk)',
                                  'Median(Bias)',
                                  'SD(Bias)',
                                  'Median(Naive Bias)',
                                  'SD(Naive Bias)',
                                  'Time', 
                                  'Active',
                                  'Model'
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def BH_statistic(method,
                 instance,
                 X,
                 Y,
                 beta,
                 l_theory,
                 l_min,
                 l_1se,
                 sigma_reid,
                 M=None):

    if M is None:
        toc = time.time()
        M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    else:
        toc = np.inf
        
    selected, active = M.select()
    try:
        if len(active) > 0:
            naive_pvalues = M.naive_pvalues(active)[1]
            naive_selected = [active[j] for j in BHfilter(naive_pvalues, q=M.q)]
        else:
            naive_selected = None
    except AttributeError:
        naive_selected = None
    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
        selection_quality = instance.discoveries(active, true_active)
        TD = instance.discoveries(selected, true_active)
        FD = len(selected) - TD
        FDP = FD / max(TD + 1. * FD, 1.)

        # naive
        if naive_selected is not None:
            nTD = instance.discoveries(naive_selected, true_active)
            nFD = len(naive_selected) - nTD
            nFDP = nFD / max(nTD + 1. * nFD, 1.)
        else:
            nTD, nFDP, nFD = np.nan, np.nan, np.nan

        ntrue_active = max(len(true_active), 1) 
        value = pd.DataFrame([[TD / ntrue_active, 
                               FD, 
                               FDP, 
                               np.maximum(nTD / ntrue_active, 1), 
                               nFD,
                               nFDP,
                               selection_quality / ntrue_active,
                               len(active)]],
                             columns=['Full Model Power',
                                      'False Discoveries',
                                      'Full Model FDP',
                                      'Naive Full Model Power',
                                      'Naive False Discoveries',
                                      'Naive Full Model FDP',
                                      'Selection Quality',
                                      'Active'])
    else:
        value = pd.DataFrame([[0, 0, 0, 0, 0, 0, tic-toc, 0, 0]],
                             columns=['Full Model Power',
                                      'False Discoveries',
                                      'Full Model FDP',
                                      'Naive Full Model Power',
                                      'Naive False Discoveries',
                                      'Naive Full Model FDP',
                                      'Time',
                                      'Selection Quality',
                                      'Active'])
    if np.isfinite(toc):
        value['Time'] = tic-toc

    return M, value

def BH_summary(result):

    nresult = result['Full Model Power'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.mean(result['Full Model Power']), 
                           np.std(result['Full Model Power']) / np.sqrt(nresult),
                           np.mean(result['False Discoveries']), 
                           np.mean(result['Full Model FDP']), 
                           np.std(result['Full Model FDP']) / np.sqrt(nresult),
                           np.mean(result['Naive Full Model FDP']), 
                           np.mean(result['Naive Full Model Power']), 
                           np.mean(result['Naive False Discoveries']), 
                           np.mean(result['Time']),
                           np.mean(result['Selection Quality']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates', 
                                  'Full Model Power', 
                                  'SD(Full Model Power)', 
                                  'False Discoveries', 
                                  'Full Model FDR', 
                                  'SD(Full Model FDR)', 
                                  'Naive Full Model FDP',
                                  'Naive Full Model Power',
                                  'Naive False Discoveries',
                                  'Time', 
                                  'Selection Quality',
                                  'Active',
                                  'Model'
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

# marginally threshold p-values at 10% by default

marginal_summary = BH_summary # reporting statistics are the same as with BHfilter

def marginal_statistic(method, 
                       instance, 
                       X, 
                       Y, 
                       beta, 
                       l_theory, 
                       l_min, 
                       l_1se, 
                       sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    try:
        active, pvalues = M.generate_pvalues()
        selected = pvalues < method.level
    except AttributeError: # some methods do not have pvalues (e.g. knockoffs for these we will run their select method
        active, selected = M.select()

    try:
        if len(active) > 0:
            naive_pvalues = M.naive_pvalues(active)[1]
            naive_selected = naive_pvalues < method.level
        else:
            naive_selected = None
    except AttributeError:
        naive_selected = None

    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
        selection_quality = instance.discoveries(active, true_active)
        TD = instance.discoveries(selected, true_active)
        FD = len(selected) - TD
        FDP = FD / max(TD + 1. * FD, 1.)

        # naive
        if naive_selected is not None:
            nTD = instance.discoveries(naive_selected, true_active)
            nFD = len(naive_selected) - nTD
            nFDP = nFD / max(nTD + 1. * nFD, 1.)
        else:
            nTD, nFDP, nFD = np.nan, np.nan, np.nan

        ntrue_active = max(len(true_active), 1) 
        return M, pd.DataFrame([[TD / ntrue_active,
                                 FD, 
                                 FDP, 
                                 np.maximum(nTD / ntrue_active, 1), 
                                 nFD,
                                 nFDP,
                                 tic-toc, 
                                 selection_quality / ntrue_active,
                                 len(active)]],
                               columns=['Full Model Power',
                                        'False Discoveries',
                                        'Full Model FDP',
                                        'Naive Full Model Power',
                                        'Naive False Discoveries',
                                        'Naive Full Model FDP',
                                        'Time',
                                        'Selection Quality',
                                        'Active'])
    else:
        return M, pd.DataFrame([[0, 0, 0, 0, 0, 0, tic-toc, 0, 0]],
                               columns=['Full Model Power',
                                        'False Discoveries',
                                        'Full Model FDP',
                                        'Naive Full Model Power',
                                        'Naive False Discoveries',
                                        'Naive Full Model FDP',
                                        'Time',
                                        'Selection Quality',
                                        'Active'])

