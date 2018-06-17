import numpy as np, pandas as pd, time
from utils import BHfilter

def interval_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    try:
        active, lower, upper = M.generate_intervals()
    except AttributeError:
        return M, None 

    if len(active) > 0:
        naive_lower, naive_upper = M.naive_intervals(active)[1:]
    else:
        naive_lower, naive_upper = None, None
    target = M.get_target(active, beta) # for now limited to Gaussian methods
    tic = time.time()

    if len(active) > 0:
        value = pd.DataFrame({'active_variable':active,
                              'lower_confidence':lower,
                              'upper_confidence':upper,
                              'target':target})
        if naive_lower is not None:
            value['naive_lower_confidence'] = naive_lower
            value['naive_upper_confidence'] = naive_upper
        value['Time'] = tic-toc
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
                                  'Median length',
                                  'Mean length',
                                  'Mean naive length',
                                  'Median naive length',
                                  'Naive coverage',
                                  'Active',
                                  'Time',
                                  'Model'])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def estimator_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)

    try:
        active, point_estimate = M.point_estimator()
    except AttributeError:
        return M, None  # cannot make point estimator

    if len(active) > 0:
        beta_naive = M.naive_estimator(active)[1]
    else:
        beta_naive = np.ones_like(point_estimate) * np.nan
    tic = time.time()

    risk = np.linalg.norm(beta - point_estimate)**2
    naive_risk = np.linalg.norm(beta_naive - point_estimate)**2
    value = pd.DataFrame({'Risk':[risk], 
                          'Naive risk':[naive_risk]})
    value['Time'] = tic-toc
    value['Active'] = len(active)
    return M, value

def estimator_summary(result):

    nresult = result['Risk'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.median(result['Risk']),
                           np.std(result['Risk']),
                           np.median(result['Naive risk']),
                           np.std(result['Naive risk']),
                           np.mean(result['Time']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates',
                                  'Risk',
                                  'SD(Risk)',
                                  'Naive risk',
                                  'SD(Naive risk)',
                                  'Time', 
                                  'Active',
                                  'Model'
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def FDR_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):
    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    selected, active = M.select()
    try:
        if len(active) > 0:
            naive_pvalues = M.naive_pvalues(active)[1]
            naive_selected = BHfilter(naive_pvalues, q=M.q)
        else:
            naive_selected = None
    except AttributeError:
        naive_selected = None
    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
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
        return M, pd.DataFrame([[TD / (len(true_active)*1.), 
                                 FD, 
                                 FDP, 
                                 np.maximum(nTD / (len(true_active)*1.), 1), 
                                 nFD,
                                 nFDP,
                                 tic-toc, 
                                 len(active)]],
                               columns=['Full model power',
                                        'False discoveries',
                                        'Full model FDP',
                                        'Naive full model power',
                                        'Naive false discoveries',
                                        'Naive full model FDP',
                                        'Time',
                                        'Active'])
    else:
        return M, pd.DataFrame([[0, 0, 0, 0, 0, 0, tic-toc, 0]],
                               columns=['Full model power',
                                        'False discoveries',
                                        'Full model FDP',
                                        'Naive full model power',
                                        'Naive false discoveries',
                                        'Naive full model FDP',
                                        'Time',
                                        'Active'])

def FDR_summary(result):

    nresult = result['Full model power'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.mean(result['Full model power']), 
                           np.std(result['Full model power']) / np.sqrt(nresult),
                           np.mean(result['False discoveries']), 
                           np.mean(result['Full model FDP']), 
                           np.std(result['Full model FDP']) / np.sqrt(nresult),
                           np.mean(result['Naive full model FDP']), 
                           np.mean(result['Naive full model power']), 
                           np.mean(result['Naive false discoveries']), 
                           np.mean(result['Time']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates', 
                                  'Full model power', 
                                  'SD(Full model power)', 
                                  'False discoveries', 
                                  'Full model FDR', 
                                  'SD(Full model FDR)', 
                                  'Naive full model FDP',
                                  'Naive full model power',
                                  'Naive false discoveries',
                                  'Time', 
                                  'Active',
                                  'Model'
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value
