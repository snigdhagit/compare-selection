import numpy as np, pandas as pd, time

def interval_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)

    active, lower, upper = M.generate_intervals()
    target = M.get_target(active, beta) # for now limited to Gaussian methods
    tic = time.time()

    if len(active) > 0:
        value = pd.DataFrame({'active_variable':active,
                             'lower_confidence':lower,
                             'upper_confidence':upper,
                             'target':target})
        value['Time'] = tic-toc
        return M, value
    else:
        return M, None

def interval_summary(result):

    length = result['upper_confidence'] - result['lower_confidence']

    def coverage_(result):
        return np.mean(np.asarray(result['lower_confidence'] <= result['target']) *
                       np.asarray(result['upper_confidence'] >= result['target']))
        
    instances = result.groupby('instance_id')
    len_cover = np.array([(len(g.index), coverage_(g)) for _, g in instances])
    active_vars, mean_coverage = np.mean(len_cover, 0)
    sd_coverage = np.std(len_cover[:,1])

    # XXX we should group by instances before averaging and computing SD

    value = pd.DataFrame([[len(np.unique(result['instance_id'])),
                           mean_coverage,
                           sd_coverage,
                           np.median(length),
                           np.mean(length),
                           active_vars,
                           result['model_target'].values[0]]],
                         columns=['Replicates',
                                  'Coverage',
                                  'SD(Coverage)',
                                  'Median length',
                                  'Mean length',
                                  'Active',
                                  'Model'])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value

def estimator_statistic(method, instance, X, Y, beta, l_theory, l_min, l_1se, sigma_reid):

    toc = time.time()
    M = method(X.copy(), Y.copy(), l_theory.copy(), l_min, l_1se, sigma_reid)
    active, point_estimate = M.point_estimator()
    tic = time.time()

    risk = np.linalg.norm(beta - point_estimate)**2
    value = pd.DataFrame({'Risk':[risk]})
    value['Time'] = tic-toc
    value['Active'] = len(active)
    return M, value


def estimator_summary(result):

    nresult = result['Risk'].shape[0]
    value = pd.DataFrame([[nresult,
                           np.median(result['Risk']),
                           np.std(result['Risk']),
                           np.mean(result['Time']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates',
                                  'Risk',
                                  'SD(Risk)',
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
    tic = time.time()
    true_active = np.nonzero(beta)[0]

    if active is not None:
        TD = instance.discoveries(selected, true_active)
        FD = len(selected) - TD
        FDP = FD / max(TD + 1. * FD, 1.)
        return M, pd.DataFrame([[TD / (len(true_active)*1.), FD, FDP, tic-toc, len(active)]],
                               columns=['Full model power',
                                        'False discoveries',
                                        'Full model FDP',
                                        'Time',
                                        'Active'])
    else:
        return M, pd.DataFrame([[0, 0, 0, tic-toc, 0]],
                               columns=['Full model power',
                                        'False discoveries',
                                        'Full model FDP',
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
                           np.mean(result['Time']),
                           np.mean(result['Active']),
                           result['model_target'].values[0]]],
                         columns=['Replicates', 
                                  'Full model power', 
                                  'SD(Full model power)', 
                                  'False discoveries', 
                                  'Full model FDR', 
                                  'SD(Full model FDR)', 
                                  'Time', 
                                  'Active',
                                  'Model'
                                  ])

    # keep all things constant over groups

    for n in result.columns:
        if len(np.unique(result[n])) == 1:
            value[n] = result[n].values[0]

    return value
