def turnpoints(x):
    """
    ported the R version of this to python.
    https://cran.r-project.org/web/packages/pastecs/index.html
    
    x: 
        number list or 1-D numpy array
    return:
        dictionary with turnpoints parameters
        
    e.g. usage:
        tp_num_array = turnpoints(num_array)
        num_turnpoints = tp_num_array['nturns']
        peak_values = np.array(num_array)[tp_num_array['pos'][tp_num_array['peaks']]]
        pit_values = np.array(num_array)[tp_num_array['pos'][tp_num_array['pits']]]
    """
    import numpy as np
    from math import gamma
    
    assert len(set(x)) > 2, 'need at least 3 unique values'
    n = len(x)
    diffs = np.not_equal(np.hstack((x[0]-1, x[0:len(x)-1])), x)
    uniques = np.array(x)[diffs]
    n2 = len(uniques)
    poss = [i for i, a in enumerate(diffs) if a == True]
    exaequos = np.array((poss[1:n2]+[n])) - poss - 1
    
    m = n2 - 2
    reper = np.repeat([3, 2, 1], repeats=m)
    for i in range(int(len(reper)/m)):
        reper[i*m:i*m+m] += range(m)
    reper = reper - 1
    ex = np.array([np.array(x)[diffs][i] for i in reper]).reshape(int(len(reper)/m), m).T
    peaks = np.hstack((False, ex.max(1) == ex[:, 1], False))
    pits = np.hstack((False, ex.min(1) == ex[:, 1], False))
    tpts = peaks | pits
    if tpts.sum() == 0:
        nturns = 0
        firstispeak = False
        peaks = np.repeat(False, n2)
        pits = np.repeat(False, n2)
        tppos = np.nan
        proba = np.nan
        info = np.nan
    else:
        tppos = (poss + exaequos)[tpts]
        tptspos = np.arange(n2)[tpts]
        firstispeak = tptspos[0] == np.arange(n2)[peaks][0]
        nturns = len(tptspos)
        if (nturns < 2):
            inter = n2 + 1
            posinter1 = tptspos[0]
        else:
            inter = np.hstack((tptspos[1:nturns], n2-1)) - np.hstack((0, tptspos[:nturns-1])) + 1
            posinter1 = tptspos - np.hstack((0, tptspos[:(nturns - 1)]))
        posinter2 = inter - posinter1
        posinter = np.array((posinter1, posinter2)).max(0)
        if type(posinter) is np.ndarray:
            proba = 2/((inter * [gamma(i) for i in posinter]) * [gamma(i) for i in np.array((inter - posinter + 1))])
        if type(posinter) is np.int64:
            proba = 2/(inter * gamma(posinter) * gamma(inter - posinter + 1))
        info = -np.log2(proba)
    
    res_dict = {'data': x, 'points': uniques, 'pos': (poss+exaequos),
                'exaequos': exaequos, 'nturns': nturns, 'firstispeak': firstispeak,
                'peaks': peaks, 'pits': pits, 'tppos': tppos, 'proba': proba, 'info': info}

    return res_dict
