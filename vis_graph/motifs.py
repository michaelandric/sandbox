import numpy as np


def vg_motif(input_series):
    """
    This is set for n=4 node motif.
    Based on:

    Iacovacci, J. and Lacasa, L. (2016) Sequential motif profile of natural visibility graphs
    https://arxiv.org/abs/1605.02645v1
    
    https://github.com/Jaia89/VisibilityGraphMotifs/blob/master/NVG_motifs.m

    input_series:
        an input series. either array or list
    returns:

    """
    Z4 = np.zeros(6)
    s = np.array(input_series)
    T = s.shape[0]

    for i in range((T-3)):
        if (s[i+3] <= (2*s[i+2]-s[i+1])) and (s[i+2]<=(2*s[i+1]-s[i])):
            Z4[0] += 1
        if ((s[i+3] <= (1.5*s[i+2]-0.5*s[i]))
            and (s[i+2] > (2*s[i+1]-s[i]))):
            Z4[1] += 1
        if ((s[i+3] > (2*s[i+2]-s[i+1]))
            and (s[i+3] <= (3*s[i+1]-2*s[i]))
            and (s[i+2] <= (2*s[i+1]-s[i]))):
            Z4[2] += 1
        if ((s[i+3] > (1.5*s[i+2] - 0.5*s[i]))
            and (s[i+3] <= (2*s[i+2] - s[i+1]))
            and (s[i+2] > (2*s[i+1] - s[i]))):
            Z4[3] += 1
        if ((s[i+3] > (3*s[i+1] - 2*s[i]))
            and (s[(i+2)] <= (2*s[i+1]-s[i]))):
            Z4[4] += 1
        if ((s[i+3] > (2*s[i+2] - s[i+1])) and (s[i+2] > (2*s[i+1] - s[i]))):
            Z4[5] += 1
    
    return Z4/(T-3)


def hvg_motif(input_series):
    """
    This is set for n=4 node motif.
    Based on:

    Iacovacci, J. and Lacasa, L. (2016) Sequential visibility graph motifs
    http://www.maths.qmul.ac.uk/~lacasa/P44.pdf    
    
    https://github.com/Jaia89/VisibilityGraphMotifs/blob/master/NVG_motifs.m

    input_series:
        an input series. either array or list
    returns:


    """
    Z4 = np.zeros(6)
    s = np.array(input_series)
    T = s.shape[0]

    for i in range((T-3)):
        if (((s[i+2] <= s[i+1]) and (s[i+3] <= s[i+2])) or ((s[i+1] > s[i]) and (s[i+2] > s[i+1]))):
            Z4[0] += 1
        if ((s[i+1] <= s[i])
            and (s[i+2] == s[i+1]) and (s[i+3] > s[i+2])):
            Z4[1] += 1
        if (((s[i+1] <= s[i])
            and (s[i+1] < s[i+2]) and (s[i+2] <=s[i]) and (s[i+3] <= s[i+2]))
            or ((s[i+1] <= s[i]) and (s[i+2] > s[i]))):
            Z4[2] += 1
        if(((s[i+1] <= s[i]) and (s[i+2] <= s[i+1]) 
            and (s[i+2] <= s[i+3]) and (s[i+3] <= s[i+1])) 
            or ((s[i+1]>s[i]) and (s[i+2] <= s[i+1]) and (s[i+3] > s[i+2]))):
            Z4[3] += 1
        if((s[i+1] <= s[i]) and (s[i+1] <= s[i+2]) and (s[i+2] <= s[i]) and (s[i+3] > s[i+2])):
            Z4[4] += 1
        if((s[i+1] <= s[i]) and (s[i+2] <= s[i+1]) and (s[i+3] > s[i+1])):
            Z4[5] += 1
    
    return Z4/(T-3)