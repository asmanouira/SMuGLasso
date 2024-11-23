import numpy as np 
import math
def pearson(s1,s2):
    """ Computes the Pearson's correlation coeffient between a list s1 and a list s2 """
    d = len(s1)
    ki = np.sum(s1)
    kj = np.sum(s2)
    expR = np.dot(ki,kj)/d 
    pi = ki/d
    pj = kj/d
    upsiloni = np.sqrt(np.dot(pi,(1-pi)))
    upsilonj = np.sqrt(np.dot(pj,(1-pj)))
    sum_list = [a + b for a, b in zip(s1, s2)]
    r = sum_list.count(2)
    similarity = (r-expR)/(d*upsiloni*upsilonj)
    if (ki==d and kj==d) or (ki==0 and kj==0):
        similarity = 1
    elif math.isnan(similarity):
        similarity = 0
    return(similarity)

def stab_index(A,func = pearson):

    """ 
    Compute the stability of the selection index.
    Args: 
        A : list of lists of selected SNPs for M samples

        func : the similarity index used to compute the stability index (such as pearson correlation)

    Returns: 
        stability : value of stability index
        
    """
    """ Computes the average pairwise similarities between the rows of A """
    M = np.size(A,0)
    stability = 0
    for i in range(M):
        for j in range(M):
            if i != j:
                stability = stability + func(A[i], A[j])
    
    return(stability/(np.dot(M,(M-1))))