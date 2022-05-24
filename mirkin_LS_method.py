#!/usr/bin/env python

'''
This script is designed for implemenation of Mirkin's method for choosing a
maximally contributing itemset from given set of elements utilizing iterative
alternative least_squares method.
'''

import numpy as np

def choose_itemset(x, max_iter, verbose=False):
    """
    :param x: the list of feature values
    :type x: list of floats
    :param max_iter: max number of iterations in the process
    :type max_iter: int
    :param verbose: verbocity indicator., wheter to print current states of
        the proccess or not
    :type verbose: bool

    :returns: tuple (S_new, pi, a, it)
        WHERE
        set S_new is the set of indices of elements in the itemset
        float pi is the parameter `pi` of the proccess
        float a is the parameter `a` of the proccess
        int is the number of iterations before convergence or number of max
            iterations
    """

    x_avg = np.mean(x)  # average value of the feature
    S_new = {np.argmax(x)}  # the first chosen item in the itemset
    for it in range(max_iter):
        p_s = len(S_new) / len(x)  # proportion of the itemset
        # x_s_avg : average value of the feature in the itemset
        x_s_avg = np.mean([x[i] for i in S_new])

        # a, pi : parameters of the procces chosen using least squares criterion
        a = (x_s_avg - x_avg) / (1 - p_s)
        pi = x_s_avg - a

        S_old = S_new  # saving the previous itemset
        # updating the itemset according to the process instruction
        S_new = {i for i in range(len(x)) if x[i] > pi + a/2}
        if S_new == S_old:  # if the itemset has not changed
            if verbose:
                print('The process converged.')
            return S_new, pi, a, it

    print('Max. num. of iterations reached')
    return S_new, pi, a, it


if __name__ == "__main__":
    # simple testing
    x = [0, 0, 0, 0.01, 0.012, 0.015, 0.02, 0.021, 0.022,
        0.2, 0.21, 0.22, 0.25, 0.3, 0.31, 0.5,
        0.8, 0.81, 0.83, .84, .89
    ]
    # np.random.shuffle(x)
    print(x)
    S, pi, a, it = choose_itemset(x, 100)
    print('pi : ', pi)
    print('a : ', a)
    print('N iterations : ', it)
    print(S)
