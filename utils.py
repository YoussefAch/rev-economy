import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def generatePossibleSequences(nbIntervals, order):

    """
       This function return the possible sequences of intervals given order
       and number of intervals

    """
    seqBase = np.arange(nbIntervals)
    final = []
    for i in range(order):
        if i==0:
            # repeat
            final.append(np.array([seqBase for _ in range(nbIntervals**(order-1))]).flatten())
        else:
            #duplicate
            seq = np.array([(nbIntervals**i)*[e] for e in seqBase]).flatten()
            #repeat
            final.append(np.array([seq for _ in range(nbIntervals**(order-i-1))]).flatten())
    return np.array(final)



def numpy_to_df(x):
    return pd.DataFrame({i+1:e for i,e in enumerate(x)}, index=range(1))


def donnes_idv(idx, test_probas, test_preds):
    new_test_probas = {}
    new_test_preds = {}
    
    for k,v in test_probas.items():
        new_test_probas[k] = test_probas[k][idx]
        new_test_preds[k] = test_preds[k][idx]
    
    
    return new_test_probas, new_test_preds






