
#########################################################################################
#########################################################################################
import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import argparse
import json
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import os.path as op
from TestUtils import saveRealClassifiers, saveRealModelsECONOMY, computeScores, computeBestGroup, evaluate, computePredictionsEconomy
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle

if __name__ == '__main__':


    #########################################################################################
    print('############################### PARAMS CONFIG ##################################')
    #########################################################################################
    # command line
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--pathToInputParams', help='path to json file with input params', required=True)
    parser.add_argument('--pathToOutputs', help='path outputs', required=True)

    args = parser.parse_args()
    pathToParams = args.pathToInputParams


    # load input params
    with open(pathToParams) as configfile:
        configParams = json.load(configfile)
    #configParams = json.load(open(pathToParams))

    # set variables
    folderRealData = op.join(os.getcwd(), configParams['folderRealData'])
    sampling_ratio = configParams['sampling_ratio']
    # et summary and sort by train set size

    classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    timeParams = configParams['timeParams']
    nbGroups = np.arange(1,configParams['nbGroups'])
    methods =  configParams['methods']
    C_m = configParams['misClassificationCost']
    misClassificationCost = np.array([[0,C_m],
                                      [C_m,0]])
    min_t = configParams['min_t']
    pathToClassifiers = op.join(os.getcwd(), configParams['pathToClassifiers'])
    allECOmodelsAvailable = configParams['allECOmodelsAvailable']
    nb_core = multiprocessing.cpu_count()
    orderGamma = configParams['orderGamma']
    ratioVal = configParams['ratioVal']
    pathToIntermediateResults = op.join(os.getcwd(), configParams['pathToIntermediateResults'])
    pathToResults = op.join(os.getcwd(), configParams['pathToResults'])
    saveClassifiers = configParams['saveClassifiers']
    pathToRealModelsECONOMY = op.join(os.getcwd(), configParams['pathToRealModelsECONOMY'])
    pathToSaveScores = op.join(os.getcwd(), configParams['pathToSaveScores'])
    pathToSavePredictions = op.join(os.getcwd(), configParams['pathToSavePredictions'])
    normalizeTime = configParams['normalizeTime']
    use_complete_ECO_model = configParams['use_complete_ECO_model']
    pathECOmodel = op.join(os.getcwd(), configParams['pathECOmodel'])
    fears = configParams['fears']
    score_chosen = configParams['score_chosen']
    feat = configParams['feat']
    datasets = configParams['Datasets']
    INF = 10000000





    ########################################################################################
    print('################################ SAVE ECONOMY ##################################')
    ########################################################################################
    if not allECOmodelsAvailable:
        func_args_eco = [(use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, min_t, classifier, fears, feat) for dataset in datasets for group in nbGroups for method in methods]
        Parallel(n_jobs=nb_core)(delayed(saveRealModelsECONOMY)(func_arg) for func_arg in func_args_eco)
    



    #########################################################################################
    print('############################### Compute scores #################################')
    #########################################################################################

    func_args_score = [(score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, min_t, paramTime) for dataset in datasets for group in nbGroups for paramTime in timeParams for method in methods]
    modelName_score = Parallel(n_jobs=nb_core)(delayed(computeScores)(func_arg) for func_arg in func_args_score)
    with open(op.join(pathToIntermediateResults, 'modelName_score.pkl'), 'wb') as outfile:
        pickle.dump(modelName_score, outfile)
    



    #########################################################################################
    print('##################### Compute best hyperparam : nbGroups #######################')
    #########################################################################################
    bestGroup = computeBestGroup(datasets, timeParams, modelName_score, INF, methods)
    with open(op.join(pathToIntermediateResults, 'bestGroup.pkl'), 'wb') as outfile:
        pickle.dump(bestGroup, outfile)
    



    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    func_args_best_score = []
    for dataset in datasets:
        for paramTime in timeParams:
            for method in methods:
                func_args_best_score.append((score_chosen, normalizeTime, C_m, pathToRealModelsECONOMY, folderRealData, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime, pathToSaveScores))
    best_score = Parallel(n_jobs=nb_core)(delayed(evaluate)(func_arg) for func_arg in func_args_best_score)
    with open(op.join(pathToIntermediateResults, 'best_score.pkl'), 'wb') as outfile:
        pickle.dump(best_score, outfile)
    





    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    results = {}
    for e in best_score:
        (modelName, score_model) = e
        results[modelName] = score_model

    with open(op.join(pathToResults, 'results.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)


    #########################################################################################
    print('############################ Compute predictions ###############################')
    #########################################################################################
    func_args_preds = []
    for dataset in datasets:
        for method in methods:
            for paramTime in timeParams:
                func_args_preds.append((normalizeTime, pathToSavePredictions, pathToIntermediateResults, folderRealData, pathToRealModelsECONOMY, dataset, method, bestGroup[method + ',' + dataset + ',' + str(paramTime)][0], paramTime))
    predictions = Parallel(n_jobs=nb_core)(delayed(computePredictionsEconomy)(func_arg) for func_arg in func_args_preds)
