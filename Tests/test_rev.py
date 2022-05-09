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
from TestUtils import computeBestGroup_REV, computeScores_REV_MC, saveRealModelsECONOMY_REV_MC, computeScores_REV, saveRealClassifiers, saveRealModelsECONOMY_REV, computeScores, computeBestGroup, evaluate, computePredictionsEconomy, computeMetricsNeeded
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
    folderRealData = configParams['folderRealData']
    sampling_ratio = configParams['sampling_ratio']
    # et summary and sort by train set size

    classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    timeParams = configParams['timeParams']
    nbGroups = np.arange(1,configParams['nbGroups'])
    methods =  configParams['methods']
    C_m = configParams['misClassificationCost']
    misClassificationCost = np.array([[0,C_m],
                                      [C_m,0]])
    C_cds = configParams['changeDecisionCost']
    min_t = configParams['min_t']
    pathToClassifiers = configParams['pathToClassifiers']
    allECOmodelsAvailable = configParams['allECOmodelsAvailable']
    nb_core = multiprocessing.cpu_count()
    orderGamma = configParams['orderGamma']
    ratioVal = configParams['ratioVal']
    pathToIntermediateResults = configParams['pathToIntermediateResults']
    pathToResults = configParams['pathToResults']
    saveClassifiers = configParams['saveClassifiers']
    pathToRealModelsECONOMY = configParams['pathToRealModelsECONOMY']
    pathToSaveScores = configParams['pathToSaveScores']
    pathToSavePredictions = configParams['pathToSavePredictions']
    normalizeTime = configParams['normalizeTime']
    use_complete_ECO_model = configParams['use_complete_ECO_model']
    pathECOmodel = configParams['pathECOmodel']
    fears = configParams['fears']
    score_chosen = configParams['score_chosen']
    datasets = configParams['Datasets']
    feat = configParams['feat']
    variantes = configParams['variantes']
    history = configParams['history']
    mc = configParams['mc']
    INF = 10000000





    ########################################################################################
    print('################################ SAVE ECONOMY ##################################')
    ########################################################################################
    if not allECOmodelsAvailable:
        if mc:
            func_args_eco = [(use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, C_m, C_cd, min_t, classifier, fears, feat) for dataset in datasets for group in nbGroups for method in methods for C_cd in C_cds]
            Parallel(n_jobs=nb_core)(delayed(saveRealModelsECONOMY_REV_MC)(func_arg) for func_arg in func_args_eco)
        else:
            func_args_eco = [(use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, np.array([[0,C_cd],[C_cd,0]]), min_t, classifier, fears, feat) for dataset in datasets for group in nbGroups for method in methods for C_cd in C_cds]
            Parallel(n_jobs=nb_core)(delayed(saveRealModelsECONOMY_REV)(func_arg) for func_arg in func_args_eco)
    



    #########################################################################################
    print('############################### Compute scores #################################')
    #########################################################################################
    preds = {}
    for dataset in datasets:
        with open(op.join(folderRealData, dataset,'val_probas.pkl') ,'rb') as inp:
            val_probas = pickle.load(inp)

        with open(op.join(folderRealData, dataset, 'val_preds.pkl') ,'rb') as inp:
            val_preds = pickle.load(inp)

        if mc:
            with open(op.join(folderRealData,'uc',dataset+'val_uc.pkl') ,'rb') as inp:
                val_uc = pickle.load(inp)

            preds[dataset] = [val_probas, val_preds, val_uc]
        else:
            preds[dataset] = [val_probas, val_preds]

    if not mc:
        func_args_score = [(True, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, np.array([[0,C_cd],[C_cd,0]]), min_t, paramTime, preds[dataset], var, hist) for dataset in datasets for group in nbGroups for paramTime in timeParams for method in methods for C_cd in C_cds for var, hist in zip(variantes, history)]
        modelName_score = Parallel(n_jobs=nb_core)(delayed(computeScores_REV)(func_arg) for func_arg in func_args_score)
    else:
        func_args_score = [(True, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, C_cd, min_t, paramTime, preds[dataset], var, hist) for dataset in datasets for group in nbGroups for paramTime in timeParams for method in methods for C_cd in C_cds for var, hist in zip(variantes, history)]
        modelName_score = Parallel(n_jobs=nb_core)(delayed(computeScores_REV_MC)(func_arg) for func_arg in func_args_score)
    with open(op.join(pathToIntermediateResults, 'modelName_score.pkl'), 'wb') as outfile:
        pickle.dump(modelName_score, outfile)
    



    #########################################################################################
    print('##################### Compute best hyperparam : nbGroups #######################')
    #########################################################################################
    
    bestGroup = computeBestGroup_REV(datasets, timeParams, modelName_score, INF, methods, C_cds, variantes, history)
    with open(op.join(pathToIntermediateResults, 'bestGroup.pkl'), 'wb') as outfile:
        pickle.dump(bestGroup, outfile)
    



    #########################################################################################
    print('############################ Compute best scores ###############################')
    #########################################################################################
    # use the best group for evaluating on data test
    # compute scores
    func_args_best_score = []
    preds={}
    for dataset in datasets:
        
        with open(op.join(folderRealData, dataset,'test_probas.pkl') ,'rb') as inp:
            test_probas = pickle.load(inp)

        with open(op.join(folderRealData, dataset,'test_preds.pkl') ,'rb') as inp:
            test_preds = pickle.load(inp)
        if not mc:
            preds[dataset] = [test_probas, test_preds]
        else:
            with open(op.join(folderRealData,'uc',dataset+'test_uc.pkl') ,'rb') as inp:
                test_uc = pickle.load(inp)
            preds[dataset] = [test_probas, test_preds, test_uc]

        for paramTime in timeParams:
            for method in methods:
                for C_cd in C_cds:
                    for variante, hist in zip(variantes, history):
                        if not mc:
                            func_args_best_score.append((False, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, bestGroup[method + ',' + dataset + ',' + str(C_cd) + ',' + str(paramTime)+','+variante+','+ str(hist)][0], misClassificationCost, np.array([[0,C_cd],[C_cd,0]]), min_t, paramTime, preds[dataset], variante, hist))
                        else:
                            func_args_best_score.append((False, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, bestGroup[method + ',' + dataset + ',' + str(C_cd) + ',' + str(paramTime)+','+variante+','+ str(hist)][0], misClassificationCost,C_cd, min_t, paramTime, preds[dataset], variante, hist))
    if not mc:
        best_score = Parallel(n_jobs=nb_core)(delayed(computeScores_REV)(func_arg) for func_arg in func_args_best_score)
    else:
        best_score = Parallel(n_jobs=nb_core)(delayed(computeScores_REV_MC)(func_arg) for func_arg in func_args_best_score)
    with open(op.join(pathToIntermediateResults, 'best_score.pkl'), 'wb') as outfile:
        pickle.dump(best_score, outfile)
    



    results = {}
    for e in best_score:
        (modelName, score_model) = e
        results[modelName] = score_model

    with open(op.join(pathToResults, 'results.pkl'), 'wb') as outfile:
        pickle.dump(results, outfile)