import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import pandas as pd
import math
import numpy as np
import glob
import os.path as op
import scipy as sp
import scipy.stats as st
from Economy_Gamma_MC_REV import Economy_Gamma_MC_REV
from Economy_K_REV import Economy_K_REV
from Economy_Gamma import Economy_Gamma
from Economy_Gamma_REV import Economy_Gamma_REV
try:
    import cPickle as pickle
except ImportError:
    import pickle
import json
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
import time


def run_predictions(arguments):

    dataset, timeparam, opt_group, folderRealData, pathModelsEconomy, pathResults = arguments
    
    with open(op.join(folderRealData, dataset, 'test_probas.pkl') ,'rb') as inp:
        test_probas = pickle.load(inp)

    with open(op.join(folderRealData, dataset, 'test_preds.pkl') ,'rb') as inp:
        test_preds = pickle.load(inp)

    filepathTest = op.join(folderRealData, dataset, dataset +'_TEST_SCORE.tsv')
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    y_test = test.iloc[:, 0]
    X_test = test.loc[:, test.columns != test.columns[0]]
    mx_t = X_test.shape[1]


    with open(op.join(pathModelsEconomy, 'Gamma,'+dataset+','+str(opt_group)+'.pkl'), 'rb') as inp:
        model = pickle.load(inp)
        
    timeCostt = (1/mx_t) * timeparam * np.arange(mx_t+1)
    setattr(model, 'timeCost', timeCostt)

    eco_rev_cu_preds = model.predict_revocable(X_test, oneIDV=False, donnes=[test_probas, test_preds], variante='C')

    
    with open(op.join(pathResults, str(timeparam)+ dataset +'eco_rev_cu_preds'+'.pkl'), 'wb') as outp:
        pickle.dump(eco_rev_cu_preds, outp)    

    return eco_rev_cu_preds, dataset, timeparam


def run_score(arguments):
    dataset, timeparam, C_cd, pathPreds, folderRealData = arguments

    with open(op.join(pathPreds, str(timeparam)+ dataset +'eco_rev_cu_preds'+'.pkl'), 'rb') as outp:
        eco_rev_cu_preds = pickle.load(outp)

    filepathTest = op.join(folderRealData, dataset, dataset +'_TEST_SCORE.tsv')
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    y_test = test.iloc[:, 0]
    X_test = test.loc[:, test.columns != test.columns[0]]
    mx_t = X_test.shape[1]
    C_m=1
    timeCostt = (1/mx_t) * timeparam * np.arange(mx_t+1)
    sc_c = score_rev(timeCostt, eco_rev_cu_preds, y_test, C_m, C_cd)
    return sc_c, dataset, timeparam, C_cd


def score_rev(timeCost, decisions, y_true, C_m, C_cd):
    score_computed = 0

    for i,decision in enumerate(decisions):
        score_computed += timeCost[decision[-1][0]]
        if y_true[i] != decision[-1][1]:
            score_computed += C_m
        for _ in range(len(decision)-1):
            score_computed += C_cd
    return score_computed/len(decisions)

def wilcoxon_test(score_A, score_B):

    # compute abs delta and sign
    delta_score = [score_B[i] - score_A[i] for i in range(len(score_A))]
    sign_delta_score = list(np.sign(delta_score))
    abs_delta_score = list(map(abs, delta_score))

    N_r = float(len(delta_score))

    # hadling scores
    score_df = pd.DataFrame({'abs_delta_score':abs_delta_score, 'sign_delta_score':sign_delta_score })

    # sort
    score_df.sort_values(by='abs_delta_score', inplace=True)
    score_df.index = range(1,len(score_df)+1)

    # adding ranks
    score_df['Ranks'] = score_df.index
    score_df['Ranks'] = score_df['Ranks'].astype('float64')

    score_df.dropna(inplace=True)

    # z : pouput value
    W = sum(score_df['sign_delta_score'] * score_df['Ranks'])
    z = W/(math.sqrt(N_r*(N_r+1)*(2*N_r+1)/6.0))

    # rejecte or not the null hypothesis
    null_hypothesis_rejected = False
    if z < -1.96 or z > 1.96:
        null_hypothesis_rejected = True

    return z, null_hypothesis_rejected

def showTests(method1, method2, timeparams, C_cds, title, figp, path):
    """
        method2: baseline
    """
    #fig, ax = plt.subplots()
    plt.figure(figsize=(6,3))
    plt.title(title)
    ticklabelpad = mpl.rcParams['xtick.major.pad']
    for i, C_cd in enumerate(C_cds):
        xFalse=[]
        xTruePerdu=[]
        xTrueGagne=[]
        yFalse=[]
        yTruePerdu=[]
        yTrueGagne=[]
        for j, timeparam in enumerate(timeparams):
            z, null_hypothesis_rejected = wilcoxon_test(method1[str(timeparam)+str(C_cd)], method2[str(timeparam)+str(C_cd)])
            if null_hypothesis_rejected:
                if z>0:
                    xTrueGagne.append(j)
                    yTrueGagne.append((i+1)/2)
                else:
                    xTruePerdu.append(j)
                    yTruePerdu.append((i+1)/2)
            else:
                xFalse.append(j)
                yFalse.append((i+1)/2)
                
        plt.scatter(xFalse, yFalse, marker='o', color='black', facecolor='white')
        plt.scatter(xTruePerdu, yTruePerdu, marker='_', color='black')
        plt.scatter(xTrueGagne, yTrueGagne, marker='+', color='black') 

    plt.annotate(r'$\alpha$', xy=(1,-0.1), xytext=(0.5, -1.5), ha='left', va='top', xycoords='axes fraction', textcoords='offset points', fontsize=18)
    plt.ylabel(r'$\beta$', fontsize=18)
    y_axis = np.linspace(1/2, len(C_cds)/2, len(C_cds))
    plt.yticks(y_axis,C_cds)
    plt.xticks([i for i in range(len(timeparams))], timeparams, rotation=90)
    plt.savefig(op.join(path,figp), bbox_inches='tight')


def Average(lst): 
    return sum(lst) / len(lst) 

def kappa_earliness(arguments):
    
    C_cd, timeparam, dataset, folderRealData, pathPredsB2B3, pathPreds, pathResults = arguments

    
    filepathTest = op.join(folderRealData, dataset, dataset+'_TEST_SCORE.tsv')
    val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

    mn = np.min(np.unique(val.iloc[:,0].values))


    # get X and y
    y_true = val.iloc[:, 0]
    X_val = val.loc[:, val.columns != val.columns[0]]
    mx_t = X_val.shape[1]
	

    # eco_rev_cu
    with open(op.join(pathResults, str(timeparam)+ dataset +'eco_rev_cu_preds'+'.pkl'), 'rb') as outp:
        vareco_rev_cu = pickle.load(outp)
    y_preds_eco_rev_cu = list(map(lambda x: x[-1][1],vareco_rev_cu))
    tau_preds_eco_rev_cu = list(map(lambda x: x[-1][0]/mx_t,vareco_rev_cu))





    with open(glob.glob(op.join(pathPredsB2B3,'EVALdecisionsGamma_rev,'+dataset+',*,'+str(C_cd)+','+str(timeparam)+',avec_cout_moindre,True.pkl'))[0] ,'rb') as inp:
        decisionseco_rev_ca = pickle.load(inp)
    y_preds_eco_rev_ca = list(map(lambda x: x[-1][1],decisionseco_rev_ca))
    tau_preds_eco_rev_ca = list(map(lambda x: x[-1][0]/mx_t,decisionseco_rev_ca))

    #Gamma 
    with open(glob.glob(op.join(pathPreds,'PREDECOGamma,'+dataset+'*,'+str(timeparam)+'.pkl'))[0] ,'rb') as inp:
        predictions_gamma = pickle.load(inp)
        key = list(predictions_gamma.keys())[0]
        predictions_gamma = predictions_gamma[key]
            
    y_preds_G = list(map(lambda x: x[2],predictions_gamma))
    tau_preds_G = list(map(lambda x: x[0]/mx_t,predictions_gamma))
		

    kap = (round(cohen_kappa_score(y_true.values.tolist(),y_preds_eco_rev_ca),2), round(cohen_kappa_score(y_true.values.tolist(),y_preds_G),2), round(cohen_kappa_score(y_true.values.tolist(),y_preds_eco_rev_cu),2))
    earl = (sum(tau_preds_eco_rev_ca)/len(tau_preds_eco_rev_ca), sum(tau_preds_G)/len(tau_preds_G), sum(tau_preds_eco_rev_cu)/len(tau_preds_eco_rev_cu))
    print(C_cd, timeparam, dataset)
    return kap, earl, C_cd, timeparam, dataset



def friedman_test(r, *args):
    """
    source: http://tec.citius.usc.es/stac/doc/_modules/stac/nonparametric_tests.html#friedman_test

        Performs a Friedman ranking test.
        Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at least two of the groups represent populations with different median values.

        Parameters
        ----------
        sample1, sample2, ... : array_like
            The sample measurements for each group.

        Returns
        -------
        F-value : float
            The computed F-value of the test.
        p-value : float
            The associated p-value from the F-distribution.
        rankings : array_like
            The ranking for each group.
        pivots : array_like
            The pivotal quantities for each group.

        References
        ----------
        M. Friedman, The use of ranks to avoid the assumption of normality implicit in the analysis of variance, Journal of the American Statistical Association 32 (1937) 674–701.
        D.J. Sheskin, Handbook of parametric and nonparametric statistical procedures. crc Press, 2003, Test 25: The Friedman Two-Way Analysis of Variance by Ranks
    """


    k = len(args)
    if k < 2: raise ValueError('Less than 2 levels')
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1: raise ValueError('Unequal number of samples')

    if r=='petit':
        rev = False
    else:
        rev = True

    rankings = []
    for i in range(n):
        row = [col[i] for col in args]
        row_sort = sorted(row, reverse=rev)
        rankings.append([row_sort.index(v) + 1 + (row_sort.count(v)-1)/2. for v in row])

    rankings_avg = [sp.mean([case[j] for case in rankings]) for j in range(k)]
    rankings_cmp = [r/sp.sqrt(k*(k+1)/(6.*n)) for r in rankings_avg]

    chi2 = ((12*n)/float((k*(k+1))))*((sp.sum(r**2 for r in rankings_avg))-((k*(k+1)**2)/float(4)))
    iman_davenport = ((n-1)*chi2)/float((n*(k-1)-chi2))

    p_value = 1 - st.f.cdf(iman_davenport, k-1, (k-1)*(n-1))

    return iman_davenport, p_value, rankings_avg, rankings_cmp


def showTestsFriedmann(friedmann, index1, index2, timeparams, C_cds, title, figp, path):
    """
        0: CA
		1: CU 
		2: Gamma 
    """
    plt.figure(figsize=(6,3))
    plt.title(title)
    for i, C_cd in enumerate(C_cds):
        xTruePerdu=[]
        xTrueGagne=[]
        yTruePerdu=[]
        yTrueGagne=[]
        for j, timeparam in enumerate(timeparams):
            
            if friedmann[str(timeparam)+str(C_cd)][index1]<friedmann[str(timeparam)+str(C_cd)][index2]:
                xTrueGagne.append(j)
                yTrueGagne.append((i+1)/2)
            else:
                xTruePerdu.append(j)
                yTruePerdu.append((i+1)/2)

                
        plt.scatter(xTruePerdu, yTruePerdu, marker='_', color='black')
        plt.scatter(xTrueGagne, yTrueGagne, marker='+', color='black') 
        
    plt.xlabel(r'$\alpha$', fontsize=18)
    plt.ylabel(r'$\beta$', fontsize=18)
    y_axis = np.linspace(1/2, len(C_cds)/2, len(C_cds))
    plt.yticks(y_axis,C_cds)
    plt.xticks([i for i in range(len(timeparams))], timeparams, rotation=90)
    plt.savefig(op.join(path,figp), bbox_inches='tight')

       
def saveRealClassifiers(arguments):
    pathToClassifiers, folderRealData, dataset, classifier = arguments

    # path to data
    filepathTrain = folderRealData + '/' + dataset + '/' + dataset + '_TRAIN.tsv'

    # read data
    train = pd.DataFrame.from_csv(filepathTrain, sep='\t', header=None, index_col=None)
    mn = np.min(np.unique(train.iloc[:,0].values))

    train.iloc[:,0] = train.iloc[:,0].apply(lambda e: 0 if e==mn else 1)

    # get X and y
    Y_train = train.iloc[:, 0]
    X_train = train.loc[:, train.columns != train.columns[0]]


    max_t = X_train.shape[1]
    classifiers = {}

    ## Train classifiers for each time step
    for t in range(1, max_t+1):

        # use the same type classifier for each time step
        classifier_t = clone(classifier)
        # fit the classifier
        classifier_t.fit(X_train.iloc[:, :t], Y_train)
        # save it in memory
        classifiers[t] = classifier_t

    # save the model
    with open(pathToClassifiers + '/classifier'+dataset+'.pkl', 'wb') as output:
        pickle.dump(classifiers, output)





def saveRealModelsECONOMY(arguments):


    use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, min_t, classifier, fears, feat = arguments


    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    pathECOmodel = pathECOmodel+modelName+'.pkl'
    if not (os.path.exists(pathToRealModelsECONOMY + '/' + modelName + '.pkl')):
        # path to data
        # read data
        train_classifs = pd.read_csv(op.join(folderRealData, dataset, dataset + '_TRAIN_CLASSIFIERS.tsv'), sep='\t', header=None, index_col=None, engine='python')
        estimate_probas = pd.read_csv(op.join(folderRealData, dataset, dataset + '_ESTIMATE_PROBAS.tsv'), sep='\t', header=None, index_col=None, engine='python')

        # read data

        mn = np.min(np.unique(train_classifs.iloc[:,0].values))
        


        mx_t = train_classifs.shape[1] - 1

        # time cost
        timeCost = 0.01 * np.arange(mx_t+1) # arbitrary value

        # choose the method


        if (method == 'Gamma'):
            model = Economy_Gamma(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        elif (method == 'K') :
            model = Economy_K(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        elif (method == 'Gamma_lite'):
            model = Economy_Gamma_Lite(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        elif (method == 'Gamma_MC'):
            model = Economy_Gamma_MC(misClassificationCost, timeCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat, folderRealData)
        else:
            model = Economy_K_multiClustering(misClassificationCost, timeCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset, feat)
        # fit the model
        pathToClassifiers = pathToClassifiers + 'classifier' + dataset
        model.fit(train_classifs, estimate_probas, ratioVal, pathToClassifiers)


        # save the model
        with open(pathToRealModelsECONOMY + '/' + modelName + '.pkl', 'wb') as output:
            pickle.dump(model, output)


def saveRealModelsECONOMY_REV_MC(arguments):


    use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, Cm, C_cd, min_t, classifier, fears, feat = arguments


    # model name
        
        
    modelName = method + ',' + dataset + ',' + str(group) + ',' + str(C_cd)
    pathECOmodel = pathECOmodel+modelName+'.pkl'
    if not (os.path.exists(op.join(pathToRealModelsECONOMY, modelName + '.pkl'))):
        # path to data
        # read data
        train_classifs = pd.read_csv(op.join(folderRealData, dataset, dataset + '_TRAIN_CLASSIFIERS.tsv'), sep='\t', header=None, index_col=None, engine='python')
        estimate_probas = pd.read_csv(op.join(folderRealData, dataset, dataset + '_ESTIMATE_PROBAS.tsv'), sep='\t', header=None, index_col=None, engine='python')

        # read data
        mx_t = train_classifs.shape[1] - 1

        # time cost
        timeCost = 0.01 * np.arange(mx_t+1) # arbitrary value
        nbCLasses=len(train_classifs.iloc[:,0].unique())

        misClassificationCost = np.ones((nbCLasses,nbCLasses))*Cm - np.eye(nbCLasses)*Cm
        changeDecisionCost = np.ones((nbCLasses,nbCLasses))*C_cd-np.eye(nbCLasses)*C_cd

        if method=='Gamma_MC':

            model = Economy_Gamma_MC_REV(misClassificationCost, timeCost, changeDecisionCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData)
        elif method=='K':
            model = Economy_K_REV(misClassificationCost, timeCost, changeDecisionCost, min_t, classifier, group, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData) 
        else:
            print('nothing')
        # fit the model
        pathToClassifiers = pathToClassifiers + 'classifier' + dataset
        model.fit(train_classifs, estimate_probas, ratioVal, pathToClassifiers)


        # save the model
        with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'wb') as output:
            pickle.dump(model, output)

def saveRealModelsECONOMY_REV(arguments):


    use_complete_ECO_model, pathECOmodel, sampling_ratio, orderGamma, ratioVal, pathToRealModelsECONOMY, pathToClassifiers, folderRealData, method, dataset, group, misClassificationCost, changeDecisionCost, min_t, classifier, fears, feat = arguments


    # model name
    modelName = method + ',' + dataset + ',' + str(group) + ',' + str(changeDecisionCost[0][1])
    pathECOmodel = pathECOmodel+modelName+'.pkl'
    if not (os.path.exists(op.join(pathToRealModelsECONOMY, modelName + '.pkl'))):
        # path to data
        # read data
        train_classifs = pd.read_csv(op.join(folderRealData, dataset, dataset + '_TRAIN_CLASSIFIERS.tsv'), sep='\t', header=None, index_col=None, engine='python')
        estimate_probas = pd.read_csv(op.join(folderRealData, dataset, dataset + '_ESTIMATE_PROBAS.tsv'), sep='\t', header=None, index_col=None, engine='python')

        # read data
        mx_t = train_classifs.shape[1] - 1

        # time cost
        timeCost = 0.01 * np.arange(mx_t+1) # arbitrary value

        # choose the method
        model = Economy_Gamma_REV(misClassificationCost, timeCost, changeDecisionCost, min_t, classifier, group, orderGamma, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData)

        # fit the model
        pathToClassifiers = pathToClassifiers + 'classifier' + dataset
        model.fit(train_classifs, estimate_probas, ratioVal, pathToClassifiers)


        # save the model
        with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'wb') as output:
            pickle.dump(model, output)




def transform_to_format_fears(X):
    nbObs, length = X.shape
    for i in range(nbObs):
        ts = X.iloc[i,:]
        data = {'id':[i for _ in range(length)], 'timestamp':[k for k in range(1,length+1)], 'dim_X':list(ts.values)}
        if i==0:
            df = pd.DataFrame(data)
        else:
            df = df.append(pd.DataFrame(data))
    df = df.reset_index(drop=True)
    return df






def score_post_optimal(model, X_test, y_test, C_m, sampling_ratio, val=None, min_t=4, max_t=50):

    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step

    if val:
        with open('RealData/'+model.dataset+'/val_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/val_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)
    if not val:
        with open('RealData/'+model.dataset+'/test_preds'+'.pkl' ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open('RealData/'+model.dataset+'/test_probas'+'.pkl' ,'rb') as inp:
            donnes_proba = pickle.load(inp)

    # We predict for every time series [label, tau*]
    for i in range(nb_observations):
        post_costs = []
        timestamps_pred = []
        
        for t in range(start, max_t+1, step):


            x = np.array(list(X_test.iloc[i, :t]))

            pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            _, cost = model.forecastExpectedCost(x,pb)
            post_costs.append(cost)
            timestamps_pred.append(t)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
        tau_post_star = timestamps_pred[np.argmin(post_costs)]

        if model.fears:
            prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
        elif model.feat:
            prediction = donnes_pred[t][i]
        else:
            prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]

        if (prediction != y_test.iloc[i]):
            score_computed += model.timeCost[tau_post_star] + C_m
        else:
            score_computed += model.timeCost[tau_post_star]
 

    return (score_computed/nb_observations)


def computeScores_REV_MC(arguments):

    val_bool, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, changeDecisionCost, min_t, paramTime, preds_cl, variante, history = arguments
    val_probas, val_preds, val_uc = preds_cl


    # model name
    modelName = method + ',' + dataset + ',' + str(group) + ',' + str(changeDecisionCost) 
    # path to data
    if val_bool:
        filepathTest = op.join(folderRealData, dataset, dataset+'_VAL_SCORE.tsv')
        stri = ''
    else:
        filepathTest = op.join(folderRealData, dataset, dataset+'_TEST_SCORE.tsv')
        stri = 'EVAL'

    if not (os.path.exists(op.join(pathToSaveScores, stri+'score'+modelName+ ',' + str(paramTime)   + ',' + variante + ',' + str(history)+'.json'))):
        # read data
        print('nooooooo')

        val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(val.iloc[:,0].values))


        # get X and y
        y_val = val.iloc[:, 0]
        X_val = val.loc[:, val.columns != val.columns[0]]
        mx_t = X_val.shape[1]
        # choose the method
        try:
            with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'rb') as inp:
                
                model = pickle.load(inp)
                if normalizeTime:
                    timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
                else:
                    timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
                setattr(model, 'timeCost', timeCostt)
        except pickle.UnpicklingError:
            print('PROOOOBLEM', modelName)

        # predictions
        if method=='Gamma_MC':
            decisions, cost_estimation = model.predict_revocable(X_val, False, [val_uc, val_preds], variante, history)
        else:
            decisions, cost_estimation = model.predict_revocable(X_val, False, [val_probas, val_preds], variante, history)
                        
                        
        score_model = score_rev(model.timeCost, decisions, y_val, C_m, changeDecisionCost)
        

        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(changeDecisionCost) + ',' + str(paramTime)   + ',' + variante + ',' + str(history)

        with open(op.join(pathToSaveScores, stri+'score'+modelName+'.json'), 'w') as outfile:
            json.dump({modelName:score_model}, outfile)

        if not val_bool:
            with open(op.join(pathToSaveScores, stri+'decisions'+modelName+'.pkl'), 'wb') as outp:
                pickle.dump(decisions, outp)

    else:
        with open(op.join(pathToSaveScores, stri+'score'+modelName+ ',' + str(paramTime)+',' + variante + ',' + str(history)+'.json')) as f:
            try:
                loadedjson = json.loads(f.read())
            except:
                print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName,score_model)

def computeScores(arguments):

    score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, min_t, paramTime = arguments

    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    # path to data
    filepathTest = op.join(folderRealData, dataset, dataset+'_VAL_SCORE.tsv')

    if not (os.path.exists(op.join(pathToSaveScores, 'score'+modelName+ ',' + str(paramTime)+'.json'))):
        # read data
        print(modelName)

        val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(val.iloc[:,0].values))

        # get X and y
        y_val = val.iloc[:, 0]
        X_val = val.loc[:, val.columns != val.columns[0]]
        mx_t = X_val.shape[1]
        # choose the method
        try:
            with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'rb') as input:
                
                model = pickle.load(input)
                if normalizeTime:
                    timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
                else:
                    timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
                setattr(model, 'timeCost', timeCostt)
        except pickle.UnpicklingError:
            print('PROOOOBLEM', modelName)
        if score_chosen == 'star':
            if method=='Gamma_MC':
                score_model = score_avgcost(model, X_val, y_val, True, C_m, model.sampling_ratio, val=True, max_t=mx_t)
            else:
                score_model = score_avgcost(model, X_val, y_val, False, C_m, model.sampling_ratio, val=True, max_t=mx_t)
        if score_chosen == 'post':
            score_model = score_post_optimal(model, X_val, y_val, C_m, model.sampling_ratio, max_t=mx_t)
        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(paramTime)

        with open(op.join(pathToSaveScores, 'score'+modelName+'.json'), 'w') as outfile:
            json.dump({modelName:score_model}, outfile)
    else:
        with open(op.join(pathToSaveScores,'score'+modelName+',' + str(paramTime)+'.json')) as f:
            try:
                loadedjson = json.loads(f.read())
            except:
                print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName,score_model)





def computeScores_REV(arguments):

    val_bool, score_chosen, normalizeTime, C_m, orderGamma, pathToSaveScores, pathToRealModelsECONOMY, folderRealData, method, dataset, group, misClassificationCost, changeDecisionCost, min_t, paramTime, preds_cl, variante, history = arguments
    val_probas, val_preds = preds_cl


    # model name
    modelName = method + ',' + dataset + ',' + str(group) + ',' + str(changeDecisionCost[0][1]) 
    # path to data
    if val_bool:
        filepathTest = op.join(folderRealData, dataset, dataset+'_VAL_SCORE.tsv')
        stri = ''
    else:
        filepathTest = op.join(folderRealData, dataset, dataset+'_TEST_SCORE.tsv')
        stri = 'EVAL'

    if not (os.path.exists(op.join(pathToSaveScores, stri+'score'+modelName+ ',' + str(paramTime)   + ',' + variante + ',' + str(history)+'.json'))):
        # read data

        val = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(val.iloc[:,0].values))


        # get X and y
        y_val = val.iloc[:, 0]
        X_val = val.loc[:, val.columns != val.columns[0]]
        mx_t = X_val.shape[1]
        # choose the method
        try:
            with open(op.join(pathToRealModelsECONOMY, modelName + '.pkl'), 'rb') as inp:
                
                model = pickle.load(inp)
                if normalizeTime:
                    timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
                else:
                    timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
                setattr(model, 'timeCost', timeCostt)
        except pickle.UnpicklingError:
            print('PROOOOBLEM', modelName)

        # predictions
        decisions, cost_estimation = model.predict_revocable(X_val, False, [val_probas, val_preds], variante, history)

        score_model = score_rev(model.timeCost, decisions, y_val, C_m, changeDecisionCost[0][1])
        

        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(changeDecisionCost[0][1]) + ',' + str(paramTime)   + ',' + variante + ',' + str(history)

        with open(op.join(pathToSaveScores, stri+'score'+modelName+'.json'), 'w') as outfile:
            json.dump({modelName:score_model}, outfile)

        if not val_bool:
            with open(op.join(pathToSaveScores, stri+'decisions'+modelName+'.pkl'), 'wb') as outp:
                pickle.dump(decisions, outp)

    else:
        with open(op.join(pathToSaveScores, stri+'score'+modelName+ ',' + str(paramTime)+',' + variante + ',' + str(history)+'.json')) as f:
            try:
                loadedjson = json.loads(f.read())
            except:
                print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName,score_model)



def computeBestGroup_REV(datasets, timeParams, modelName_score, INF, methods, C_cds, variantes, history):
    bestGroup = {method + ',' + dataset + ',' + str(C_cd) + ',' + str(timeparam)   + ',' + variante + ',' + str(hist):[1,INF] for method in methods for dataset in datasets  for timeparam in timeParams for C_cd in C_cds for variante in variantes for hist in history}
    for e in modelName_score:
        (modelName, score_model) = e
        method, dataset, group, C_cd, paramTime, variante, hist = modelName.split(',')
        group = int(group)
        paramTime = float(paramTime)
        if paramTime == 1.0:
            paramTime = int(paramTime)
        if (bestGroup[method + ',' + dataset + ',' + C_cd + ',' + str(paramTime)   + ',' + variante + ',' + hist][1] > score_model):
            bestGroup[method + ',' + dataset + ',' + str(C_cd) + ',' + str(paramTime)   + ',' + variante + ',' + hist][0] = group
            bestGroup[method + ',' + dataset + ',' + str(C_cd) + ',' + str(paramTime)   + ',' + variante + ',' + hist][1] = score_model
    return bestGroup



def computeBestGroup(datasets, timeParams, modelName_score, INF, methods):
    bestGroup = {method + ',' + dataset + ',' + str(timeparam):[1,INF] for method in methods for dataset in datasets  for timeparam in timeParams}
    for e in modelName_score:
        (modelName, score_model) = e
        method, dataset, group, paramTime = modelName.split(',')
        group = int(group)
        paramTime = float(paramTime)
        if paramTime == 1.0:
            paramTime = int(paramTime)
        if (bestGroup[method + ',' + dataset + ',' + str(paramTime)][1] > score_model):
            bestGroup[method + ',' + dataset + ',' + str(paramTime)][0] = group
            bestGroup[method + ',' + dataset + ',' + str(paramTime)][1] = score_model
    return bestGroup

def score_avgcost(model, X_test, y_test, uc, C_m, sampling_ratio, val=None, min_t=4, max_t=50):
    nb_observations, _ = X_test.shape
    score_computed = 0
    step = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1
    start = step
    # We predict for every time series [label, tau*]

    if val:
        with open(op.join(model.folderRealData, model.dataset, 'val_preds.pkl') ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open(op.join(model.folderRealData, model.dataset, 'val_probas.pkl') ,'rb') as inp:
            donnes_proba = pickle.load(inp)
        if uc:
            with open(op.join(model.folderRealData, 'uc', model.dataset+'val_uc.pkl') ,'rb') as inp:
                donnes_uc = pickle.load(inp)
    if not val:
        with open(op.join(model.folderRealData, model.dataset, 'test_preds.pkl') ,'rb') as inp:
            donnes_pred = pickle.load(inp)
        with open(op.join(model.folderRealData, model.dataset, 'test_probas.pkl') ,'rb') as inp:
            donnes_proba = pickle.load(inp)
        if uc:
            with open(op.join(model.folderRealData, 'uc', model.dataset+'test_uc.pkl') ,'rb') as inp:
                donnes_uc = pickle.load(inp)

    
    for i in range(nb_observations):
        # The approach is non-moyopic, for each time step we predict the optimal
        # time to make the prediction in the future.

        for t in range(start, max_t+1, step):
            # first t values of x

            x = np.array(list(X_test.iloc[i, :t]))
            if uc:
                pb = donnes_uc[t][i]
            else:
                pb = donnes_proba[t][i]
            # compute cost of future timesteps (max_t - t)
            send_alert, cost, cst = model.forecastExpectedCost(x,pb)
            #compute tau*
            # predict the label of our time series when tau* = 0 or when we
            # reach max_t
            if (send_alert):
                if model.fears:
                    prediction = model.classifiers[t].predict(transform_to_format_fears(x.reshape(1, -1)))[0]
                elif model.feat:
                    prediction = donnes_pred[t][i]
                else:
                    prediction = model.classifiers[t].predict(x.reshape(1, -1))[0]


                if (prediction != y_test.iloc[i]):
                    score_computed += model.timeCost[t] + C_m
                else:
                    score_computed += model.timeCost[t]
                break
    return (score_computed/nb_observations)


def evaluate(arguments):

    score_chosen, normalizeTime, C_m, pathToRealModels, folderRealData, dataset, method, group, paramTime, pathToSaveScores = arguments

    # model name
    modelName = method + ',' + dataset + ',' + str(group)
    # path to data
    filepathTest = op.join(folderRealData, dataset, dataset+'_TEST_SCORE.tsv')

    if not (os.path.exists(op.join(pathToSaveScores, 'EVALscore'+modelName+ ',' + str(paramTime)+'.json'))):
        # read data
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')

        mn = np.min(np.unique(test.iloc[:,0].values))

        
        # get X and y
        y_test = test.iloc[:, 0]
        X_test = test.loc[:, test.columns != test.columns[0]]
        mx_t = X_test.shape[1]
        modelName = method + ',' + dataset + ',' + str(group)

        with open(op.join(pathToRealModels ,modelName + '.pkl'), 'rb') as input:
            model = pickle.load(input)
            if normalizeTime:
                timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
            else:
                timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
            setattr(model, 'timeCost', timeCostt)

        if score_chosen == 'star':
            if method == 'Gamma_MC':
                score_model = score_avgcost(model, X_test, y_test, True, C_m, model.sampling_ratio, val=False, max_t=mx_t)
            else:
                score_model = score_avgcost(model, X_test, y_test, False, C_m, model.sampling_ratio, val=False, max_t=mx_t)

            with open(op.join(pathToSaveScores,'EVALscore'+modelName+'.json'), 'w') as outfile:
                json.dump({modelName:score_model}, outfile)
        if score_chosen == 'post':
            score_model = score_post_optimal(model, X_test, y_test, C_m, model.sampling_ratio, val=False, max_t=mx_t)
            with open(pathToSaveScores+'/EVALscorePOST'+modelName+'.json', 'w') as outfile:
                json.dump({modelName:score_model}, outfile)
        modelName = method + ',' + dataset + ',' + str(group) + ',' + str(paramTime)

        
    else:
        if score_chosen == 'star':
            with open(op.join(pathToSaveScores,'EVALscore'+modelName+ ',' + str(paramTime)+'.json')) as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        else:
            with open(op.join(pathToSaveScores, 'EVALscorePOST'+modelName+ ',' + str(paramTime)+'.json')) as f:
                try:
                    loadedjson = json.loads(f.read())
                except:
                    print('BUUUUUUUUUUUUUUUUUUUUUG',modelName)
        modelName = list(loadedjson.keys())[0]
        score_model = list(loadedjson.values())[0]
    return (modelName, score_model)


    

    


    for i in range(nb_observations_val):

        for t in timestamps:

            proba1 = val_probas[t][i]
            proba2 = 1.0 - proba1
            if proba1 > proba2:
                maxiProba = proba1
                scndProba = proba2
            else:
                maxiProba = proba2
                scndProba = proba1

            # Stopping rule
            sr = moriParams[0] * maxiProba + moriParams[1] * (maxiProba-scndProba) + moriParams[2] * (t / max_t)

            if sr > 0 or t==timestamps[-1]:
                
                if y_test_val.iloc[i] == val_preds[t][i]:
                    score += timecost[t]
                else:
                    score += timecost[t] + 1 #C_m = 1
                break
    print('------------FINISH---------- : ', timecostParam)
    return score / (nb_observations_ep + nb_observations_val)

            
    

def computePredictionsEconomy(arguments):
    normalizeTime, pathToSavePredictions, pathToIntermediateResults, folderRealData, pathToRealModels, dataset, method, group, paramTime = arguments


    # path to data
    filepathTest = op.join(folderRealData,dataset,dataset+'_TEST_SCORE.tsv')
    
    modelName = method + ',' + dataset + ',' + str(group)+',' + str(paramTime)
    if not (os.path.exists(op.join(pathToSavePredictions,'PREDECO'+modelName+'.pkl'))):

        modelName = method + ',' + dataset + ',' + str(group)

        # read data
        test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
        mn = np.min(np.unique(test.iloc[:,0].values))

        # get X and y
        y_test = test.iloc[:, 0]
        X_test = test.loc[:, test.columns != test.columns[0]]
        mx_t = X_test.shape[1]

        with open(op.join(pathToRealModels, modelName + '.pkl'), 'rb') as input:
            model = pickle.load(input)
            if normalizeTime:
                timeCostt = (1/mx_t) * paramTime * np.arange(model.timestamps[-1] + 1)
            else:
                timeCostt = paramTime * np.arange(model.timestamps[-1]+1)
            setattr(model, 'timeCost', timeCostt)

        with open(op.join(folderRealData,dataset,'test_probas.pkl') ,'rb') as inp:
            test_probas = pickle.load(inp)
        with open(op.join(folderRealData,dataset,'test_preds.pkl'),'rb') as inp:
            test_preds = pickle.load(inp)
        if method=='Gamma_MC':
            with open(op.join(folderRealData,'uc',dataset+'test_uc.pkl'),'rb') as inp:
                test_uc = pickle.load(inp)
            preds_tau = model.predict(X_test, oneIDV=False, donnes=[test_uc, test_preds])
        else:
            preds_tau = model.predict(X_test, oneIDV=False, donnes=[test_probas, test_preds])

        #preds_post = model.predict_post_tau_stars(X_test, [test_probas, test_preds])

        #preds_optimal = model.predict_optimal_algo(X_test, y_test, test_preds)
        #metric = preds_tau, preds_post, preds_optimal
        modelName = method + ',' + dataset + ',' + str(group)+',' + str(paramTime)


        with open(op.join(pathToSavePredictions, 'PREDECO'+modelName+'.pkl'), 'wb') as outfile:
            pickle.dump({modelName: preds_tau}, outfile)

        return preds_tau, modelName
    else:
        with open(op.join(pathToSavePredictions, 'PREDECO'+modelName+'.pkl'), 'rb') as outfile:
            loadedjson = pickle.load(outfile)
        
    
        return list(loadedjson.values())[0], list(loadedjson.keys())[0]


def computeMetricsNeeded(arguments):

    sampling_ratio = arguments[0]
    preds_tau, preds_post, preds_optimal, modelName = arguments[1]
    pathToIntermediateResults = arguments[2]
    folderRealData = arguments[3]

    method, dataset, group, timeparam = modelName.split(',')

    organized_metrics = {dataset+','+method:[]}
    tau_et = np.array(preds_tau)[:,0]
    tau_post = np.array(preds_post)[:,0]

    tau_opt = np.array(preds_optimal)[:,0]
    f_et = np.array(preds_tau)[:,1]

    f_post = np.array(preds_post)[:,1]

    f_opt = np.array(preds_optimal)[:,1]

    objectives = [tau_et, tau_post, tau_opt, f_et, f_post, f_opt]
    organized_metrics[dataset+','+method].append(timeparam)
    for e in objectives:
        organized_metrics[dataset+','+method].append(np.mean(e))
        organized_metrics[dataset+','+method].append(np.std(e))

    for e1, e2 in zip(objectives[1:3], objectives[4:]):
        organized_metrics[dataset+','+method].append(np.mean(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.std(abs(tau_et-e1)))
        organized_metrics[dataset+','+method].append(np.mean(abs(f_et-e2)))
        organized_metrics[dataset+','+method].append(np.std(abs(f_et-e2)))

    organized_metrics[dataset+','+method].append(group) #group



    # AUC import packages ???
    filepathTest = folderRealData+'/'+dataset+'/'+dataset+'_TEST_SCORE.tsv'
    test = pd.read_csv(filepathTest, sep='\t', header=None, index_col=None, engine='python')
    mn = np.min(np.unique(test.iloc[:,0].values))
    test.iloc[:,0] = test.iloc[:,0].apply(lambda e: 0 if e==mn else 1)
    y_test = test.iloc[:, 0]
    X_test = test.loc[:, test.columns != test.columns[0]]
    max_t = X_test.shape[1]
    # pourcentage des prediction à min_t
    min_t = int(max_t*sampling_ratio) if int(max_t*sampling_ratio)>0 else 1

    for i in range(min_t, max_t+1, min_t):
        mx_t = i
    count_min_t = 0
    count_max_t = 0
    for elem in tau_et:
        if (elem == min_t):
            count_min_t += 1
        if (elem == mx_t):
            count_max_t += 1

    count_min_t /= len(tau_et)
    count_max_t /= len(tau_et)
    organized_metrics[dataset+','+method].append(count_min_t)
    organized_metrics[dataset+','+method].append(count_max_t)

    kappa_tau = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_tau)[:,2]))),2)
    kappa_opt = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_post)[:,2]))),2)
    kappa_post = round(cohen_kappa_score(y_test.values.tolist(), list(map(int, np.array(preds_optimal)[:,2]))),2)

    organized_metrics[dataset+','+method].append(kappa_tau)
    organized_metrics[dataset+','+method].append(kappa_post)
    organized_metrics[dataset+','+method].append(kappa_opt)


    # pourcentage des cas ou on est précoce au post optimal
    counters = [0,0,0,0]
    for tauu, tau_ppost, tau_opti in zip(tau_et, tau_post, tau_opt):
        if tauu < tau_ppost:
            counters[0] = counters[0] + 1
        if tauu < tau_opti:
            counters[1] = counters[1] + 1
        if tauu >= tau_ppost:
            counters[2] = counters[2] + 1
        if tauu >= tau_opti:
            counters[3] = counters[3] + 1
    for i in range(4):
        counters[i] = round(counters[i] / len(tau_et),2)

    organized_metrics[dataset+','+method].append(counters[0])
    organized_metrics[dataset+','+method].append(counters[1])
    organized_metrics[dataset+','+method].append(counters[2])
    organized_metrics[dataset+','+method].append(counters[3])

    # compute medians tau_et, tau_post, tau_opt, f_et, f_post, f_opt
    organized_metrics[dataset+','+method].append(np.median(tau_et))
    organized_metrics[dataset+','+method].append(np.median(tau_post))
    organized_metrics[dataset+','+method].append(np.median(tau_opt))
    organized_metrics[dataset+','+method].append(np.median(f_et))
    organized_metrics[dataset+','+method].append(np.median(f_post))
    organized_metrics[dataset+','+method].append(np.median(f_opt))

    # compute medians differences
    organized_metrics[dataset+','+method].append(np.median(abs(tau_et-tau_post)))
    organized_metrics[dataset+','+method].append(np.median(abs(f_et-f_post)))




    with open(pathToIntermediateResults+'/MetricsNeeded'+modelName+'.pkl', 'wb') as outfile:
        pickle.dump(organized_metrics, outfile)

    return organized_metrics
