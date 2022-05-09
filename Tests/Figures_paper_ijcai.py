import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy as sp
import scipy.stats as st
import itertools as it
import pickle 
import os.path as op
import math
import glob 
import csv
from sklearn.metrics import cohen_kappa_score
import sys 
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Economy_Gamma import Economy_Gamma 
from TestUtils import run_predictions, run_score, score_rev, wilcoxon_test, showTests, Average, kappa_earliness, showTestsFriedmann, friedman_test
import multiprocessing
from joblib import Parallel, delayed
import argparse



parser = argparse.ArgumentParser(description='test')
parser.add_argument('--pathToInputParams', help='path to json file with input params', required=True)

    
args = parser.parse_args()
pathToParams = args.pathToInputParams


# load input params
with open(pathToParams) as configfile:
    configParams = json.load(configfile)

timeparams = configParams['timeParams']
datasets = configParams['Datasets']
Cm = configParams['misClassificationCost']
C_cds = configParams['changeDecisionCost']
folderRealData = op.join(os.getcwd(), configParams['folderRealData'])

sampling_ratio = configParams['sampling_ratio']
nb_core = multiprocessing.cpu_count()


########################################################################################################################################
########################################################################################################################################
########################################################## Exp ECO-REV-CU ##############################################################
########################################################################################################################################
########################################################################################################################################





pathexp_gamma = op.join(os.getcwd(), 'experiments\\experiment1')
pathModelsEconomy = op.join(os.getcwd(), 'experiments\\experiment1\\modelsECONOMY')
pathPreds = op.join(os.getcwd(), 'experiments\\experiment1\\intermediate_results\\predictions')
pathBestGroups = op.join(os.getcwd(), 'experiments\\experiment1\\intermediate_results\\bestGroup.pkl')

pathResults = op.join(os.getcwd(), 'experiments\\experiment3')


with open(pathBestGroups, 'rb') as inp:
    bestGroup = pickle.load(inp)

results = Parallel(n_jobs=nb_core)(delayed(run_predictions)(func_arg) for func_arg in [(dataset, timeparam, bestGroup['Gamma,'+dataset+','+str(timeparam)][0], folderRealData, pathModelsEconomy, pathResults) for dataset in datasets for timeparam in timeparams])




results = Parallel(n_jobs=nb_core)(delayed(run_score)(func_arg) for func_arg in [(dataset, timeparam, C_cd, pathResults, folderRealData) for dataset in datasets for timeparam in timeparams for C_cd in C_cds])

with open(op.join(pathResults, 'Results_eco_rev_cu_preds.pkl'),'wb') as outp:
    pickle.dump(results, outp)

########################################################################################################################################
########################################################################################################################################
########################################################################################################################################
########################################################################################################################################


# eco_gamma
with open(op.join(pathexp_gamma, 'results.pkl'),'rb') as outp:
    results_eco_gamma_preds = pickle.load(outp)

with open(pathBestGroups,'rb') as outp:
    bestG_eco_gamma_preds = pickle.load(outp)


results_eco_gamma_dict = {}



resultats_f_gamma = {str(timeparam)+str(C_cd):[] for timeparam in timeparams for C_cd in C_cds}
for C_cd in C_cds:
    for timeparam in timeparams:
        for dataset in datasets:
            resultats_f_gamma[str(timeparam)+str(C_cd)].append(results_eco_gamma_preds['Gamma,'+dataset+','+ str(bestG_eco_gamma_preds['Gamma,'+dataset+','+str(timeparam)][0])+','+str(timeparam)])




# eco_rev_cu
with open(op.join(pathResults, 'Results_eco_rev_cu_preds.pkl'),'rb') as outp:
    results_eco_rev_cu_preds = pickle.load(outp) 

results_eco_rev_cu_dict = {}


for e in results_eco_rev_cu_preds:
    sc_b, dataset, timeparam, C_cd = e
    results_eco_rev_cu_dict[dataset+str(timeparam)+str(C_cd)] = sc_b
    

resultats_f_eco_rev_cu = {str(timeparam)+str(C_cd):[] for timeparam in timeparams for C_cd in C_cds}

for C_cd in C_cds:
    for timeparam in timeparams:
        for dataset in datasets:
            resultats_f_eco_rev_cu[str(timeparam)+str(C_cd)].append(results_eco_rev_cu_dict[dataset+str(timeparam)+str(C_cd)]) 



# eco_rev_ca
pathBestGroups_eco_rev_ca = op.join(os.getcwd(), 'experiments\\experiment2\\intermediate_results\\bestGroup.pkl')
resultsEXP_eco_rev_ca = op.join(os.getcwd(), 'experiments\\experiment2\\results.pkl')

	
with open(pathBestGroups_eco_rev_ca, 'rb') as inp:
    bestGroups_eco_rev_ca = pickle.load(inp)
with open(resultsEXP_eco_rev_ca, 'rb') as inp:
    resultats_eco_rev_ca = pickle.load(inp)

resultats_f_eco_rev_ca = {str(timeparam)+str(C_cd):[] for timeparam in timeparams for C_cd in C_cds}

for C_cd in C_cds:
    for timeparam in timeparams:
        for dataset in datasets:
            resultats_f_eco_rev_ca[str(timeparam)+str(C_cd)].append(resultats_eco_rev_ca['Gamma_rev,'+dataset+','+ str(bestGroups_eco_rev_ca['Gamma_rev,'+dataset+','+str(C_cd)+','+str(timeparam)+',avec_cout_moindre,True'][0])+','+str(C_cd)+','+str(timeparam)+',avec_cout_moindre,True']) 


########################################################################################################################################
########################################################################################################################################
############################################################## wilcoxon  ###############################################################
########################################################################################################################################
########################################################################################################################################




	

	
pathfigures = op.join(os.getcwd(), 'experiments\\Figures')

showTests(resultats_f_eco_rev_cu, resultats_f_gamma, timeparams, C_cds, r'ECO-REV-CU vs Economy-$\gamma$', 'ECO-REV-CU_vs_gamma.png', pathfigures)

showTests(resultats_f_eco_rev_ca, resultats_f_gamma, timeparams, C_cds, r'ECO-REV-CA vs Economy-$\gamma$', 'ECO-REV-CA_vs_gamma.png', pathfigures)

showTests(resultats_f_eco_rev_ca, resultats_f_eco_rev_cu, timeparams, C_cds, 'ECO-REV-CA vs ECO-REV-CU', 'ECO-REV-CA_vs_ECO-REV-CU.png', pathfigures)





########################################################################################################################################
########################################################################################################################################
############################################################## Kappa and moment  #######################################################
########################################################################################################################################
########################################################################################################################################


Kappa = {}
Taus = {}
pathPredsB2B3 = op.join(os.getcwd(), 'experiments\\experiment2\\intermediate_results\\scores')


kap_earl = Parallel(n_jobs=nb_core)(delayed(kappa_earliness)(func_arg) for func_arg in [(C_cd, timeparam, dataset,folderRealData, pathPredsB2B3, pathPreds, pathResults) for dataset in datasets for timeparam in timeparams for C_cd in C_cds])

with open(op.join(pathResults, 'kap_earl_avg.pkl'),'wb') as outp:
    pickle.dump(kap_earl, outp)


for e in kap_earl:
    kap, earl, C_cd, timeparam, dataset = e
    Kappa[dataset+str(timeparam)+str(C_cd)] = kap
    Taus[dataset+str(timeparam)+str(C_cd)] = earl

aggTau_eco_rev_cu = {}
aggKappa_eco_rev_cu = {}
aggTau_eco_rev_ca = {}
aggKappa_eco_rev_ca = {}
aggTau_G = {}
aggKappa_G = {}

for C_cd in C_cds:
    for timeparam in timeparams:
        aTau_eco_rev_ca = []
        aKappa_eco_rev_ca = []
        aTau_eco_rev_cu = []
        aKappa_eco_rev_cu = []
        aTau_G = []
        aKappa_G = []

        for dataset in datasets:
            aTau_eco_rev_ca.append(Taus[dataset+str(timeparam)+str(C_cd)][0])
            aKappa_eco_rev_ca.append(Kappa[dataset+str(timeparam)+str(C_cd)][0])
            aTau_G.append(Taus[dataset+str(timeparam)+str(C_cd)][1])
            aKappa_G.append(Kappa[dataset+str(timeparam)+str(C_cd)][1])

            aTau_eco_rev_cu.append(Taus[dataset+str(timeparam)+str(C_cd)][2])
            aKappa_eco_rev_cu.append(Kappa[dataset+str(timeparam)+str(C_cd)][2])

        aggTau_eco_rev_ca[str(timeparam)+str(C_cd)] = np.mean(np.array(aTau_eco_rev_ca))
        aggKappa_eco_rev_ca[str(timeparam)+str(C_cd)] = np.mean(np.array(aKappa_eco_rev_ca))
        aggTau_G[str(timeparam)+str(C_cd)] = np.mean(np.array(aTau_G))
        aggKappa_G[str(timeparam)+str(C_cd)] = np.mean(np.array(aKappa_G))
        aggTau_eco_rev_cu[str(timeparam)+str(C_cd)] = np.mean(np.array(aTau_eco_rev_cu))
        aggKappa_eco_rev_cu[str(timeparam)+str(C_cd)] = np.mean(np.array(aKappa_eco_rev_cu))
            
# figures
alp = r'$\alpha$ = '
for C_cd in C_cds:
    plt.figure()
    plt.title(r'Kappa vs Earliness for $\beta=$'+str(C_cd))
    plt.plot([aggTau_eco_rev_ca[str(timeparam)+str(C_cd)] for timeparam in timeparams], [aggKappa_eco_rev_ca[str(timeparam)+str(C_cd)] for timeparam in timeparams], color='grey', label='ECO-REV-CA', marker='d', markerfacecolor='black')
    
    plt.plot([aggTau_eco_rev_cu[str(timeparam)+str(C_cd)] for timeparam in timeparams], [aggKappa_eco_rev_cu[str(timeparam)+str(C_cd)] for timeparam in timeparams], color='grey', label='ECO-REV-CU', marker='d', markerfacecolor='white')
    plt.plot([aggTau_G[str(timeparam)+str(C_cd)] for timeparam in timeparams], [aggKappa_G[str(timeparam)+str(C_cd)] for timeparam in timeparams], color='grey', label=r'ECONOMY-$\gamma$', marker='o', markerfacecolor='white')
    indices =  [k for k in range(1,9,3)] + [k for k in range(9,len(timeparams))]
    """for i in indices:
        plt.annotate(alp + str(timeparams[i]), ([aggTau_G[str(timeparam)+str(C_cd)] for timeparam in timeparams][i]+0.005,  [aggKappa_G[str(timeparam)+str(C_cd)] for timeparam in timeparams][i]-0.01),  fontsize=7)
    """
    plt.legend()
    plt.xlabel(r'$Earliness$', fontsize=16)
    plt.ylabel(r'$Kappa$', fontsize=16)
    plt.savefig(op.join(pathfigures,"figure_C_cd_"+str(C_cd)+".png"))
    plt.close()





########################################################################################################################################
########################################################################################################################################
############################################################## Friedmann  ##############################################################
########################################################################################################################################
########################################################################################################################################

friedmann = {}
for C_cd in C_cds:
    for i,timeparam in enumerate(timeparams):
        iman_davenport, p_value, rankings_avg, rankings_cmp = friedman_test('petit', *[resultats_f_eco_rev_ca[str(timeparam)+str(C_cd)], resultats_f_eco_rev_cu[str(timeparam)+str(C_cd)], resultats_f_gamma[str(timeparam)+str(C_cd)]])
        friedmann[str(timeparam)+str(C_cd)] = rankings_avg

		
showTestsFriedmann(friedmann, 0, 1, timeparams, C_cds, 'ECO-REV-CA vs ECO-REV-CU', 'FRIEDMANNECO-REV-CA_vs_ECO-REV-CU.png', pathfigures)


