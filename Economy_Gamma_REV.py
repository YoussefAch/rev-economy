import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import clone
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from utils import generatePossibleSequences
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import json
from collections import defaultdict
from Economy import Economy
import os.path as op

class Economy_Gamma_REV(Economy):

    """
    Economy_Gamma inherits from Economy

    ATTRIBUTES :

        - nbIntervals : number of intervals.
        - order       : order of marcov chain.
        - thresholds  : dictionary of thresholds for each time step.
        - transitionMatrices   : transition matrices for each sequence (t,t+1).
        - complexTransitionMatrices : transition matrices for each sequence (t-ordre..t,t+1).
        - indices     : indices of data associated to each time step and each interval
        - labels     : list of labels observed on the data set.
    """

    def __init__(self, misClassificationCost, timeCost, changeDecisionCost, min_t, classifier, nbIntervals, order, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData):
        super().__init__(misClassificationCost, timeCost, min_t, classifier)
        self.nbIntervals = nbIntervals
        self.order = order
        self.thresholds = {}
        self.transitionMatrices = {}
        self.complexTransitionMatrices = {}
        self.indices = {}
        self.P_y_gamma = {}
        self.use_complete_ECO_model = use_complete_ECO_model
        self.pathECOmodel = pathECOmodel
        self.sampling_ratio = sampling_ratio
        self.fears = fears
        self.dataset = dataset
        self.feat = feat
        self.changeDecisionCost = changeDecisionCost
        self.folderRealData = folderRealData



    def fit(self, train_classifiers, estimate_probas, ratioVal, path, usePretrainedClassifs=True):
        """
           This procedure fits our model Economy

           INPUTS :
                X : Independent variabless
                Y : Dependent variable

           OUTPUTS :
                self.thresholds : dictionary of thresholds for each time step.
                self.transitionMatrices   : transition matrices for each sequence (t,t+1).
                self.P_yhat_y : confusion matrices
                self.indices  : indices associated to each time step and each interval
        """

        self.max_t = train_classifiers.shape[1] - 1

        step = int(self.max_t*self.sampling_ratio) if int(self.max_t*self.sampling_ratio)>0 else 1
        self.timestamps = [t for t in range(step, self.max_t + 1, step)]

        if self.use_complete_ECO_model:

            with open(self.pathECOmodel, 'rb') as input:
                fullmodel = pickle.load(input)



            for t in self.timestamps:
                self.classifiers[t] = fullmodel.classifiers[t]
            self.labels = fullmodel.labels

            for i,t in enumerate(self.timestamps):

                self.thresholds[t] = fullmodel.thresholds[t]

                self.indices[t] = fullmodel.indices[t]


                # compute simple transition matrices for t > min_t
                if (i>0):
                    self.transitionMatrices[t] = self.computeTransitionMatrix(self.indices[t-step*i], self.indices[t])

                # compute complex transition matrices
                if (t>= self.min_t + (self.order-self.min_t)*(self.order>self.min_t) and self.order != 1):
                    self.complexTransitionMatrices[t] = self.computeComplexTransitionMatrix([self.indices[t-i-1] for i in range(self.order-1)], self.indices[t])

                if (t>=self.min_t):
                    self.P_yhat_y[t] = fullmodel.P_yhat_y[t]
                    self.P_y_gamma[t] = fullmodel.P_y_gamma[t]


        else:

            ## Split into train and val
            Y_train = train_classifiers.iloc[:, 0]
            X_train = train_classifiers.loc[:, train_classifiers.columns != train_classifiers.columns[0]]
            Y_val = estimate_probas.iloc[:, 0]
            X_val = estimate_probas.loc[:, estimate_probas.columns != estimate_probas.columns[0]]

            

            

            # labels seen on the data set
            self.labels = Y_train.unique()


            # time step to start train classifiers
            starting_timestep = self.min_t

            # fit classifiers
            #self.fit_classifiers(X_train, Y_train, starting_timestep, usePretrainedClassifs, path)


            # iterate over all time steps 1..max_t
            for i,t in enumerate(self.timestamps):

                #compute thresholds and indices
                self.thresholds[t], self.indices[t] = self.computeThresholdsAndindices(X_val.iloc[:, :t])

                # compute simple transition matrices for t > min_t
                if (i>0):
                    self.transitionMatrices[t] = self.computeTransitionMatrix(self.indices[t-step*i], self.indices[t])
                # compute complex transition matrices
                if (t>= self.min_t + (self.order-self.min_t)*(self.order>self.min_t) and self.order != 1):
                    self.complexTransitionMatrices[t] = self.computeComplexTransitionMatrix([self.indices[t-i-1] for i in range(self.order-1)], self.indices[t])

                # compute confusion matricies
                if (t>=self.min_t):
                    self.P_yhat_y[t] = self.compute_P_yhat_y_gammak(X_val, Y_val, t, self.indices[t])

                    self.P_y_gamma[t] = self.compute_P_y_gamma(Y_val, self.indices[t])

            self.recodeTS(X_val)

            self.decisionProbs = self.change_decision_probas()
            self.priorPreds = self.probas_prior_predictions()

    def change_decision_probas(self):
        #probabilities = {(gamma_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for gamma_k in range(self.nbIntervals)}
        probabilities = defaultdict(int)

        # load files
        with open(op.join(self.folderRealData, self.dataset, 'ep_preds.pkl'), 'rb') as inp:
            predictions = pickle.load(inp)

        for t_star in self.timestamps:
            for gamma_k in range(self.nbIntervals):
                
                indices_gamma_k = self.recodedTS[self.recodedTS[t_star-1]==gamma_k+1].index.values
                for t_index in range(self.timestamps.index(t_star)+1, len(self.timestamps)):
                    t_prime = self.timestamps[t_index]  
                    
                    for y_star, y_prime in zip(predictions[t_star][indices_gamma_k], predictions[t_prime][indices_gamma_k]):
                        probabilities[t_star, t_prime, gamma_k, y_star, y_prime] += 1
                    for y_star in self.labels:
                        for y_prime in self.labels:
                            pred = predictions[t_star][indices_gamma_k]
                            if len(pred[pred==y_star]) !=0:
                                probabilities[t_star, t_prime, gamma_k, y_star, y_prime] /= len(pred[pred==y_star])
        return  probabilities        
                    

    def probas_prior_predictions(self):
        
        probabilities = defaultdict(int)
        
        with open(op.join(self.folderRealData, self.dataset, 'ep_preds.pkl'), 'rb') as inp:
            predictions = pickle.load(inp)
            
        for t in self.timestamps:    
            for gamma_k in range(self.nbIntervals):
                indices_gamma_k = self.recodedTS[self.recodedTS[t-1]==gamma_k+1].index.values
                for y in predictions[t][indices_gamma_k]:
                    probabilities[t,y,gamma_k] += 1
                if len(indices_gamma_k) != 0:
                    probabilities[t,y,gamma_k] /= len(indices_gamma_k)
        return probabilities
    

    def recodeTS(self, X_val):
        X_val_new = X_val.copy()
        nb_observations, _ = X_val.shape

        if not self.feat:
            # We predict for every time series [label, tau*]
            for i in range(nb_observations):
                x = X_val.iloc[i, :].values
                intervals = self.findIntervals(x, self.max_t)
                for j in range(len(intervals)):
                    X_val_new.iloc[i, j] = intervals[j]
            X_val_new = X_val_new.astype(int)
        else:
            for t in self.timestamps:
                with open(op.join(self.folderRealData, self.dataset, 'ep_probas_'+str(t)+'.pkl'), 'rb') as inp:
                    X_val_t = pickle.load(inp)

                X_val_new.at[:,t] = self.findIntervalDataset(X_val_t,t)

        self.recodedTS = X_val_new
        self.recodedTS.columns = [i for i,col in enumerate(self.recodedTS.columns)]
        with open('CBFrecoded.pkl', 'wb') as rec:
            pickle.dump(self.recodedTS, rec)




    def computeThresholdsAndindices(self, X_val):

        """
           This procedure computes thresholds and indices of data associatied
           to each interval.

           INPUTS :
                X_val : validation data

           OUTPUTS :
                thresholds : dictionary of thresholds for each time step.
                indices  : indices associated to each time step and each interval
        """

        _, t = X_val.shape
        # Predict classes
        if self.fears:
            predictions = self.handle_my_classifier(t, transform_to_format_fears(X_val), proba=True)
            predictions = predictions.values
        elif self.feat:
            with open(op.join(self.folderRealData, self.dataset, 'ep_probas_'+str(t)+'.pkl') ,'rb') as inp:
                predictions = pickle.load(inp)
        else:
            predictions = self.classifiers[t].predict_proba(X_val)

        # Sort according to probabilities

        if self.feat:
            sortedProbabilities = [(i,val) for i,val in zip(np.argsort(predictions)[::-1], sorted(predictions, reverse=True))]
        else:
            sortedProbabilities = [(i,val) for i,val in zip(np.argsort(predictions[:, 1])[::-1], sorted(predictions[:, 1], reverse=True))]

        # equal frequence
        frequence = len(sortedProbabilities) // self.nbIntervals
        #compute thresholds
        thresholds = []
        indices = [[idx[0] for idx in sortedProbabilities[0:frequence]]]
        for i in range(1, self.nbIntervals):
            thresholds.append(sortedProbabilities[i*frequence][1])
            if (i==self.nbIntervals):
                indices.append([idx[0] for idx in sortedProbabilities[i*frequence:]])
            else:
                indices.append([idx[0] for idx in sortedProbabilities[i*frequence:(i+1)*frequence]])

        return thresholds,indices


    def compute_P_yhat_y_gammak(self, X_val, Y_val, timestep, indicesData):
        """
           This function computes P_t(ŷ/y,c_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep
           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        occurences = {}

        # initialise probabilities to 0
        probabilities = {(gamma_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for gamma_k in range(self.nbIntervals)}

        # Iterate over intervals
        for gamma_k in range(self.nbIntervals):

            indices_gamma_k = indicesData[gamma_k]
            # Subset of Validation set in interval gamma_k
            X_val_ck = X_val.loc[indices_gamma_k,:]

            # Subset of Validation set in interval gamma_k
            if (X_val_ck.shape[0]>0):
                if self.fears:
                    predictions = self.handle_my_classifier(timestep, transform_to_format_fears(X_val_ck.iloc[:, :timestep]))
                elif self.feat:
                    with open(op.join(self.folderRealData, self.dataset, 'ep_preds_'+str(timestep)+'.pkl') ,'rb') as inp:
                        predictions = list(pickle.load(inp))
                        predictions = [predictions[ii] for ii in indices_gamma_k]


                else:
                    predictions = self.classifiers[timestep].predict(X_val_ck.iloc[:, :timestep])

                for y_hat, y in zip(predictions, Y_val.loc[indices_gamma_k]):
                    # frequenceuence
                    probabilities[gamma_k, y, y_hat] += 1
        # normalize
        for gamma_k, y, y_hat in probabilities.keys():
            Y_val_gamma = Y_val.loc[indicesData[gamma_k]]

            # number of observations in gammak knowing y
            sizeCluster_gamma = len(Y_val_gamma[Y_val_gamma==y])

            if (sizeCluster_gamma != 0):
                probabilities[gamma_k, y, y_hat] /= sizeCluster_gamma

        return probabilities


    
        


    def compute_P_y_gamma(self, Y_val, indicesData):
        """
           This function computes P_t(y|gamma_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached
                indicesData  : indices of data associated to each interval / timestep
           OUTPUTS :
                probabilities : P_t(ŷ/y,gamma_k)

        """

        # Initialize all probabilities with 0
        probabilities = {(gamma_k, y):0 for y in self.labels for gamma_k in range(self.nbIntervals)}

        for gamma_k,e in enumerate(indicesData):
            for ts in e:
                probabilities[gamma_k, Y_val.iloc[ts]] += 1

            if len(e) != 0:
                for y in self.labels:
                    probabilities[gamma_k, y] /= len(e)
        return probabilities




    def computeComplexTransitionMatrix(self, indices_t_preced, indices_t):

        """
           This function computes a transition matrix between indices_t_preced and
           indices_t. (N^delta x N) matrix


           INPUTS :
                indices_t_preced : indices of data individuals in each interval at
                                   time steps considered in the past
                indices_t        : indices of data individuals in each interval at
                                   time step t

           OUTPUTS :
                transMatrix : transition matrix

        """
        transMatrix = np.zeros((self.nbIntervals**self.order, self.nbIntervals))

        # generate possible sequences to have the same order (useful for
        # transition matrix 1..t = t+1)
        possibleSequences = generatePossibleSequences(self.nbIntervals, self.order)

        for i in range(self.nbIntervals**self.order):
            possibleSequence = possibleSequences[:,i]
            # compute the ratio of the time series in gamma_j that were in gamma_i
            for j in range(self.nbIntervals):
                proportion = 0
                for e in indices_t[j]:
                    cond = [e in indices_t_preced[ord][possibleSequence[ord]] for ord in range(self.order-1)]
                    #if all true
                    if sum(cond)==self.order-1:
                        proportion += 1
                size = [len(indices_t_preced[ord][possibleSequence[ord]]) for ord in range(self.order-1)]
                transMatrix[i][j] = proportion/sum(size)
        return transMatrix



    def computeTransitionMatrix(self, indices_t_preced, indices_t):
        """
           This function computes a transition matrix between indices_t_preced and
           indices_t.


           INPUTS :
                indices_t_preced : valdiation data
                indices_t        : timestep reached

           OUTPUTS :
                transMatrix : prior probabilities of label y given a cluster ck.

        """
        transMatrix = np.zeros((self.nbIntervals, self.nbIntervals))
        for i in range(self.nbIntervals):
            for j in range(self.nbIntervals):
                proportion = 0
                for e in indices_t[j]:
                    if e in indices_t_preced[i]:
                        proportion += 1
                if (len(indices_t_preced[i]) > 0):
                    transMatrix[i][j] = proportion/len(indices_t_preced[i])
        return transMatrix

    


    def forecastExpectedCost(self, x_t, pb, decisions, history=False):
        """
           This function computes expected cost for future time steps given
           a time series xt


           INPUTS :
                x_t : time series

           OUTPUTS :
                totalCosts : list of (max_t - t) values that contains total cost
                             for future time steps.
        """
        t_current = len(x_t)
        cost_estimation_t = {'C_m': [], 'C_cd': [], 'history': [], 'C_total': []}

        # we initialize total costs with time cost
        forecastedCosts = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]

        send_alert = True
        p_gamma = np.zeros((1,self.nbIntervals**self.order))

        if (self.order != 1):
            # compute p
            intervals = np.array(self.findIntervals(x_t)) - 1
            seqs = generatePossibleSequences(self.nbIntervals, self.order).T
            # seqs as index in pandas important
            index = 0
            for i,e in enumerate(seqs):
                if (np.array_equal(e,intervals)):
                    index = i
                    break

            # if order != 1, we introduce a transition matrix to take into consideration order time steps in the past
            transitionMatrix = self.complexTransitionMatrices[t_current][i,:]
            transitionMatrix = transitionMatrix.reshape(1,self.nbIntervals)

        else:
            # compute p
            if self.feat:
                interval = self.findInterval(x_t, pb)
            else:
                interval = self.findInterval(x_t)
            p_gamma[:, interval-1] = 1

            # we just take into consideration the current instant
            transitionMatrix = p_gamma
            
        
        cost_cd = []
        cost_time = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]
        for i,t in enumerate(self.timestamps[self.timestamps.index(t_current):]):

            # Compute p_transition
            if (t>t_current):
                transitionMatrix = np.matmul(transitionMatrix, self.transitionMatrices[t])
            #iterate over intervals
            for gamma_k in range(self.nbIntervals):
                #iterate over possible labels
                for y in self.labels:
                    P_y_gamma = self.P_y_gamma[t]
                    # iterate over possible predictions
                    for y_hat in self.labels:
                        tem = transitionMatrix[:, gamma_k] * self.P_yhat_y[t][gamma_k, y, y_hat] * P_y_gamma[gamma_k, y] * self.misClassificationCost[y_hat][y]
                        forecastedCosts[i] += tem[0]
    
            cost_estimation_t['C_m'].append(forecastedCosts[i]-self.timeCost[t])

            if decisions:

                for gamma_k in range(self.nbIntervals):
                    for y_prime in self.labels:
                        tm = transitionMatrix[:, gamma_k] * self.decisionProbs[decisions[-1][0], t, gamma_k, decisions[-1][1], y_prime] * self.changeDecisionCost[decisions[-1][1]][y_prime] 
                        #print(transitionMatrix[:, gamma_k] , self.decisionProbs[decisions[0], t, gamma_k, decisions[1], y_prime] , self.changeDecisionCost[decisions[1]][y_prime], '     :' ,tm)
                        forecastedCosts[i] += tm[0]
                cost_estimation_t['C_cd'].append(forecastedCosts[i]-self.timeCost[t]-cost_estimation_t['C_m'][i]) 

                if len(decisions) > 1 and history:
                    for j in range(len(decisions)-1):
                        forecastedCosts[i] += self.changeDecisionCost[decisions[j+1][1]][decisions[j][1]]
                    cost_estimation_t['history'].append(forecastedCosts[i]-self.timeCost[t]-cost_estimation_t['C_m'][i]-cost_estimation_t['C_cd'][i]) 

            else:
                cost_estimation_t['C_cd'].append(0)  


            if (i>0):
                if (forecastedCosts[i] < forecastedCosts[0]):
                    send_alert = False
            cost_estimation_t['C_total'].append(forecastedCosts[i])
                    
        return send_alert, cost_estimation_t


    def findIntervalDataset(self, X,t):
        l=[]
        for proba in X:
            l.append(self.determineInterval(t, proba))
        return l
    def determineInterval(self, t_current, proba):
        # search for interval given probability
        ths = self.thresholds[t_current]
        for i,e in enumerate(sorted(ths, reverse=False)):
            if (proba <= e):
                return self.nbIntervals - i
        return 1
    def findInterval(self, x_t, pb=None):
        """
           This function finds interval associated with a timeseries given its
           probability


           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                interval of x_t
        """
        # we could use binary search for better perf
        t_current = len(x_t)
        # predict probability
        if self.fears:
            probadf = self.handle_my_classifier(t_current, transform_to_format_fears(numpy_to_df(x_t)), proba=True)
            proba = probadf['ProbNone1'].values[0]
        elif self.feat:
            proba=pb

        else:
            probadf = self.classifiers[t_current].predict_proba(x_t.reshape(1, -1))
            proba = probadf[0][1] # a verifier



        # search for interval given probability
        ths = self.thresholds[t_current]
        for i,e in enumerate(sorted(ths, reverse=False)):
            if (proba <= e):
                return self.nbIntervals - i
        return 1
        # nbInt = 3
        # 1
        # 2
        # 3



    def findIntervals(self, x_t):
        """
           This function finds intervals associated with a timeseries given its
           probabilities for last order values


           INPUTS :
                proba : probability given by the classifier at time step t

           OUTPUTS :
                interval of x_t
        """
        intervals = []
        t = len(x_t)
        for i in range(self.order):
            intervals.append(self.findInterval(x_t[:t-i]))
        intervals.reverse()
        return intervals

    def predict_revocable(self, X_test, oneIDV=None, donnes=None, variante='sans_cout_moindre', history=False):
        """
        This function predicts for every time series in X_test the optimal time
        to make the prediction and the associated label.

        INPUTS :
            - X_test : Independent variables to test the model

        OUTPUTS :
            - predictions : list that contains [label, tau*] for every
                            time series in X_test.

        """
        if not oneIDV:
            nb_observations, _= X_test.shape
        predictions = []
        if donnes != None:
            test_probas, test_preds = donnes

        # We predict for every time series [label, tau*]
        costs_timestamps = {}
        costs_rev = []
        decisions = []

        

        if oneIDV:
            cost_estimation = {}
            for t in self.timestamps:
                # first t values of x
                #x = np.array(list(X_test.iloc[i, :t]))
                x = X_test.values[:t]
                
                # compute cost of future timesteps (max_t - t)
                if self.feat:
                    probass = test_probas[t]
                    proba = probass
                    if decisions:
                        send_alert, cost = self.forecastExpectedCost(x,proba,decisions, history)
                    else:
                        send_alert, cost = self.forecastExpectedCost(x,proba,None,history)
                else:
                    send_alert, cost = self.forecastExpectedCost(x)
                costs_timestamps[t] = cost
                cost_estimation[t] = cost

                if variante=='sans_cout_moindre':
                    if send_alert:
                        if decisions:
                            if decisions[-1][1] != test_preds[t]:
                                decisions.append((t,test_preds[t]))
                        else:
                            decisions.append((t,test_preds[t]))
                else:
                    if send_alert:
                        if decisions:
                            if decisions[-1][1] != test_preds[t] and cost['C_total'][0] < lastcost:
                                decisions.append((t,test_preds[t]))
                        else:
                            lastcost = cost['C_total'][0]
                            decisions.append((t,test_preds[t]))
                    
        else:
            decisions = []
            cost_estimation = []
            for i in range(nb_observations):
                cost_individual = {}
                dec = []
                rev_cost = []
                # The approach is non-moyopic, for each time step we predict the optimal
                # time to make the prediction in the future.
                for t in self.timestamps:
                    # first t values of x
                    #x = np.array(list(X_test.iloc[i, :t]))
                    x = X_test.iloc[i, :t].values

                    # compute cost of future timesteps (max_t - t)
                    if self.feat:
                        probass = test_probas[t]
                        proba = probass[i]
                        if dec:
                            send_alert, cost = self.forecastExpectedCost(x,proba,dec,history)
                        else:
                            send_alert, cost = self.forecastExpectedCost(x,proba,None,history)
                    else:
                        send_alert, cost = self.forecastExpectedCost(x)

                    cost_individual[t] = cost
                    if variante=='sans_cout_moindre':
                        if send_alert:
                            if dec:
                                if dec[-1][1] != test_preds[t][i]:
                                    dec.append((t, test_preds[t][i]))
                            else:
                                
                                dec.append((t, test_preds[t][i]))
                    else:
                        if send_alert:
                            if dec:
                                if dec[-1][1] != test_preds[t][i] and cost['C_total'][0] < lastcost:
                                    dec.append((t, test_preds[t][i]))
                            else:
                                lastcost = cost['C_total'][0]
                                dec.append((t, test_preds[t][i]))
                decisions.append(dec)
                cost_estimation.append(cost_individual)
        return decisions, cost_estimation
                        

