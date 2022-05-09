from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial.distance import cdist
from Economy import Economy
from sklearn.base import clone
from scipy.linalg import norm
import os.path as op
try:
    import cPickle as pickle
except ImportError:
    import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import json
from collections import defaultdict



class Economy_K_REV(Economy):

    """
    Economy_K inherits from Economy

    ATTRIBUTES :

        - nbClusters : number of clusters.
        - lmbda      : parameter of our model estimating the membership probability
                       to a given cluster
        - clustering : clustering model
        - clusters   : list of clusters
        - P_y_ck     : prior probabilities of a label y given a cluster.
        - labels     : list of labels observed on the data set.

    """

    def __init__(self, misClassificationCost, timeCost, changeDecisionCost, min_t, classifier, nbClusters, sampling_ratio, use_complete_ECO_model, pathECOmodel, fears, dataset , feat, folderRealData):
        super().__init__(misClassificationCost, timeCost, min_t, classifier)
        self.nbClusters = nbClusters
        self.sampling_ratio = sampling_ratio
        self.use_complete_ECO_model = use_complete_ECO_model
        self.pathECOmodel = pathECOmodel
        self.fears = fears
        self.dataset = dataset
        self.feat = feat
        self.folderRealData = folderRealData
        self.ifProba = False
        self.changeDecisionCost = changeDecisionCost

    def clusteringTrainset(self, X_train):
        """
           This procedure performs a clustering of our train set

           INPUTS :
                X_train : train set

           OUTPUTS :
                self.clustering : clustering model.
                self.clusters : list of clusters.
        """
        ## Identify a finite set of clusters (Kmeans)
        self.clustering = KMeans(n_clusters=np.min([self.nbClusters,X_train.shape[0]]), init="k-means++", algorithm="elkan")
        self.clustering.fit(X_train)
        # list of clusters
        self.clusters = range(np.min([self.nbClusters,X_train.shape[0]]))





    def fit(self, train_classifiers, estimate_probas, ratioVal, path, usePretrainedClassifs=True):
        """
           This procedure fits our model Economy

           INPUTS :
                X : Independent variables
                Y : Dependent variable

           OUTPUTS :
                self.P_yhat_y : dictionary which contains for every time step the confusion matrix
                                associated to a given cluster at time step t.
                self.P_y_ck   : prior probabilities of a label y given a cluster.
        """
        self.max_t = train_classifiers.shape[1] - 1

        step = int(self.max_t*self.sampling_ratio) if int(self.max_t*self.sampling_ratio)>0 else 1
        self.timestamps = [t for t in range(step, self.max_t + 1, step)]

        if self.use_complete_ECO_model:

            with open(self.pathECOmodel, 'rb') as input:
                fullmodel = pickle.load(input)

            self.clusters = fullmodel.clusters
            self.clustering = fullmodel.clustering

            for t in self.timestamps:
                self.classifiers[t] = fullmodel.classifiers[t]
            self.labels = fullmodel.labels

            for t in self.timestamps:
                self.P_yhat_y[t] = fullmodel.P_yhat_y[t]
            self.P_y_ck = fullmodel.P_y_ck


        else:


            ## Split into train and val
            Y_train = train_classifiers.iloc[:, 0]
            X_train = train_classifiers.loc[:, train_classifiers.columns != train_classifiers.columns[0]]
            Y_val = estimate_probas.iloc[:, 0]
            X_val = estimate_probas.loc[:, estimate_probas.columns != estimate_probas.columns[0]]

            # labels seen on the data set
            self.labels = Y_train.unique()


            # perform clustering
            self.clusteringTrainset(X_train)

            # fit classifiers
            #self.fit_classifiers(X_train, Y_train, self.min_t, usePretrainedClassifs, path)


            # Compute probabilities (confusion matricies) for each time step
            for t in self.timestamps:
                self.P_yhat_y[t] = self.compute_P_yhat_y_ck(X_val, Y_val, t)

            # Compute prior probabilities given a cluster
            self.P_y_ck = self.compute_P_y_ck(X_val, Y_val)
            self.decisionProbs = self.change_decision_probas()


    def compute_P_ck_xt(self, x_t, clusters):
        """
           This function computes membership probablities to a cluster given a sequence
           of values.

           INPUTS :
                x_t      : sequence of values (time series).
                clusters : centers of our clusters.
                lmbda    : parameter of the logistic function

           OUTPUTS :
                probabilities : list of nbClusers values which contains the probabilty
                                of membership to the different clusters identified.
        """
        # Compute the average of distances beween x_t and all the clusters
        # using the euclidean distance.


        # the distances between x_t and all the clusters
        distances = cdist([x_t], clusters, metric='euclidean')
        for i,e in enumerate(distances[0]):
            if e < 0.000001:
                distances[0] = 0.000001

        distances = 1./distances
        probabilities = distances / np.sum(distances)
        return probabilities[0]






    def compute_P_y_ck(self, X_train, Y_train):
        """
           This function computes prior probabilities of label y given a cluster ck


           INPUTS :
                X_train, Y_train : train data

           OUTPUTS :
                probabilities : P(y|ck) prior probabilities of label y given a cluster ck
        """

        occurences = {}
        # Initialize all probabilities with 0
        probabilities = {(c_k, y):0 for y in self.labels for c_k in self.clusters}

        # for every observation in train set
        cl = self.clustering.predict(X_train)
        for c_k, y in zip(cl, Y_train):
            # compute frequence of (ck,y)
            probabilities[c_k, y] += 1
            # compute size of cluster ck
            occurences[c_k] = occurences.get((c_k), 0) + 1

        # normalize
        for c_k, y in probabilities.keys():
            # avoid div by 0
            if (c_k in occurences.keys()):
                probabilities[c_k, y] /= occurences[c_k]

        return probabilities







    def compute_P_yhat_y_ck(self, X_val, Y_val, timestep):
        """
           This function computes P_t(y/y,c_k)


           INPUTS :
                X_val, Y_val : valdiation data
                timestep     : timestep reached

           OUTPUTS :
                probabilities : probabilities of label y given a cluster ck.
        """

        ############## INITS
        occurences = {}
        probabilities = {}
        subsets = {}


        # clusters associated to time series
        # � modifier les noms de variables & noms de fonctions (id ou clusters etc)
        clusters_data = self.clustering.predict(X_val)

        # initialise probabilities to 0
        probabilities = {(c_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for c_k in self.clusters}

        # for each cluster we associate indices of data corresponding to this cluster
        indices_data_cluster = {c_k:[] for c_k in self.clusters}

        for index, value in enumerate(clusters_data):
            # indices id ?
            indices_data_cluster[value].append(index)
        self.indices_data_cluster = indices_data_cluster
        ############## OCCURENCES
        for c_k in self.clusters:

            indices_ck = indices_data_cluster[c_k]

            # Subset of Validation set in cluster C_k
            X_val_ck = X_val.iloc[indices_ck]

            if (len(indices_ck)>0):
                # predict labels for this subset
                if self.fears:
                    predictions = self.handle_my_classifier(timestep, transform_to_format_fears(X_val_ck.iloc[:, :timestep]))
                elif self.feat:
                    with open(op.join(self.folderRealData,self.dataset,'ep_preds_'+str(timestep)+'.pkl') ,'rb') as inp:
                        predictions = pickle.load(inp)
                        predictions = [predictions[ii] for ii in indices_ck]
                else:
                    predictions = self.classifiers[timestep].predict(X_val_ck.iloc[:, :timestep])

                for y_hat, y in zip(predictions, Y_val.iloc[indices_ck]):
                    # compute frequence
                    probabilities[c_k, y, y_hat] += 1

        ############## NORMALIZATION KNOWING Y
        for c_k, y, y_hat in probabilities.keys():

            # subset ck
            Y_val_ck = Y_val.iloc[indices_data_cluster[c_k]]
            # number of observations in this subset that have label y
            sizeCluster_y = len(Y_val_ck[Y_val_ck==y])
            if sizeCluster_y != 0:
                probabilities[c_k, y, y_hat] /= sizeCluster_y

        return probabilities



    def change_decision_probas(self):
        #probabilities = {(gamma_k, y, y_hat):0 for y in self.labels for y_hat in self.labels for gamma_k in range(self.nbIntervals)}
        probabilities = defaultdict(int)

        # load files
        with open(op.join(self.folderRealData, self.dataset, 'ep_preds.pkl'), 'rb') as inp:
            predictions = pickle.load(inp)

        for t_star in self.timestamps:
            for c_k in self.clusters:
                
                indices_c_k = self.indices_data_cluster[c_k]
                for t_index in range(self.timestamps.index(t_star)+1, len(self.timestamps)):
                    t_prime = self.timestamps[t_index]  
                    
                    for y_star, y_prime in zip(predictions[t_star][indices_c_k], predictions[t_prime][indices_c_k]):
                        probabilities[t_star, t_prime, c_k, y_star, y_prime] += 1
                    for y_star in self.labels:
                        for y_prime in self.labels:
                            pred = predictions[t_star][indices_c_k]
                            if len(pred[pred==y_star]) !=0:
                                probabilities[t_star, t_prime, c_k, y_star, y_prime] /= len(pred[pred==y_star])
        return  probabilities



    def forecastExpectedCost(self, x_t, pb, decisions, history=False):
        """
           This function computes expected cost for future time steps given
           a time series xt


           INPUTS :
                x_t : time series

           OUTPUTS :
                forecastedCosts : list of (max_t - t) values that contains total cost
                             for future time steps.
        """

        t_current = len(x_t)
        cost_estimation_t = {'C_m': [], 'C_cd': [], 'history': [], 'C_total': []}
        # compute membership probabilites
        P_ck_xt = self.compute_P_ck_xt(x_t, self.clustering.cluster_centers_[:, :t_current])

        # we initialize total costs with time cost
        forecastedCosts = [self.timeCost[t] for t in self.timestamps[self.timestamps.index(t_current):]]
        send_alert = True
        # iterate over future time steps

        for i,t in enumerate(self.timestamps[self.timestamps.index(t_current):]):
            # iterate over clusters

            for c_k, P_ck in zip(self.clusters, P_ck_xt):
                # iterate over possible labels
                for y in self.labels:
                    # iterate over possible predictions
                    for y_hat in self.labels:
                        forecastedCosts[i] += P_ck * self.P_y_ck[c_k, y] * self.P_yhat_y[t][c_k, y, y_hat] * self.misClassificationCost[y_hat-1, y-1]

            cost_estimation_t['C_m'].append(forecastedCosts[i]-self.timeCost[t])
            if decisions:

                for c_k, P_ck in zip(self.clusters, P_ck_xt):
                    for y_prime in self.labels:
                        tm = P_ck * self.decisionProbs[decisions[-1][0], t, c_k, decisions[-1][1], y_prime] * self.changeDecisionCost[y_prime-1][decisions[-1][1]-1] 
                        forecastedCosts[i] += tm
                cost_estimation_t['C_cd'].append(forecastedCosts[i]-self.timeCost[t]-cost_estimation_t['C_m'][i]) 

                if len(decisions) > 1 and history:
                    for j in range(len(decisions)-1):
                        forecastedCosts[i] += self.changeDecisionCost[decisions[j+1][1]-1][decisions[j][1]-1]
                    cost_estimation_t['history'].append(forecastedCosts[i]-self.timeCost[t]-cost_estimation_t['C_m'][i]-cost_estimation_t['C_cd'][i]) 

            else:
                cost_estimation_t['C_cd'].append(0)  


            if (i>0):
                if (forecastedCosts[i] < forecastedCosts[0]):
                    send_alert = False
                    break
            cost_estimation_t['C_total'].append(forecastedCosts[i])
        return send_alert, cost_estimation_t



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
                        send_alert, cost = self.forecastExpectedCost(x,None,decisions, history)
                    else:
                        send_alert, cost = self.forecastExpectedCost(x,None,None,history)
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
                            send_alert, cost = self.forecastExpectedCost(x,None,dec,history)
                        else:
                            send_alert, cost = self.forecastExpectedCost(x,None,None,history)
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
    

    

