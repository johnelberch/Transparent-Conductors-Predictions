## Script to ease the elaboration of models and their fitting

###########################
## P R E L I M I N A R S ##
###########################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Type, Optional, List, Union

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, _search
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


#######################################
## H E L P I N G : F U N C T I O N S ##
#######################################

## RMSLE Score
def rmsle(y_true : np.ndarray, y_pred : np.ndarray):
    power = np.power ( np.log(y_pred + 1) - np.log(y_true + 1) , 2)
    return np.sqrt(1/len(y_true) * np.sqrt(np.sum(power)))

score_rmsle = make_scorer(rmsle, greater_is_better=False)

## Plot from results
def Build_Summary_CV( results : _search.GridSearchCV):
    #List of params to plot
    List_of_params = [par[17:] for par in list(results.cv_results_.keys()) if 'param_estimator' in par]

    #Create an empty Dictionary
    Dictio = {}

    #Populate the dictionary
    for param in List_of_params:
        Dictio[param] = results.cv_results_[f'param_estimator__{param}'].data.astype('float64')

    # Get summaries
    Dictio['mean_test_score'] = results.cv_results_['mean_test_score']
    Dictio['std_test_score'] = results.cv_results_['std_test_score']

    return(pd.DataFrame(Dictio))


#################################
## DataFit : N E W : C L A S S ##
#################################
class DataFit:
    '''
    This is a class to help streamline the entire model-fitting process and make it in the cleanest way possible
    '''

    ## Initialize the class, the main attributes to consider are the DataFrame object, as well as lists of the continous, categorical inputs to consider, and the output to focus on
    def __init__(self, df : pd.DataFrame, continous_inputs : List[str], categorical_inputs : List[str], output : str, use_PCA : bool = False):
        #Create a copy of the originally provided DataFrame and select only the variables we will focus on. Store this new DataFrame into the df attribute
        self.df = df.loc[:,continous_inputs + categorical_inputs + [output]].copy()

        #Convert the categorical inputs into a categorical data type
        self.df.loc[:,categorical_inputs] = self.df.loc[:,categorical_inputs].astype("category")

        #Store the lists of inputs by type, as well as the outputs
        self.continous = continous_inputs
        self.categorical = categorical_inputs
        self.output = output

        #Define the resampling scheme
        self.my_cv = StratifiedKFold(n_splits=5, random_state=101, shuffle=True)

        #Define the scoring metric
        self.score = 'neg_mean_squared_error'
        # self.score = score_rmsle
        

        self.use_PCA = use_PCA


    ###########
    ## Define a method that will prepare the preprocessing based on user-defined pipelines
    def DefinePreprocessing(self, n_components : int = 5, remainder : Optional[Union[str, Pipeline]] = 'drop', _PolyFeatures : bool = False, consider_PCA = False):
        '''
        Method that stores Pipelines with basic preprocessing operations for inputs, outputs, and categorical variables
        ColumnTransformer objects will be used to standarize the input and output variables when fitting the models (to avoid potential data leakage)
        '''

        # Define the main preprocessing Pipeline for the categorical variables
        cat_transform = Pipeline(steps = [ ('dummy', OneHotEncoder(drop='first')) ])

        # Work with a list first (appending as needed) and then create the Pipeline for continous inputs
        _num_steps = [('std_input', StandardScaler())]

        # Get the PCA
        if consider_PCA == False:
            use_PCA = False
        else:
            use_PCA = self.use_PCA


        if use_PCA == True:
            _num_steps.append( ('PCA', PCA(n_components= n_components)) )
        
        if _PolyFeatures ==True:
            _num_steps.append(('Polynomial_Interactions', PolynomialFeatures(degree=2)))

        if use_PCA == True or _PolyFeatures == True:
            _num_steps.append(('std_features', StandardScaler()))

        num_transform = Pipeline(steps = _num_steps)

        Preprocessing = ColumnTransformer( transformers = [ ('Continous_inputs' , num_transform, self.continous),
                                                            ('Categorical_inputs', cat_transform, self.categorical)],
                                           remainder = remainder)
            
        return (Preprocessing)
        

    ###########
    ## Define a method that will take care of the splitting for us
    def DefineSplit(self):
        '''
        Since StratifiedKfold doesn't automatically recognize the groups in the training data (specially once the OneHotEncoding is performed), we need to use the split() method to provide a cross-validation generator
        (https://scikit-learn.org/stable/glossary.html#term-CV-splitter)
        This method will take the identified categorical variable and use it to generate the stratified k-fold scheme
        '''

        # Select the categorical data to use
        X_cat = self.df.loc[:,self.categorical].to_numpy(dtype=str).ravel()

        # Build the splits and return the indices
        Split_Generator = self.my_cv.split(self.df, X_cat)

        return(Split_Generator)


    ###########
    ## Define a function for the first ENET
    def FirstEnet(self):
        #Basic Estimator → Elastic Net
        _Estimator = ElasticNet(fit_intercept = True, max_iter = 50000)

        #Param_grid → For the enet, we test the l1 ratio and alpha
        _Grid = {'estimator__l1_ratio' : np.linspace(0.00001,1, num=3),
                     'estimator__alpha' : np.exp( np.linspace(-5, 5, num=5))}

        #Score → RMSLE, but Sklearn has the MSLE available, will transform it later
        Score = self.score

        #Build the Preprocessing ColumnTransformer
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=False, consider_PCA = True)

        #Complete Estimator → preprocess first,  then fit the enet
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._FirstEnet_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.FirstEnet_results = self._FirstEnet_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())


    ###########
    ## Define a function for the second ENET
    def SecondEnet(self):
        #Basic Estimator → Elastic Net
        _Estimator = ElasticNet(fit_intercept = True, max_iter = 50000)

        #Param_grid → For the enet, we test the l1 ratio and alpha
        _Grid = {'estimator__l1_ratio' : np.linspace(0.00001,1, num=3),
                     'estimator__alpha' : np.exp( np.linspace(-5, 5, num=5))}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=True, consider_PCA = True)

        #Complete Estimator → preprocess first,  then fit the enet
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._SecondEnet_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.SecondEnet_results = self._SecondEnet_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())
        
    
    ###########
    ## Define a function for the Neural Network
    def NN(self):
        #Basic Estimator → Neural Network
        _Estimator = MLPRegressor(activation='tanh', solver='sgd', batch_size = 25,
                                   max_iter=5001, learning_rate_init=0.001, random_state=101)

        #Param_grid → For the NN, we test the number of hidden layers, the batch size and the alpha
        # nn_grid = {'nn__hidden_layer_sizes' : np.linspace(5,30, num=6, dtype=int),
        #            'nn__batch_size' : np.linspace(5,30, num=6, dtype=int),
        #            'nn__alpha' : 10**(-np.linspace(0,5, num=6))}

        _Grid = {'estimator__hidden_layer_sizes' : [3,5],
                 'estimator__alpha' : [0.0001, 0.01]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=False, consider_PCA = True)

        #Complete Estimator → preprocess first,  then fit the NN
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._NN_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.NN_results = self._NN_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())
        
        
    ###########
    ## Define a function for the SVM
    def SVM(self):
        #Basic Estimator → Supported Vector Machine Regressor
        _Estimator = SVR(gamma='scale', max_iter = 100000, kernel='poly')

        #Param_grid → For the SVM we fit the degree of the polynomial and the C. We could also test the kernel (more computational cost)
        _Grid = {'estimator__degree' : [2,3],
                 'estimator__C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=False, consider_PCA = True)

        #Complete Estimator → preprocess first,  then fit the SVM
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._SVM_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.SVM_results = self._SVM_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())
        

    ###########
    ## Define a function for the Random Forest
    def RF(self):
        #Basic Estimator → Random Forest
        _Estimator = RandomForestRegressor()

        _Grid = {'estimator__n_estimators' : [25,100,250,400],
                 'estimator__max_features' : [1,2,3]}

        #Score → RMSLE, but Sklearn has the MSLE available, will transform it later
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=False, consider_PCA = False)

        #Complete Estimator → preprocess first,  then fit the enet
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._RF_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.RF_results = self._RF_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())
        

    ###########
    ## Define a function for the Gradient Boosted Trees
    def GBM(self):
        #Basic Estimator → GB regressor
        _Estimator = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1)

        _Grid = {'estimator__n_estimators' : [25,100,250,400],
                 'estimator__max_depth' : [1,3,6],
                 'estimator__learning_rate' : [0.1/5, 0.1, 0.5]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(remainder='drop', _PolyFeatures=False, consider_PCA = False)

        #Complete Estimator → preprocess first,  then fit the enet
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._GBM_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.GBM_results = self._GBM_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
                                                          y=self.df.loc[:,self.output].copy())
        

    ###########
    ## Define a wrapper method capable of doing all the fits at once
    def FitAll(self):
        # Fit each model, one by one
        if self.use_PCA == False:
            for i , model in enumerate([self.FirstEnet(), self.SecondEnet(), self.NN(), self.SVM(), self.RF(), self.GBM()]):
                model
        else:
            for i , model in enumerate([self.FirstEnet(), self.SecondEnet(), self.NN(), self.SVM()]):
                model
            


    ############
    ## Define a function to plot all the results
    def PlotAll(self):

        #Item list of results (depends on the use of PCA)
        if self.use_PCA == False:
            Item_list = [self.FirstEnet_results,
                        self.SecondEnet_results,
                        self.NN_results,
                        self.SVM_results,
                        self.RF_results,
                        self.GBM_results]
            
            Names = ['First Enet', 'Second Enet', 'Neural Network', 'Supported Vector Machine', 'Random Forest', 'Gradient Boosted']
        else:
            Item_list = [self.FirstEnet_results,
                        self.SecondEnet_results,
                        self.NN_results,
                        self.SVM_results]
            
            Names = ['First Enet', 'Second Enet', 'Neural Network', 'Supported Vector Machine']
            
        # Store the scores
        Scores = [item.best_score_ for item in Item_list]
        
        # Get the standard error in the scores
        Errors = [item.cv_results_['std_test_score'][np.argmin(self.FirstEnet_results.cv_results_['rank_test_score'])]
                  for item in Item_list]


        # Find the best score overall to highlight in the figure
        Best_Score_overall = max(Scores)
        Min_index = Scores.index(Best_Score_overall)

        # Colors for the bars
        colors = np.array([[152/255,251/255,152/255,0.4] for i in range(len(Scores))])
        colors[Min_index,:] = [50/255,205/255,50/255,1]

        #Plot
        _ , ax = plt.subplots(figsize=(6,4))
        ax.bar(Names, Scores, yerr=Errors, color = colors, edgecolor='black')
        ax.tick_params(axis='x',which='both',rotation=90)
        ax.set_title('Scores of all models')
        ax.set_ylabel('Score = neg RMSLE')
        plt.tight_layout()
        plt.show()

        self.Performances = [i for i in zip(Names,Scores)]


    ############
    ## Define a function for the random forest, that will give you the most important feature
    #Code based on the example provided in https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    def EvaluateRandomForest(self):
        #Retrieve the forest from the best estimator case
        forest =  self.RF_results.best_estimator_.named_steps['estimator']

        #Retrieve the importances for that forest
        importances = forest.feature_importances_

        #Compute the standard error for the importances
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

        #Get the feature names
        feature_names = self.RF_results.best_estimator_.named_steps.preprocessing.get_feature_names_out()

        #Store results in a Series
        forest_importances = pd.Series(importances, index=feature_names)

        #Plot
        _, ax = plt.subplots(figsize = (8,6))
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        plt.tight_layout()
        plt.show()
        
    
    ############
    ## Define a function that will automate generating DataFrames with key information about each model
    def DataFrameForEachModel(self):
        #Item list of results (depends on the use of PCA)
        if self.use_PCA == False:
            Item_list = [self.FirstEnet_results,
                        self.SecondEnet_results,
                        self.NN_results,
                        self.SVM_results,
                        self.RF_results,
                        self.GBM_results]
            
            Names = ['First Enet', 'Second Enet', 'Neural Network', 'Supported Vector Machine', 'Random Forest', 'Gradient Boosted']
        else:
            Item_list = [self.FirstEnet_results,
                        self.SecondEnet_results,
                        self.NN_results,
                        self.SVM_results]
            
            Names = ['First Enet', 'Second Enet', 'Neural Network', 'Supported Vector Machine']

        # Put everything together
        return( [(name, Build_Summary_CV( result )) for (result, name) in zip(Item_list, Names)])
    

    ##########
    ## Automate the figures of the hyper parameters
    def PlotEachModel(self, height=2.5, aspect=1.5):

        DF_Results = self.DataFrameForEachModel()

        sns.relplot(data = DF_Results[0][1], x='alpha', y='mean_test_score', hue='l1_ratio', height=height,
                    palette='coolwarm', kind='line', aspect=aspect).set(title = DF_Results[0][0])

        plt.show()

        sns.relplot(data = DF_Results[1][1], x='alpha', y='mean_test_score', hue='l1_ratio', height=height,
                    palette='coolwarm', kind='line', aspect=aspect).set(title = DF_Results[1][0])

        plt.show()

        sns.relplot(data = DF_Results[2][1], x='alpha', y='mean_test_score', hue='hidden_layer_sizes', height=height,
                    palette='coolwarm', kind='line', aspect=aspect).set(title = DF_Results[2][0])

        plt.show()

        sns.relplot(data = DF_Results[3][1], x='C', y='mean_test_score', hue='degree', height=height,
                    palette='coolwarm', kind='line', aspect=aspect).set(title = DF_Results[3][0])

        plt.show()

        if self.use_PCA == False:
            sns.relplot(data = DF_Results[4][1], x='n_estimators', y='mean_test_score', hue='max_features', height=height,
                        palette='coolwarm', kind='line', aspect=aspect).set(title = DF_Results[4][0])

            plt.show()

            sns.relplot(data = DF_Results[5][1], x='n_estimators', y='mean_test_score', hue='max_depth', col = 'learning_rate', height=height,
                        palette='coolwarm', kind='line', aspect=aspect)

            plt.show()