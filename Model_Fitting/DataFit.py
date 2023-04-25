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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from tqdm import tqdm


#######################################
## H E L P I N G : F U N C T I O N S ##
#######################################

## RMSLE Score
def rmsle(y_true : np.ndarray, y_pred : np.ndarray):
    power = np.power ( np.log(y_pred + 1) - np.log(y_true + 1) , 2)
    return np.sqrt(1/len(y_true) * np.sqrt(np.sum(power)))

score_rmsle = make_scorer(rmsle, greater_is_better=False)


#################################
## DataFit : N E W : C L A S S ##
#################################
class DataFit:
    '''
    This is a class to help streamline the entire model-fitting process and make it in the cleanest way possible
    '''

    ## Initialize the class, the main attributes to consider are the DataFrame object, as well as lists of the continous, categorical inputs to consider, and the output to focus on
    def __init__(self, df : pd.DataFrame, continous_inputs : List[str], categorical_inputs : List[str], output : str):
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
        # self.score = score_rmsle
        self.score = 'neg_mean_squared_error'


    ###########
    ## Define a method that will prepare the preprocessing based on user-defined pipelines
    def DefinePreprocessing(self, PCA : bool = False, n_components : int = 7, remainder : Optional[Union[str, Pipeline]] = 'drop', _PolyFeatures : bool = False):
        '''
        Method that stores Pipelines with basic preprocessing operations for inputs, outputs, and categorical variables
        ColumnTransformer objects will be used to standarize the input and output variables when fitting the models (to avoid potential data leakage)
        '''

        # Define the main preprocessing Pipeline for the categorical variables
        cat_transform = Pipeline(steps = [ ('dummy', OneHotEncoder(drop='first')) ])

        # Work with a list first (appending as needed) and then create the Pipeline for continous inputs
        _num_steps = [('std_input', StandardScaler())]

        if PCA == True:
            _num_steps.append( ('PCA', PCA(n_components= n_components)) )
        
        if _PolyFeatures ==True:
            _num_steps.append(('Polynomial_Interactions', PolynomialFeatures(degree=2)))

        if PCA == True or _PolyFeatures == True:
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
        _Grid = {'estimator__l1_ratio' : np.linspace(0.00001,1, num=5),
                     'estimator__alpha' : np.exp( np.linspace(-6, 6, num=11))}

        #Score → RMSLE, but Sklearn has the MSLE available, will transform it later
        Score = self.score

        #Build the Preprocessing ColumnTransformer
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=False)

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
        _Grid = {'estimator__l1_ratio' : np.linspace(0,1, num=5),
                     'estimator__alpha' : np.exp( np.linspace(-6, 6, num=11))}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=True)

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
        _Estimator = MLPRegressor(activation='tanh', solver='sgd', 
                                   max_iter=5001, learning_rate_init=0.001, random_state=101)

        #Param_grid → For the NN, we test the number of hidden layers, the batch size and the alpha
        # nn_grid = {'nn__hidden_layer_sizes' : np.linspace(5,30, num=6, dtype=int),
        #            'nn__batch_size' : np.linspace(5,30, num=6, dtype=int),
        #            'nn__alpha' : 10**(-np.linspace(0,5, num=6))}

        _Grid = {'estimator__hidden_layer_sizes' : [3,4,5],
                   'estimator__batch_size' : [25,30],
                   'estimator__alpha' : [0.0001, 0.001]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=False)

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
        _Estimator = SVR(gamma='scale', max_iter = 100000)

        #Param_grid → For the SVM we fit the degree of the polynomial and the C. We could also test the kernel (more computational cost)
        _Grid = {'estimator__kernel' : ['poly'],
                 'estimator__degree' : [2,3],
                 'estimator__C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=False)

        #Complete Estimator → preprocess first,  then fit the SVM
        _Estimator_wflow = Pipeline(steps = [('preprocessing', Preprocessing),
                                           ('estimator', _Estimator)])
        
        #Define the GridSearchCV object
        self._SVM_grid = GridSearchCV(estimator = _Estimator_wflow,
                                        param_grid = _Grid,
                                        scoring=Score,
                                        cv=self.DefineSplit())

        #Fit
        self.SVM_results = self._svm_grid.fit(X=self.df.drop(self.output, axis=1).copy(),
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
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=False)

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
                 'estimator__learning_rate' : [0.1/5, 0.1, 0.5*5]}

        #Score → RMSLE
        Score = self.score

        #Build the column transformer for preprocessing
        Preprocessing = self.DefinePreprocessing(PCA=False, n_components=7, remainder='drop', _PolyFeatures=False)

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
        for model in tqdm([self.FirstEnet(), self.SecondEnet(), self.NN(), self.SVM(), self.RF(), self.GBM()]):
            model
        