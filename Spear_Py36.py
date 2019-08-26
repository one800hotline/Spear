{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 0
}

import datetime as dt
import os as os



def f_welcome(exec_f
              ,platform):
    
    if exec_f:
        import os as os
        import datetime as dt
        import getpass

        print ("------------------------------------------")
        print ("Loading %s utility platform!" % platform)
        print ("------------------------------------------")
        
        # Set some parameters
        date = dt.datetime.today().strftime("%y-%m-%d")
        time = dt.datetime.today().strftime("%H:%M:%S")
        time_h = dt.datetime.today().hour
        
        # Are we fetching user (Linux) or username (Windows)?
        try:
            user = getpass.getuser()
        except:
            pass

        try:
            user = getpass.getuser()
        except:
            pass
        
        # Some nice welcoming rules based on time of day
        if 6 < time_h <= 12:
            
            print ("Good morning, %s, the time is %s and the date is: %s" % (user, time, date))
            
        elif 12 < time_h <= 18:
            
            print ("Good afternoon, %s, the time is %s and the date is: %s" % (user, time, date))
        
        elif 18 < time_h <= 24:
            
            print ("Good evening, %s, the time is %s and the date is: %s" % (user, time, date))
            
        else:
            
            print ("Its late, go home... ")
        
    else:
        print ("No execution, ending...")
        
        
f_welcome(True
         ,'Spear')
    
    


#----------------------------------
# SPEAR functions
#----------------------------------    


def f_model_performance_shap(exec_f
                             ,array_y
                             ,array_y_pred
                             ,tuple_model_obj
                             ,list_feat_names
                             ,indata_X_train
                             ,indata_X_test=None
                             ,param_force_plt=None):

    """
    This function looks at a model prediction, and its actuals ground truth label, and takes out the biggest differences (calculated as 'actual'-'prediction'). Given these, and the biggest deviations, an overview of the models SHAP
    values is given, and a detailed break-down of the prediction given the most influential features. 
    
    Parameters:
    
    array_y                    Numpy array/Pandas Series (n*1): Actual ground truth labels, i.e. target variable
    array_y_pred               Numpy array/Pandas Series (n*1): Model object predictions
    tuple_model_obj            Tuple. String/sklearn estimator object/**kwarg (3*1): Tuple containing model object with estimated parameters from indata_X_trn and a description of model type. Two options are supported, linear and tree-based.
                                                                                     If 'linear' is passed as parameter 0, parameter 3 needs to be either 'independent or 'correlation'
    indata_X_trn               Numpy array/Pandas DataFrame (n*m): Design matrix where one row is an observation and one column is an descriptive feature.
    indata_X_test              Numpy array/Pandas DataFrame or None (n*m): Hold-out test/validation data from model object estimation process. If None, train data is used for SHAP values
    list_feat_names            List of strings (n*1): List of strings with all feature names going into model object estimation
    param_force_plt            Scalar. integer/string/None: Parameter for controling force plot. If string ('max'), first element of max model prediction difference is selected from DataFrame. If integer, index row is selected from Train/Test DataFrame. 
                                                            If None, force plot is skipped.
    
    """

    if exec_f:

        #-------------------
        # Base parameters
        #-------------------

        # Modules
        import pandas as pd
        import numpy as np
        import shap
        from IPython.core.display import HTML, display
        
        try:
            # Time
            print ("DateTime now is: {}".format(f_dt_now()))
            
        except:
            print ("Time flies....")
        
        tick=time.time()

        
        shap.initjs()
        
        # Indata
        X_train=indata_X_train
        X_test=indata_X_test
        
        # Target labels and model predictions
        y_actual=array_y
        y_pred=array_y_pred

        # list_feat_names
        list_feat_names=list_feat_names

        #-----------------------------
        # Model actual vs prediction
        #-----------------------------

        # Difference in prediction
        array_diff_pred_actual=y_actual-y_pred

        # Matrix with index, difference, prediction and actual value
        array_diff_pred_actual=np.array(list(zip([idx for idx in range(0, len(y_actual))]
                                            ,array_diff_pred_actual
                                            ,y_actual
                                            ,y_pred
                                                )
                                            )
                                       )

        # DataFrame sorted to show top 20 biggest differences
        df_top_diff_pred_vs_actual=pd.DataFrame(array_diff_pred_actual).sort_values(by=1)
        df_top_diff_pred_vs_actual.columns=['idx_row', 'diff_pred_vs_actual', 'y_actual', 'y_pred']
        df_top_diff_pred_vs_actual.reset_index(drop=True, inplace=True)

        # Push top 10 rows with biggest differences
        display(df_top_diff_pred_vs_actual[:10])

        #-----------------------------------
        # Model explanation: SHAP values
        #-----------------------------------
        def f_shap_values():

            print ("\nGenerating SHAP values, and visualizations given parameters...\n")

            # No test data, only train 
            if X_test==None:

                # We use X_train to create SHAP values
                X_shap_input=X_train

            # We have test data!
            else:

                # We use X_train to create SHAP values
                X_shap_input=X_test

            # Get shap values, given data specified
            shap_values = explainer.shap_values(X_shap_input)

            #------------------------------------------------------
            # Summary plot of model variable importance, 
            # and force plot for individual prediction break-down
            #------------------------------------------------------
            
            # Summary plot
            display(shap.summary_plot(shap_values, X_shap_input, feature_names=list_features_X))

            #----------------------------------
            # Force plot for top diff actual 
            # vs. prediction IDX row
            #----------------------------------

            # Pull out a specifix idx value, given data, and check its prediction break-down given input features from model object above and SHAP values
            def f_force_plot():
                display(shap.force_plot(explainer.expected_value
                                ,shap_values[idx,:]
                                ,X_shap_input[idx,:]
                                ,feature_names=list_features_X)
                       )

            # Skip plot
            if param_force_plt==None:
                
                    print ("\nNo execution of prediction row force plot, ending....\n")

            # Force plot for user specified actual vs. prediction IDX row
            elif type(param_force_plt)==int:

                try:
                    # index row, and force plot to follow
                    idx=param_force_plt
                    f_force_plot()

                # Some error in the provided index
                except ValueError:
                    UserWarning('Wrong input for data extraction to force_plot, ending.....')
                    
            # We pull the value with biggest difference
            elif (type(param_force_plt)==str):
                
                if param_force_plt=='max': 
                    idx=int(df_top_diff_pred_vs_actual.loc[0, 'idx_row'])
                    f_force_plot()
                    
                else:
                    print ("'param_force_plt' string value can only be 'max', re-check parameter.....")

                

        #-----------------------------------------------------
        # We have a linear model, i.e. something a long 
        # side logistic regression, linear regression, etc.
        #-----------------------------------------------------            
        if tuple_model_obj[0]=='linear':

            try:
                # Explainer for linear model
                explainer = shap.LinearExplainer(tuple_model_obj[1], X_train, feature_dependence=tuple_model_obj[2])
                
                # Generate SHAP values and visualizations
                f_shap_values()

            except:
                print ("Input to LinearExplainer is wrong, check parameters for 'tuple_model_obj'.")

        #-----------------------------------------------------------
        # We have a tree-based model, i.e. something like boosted or 
        # bagged trees (RandomForrest, XGBoost, LGBM, Catboost, etc...)
        #-----------------------------------------------------------
        elif tuple_model_obj[0]=='tree':

            explainer=shap.TreeExplainer(tuple_model_obj[1])

            # Generate SHAP values and visualizations
            f_shap_values()

        # Catch the rest....
        else:
            raise UserWarning("\nModel input for SHAP calculation not correctly specified, ending....\n")

        
        return df_top_diff_pred_vs_actual


        # Time elapsed
        print ("Total elapsed time is: {}s".format(round(time.time()-tick, 2)
                                                  )
              )

    else:
        print ("No execution of function, ending....")
          






def f_test_mean(exec_f
                ,indata
                ,list_test_x
                ,list_test_y
                ,bool_print_result
                ):

    """
    Function for using linear model to test sample.mean on "A/B" test population 
    
    indata                   Data holding variables to test, and target separating the groups
    list_test_x              List of string(s): indicator (dummy) flag which separates the two groups
    list_test_y              List of string(s): The outcome variable whoes.mean we want to test between the two groups in 'list_test_y'
    bool_print_result        Boolean(True/False): Print result or not
    """
    
    if exec_f:
        
        # Module needed
        import statsmodels.api as sm
        
        # copy up
        df_temp=indata.copy()
        
        # One test is done: x vs  y on.mean
        if isinstance(list_test_y,list) and len(list_test_y)==1:
        
            X=df_temp[list_test_x]

            y=df_temp[list_test_y]

            model=sm.OLS(y, X)
            results=model.fit()
            
            print(results.summary())

        # a List is provided for outcomes to test, i e  multipple y, a loop is performed over each of them  Also for multipple x
        elif isinstance(list_test_y, list) and len(list_test_y) > 1:
            
            list_hold_all=list()
            
            list_list_hold_para_n=list()
            list_list_hold_para_v=list()
            list_list_hold_para_y=list()
            list_hold_t_val=list()
            list_hold_p_val=list()
            
            for idx, var_y in enumerate(list_test_y):
                
                print (idx, var_y)
                
                X=df_temp[list_test_x]
                y=df_temp[var_y]
                
                model=sm.OLS(y, X)
                
                result=model.fit()
                
                list_list_hold_para_n.append(result.params.index[0])
                list_list_hold_para_n.append(result.params.index[1])
                
                list_list_hold_para_y.append(list_test_y[idx])
                list_list_hold_para_y.append(list_test_y[idx])
                
                list_list_hold_para_v.append(result.params[0])
                list_list_hold_para_v.append(result.params[1])
                
                list_hold_t_val.append(result.tvalues[0])
                list_hold_t_val.append(result.tvalues[1])
                
                list_hold_p_val.append(result.pvalues[0])
                list_hold_p_val.append(result.pvalues[1])
                
                if bool_print_result:
                
                    print("-------------------------------------------------------------------------------------")
                    print ("Test of outcome n %s, y: %s vs  x: %s" % ((idx + 1)
                                                                      ,var_y
                                                                      ,list_test_x)
                          )
                    print("-------------------------------------------------------------------------------------")
                    print (result.summary())
                    print ("\n")
    
    
            out=pd.DataFrame({'para' : list_list_hold_para_n
                                ,'para_y' : list_list_hold_para_y
                                ,'para_v' : list_list_hold_para_v
                                ,'para_t' : list_hold_t_val
                                ,'para_p' : list_hold_p_val})
            
            return (out)
        
    else:
        print ("No exectuion of function, ending     ")





def f_test_z_1prop(success_sample, n_sample, pop_mean=0.5, one_sided=False):

    """
    This functions performs a one-sample (proportion) z-test vs  a stated population hypothesis 
    
    In-parameters:
    
    sucecess_sample           Scalar, integer: Nr of 'positive' outcomes (or nr of 'bads') of the given test/treatment
    n_sample                  Scalar, integer: Nr of subjects for sample, i e  'n'
    pop_mean                  Population proportion being tested
    one_sided                 Boolean (True/False): If False, one sided hypothesis, else two-sided
    """
    
    
    import numpy as np
    import scipy.stats
    
    
    se=pop_mean*(1-pop_mean)/n_sample
    p_1=success_sample/n_sample
    se=np.sqrt(se)
    z=(p_1-pop_mean)/se
    p_val=1-scipy.stats.norm.cdf(abs(z))
    p_val *= 2-one_sided # if not one_sided: p *= 2

    print(' z-stat={z} \n p-value={p}'.format(z=z,p=p_val))



def f_test_z_2prop(x1, n1, x2, n2, one_sided=False):
    
    """
    This functions performs a two-sample (proportion) z-test, in a 'A/B''test type of set-up, on the difference between the proportions of sample 1.and 2 
    
    In-parameters:
    
    x1.                       Scalar, integer: Nr of 'positive' outcomes (or nr of 'bads') of the given test/treatment in sample 1
    n1.                       Scalar, integer: Nr of subjects, i e  'n', in sample 1
    x2                        Scalar, integer: Nr of 'positive' outcomes (or nr of 'bads') of the given test/treatment in sample 2
    n2                        Scalar, integer: Nr of subjects, i e  'n', in sample 2
    one_sided                 Boolean (True/False): If False, one sided hypothesis, else two-sided
    """
    
    
    import numpy as np
    import scipy.stats
    
    
    p1=x1/n1
    p2=x2/n2    
    
    p=(x1+x2)/(n1+n2)
    se=p*(1-p)*(1/n1+1/n2)
    se=np.sqrt(se)
    
    z=(p1-p2)/se
    p_val=1-scipy.stats.norm.cdf(abs(z))
    p_val *= 2-one_sided # if not one_sided: p *= 2
    
    print(' z-stat={z} \n p-value={p}'.format(z=z,p=p_val))
        
    #return z, p







def f_roc_curve(exec_f,target, prediction):

    """
    Receiver operating characterisitc curve (ROC) for a binart mode, to illustrate lift

    Parameters:

    array_target                       pd.Series or np.array (dim n): Actual array_target labels
    array_prediction                   pd.Series or np.array (dim n): Predicted, by model, array_target label. Make sure to use predict-method and not predict proba.

    """
    
    if exec_f:
        
        # Modules
        import numpy as np
        import pandas as pd
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        
        # Arrays
        array_target=target
        array_prediction=prediction

        # Pre-processsing of input types to be able to handle separate objects coming in, i e  Pandas Series or numpy ndarray;
        if isinstance(array_prediction, pd.Series):
            array_prediction=array_prediction.as_matrix()
            print ("array_prediction converted to array")
            print ("\n")

        elif isinstance(array_prediction, np.ndarray):
            print ("Model array_prediction %s is of correct form, proceeding   " % type(array_prediction))
            print ("\n")

        else: 
            print("Ending exeuction due to not having a correct variable type")
            print ("\n")

        if isinstance(array_target, pd.Series):
            array_target=array_target.as_matrix()
            print ("array_target converted to array")
            print ("\n")

        elif isinstance(array_target, np.ndarray):
            print ("Model array_target variable %s is of correct form, proceeding   " % type(array_target))
            print ("\n")

        else:
            print("Ending exeuction due to not having a correct variable type")
            print ("\n")


        # ROC-curve;
        fpr=dict()
        tpr=dict()
        thresholds=dict()
        roc_auc=dict()

        # Object holding train array_target and itv;
        array_target=[array_target]
        array_prediction=[array_prediction]

        # Itteration over train and ITV to calculate metrics;
        for i in range(len(array_target)):
            fpr[i], tpr[i], thresholds[i]=roc_curve(array_target[i], array_prediction[i])
            roc_auc[i]=auc(fpr[i], tpr[i])

        # Plot train and ITV ROC curve;
        for j in range(len(array_target)):    
            plt.figure()
            plt.figure(figsize=(12, 6))
            plt.plot(fpr[j], tpr[j], label='ROC curve (area=%0.2f)' % roc_auc[j])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC curve')
            plt.legend(loc="lower right")
            plt.show()
            
    else:
        print ("No execution of ROC-curve function, ending....")


def f_prec_recall_curve(exec_f, array_target, array_pred):

    """
    This function plots a precision/recall over the whole model population, alongside a univariate average precision/recall measurement.
    
    Parameters:

    array_target                       np.array (dim n): Actual target labels
    array_pred                         np.array (dim n): Predicted, by model, target label. Make sure to use predict-method and not predict proba.
    
    
    """

    if exec_f:

        import numpy as np
        import pandas as pd
        from sklearn.metrics import precision_recall_curve
        from sklearn.metrics import average_precision_score
        import matplotlib.pyplot as plt
        from inspect import signature

        # Indata
        array_tmp_target=array_target
        array_tmp_prediction=array_pred


        #----------------
        # Univariate
        #----------------
        print ("Average precision is: {}".format(average_precision_score(array_tmp_target
                                                                         ,array_tmp_prediction
                                                                        )
                                                )
              )

        #--------------------------------
        # Plot Precision/Recall curve
        #--------------------------------

        fig, ax=plt.subplots(ncols=1
                            ,nrows=1
                            ,figsize=(12,6))

        precision, recall, _ = precision_recall_curve(array_tmp_target, array_tmp_prediction)

        # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Binary-class Precision/Recall curve: AP={0:0.2f}'.format(average_precision_score(array_tmp_target
                                                                                                    ,array_tmp_prediction
                                                                                                   )
                                                                           )
                 )

    else:
        print ("No execution of precision/recall curve, ending....")




def f_lift_table(target
                ,prediction
                ,n_bins
                ,print_desc=False):

    """
    Lift table for binary classification output  To review model performance over the whole population
    for a classifier with a predict-method 

    Parameters:

    target         Array/Series, integer (binary): Actual target labels
    prediction     Array/Series, float (continious): Predicted class label probability from classifier built on target
    n_bins         Scalar(Integer) or List: How many groups, standard deciles (demi-deciles) given scalar OR binned given list input on range (0,1)
    print_desc     Boolean (True/False)): Print.info, or not    

    """
    import pandas as pd
    import numpy as np


    # Pre-processsing of input types to be able to handle separate objects coming in, i e  Pandas Series or numpy ndarray;
    
    #-----------------------------
    # Check prediction object type
    #-----------------------------
    if isinstance(prediction, pd.Series):
        print ("Model prediction %s is of correct form, proceeding   " % type(prediction))
        
    elif isinstance(prediction, np.ndarray):
        print ("Model prediction %s is of incorrect form, changing   " % type(prediction))
        prediction=pd.Series(prediction)
        
    else: 
        print("Ending exeuction due to not having a correct variable type")

    #-----------------------------
    # Check prediction object type
    #-----------------------------
    if isinstance(target, pd.Series):
        print ("Model target variable %s is of correct form, proceeding   " % type(target))

    elif isinstance(target, np.ndarray):
        print ("Model prediction %s is of incorrect form, changing   " % type(prediction))
        target=pd.Series(target)
        
    else:
        print("Ending exeuction due to not having a correct variable type")
        
        
    print ("Target is type: %s" % type(target))
    print ("Prediction is type: %s" % type(prediction))
     
    print ("\n")
        
    target=target.reset_index(drop=True)
    
    # Combine to make calculations
    common=pd.concat([target, prediction], axis=1, ignore_index=True, sort=False)
    
    #.rename to make code match
    common=pd.DataFrame(common.rename(columns={common.columns[0] : 'target'
                                                   ,common.columns[1]:'prediction'})
                         )

    # List for holding labels
    num_label=list()
    
    # Scalar
    if isinstance(n_bins, int):
        for i in range(1, (n_bins-1)):
            num_label.append(str(i))

    # List
    elif isinstance(n_bins, list):
        for i in range(1, (len(n_bins))):
            num_label.append(str(i))
        

    common=common.sort_values(by='prediction')
    
    #-------------------------------------
    # Check bins type input, given scalar
    # or list we apply qcut or cut
    #-------------------------------------
    
    # If scalar --> equally sized bins given integer
    if isinstance(n_bins, int):
        common['bin']=pd.qcut(common['prediction'], n_bins)

    elif isinstance(n_bins, list):
        common['bin']=pd.cut(common['prediction'], n_bins, include_lowest=True)


    # Copy it and proceed
    common2=common.copy()

    
    #Unique values + Rank order index (1.being lowest score decile (if 10.groups), 10.being the highest)
    common_mask=common2['bin'].drop_duplicates().reset_index()
    common_mask['rank_order']=(common_mask.index+1)

    #.merge back rank order index
    common_desc=common2.merge(common_mask[['bin', 'rank_order']], how='inner', left_on='bin', right_on='bin')


    #-----------------------------------------
    # Target, i e  actual stat 
    #-----------------------------------------
    actual=common_desc[['rank_order','target']].groupby(['rank_order']).agg([pd.Series.count, np.sum]).reset_index()
    actual['bad_tot_sum'], actual['pop_tot_sum']=actual['target']['sum'].cumsum(), actual['target']['count'].cumsum()
    actual['actual_bad_rate']=actual['target']['sum']/actual['target']['count']

    #-----------------------------------------
    # Prediction stat
    #-----------------------------------------
    prediction=common_desc[['rank_order','prediction']].groupby(['rank_order']).agg([np.min, np.max, np.mean]).reset_index()
    
    #---------------------------------------------
    # badrate distribution and % cummulative.sum
    #---------------------------------------------
    bads=actual['target']['sum']/np.sum(actual['target']['sum'])
    bad_cumsum=np.cumsum(bads.sort_index(ascending=False))
    
    # fix metadata
    bads.rename('badrate_dist'
               ,inplace=True)
    
    bad_cumsum.rename('badrate_cumsum'
                     ,inplace=True)

    #----------------------------------------------------------
    # population overall distribution and % cummulative.sum
    #----------------------------------------------------------
    pop=actual.iloc[:, 4]/np.sum(actual.iloc[:, 4])
    pop_cumsum=np.cumsum(pop.sort_index(ascending=False))
    
    # fix metadata
    pop.rename('pop_dist'
              ,inplace=True)
    pop_cumsum.rename('pop_cumsum'
                     ,inplace=True)

    
    #----------------------------------------------------------
    # Do we want to look at the names of the tables?
    #----------------------------------------------------------
    if print_desc:
    
        print ("\n")
        print ("-----------------------------")
        print ("|         PREDICTIONS       |")
        print ("-----------------------------")
        prediction.info()
        print ("\n")
        print ("-----------------------------")
        print ("|          ACTUALS          |")
        print ("-----------------------------")
        actual.info()
        
        
    # Common table
    common_lift=pd.concat([actual
                             ,prediction['prediction']
                             ,bads
                             ,pop
                             ,bad_cumsum
                             ,pop_cumsum]
                            ,axis=1, sort=False).sort_index(ascending=False)
    
    #.rename
    list_col_to=['decile', 'n_count', 'y_count', 'y_cum_sum', 'n_cum_sum', 'y_actual_%', 'min_pred', 'max_pred', 'mean_pred', 'y_dist', 'n_dist', 'y_dist_cumsum', 'n_dist_cumsum']
    
    common_lift.rename(columns=dict(zip([col for col in common_lift.columns], list_col_to)), inplace=True)
    
    print ("Bad rate in data is: {}%".format(round(np.sum(target)/len(target),3)*100))
    return common_lift





def f_conf_mtrx(exec_f 
                ,y
                ,y_pred
                ,lvl_cutoff=10):
    
    """   
        Confusion matrix for reviewing false positive/true postive vs  false negative/true negative in a binary classification setting 
        
        
        Parameters:
        
        exec_f:                    If None then do not execute, else execute (for "commenting out" the function)
        lvl_cutoff:                If number of distinct levels in target > cut-off, raise error and dont execute
        target:                    Target vector
        predicted_target:          Predicted target class labels
    
    """
    if exec_f:
        
        import numpy as np
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        
        print ("Executing function   ")
        print ("\n")
        
        if len(np.unique(y)) > lvl_cutoff:
                
            raise ValueError("Distinct levels of classification target variable (%s) exceeeds threshold %s" % (len(np.unique(y)),lvl_cutoff))
                
        else:
                
            print ("Target variable, y, has %s distinct levels, proceeding with confusion matrix creation:" % (len(np.unique(y))))

            
            # Create subplots to hold 2 figures
            fig, ax=plt.subplots(nrows=1
                                   ,ncols=2
                                   ,figsize=(12, 4)
                                  )
        
            # Some options and control parameters
            fig.tight_layout()
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        
        
            #--------------------------------------------
            #    Absolute volumes
            #--------------------------------------------
            conf_mat=confusion_matrix(y_true=y
                                         ,y_pred=y_pred)

            
            ax[0].matshow(conf_mat
                       ,cmap=plt.cm.Blues
                       ,alpha=0.3)

            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    ax[0].text(x=j
                            ,y=i
                            ,s=conf_mat[i, j]
                            ,va='center'
                             ,ha='center'
                            ,fontsize=16)
                    

            #--------------------------------------------
            #    % distribution of absolut volumes
            #--------------------------------------------
            conf_mat_prc=conf_mat/len(y)

            
            ax[1].matshow(conf_mat_prc
                        ,cmap=plt.cm.Blues
                        ,alpha=0.3)

            precision=2
            for i in range(conf_mat_prc.shape[0]):
                for j in range(conf_mat_prc.shape[1]):
                    ax[1].text(x=j
                            ,y=i
                            ,s=str(round(conf_mat_prc[i, j]*100,precision)) + "%"
                            ,va='center'
                            ,ha='center'
                            ,fontsize=16)

                    
            plt.xlabel('Predicted % dist')
            plt.ylabel('Actual % dist')
            plt.show()
            
            return conf_mat, conf_mat_prc
                
    else:
        print ("No exeuction of function, ending   ")



def f_plt_learning_curve(algorithm_obj
                        ,title
                        ,model_x_train
                        ,y
                        ,ylim=None
                        ,cv=None
                        ,n_jobs=1
                        ,train_sizes=[0.20, 0.40, 0.60, 0.80, 1.0]):
    
    """
    ----------------------------------------------------------------------------------------------------------------------
    Plot of the learning curve with varying sizes on the training sample (recomended default is used, 5 groups) then 
    using k-fold cross validation for each size 
    
    Steps:          1. Sample train groups (Default 5 groups, with increasing size of data)
                    2  For each sample size, the following is performed
                           a  Training model on k-1.folds --> Output performance metric
                           b  Apply model on 1.fold of validation data --> output
                    3  Step 2a and 2b are repeated "cv times" for each sample size, average on performance metric is 
                       created for each sample size 

    
    Note:   What happends to the last 10% of the data when doing, e g  10.fold cross-validation? It dosent seemt to be a 
            part of the model training/validation cycle
    
    ----------------------------------------------------------------------------------------------------------------------

    algorithm_obj:                    Object type that implements the "fit" and "predict" methods, i e  logistic regression or SVM, etc   
                                    An object of that type which is cloned for each validation 

    title:                          As string 
                                    Title for the chart 

    model_x_train:                  array-like,.shape (n_samples, n_features)
                                    Training vector, where n_samples is the number of samples and
                                    n_features is the number of features 

    model_y_train:                  array-like,.shape (n_samples) or (n_samples, n_features)
                                    Target relative to model_x_train for classification or regression;
            
    ylim:                              tuple,.shape (ymin, ymax), optional
                                    Defines minimum and maximum yvalues plotted 

    cv:                             integer, cross-validation generator, optional
                                    If an integer is passed, it is the number of folds (defaults to 3) 

    n_jobs :                         integer, optional
                                    Number of jobs to run in parallel (default 1) 
        
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import learning_curve


    plt.figure(figsize=(12, 6))
    plt.title(title)
    
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    
    plt.ylabel("Score metric output")
    
    plt.rcParams['axes facecolor']='0.99'
    
    print ("N rows in training data in is: %s" % len(model_x_train))
    print ("\n")
    
    print ("Varying train sizes to be used is:")
    print (train_sizes)
    print ("\n")
    
    
    # Compute arrays from sampling and estimation of test/validation outpyt, given model (with a specific performance metric) and train_sizes array 
    train_sizes, train_scores, validation_scores =learning_curve(algorithm_obj
                                                                    ,model_x_train
                                                                    ,y
                                                                    ,cv=cv
                                                                    ,n_jobs=n_jobs
                                                                    ,train_sizes=train_sizes)

    print ("If it feels confusing, have a look at page 175 in Raschka, 2015")
    print (train_sizes)
    print ("\n")
    print ("Shape size on train_score matrix is:")
    print (train_scores.shape)
#     print (train_scores)
    print ("\n")
    print ("Shape size on validation_score matrix is:")
    print (validation_scores.shape)
#     print (validation_scores)
    print ("\n")
    
    # Stats on each training/validation output
    train_scores_mean=np.mean(train_scores, axis=1)
    train_scores_std=np.std(train_scores, axis=1)
    validation_scores_mean=np.mean(validation_scores, axis=1)
    validation_scores_std=np.std(validation_scores, axis=1)
    
    
    plt.grid()

    # Train: Set fill colour, given standard deviation and.mean 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    
    # Validation: Set fill colour, given standard deviation and.mean 
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    
    return plt
    
        
    
def f_partial_plot(exec_f
                   ,X
                   ,y
                   ,feat_names):
    
    """
        Partial plot for continious variable and binary classifcation target 
        
        exec_f:                    Boolean (True/False)  If False then do not execute, else execute (for "commenting out" the function)        
        X:                         Feature matrix (D number of features)
        y:                         Target variable
        feat_names:                Names of.columns in feature matrix (D number of names for feature matrix)
    """
    
    if exec_f:
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
    
        print ("Executing function   ")
        print ("\n")

    
        if len(np.unique(y)) == 2:

            print ("Binary classication target variable  Number of distinct levels: %s" % len(np.unique(y)))

            from sklearn.ensemble import GradientBoostingClassifier

            clf_tree=GradientBoostingClassifier(random_state=0.1
                                                                    ,n_estimators=100
                                                                    ,max_depth=6
                                                                    ,max_features=8
                                                                    ,min_samples_split=2)
            fit_clf=clf_tree.fit(X, y)
            y_pred_proba = fit_clf.predict_proba(X)
            y_pred = fit_clf.predict(X)


            features=[i for i in range(X.shape[1])]

            names=feat_names

            plt.figure()

            figure, axs=plot_partial_dependence(fit_clf
                                                  ,X
                                                  ,features
                                                  ,feature_names=names
                                                  ,figsize=(15, 9)
                                                 )

            plt.subplots_adjust(top=1.2)                                 
            plt.show()

        elif len(np.unique(y)) > 2:

            print ("Continious target variable  Number of distinct values: %s" % len(np.unique(y)))

            from sklearn.ensemble import GradientBoostingRegressor

            clf_tree=GradientBoostingRegressor(random_state=1)
            fit_clf=clf_tree.fit(X, y)
            y_pred = fit_clf.predict(X)


            features=[i for i in range(X.shape[1])]

            names=feat_names

            plt.figure()

            figure, axs=plot_partial_dependence(fit_clf
                                                  ,X
                                                  ,features
                                                  ,feature_names=names
                                                  ,figsize=(15, 9))

            plt.subplots_adjust(top=1.2)                                 
            plt.show() 
            
            
    else:
        print ("No exeuction of function, ending   ")
        
    


def f_desc_stat(exec_f, dist):

    '''
    Compute and present simple descriptive stats for a dist

    Parameter:
    
    exec_f      Boolean(True/False): Execute or not
    dist        List/Array: Distribution to be analyzed

    Reference:

    https://www kdnuggets com/2017/05/descriptive-statistics-key-terms-explained html

    '''

    if exec_f:
        import numpy as np
        import seaborn as sns
        
        # Convert dist as numpy ndarray
        dist=np.array(dist)

        print ('Descriptive statistics for dist:\n', dist)
        print ('\nNumber of scores:', len(dist))
        print ('Number of unique scores:', len(np.unique(dist)))
        print ('Sum:',sum(dist))
        print ('Min:', min(dist))
        print ('Max:', max(dist))
        print ('Range:', max(dist)-min(dist))
        print ('Mean:', np.mean(dist, axis=0))
        print ('Median:', np.median(dist, axis=0))
        print ('Mode:', scipy.stats.mode(dist)[0][0])
        print ('Variance:', np.var(dist, axis=0))
        print ('Standard deviation:', np.std(dist, axis=0))
        print ('1st quartile:', np.percentile(dist, 25))
        print ('3rd quartile:', np.percentile(dist, 75))
        print ('dist skew:', scipy.stats.skew(dist))

        
        sns.distplot(dist)        
        plt.title('Distribution of score')
        plt.show()
        
    else:
         print ("No execution of function for reviewing dist, ending   ")
        
        
        

class c_swoe_iv(object):
    
    """
    Parameters and formula for calculating smoothed Weight of Evidence (sWoE) and.information value (IV) for binary 
    target variable and given target variable  Requires import of Numpy and Panda modules:
    
    In parameters:
    
        data_in:       Input data, containing target variable and category variable
        target:        Name of target variable
        cat:           Category variable of choice
        smooth_p:      Smoothing parameter used to normalize the.mean value  Pre-selected value is 24
    
    Calculated parameters:
        cat_n:         Number of entities in given varible category 
        cat_sum:      .sum of target variable in given variable category
        
        prop_bad:      Proportion bad in cat i 
        prop_good:     Proportion good in cat i
        
       .mean_t:       .mean for target variable in development data

    Formula: 

        sWoE(i)=(Cat_n +.mean_target * smooth_param)/(cat_n - cat_sum + (1.-.mean_target) * smooth_param)
        IV(i)=(prop_good - prop_bad) * WoE
        WoE(i)=np.log(prop_good/prop_bad)
        
    Outparameters:
    
        IV=The.information value for the given category variable (or binned continious variable)
    
    """

    def __init__(self, data_in):
        self.data_in=data_in
        
    def swoe_iv(self
                        ,target_var
                        ,cat
                        ,smooth_p):
                        
        print ("Exectuting WoE estimation on input parameters")
        print ("\n")
        
        self.target_var=target_var
        self.mean_t=self.data_in[target_var].mean()
        self.smooth_p=smooth_p
        
        print ("Mean value for data: %s and smoothing parameter: %s" % (self.mean_t, self.smooth_p))
        
        # Group covariate with target and calculate N in each bin and.sum(target);
        self.woe_holder=self.data_in.groupby([cat]).agg([len, np.sum])[target_var]
        self.woe_holder['cat_good']=(self.woe_holder['len'] - self.woe_holder['sum'])

        #.rename to better convention;
        self.woe_holder.rename(columns={'len': 'cat_n', 'sum': 'cat_bad'}, inplace=True)
        
        #.sum the total number of elements, and the bad elements in data for given category;
        tot_n=(self.woe_holder['cat_n'].sum())
        tot_bad=(self.woe_holder['cat_bad'].sum())

        # Proportion bad and good - nice twist on the calculation there     ;
        self.woe_holder['prop_bad'], self.woe_holder['prop_good']=(self.woe_holder['cat_bad']/tot_n), (self.woe_holder['cat_good']/tot_n)

        print ("Total n is: %s and the total number of bads is: %s" % (tot_n, tot_bad))
        print ("\n")

        # Calculate WoE value for each category;
        self.woe_holder['swoe_' + cat]=(self.woe_holder['cat_bad'] + self.mean_t * self.smooth_p)/((self.woe_holder['cat_n'] - self.woe_holder['cat_bad']) + (1-self.mean_t) * self.smooth_p)

        # Logarithm of woe value;
        self.woe_holder['swoe_log_' + cat]=np.log(self.woe_holder['swoe_' + cat])
     
  
        # Calculate IV value for each category, if inf or nan then 0;
        self.woe_holder['IV_' + cat]=(self.woe_holder['prop_good'] - self.woe_holder['prop_bad']) * (np.log(self.woe_holder['prop_good']/self.woe_holder['prop_bad']))
       
        
        #.replace inf with 0.for IV;
        self.woe_holder['IV_' + cat]=self.woe_holder['IV_' + cat].replace(np.inf, 0)

        # Look through IV-vector and see if inf values exists, if True then.replace to 0;
        for i in range(1,len(np.isinf(self.woe_holder['IV_' + cat]))):
            if i == True:
                true_story=True

        # Given check above on itteratinon over rows;
        if true_story:
            print ("Inf function scan shows True value, replacing inf with 0.since no IV can be obtained for category i")
            print ("\n")
            self.woe_holder['IV_' + cat]=self.woe_holder['IV_' + cat].replace(np.inf, 0)
        else:
            print ("No inf value, proceeding with output")
            print ("\n")
        
        #.sum the IV to represent the IV-Value of the 
        print ("IV-value for variable: %s is %s" % (cat, self.woe_holder['IV_' + cat].sum()))
        print ("\n")
    
        # Output table is set to same name as parameter cat;
        self.woe_holder.info()
        
        # reset index in order to be able to join with original categorical variable in the input data frame;
        self.woe_holder.reset_index(level=0
                                    ,inplace=True
                                   )

        # Spit it out! 
        return self.woe_holder 




