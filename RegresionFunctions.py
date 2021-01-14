
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import RandomizedSearchCV


def test_regresion(regressor,X_train,X_test,y_train,y_test):
    regressor.fit(X_train,y_train)
    pred=regressor.predict(X_test)
    r2=r2_score(y_test,pred)
    MAE=mean_absolute_error(y_test,pred)
    MSE=mean_squared_error(y_test,pred)
    return r2,MAE,MSE

def cv_scorer(regressor,X,y,cv=5,scoring='r2'):
    clf = make_pipeline(StandardScaler(), regressor)
    cvs=cross_val_score(clf,X,y, cv=cv,scoring=scoring)
    return cvs.mean()

def cv_scorer_pca(regressor,X,y,value,cv=5,scoring='r2'):
    clf = make_pipeline(StandardScaler(),PCA(value), regressor)
    cvs=cross_val_score(clf,X,y, cv=cv,scoring=scoring)
    return cvs.mean()


def insert_into_summary(summaryDF, regression_model, label, value):
    row_loc = summaryDF.index
    colum_loc=summaryDF.columns
    summaryDF.iat[row_loc.get_loc(regression_model),colum_loc.get_loc(label)]=value

def regresion_analysis(regressor,X,y,label,summaryDF,cv=5,cv_score='r2',train=0.8):
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=train, random_state=1)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    X_train=pd.DataFrame(X_train,columns=list(X))
    X_test=pd.DataFrame(X_test,columns=list(X))
    print(label+': ')
    results=test_regresion(regressor,X_train,X_test,y_train,y_test)
    cv_score=cv_scorer(regressor,X,y,cv=5,scoring='r2')    
    insert_into_summary(summaryDF,label,'R2',results[0])
    insert_into_summary(summaryDF,label,'MAE',results[1])
    insert_into_summary(summaryDF,label,'MSE',results[2])
    insert_into_summary(summaryDF,label,'CV-R2',cv_score)



def FeatureSelector(stepF,X):
    RfcFeatureReport=pd.DataFrame.from_dict(stepF.get_metric_dict()).T
    RfcFeatureReport['avg_score']=RfcFeatureReport['avg_score'].astype(float)
    RfcFeatureReport.plot(y='avg_score',style='-r.')
    lst=list(RfcFeatureReport.iloc[RfcFeatureReport['avg_score'].idxmax()-1].feature_names)
    return lst

def stepFeatureSelect(X,y,regressor, num_features=10,direction=False):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    X_train=pd.DataFrame(X_train,columns=list(X))
    X_test=pd.DataFrame(X_test,columns=list(X))
    stepF=sfs(regressor,k_features=num_features, forward=direction, floating=False, verbose=2,scoring='r2', cv=3,n_jobs=-1).fit(X_train,y_train)
    return FeatureSelector(stepF,X)


def find_PCA(X):
    X_train,X_test=train_test_split(X,test_size=0.2,random_state=1)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    size=X.shape[1]
    summaryDF=pd.DataFrame(columns=['PCA%','num_features'], index=range(1,size))
    
    for value in range(1,size):
        pca = PCA(n_components=value)
        pca.fit(X_train)
        insert_into_summary(summaryDF,value,'PCA%',pca.explained_variance_ratio_.sum())
        insert_into_summary(summaryDF,value,'num_features',value)
    plt.figure(figsize=(10, 7))
    plt.axhline(y = .95, color='k', linestyle='--', label = '95% Explained Variance')
    plt.axhline(y = .90, color='c', linestyle='--', label = '90% Explained Variance')
    plt.axhline(y = .85, color='r', linestyle='--', label = '85% Explained Variance')
    plt.axhline(y = .99, color='g', linestyle='--', label = '99% Explained Variance')
    plt.plot(summaryDF['num_features'],summaryDF['PCA%'])
    plt.xlabel('Number of features')
    plt.ylabel('Variance')
    plt.show()

    return summaryDF


def test_PCA(X,y,regressor,val_dic):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    X_train=pd.DataFrame(X_train,columns=list(X))
    X_test=pd.DataFrame(X_test,columns=list(X))
    summaryDF=pd.DataFrame(columns=['PCA%','R2','MAE','MSE','CV-R2','num_features'], index=val_dic.keys())

    for value in val_dic.keys():
        pca = PCA(n_components=val_dic.get(value))
        pca.fit(X_train)
        X_test_pca = pca.transform(X_test)
        X_train_pca = pca.transform(X_train)

        results=test_regresion(regressor,X_train_pca,X_test_pca,y_train,y_test)
        cv_score=cv_scorer_pca(regressor,X,y,val_dic.get(value),cv=5,scoring='r2') 

        insert_into_summary(summaryDF,value,'R2',results[0])
        insert_into_summary(summaryDF,value,'MAE',results[1])
        insert_into_summary(summaryDF,value,'MSE',results[2])
        insert_into_summary(summaryDF,value,'CV-R2',cv_score)
        insert_into_summary(summaryDF,value,'PCA%',pca.explained_variance_ratio_.sum())
        insert_into_summary(summaryDF,value,'num_features',pca.n_components_)

    return summaryDF


def print_boxplots(data):
    number_of_plots=data.shape[1]
    number_of_columns=5
    number_of_rows=math.ceil(number_of_plots/number_of_columns)
    columns=list(data.columns)
    fig, axes = plt.subplots(number_of_rows,number_of_columns, figsize=(25, 3*number_of_rows))
    plt.subplots_adjust(hspace = 0.5)
    fig.suptitle('DATSET BOX PLOTS')
   

    for i in range(0,number_of_rows):
        for j in range(0,number_of_columns):
            index=i*number_of_columns+j
            if(index<number_of_plots):
                sns.boxplot(ax=axes[i, j], data=data, x=columns[index])
            else:
                fig.delaxes(axes[i][j])

    

def RandSearch(X,y,regressor,param_dict,cv=5,n_iter=10):
    clf = make_pipeline(StandardScaler(), regressor)
    search = RandomizedSearchCV(clf, param_dict, n_iter=n_iter, scoring='r2', n_jobs=-1, cv=cv, random_state=1)
    result=search.fit(X, y)
    return result.best_score_,result.best_params_


