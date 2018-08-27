import librosa
import os
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

def read_audio(path):
    """
    busca por todos os arquivos .wav em um diretório
    lê cada arquivo e divide o sinal do captcha a cada 2 segundos (quatro caracteres)

    path -- caminho para o diretorio contendo os arquivos .wav
    
    retorna -- lista de caracteres na forma de tuplas (sinal, taxa de amostragem, rotulo do caractere, rotulo do captcha)
    """
    wavs = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file.endswith('.wav')]
    chars = []
    for j,wav in enumerate(wavs):
        signal, sampling_rate = librosa.load(os.path.join(path,wav), None)
        labels = wav.split('.wav')[0]
        chars += [(signal[sampling_rate*2*i:sampling_rate*2*(i+1)],sampling_rate,labels[i],path+'_'+labels) for i in range(4)]
    return chars


def extract_features(chars):
    """
    extrai mfccs de cada caractere e computa estatisticas básicas
    
    chars -- lista de caracteres na forma de tuplas (sinal, taxa de amostragem, rotulo do caractere, rotulo do captcha)

    retorna -- tabela atributo valor
    """
    data = pd.DataFrame()
    for j, char in enumerate(chars):
        signal = char[0]
        sampling_rate=char[1]
        char_label=char[2]
        captcha_label  = char[3]
        row = pd.DataFrame()
        row['char_label'] = [char_label]
        row['captcha_label'] = [captcha_label]
        mfcc = librosa.feature.mfcc(signal,sampling_rate, n_mfcc=40)
        for i,mfcc in enumerate(mfcc):
            row['mfcc_'+str(i)+'_mode'] = [np.mean(sp.stats.mode(mfcc))]
            row['mfcc_'+str(i)+'_min'] = [np.min(mfcc)]
            row['mfcc_'+str(i)+'_max'] = [np.max(mfcc)]
            row['mfcc_'+str(i)+'_mean'] = [np.mean(mfcc)]
            row['mfcc_'+str(i)+'_std'] = [np.std(mfcc)]
            row['mfcc_'+str(i)+'_median'] = [np.median(mfcc)]
            row['mfcc_'+str(i)+'_iqr'] = [sp.stats.iqr(mfcc)]
            row['mfcc_'+str(i)+'_kutosis'] = [sp.stats.kurtosis(mfcc)]
            row['mfcc_'+str(i)+'_skewness'] = [sp.stats.skew(mfcc)]
        data = data.append(row)
    return data.reset_index(drop=True)


def select_model(data):
    """
    busca o melhor modelo e combinação de hiperparâmetros

    data -- dados para validação cruzada

    retorna -- ensemble dos melhores modelos
    """
    seed=42 
    
    X = data.drop(['captcha_label','char_label'],axis=1)
    y = data['char_label']

    estimators = [Pipeline(steps=[('scaler', StandardScaler())
                                  ,('selector',SelectKBest(score_func=mutual_info_classif)),
                                  ('model',KNeighborsClassifier())]),
              Pipeline(steps=[('scaler', StandardScaler()),
                              ('model',LogisticRegression())]),
              Pipeline(steps=[('model',RandomForestClassifier())])
                 ]
                  
    param_grids = [{'selector__k':[50,75,100],
                    'model__n_neighbors':[2,3,5], 
                    'model__weights':['distance'],
                    'model__p':[1,2]},
                   
               {'model__C':np.logspace(-4,4,9),
                'model__penalty':['l1','l2'],
                'model__random_state':[seed]},
                   
               {'model__max_depth':np.arange(2,20),
                'model__n_estimators':[500],
                'model__random_state':[seed]}]

    best_score = 0
    best_estimators=[]
    for estimator,param_grid in zip(estimators,param_grids):
        gridsearch = GridSearchCV(estimator,param_grid,scoring='accuracy',cv=5,error_score=0,verbose=True)
        gridsearch.fit(X,y)
        best_estimators.append(gridsearch.best_estimator_)
    final_estimator = VotingClassifier([('knn',best_estimators[0]),('lr',best_estimators[1]),('rf',best_estimators[2])],voting='soft')
    return final_estimator

def train_model(estimator,data):
    """
    treina o modelo no conjunto de dados

    estimator -- modelo a ser treinado
    
    data -- dados de treinamento

    retorna -- modelo treinado
    """
    X = data.drop(['captcha_label','char_label'],axis=1)
    y = data['char_label']
    estimator.fit(X,y)
    return estimator

def evaluate_model(estimator,data):
    """
    imprime report de avaliação do modelo no conjunto de dados
    
    estimator -- modelo 
    
    data -- dados de treinamento
    
    """
    X = data.drop(['captcha_label','char_label'],axis=1)
    y = data['char_label']
    y_pred = estimator.predict(X)
    accuracy = accuracy_score(y,y_pred)
    metrics = classification_report(y,y_pred)
    conf = pd.DataFrame(confusion_matrix(y,y_pred,labels=y.unique()),columns=y.unique(),index=y.unique())      
    hits = 0
    captchas = data['captcha_label'].unique()
    for captcha in captchas:
        X = data[data.captcha_label==captcha].drop(['captcha_label','char_label'],axis=1)
        y = data[data.captcha_label==captcha]['char_label']
        y_pred = estimator.predict(X)
        if np.equal(y_pred,y).sum() == 4:
            hits+=1
    print('confusion matrix' + '\n')
    print(conf)
    print('\n')
    print(metrics)
    print('\n' + 'accuracy {0:.2f}'.format(accuracy) + '\n')
    print('\n' + 'captchas hit rate ' + "{0:.2f}".format(hits/len(captchas)))


