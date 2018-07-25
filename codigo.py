import librosa
import pandas as pd
import numpy as np
import scipy as sp

import os
import time

from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import classification_report, confusion_matrix


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
        chars += [(signal[sampling_rate*2*i:sampling_rate*2*(i+1)],sampling_rate,labels[i],labels) for i in range(4)]
    return chars


def extract_features(chars, n_mfcc=20):
    """
    extrai n mfccs de cada caractere e computa as estatísticas:
        *moda
        *minimo
        *maximo
        *media
        *desvio padrao
        *mediana
        *intervalo interquartil
        *curtose
        *assimetria

    chars -- lista de caracteres na forma de tuplas (sinal, taxa de amostragem, rotulo do caractere, rotulo do captcha)
    n_mfcc -- numero de mfccs a serem extraidos

    retorna -- tabela atributo valor (linhas: len(caracteres), colunas: n_mfcc*9 + char_label + captcha_label)
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
        mfcc = librosa.feature.mfcc(signal,sampling_rate, n_mfcc=n_mfcc)
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


def train_model(train_data,valid_data,transformer,estimators,param_grids,metric):
    """
    busca o melhor modelo e combinação de hiperparâmetros

    train_data -- dados de treinamento
    valid_data -- dados de validacao
    estimators -- lista de modelos
    param_grid -- lista de dicionarios de hiperparâmetros

    retorna -- transformador ajustado, melhor modelo treinado, tabela de resultados
    """
    X_train = train_data.drop(['captcha_label','char_label'],axis=1)
    y_train = train_data['char_label']

    X_valid = valid_data.drop(['captcha_label','char_label'],axis=1)
    y_valid = valid_data['char_label']

    X_train = pd.DataFrame(transformer.fit_transform(X_train),columns=X_train.columns)
    X_valid = pd.DataFrame(transformer.transform(X_valid),columns=X_valid.columns)

    X = np.concatenate((X_train.values,X_valid.values),axis=0)
    y = np.concatenate((y_train.values,y_valid.values),axis=0)
    test_fold = []
    for i in range(len(X_train)):
        test_fold.append(-1)
    for i in range(len(X_valid)):
        test_fold.append(0)
    cv = PredefinedSplit(test_fold=test_fold)

    metrics = {'estimator':[],'training metric':[],'validation metric':[]}
    best_estimator = None
    best_score = 0
    for estimator,param_grid in zip(estimators,param_grids):
        gridsearch = GridSearchCV(estimator,param_grid,scoring=metric,cv=cv,return_train_score=True)
        gridsearch.fit(X,y)
        metrics['estimator'].append(str(gridsearch.best_estimator_).split('(')[0])
        metrics['validation metric'].append(gridsearch.best_score_)
        results = pd.DataFrame(gridsearch.cv_results_)
        metrics['training metric'].append(results[results.rank_test_score == 1]['mean_train_score'].values[0])
        if gridsearch.best_score_ > best_score:
            best_score = gridsearch.best_score_
            best_estimator = gridsearch.best_estimator_
    best_estimator.fit(X_train,y_train)
    return transformer, best_estimator, pd.DataFrame(metrics)

def evaluation_per_character(transformer,estimator,data):
    X = data.drop(['captcha_label','char_label'],axis=1)
    y = data['char_label']
    X = pd.DataFrame(transformer.transform(X),columns=X.columns)
    y_pred = estimator.predict(X)
    metrics = classification_report(y,y_pred)
    conf = pd.DataFrame(confusion_matrix(y,y_pred,labels=y.unique()),columns=y.unique(),index=y.unique())
    return metrics, conf

def evaluation_per_captcha(transformer,estimator,data):
    accuracy = 0
    captchas = data['captcha_label'].unique()
    for captcha in captchas:
        X = data[data.captcha_label==captcha].drop(['captcha_label','char_label'],axis=1)
        y = data[data.captcha_label==captcha]['char_label']
        X = pd.DataFrame(transformer.transform(X),columns=X.columns)
        y_pred = estimator.predict(X)
        if np.equal(y_pred,y).sum() == 4:
            accuracy+=1
    return accuracy/len(captchas)
