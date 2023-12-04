from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression, r_regression
from __init__ import *

import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures, minmax_scale
from preprocessing.constants import CODERS, HOUSEHOLD_CLASS, MORALITY_ORIGIN

from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews, quantize_classes=False)

    for c in config:
        if c == 1:
            pass


inputs=['Model', 'Coders']
wave_list=['Wave 1', 'Wave 3']

#Prepare data
interviews = merge_codings(interviews)
codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
interviews[[mo + '_' + inputs[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
interviews[[mo + '_' + inputs[1] for mo in MORALITY_ORIGIN]] = codings
interviews['Income'] = interviews['Income'].apply(lambda i: i if i in HOUSEHOLD_CLASS.keys() else np.nan)
interviews = merge_matches(interviews, wave_list=wave_list)
interviews['Household Income Diff'] = (interviews[wave_list[1] + ':Income'] - interviews[wave_list[0] + ':Income'])

for input in inputs:
    input_interviews = pd.DataFrame(interviews[[wave_list[0] + ':' + mo + '_' + input  for mo in MORALITY_ORIGIN]+[wave_list[0] + ':Income', wave_list[0] + ':Parent Education', wave_list[1] + ':Income']].values, columns=MORALITY_ORIGIN+['Income', 'Parent Education', 'Income Prediction']).dropna()
    input_interviews[['Income', 'Parent Education', 'Income Prediction']] = minmax_scale(input_interviews[['Income', 'Parent Education', 'Income Prediction']])

    X = input_interviews[MORALITY_ORIGIN+['Income', 'Parent Education']]
    y = input_interviews['Income Prediction']

    # ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10], cv=2).fit(X, y) 
    # print(input, ' Score:', ridge.best_score_)

    # anova = f_classif(X, y)
    # for i, mo in enumerate(MORALITY_ORIGIN+['Income', 'Parent Education']):
    #     importance =  '***' if float(anova[1][i])<.005 else '**' if float(anova[1][i])<.01 else '*' if float(anova[1][i])<.05 else None
    #     if importance:
    #         print(input, mo, importance)
    

    print(input, 'Accuracy:', cross_val_score(RandomForestRegressor(n_estimators=20, max_depth=1, random_state=42), X, y, cv=5).mean())