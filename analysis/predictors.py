import numpy as np
import pandas as pd
import seaborn as sns
from __init__ import *
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys

def action_prediction(interviews, actions, wave_list=['Wave 1', 'Wave 3'], inputs=['Model', 'Coders']):
    #Prepare data
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + inputs[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + inputs[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=wave_list)

    #Train Classifier
    action_prediction = []
    for action in actions:
        for input in inputs:
            input_interviews = interviews[[wave_list[0] + ':' + mo + '_' + input  for mo in MORALITY_ORIGIN]+[wave_list[0] + ':' + action]].dropna()
            y = input_interviews[wave_list[0] + ':' + action].apply(lambda d: False if d == 1 else True).values
            X = StandardScaler().fit_transform(input_interviews.drop([wave_list[0] + ':' + action], axis=1).values)
        
            clf = GridSearchCV(RandomForestClassifier(random_state=42), {'n_estimators': [1, 2, 3, 4, 5, 10, 20, 30], 'max_depth': [1, 2, 3, 4, 5]}, scoring='f1', cv=2)
            clf.fit(X, y)

            feature_importances = RandomForestClassifier(**clf.best_params_, random_state=42).fit(X, y).feature_importances_
            feature_importances = {item[0]:item[1] for item in sorted(zip(MORALITY_ORIGIN, feature_importances), key=lambda x: x[1], reverse=True)}
            action_prediction.append({'Action' : action, 'Input' : input, 'F1 Score' : clf.best_score_, 'Feature_Importances' : feature_importances})
    action_prediction = pd.DataFrame(action_prediction)


    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(action_prediction, x='F1 Score', y='Action', hue='Input', hue_order=inputs, orient='h', palette=sns.color_palette('Set2'))
    plt.savefig('data/plots/predictors-action_prediction.png', bbox_inches='tight')
    plt.show()

    print(action_prediction.groupby('Action')['Feature_Importances'].apply(list).apply(lambda l: (pd.Series(l[0]) + pd.Series(l[1])).idxmax()))


if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    actions=['Pot', 'Drink', 'Cheat']
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews, quantize_classes=False)

    for c in config:
        if c == 1:
            action_prediction(interviews, actions=actions)