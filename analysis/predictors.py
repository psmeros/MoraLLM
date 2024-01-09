import pandas as pd
import seaborn as sns
from __init__ import *
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys
from scipy.stats import pearsonr

def probability_mass_reallocation(prob, factor, nprobs):
    if factor is not None:
        max_prob = prob.max() + (1.0 - prob.max()) * factor
        max_prob_index = prob.idxmax()
        prob = prob - prob * factor
        prob[max_prob_index] = max_prob
    elif nprobs is not None:
        index = prob.nlargest(nprobs).index
        prob.loc[:] = 0
        prob.loc[index] = 1.0
    return prob

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
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig('data/plots/predictors-action_prediction.png', bbox_inches='tight')
    plt.show()

    action_prediction = action_prediction.groupby('Action')['Feature_Importances'].apply(list).apply(lambda l: (pd.Series(l[0]) + pd.Series(l[1])).idxmax())
    action_prediction = pd.DataFrame(action_prediction.reindex(actions).reset_index().values, columns=['Action', 'Key Morality Origin'])
    print(action_prediction)

def moral_consciousness(interviews, factor, nprobs, wave_list=['Wave 1', 'Wave 3'], inputs=['Model', 'Coders']):
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + inputs[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN].apply(lambda mo: probability_mass_reallocation(mo, factor=factor, nprobs=nprobs), axis=1)
    interviews[[mo + '_' + inputs[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=wave_list)
    desicion_taking = pd.get_dummies(interviews[wave_list[0] + ':' + 'Decision Taking'])
    Age = interviews[wave_list[0] + ':Age'].dropna().astype(int)
    Grades = interviews[wave_list[0] + ':Grades'].dropna().astype(int)
    Gender = pd.factorize(interviews[wave_list[0] + ':Gender'])[0]
    Race = pd.factorize(interviews[wave_list[0] + ':Race'])[0]
    Church_Attendance = interviews[wave_list[0] + ':Church Attendance'].dropna()
    Parent_Education = interviews[wave_list[0] + ':Parent Education'].dropna()
    Parent_Income = interviews[wave_list[0] + ':Income'].dropna()

    compute_correlation = lambda x: str(round(x[0], 3)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')

    data = []
    for input in inputs:
        correlations = pd.DataFrame(columns=wave_list, index=['Intuitive - Consequentialist', 'Social - Consequentialist', 'Intuitive - Social', 'Intuitive - Expressive Individualist', 'Intuitive - Utilitarian Individualist', 'Intuitive - Relational', 'Intuitive - Theistic', 'Intuitive - Age', 'Intuitive - GPA', 'Intuitive - Gender', 'Intuitive - Race', 'Intuitive - Church Attendance', 'Intuitive - Parent Education', 'Intuitive - Parent Income'])
        for wave in wave_list:
            Intuitive = interviews[wave + ':Experience_' + input]
            Consequentialist = interviews[wave + ':Consequences_' + input]
            Social = interviews[[wave + ':' + mo + '_' + input for mo in ['Family', 'Community', 'Friends']]]
            Social = Social.max(axis=1) if input == 'Model' else Social.any(axis=1) if input == 'Coders' else None

            correlations[wave].loc['Intuitive - Consequentialist'] = compute_correlation(pearsonr(Intuitive, Consequentialist))
            correlations[wave].loc['Social - Consequentialist'] = compute_correlation(pearsonr(Social, Consequentialist))
            correlations[wave].loc['Intuitive - Social'] = compute_correlation(pearsonr(Intuitive, Social))

            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Consequentialist.values), pd.Series(['Intuitive - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([input]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Social.values), pd.Series(Consequentialist.values), pd.Series(['Social - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([input]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Social.values), pd.Series(['Intuitive - Social']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([input]*len(interviews))], axis=1))
            
            correlations[wave].loc['Intuitive - Expressive Individualist'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Expressive Individualist']))
            correlations[wave].loc['Intuitive - Utilitarian Individualist'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Utilitarian Individualist']))
            correlations[wave].loc['Intuitive - Relational'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Relational']))
            correlations[wave].loc['Intuitive - Theistic'] = compute_correlation(pearsonr(Intuitive, desicion_taking['Theistic']))

            correlations[wave].loc['Intuitive - Age'] = compute_correlation(pearsonr(Intuitive.loc[Age.index], Age))
            correlations[wave].loc['Intuitive - GPA'] = compute_correlation(pearsonr(Intuitive.loc[Grades.index], Grades))
            correlations[wave].loc['Intuitive - Gender'] = compute_correlation(pearsonr(Intuitive, Gender))
            correlations[wave].loc['Intuitive - Race'] = compute_correlation(pearsonr(Intuitive, Race))
            correlations[wave].loc['Intuitive - Church Attendance'] = compute_correlation(pearsonr(Intuitive.loc[Church_Attendance.index], Church_Attendance))
            correlations[wave].loc['Intuitive - Parent Education'] = compute_correlation(pearsonr(Intuitive.loc[Parent_Education.index], Parent_Education))
            correlations[wave].loc['Intuitive - Parent Income'] = compute_correlation(pearsonr(Intuitive.loc[Parent_Income.index], Parent_Income))

        print(input)
        print(correlations)
    data = pd.concat(data, axis=0, ignore_index=True)
    data.columns = ['x', 'y', 'Correlation', 'Wave', 'Input']

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=data[data['Input'] == 'Model'], x='x', y='y', col='Correlation', row='Wave')
    g.set_titles('{row_name}' + '\n' + '{col_name}')
    g.fig.subplots_adjust(wspace=0.1)
    plt.xlim(0, .8)
    plt.ylim(0, .8)
    plt.savefig('data/plots/predictors-correlations', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [2]
    actions=['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews, quantize_classes=False)

    for c in config:
        if c == 1:
            action_prediction(interviews, actions=actions)
        elif c == 2:
            moral_consciousness(interviews, factor=None, nprobs=None)