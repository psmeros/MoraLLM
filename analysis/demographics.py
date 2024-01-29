import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale
from __init__ import *

from preprocessing.constants import CODERS, MERGE_MORALITY_ORIGINS, MORALITY_ORIGIN, CODED_WAVES, MORALITY_ESTIMATORS
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys


#Plot morality evolution
def plot_morality_evolution(interviews, attribute):
    #Prepare data for computing evolution
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)

    #Compute evolution for each data slice
    interviews_list = []
    for estimator in MORALITY_ESTIMATORS:
        for attribute_value in attribute['values']:
            filtered_interviews = interviews[interviews[CODED_WAVES[0] + ':' + attribute['name']] == attribute_value]
            N = len(filtered_interviews)
            filtered_interviews = pd.concat([pd.DataFrame(filtered_interviews.filter(regex='^' + wave + '.*(' + estimator + '|Wave)$').values, columns=['Wave']+MORALITY_ORIGIN) for wave in CODED_WAVES])
            filtered_interviews['estimator'] = estimator
            filtered_interviews[attribute['name']] = attribute_value + ' (N = ' + str(N) + ')'
            interviews_list.append(filtered_interviews)
    interviews = pd.concat(interviews_list)

    #Prepare data for plotting
    interviews = pd.melt(interviews, id_vars=['Wave', 'estimator', attribute['name']], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Value')
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.relplot(data=interviews, y='Value', x='Wave', hue='Morality Origin', col=attribute['name'], row='estimator', kind='line', linewidth=4, palette='Set2')
    g.fig.subplots_adjust(wspace=0.05)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_ylabels('')
    g.set_titles('\n' + attribute['name'] + ': {col_name}\n Estimator: {row_name}')
    plt.xticks(interviews['Wave'].unique())
    plt.xlim(interviews['Wave'].min(), interviews['Wave'].max())
    legend = g._legend
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig('data/plots/demographics-morality_evolution_by_'+attribute['name'].lower()+'.png', bbox_inches='tight')
    plt.show()

#Compute morality shifts across waves
def compute_morality_shifts(interviews, estimator, shift_threshold, attribute_name=None, attribute_value=None):
    interviews = merge_codings(interviews)
    if estimator == 'Coders':
        codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
        interviews[MORALITY_ORIGIN] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)
    if attribute_name is not None:
        interviews = interviews[interviews[CODED_WAVES[0] + ':' + attribute_name] == attribute_value]
    N = len(interviews)

    wave_source = interviews[[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN]]
    wave_target = interviews[[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN]]
    wave_source.columns = MORALITY_ORIGIN
    wave_target.columns = MORALITY_ORIGIN

    #Combine outgoing percentage, incoming coefficients, and remaining percentage
    outgoing = (wave_source - wave_target > 0) * abs(wave_source - wave_target)
    incoming = (wave_source - wave_target < 0) * abs(wave_source - wave_target)

    outgoing_percentage = (outgoing / wave_source).fillna(0)
    coefs = incoming.div(incoming.sum(axis=1), axis=0).fillna(0)

    remaining_percentage = (1 - outgoing_percentage)
    remaining_percentage = pd.DataFrame(np.diag(remaining_percentage.sum()), index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Normalize shift
    shift = ((outgoing_percentage.T @ coefs) + remaining_percentage) / len(interviews)

    #Reshape shift
    shift = pd.DataFrame(shift.values, index=[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN], columns=[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN])
    shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})

    #Compute the prior shift
    shift = shift.merge(pd.DataFrame(interviews[[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN]].mean().reset_index().values, columns=['source', 'prior']))
    shift['value'] = shift['value'] * shift['prior']

    #Apply threshold
    shift = shift[shift['value'] > shift_threshold]

    return shift, N

#Plot morality shift
def plot_morality_shift(interviews, shift_threshold):
    
    figs = []
    for estimator, position in zip(MORALITY_ESTIMATORS, [[0, .45], [.55, 1]]):
        shifts, _ = compute_morality_shifts(interviews, estimator, shift_threshold)
        #Prepare data
        sns.set_palette('Set2')
        mapping = {wave+':'+mo:j+i*len(MORALITY_ORIGIN) for i, wave in enumerate(CODED_WAVES) for j, mo in enumerate(MORALITY_ORIGIN)}
        shifts = shifts.replace(mapping)
        label = pd.DataFrame([(i,j/(.69*len(MORALITY_ORIGIN))) for i, _ in enumerate(CODED_WAVES) for j, _ in enumerate(MORALITY_ORIGIN)], columns=['x', 'y']) + 0.001
        label['name'] = pd.Series({v:k for k, v in mapping.items()}).apply(lambda x: x.split(':')[-1])
        label['color'] = list(sns.color_palette("Set2", len(MORALITY_ORIGIN)).as_hex()) * len(CODED_WAVES)

        #Create Sankey
        node = dict(pad=10, thickness=30, line=dict(color='black', width=0.5), label=label['name'], color=label['color'], x=label['x'], y=label['y'])
        link = dict(source=shifts['source'], target=shifts['target'], value=shifts['value'], color=label['color'].iloc[shifts['target']])
        domain = dict(x=position)
        fig = go.Sankey(node=node, link=link, domain=domain)
        figs.append(fig)

    #Plot
    fig = go.Figure(data=figs, layout=go.Layout(height=400, width=800, font_size=14))
    subtitle = '<br><sup>Shift Threshold: '+str(int(shift_threshold*100))+'%</sup>' if shift_threshold > 0 else ''
    fig.update_layout(title=go.layout.Title(text='Morality Shift by Model (left) and Coders (right)'+subtitle, x=0.08, xanchor='left'))
    fig.write_image('data/plots/demographics-morality_shift.png')
    fig.show()

#Plot morality shift by attribute
def plot_morality_shift_by_attribute(interviews, attribute, shift_threshold):
    #Prepare data
    shifts = []
    col_order = []
    for estimator in MORALITY_ESTIMATORS:
        for attribute_value in attribute['values']:
            shift, N = compute_morality_shifts(interviews, estimator=estimator, shift_threshold=shift_threshold, attribute_name=attribute['name'], attribute_value=attribute_value)
            if not shift.empty:
                shift[attribute['name']] = attribute_value + ' (N = ' + str(N) + ')'
                col_order.append(shift[attribute['name']].loc[0])
                shift['estimator'] = estimator
                shifts.append(shift)
    shifts = pd.concat(shifts)
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts = shifts.groupby(['morality', 'estimator', attribute['name']])['value'].sum().reset_index()
    shifts['value'] = shifts['value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=1)
    plt.figure(figsize=(10, 10))
    g = sns.FacetGrid(shifts, col=attribute['name'], col_order=pd.Series(col_order).drop_duplicates())
    g.map_dataframe(sns.barplot, x='value', y='morality', hue='estimator', hue_order=MORALITY_ESTIMATORS, orient='h', order=MORALITY_ORIGIN, palette=sns.color_palette('Set2'), errorbar=None)
    g.fig.subplots_adjust(wspace=0.3)
    g.add_legend()
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_titles(attribute['name'] + ': {col_name}')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/demographics-morality_shift_by_'+attribute['name'].lower()+'.png', bbox_inches='tight')
    plt.show()

def plot_ecdf(interviews):
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)

    codings = pd.DataFrame(codings.values, columns=MORALITY_ORIGIN).unstack().reset_index().rename(columns={'level_0':'Morality', 0:'Value'}).drop('level_1', axis=1)
    codings['Estimator'] = 'Coders'
    interviews = interviews[MORALITY_ORIGIN].unstack().reset_index().rename(columns={'level_0':'Morality', 0:'Value'}).drop('level_1', axis=1)
    interviews['Estimator'] = 'Model'
    interviews = pd.concat([interviews, codings])
    
    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(15, 10))
    sns.displot(data=interviews, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=3, palette=sns.color_palette('Set2'))
    plt.savefig('data/plots/demographics-morality_ecdf.png', bbox_inches='tight')
    plt.show()

def plot_class_movement(interviews):
    #Prepare data
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)
    interviews['Household Income Diff'] = (interviews[CODED_WAVES[1] + ':Income (raw)'] - interviews[CODED_WAVES[0] + ':Income (raw)'])
    interviews['Household Income Diff'] = pd.to_numeric(interviews['Household Income Diff'])

    interviews = pd.concat([pd.melt(interviews, id_vars=['Household Income Diff'], value_vars=[CODED_WAVES[0] + ':' + mo + '_' + e for mo in MORALITY_ORIGIN], var_name='Morality Origin', value_name='Value').dropna() for e in MORALITY_ESTIMATORS])
    interviews['Estimator'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[1])
    interviews['Morality Origin'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[0].split(':')[1])
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(15, 10))
    g = sns.lmplot(data=interviews, x='Household Income Diff', y='Value', hue='Estimator', col='Morality Origin', col_wrap=3, truncate=False, x_jitter=.3, seed=42, palette=sns.color_palette('Set2'))
    g.set_ylabels('')
    g.set_titles('Morality: {col_name}')
    g.fig.subplots_adjust(wspace=0.1)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_xlim(-abs(interviews['Household Income Diff']).max(), abs(interviews['Household Income Diff']).max())
    plt.savefig('data/plots/demographics-class_movement.png', bbox_inches='tight')
    plt.show()

def plot_action_probability(interviews, n_clusters, actions):
    #Prepare data
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)

    # Perform clustering, dimensionality reduction, and probability estimation
    embeddings_list = []
    for action in actions:
        for estimator in MORALITY_ESTIMATORS:
            embeddings = interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator for mo in MORALITY_ORIGIN]].values
            clusters = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42).fit_predict(embeddings)
            embeddings = pd.DataFrame(TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(embeddings))
            embeddings['Clusters'] = clusters
            embeddings['Value'] = minmax_scale(interviews[CODED_WAVES[0] + ':' + action])
            embeddings['Value'] = embeddings['Clusters'].apply(lambda c: embeddings.groupby('Clusters')['Value'].mean()[c])
            embeddings['Estimator'] = estimator
            embeddings['Action'] = action
            embeddings_list.append(embeddings)
    embeddings = pd.concat(embeddings_list)

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    color_palette = sns.color_palette('coolwarm', as_cmap=True)
    g = sns.displot(data=embeddings, col='Action', row='Estimator', kind='kde', facet_kws=dict(sharex=False, sharey=False), common_norm=False, x=0, y=1, hue='Value', hue_norm=(0, .25), fill=True, thresh=.2, alpha=.5, legend=False, palette=color_palette)

    cbar_ax = g.fig.add_axes([1.0, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=color_palette), cax=cbar_ax)
    cbar.ax.get_yaxis().set_ticks([])
    cbar.ax.get_yaxis().set_ticks([0, 1])
    cbar.ax.get_yaxis().set_ticklabels(['Low', 'High'])
    g.set_axis_labels('', '')
    for ax in g.axes.flat:
        ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=4))
    g.set_titles('Estimator: {row_name}' + '\n' + 'Action: {col_name}')
    plt.savefig('data/plots/demographics-action_probability', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [2,3,4,5]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews['Race'] = interviews['Race'].apply(lambda x: x if x in ['White'] else 'Other')
    interviews = merge_surveys(interviews)
    attributes = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                  {'name' : 'Race', 'values' : ['White', 'Other']},
                  {'name' : 'Income', 'values' : ['Lower', 'Middle', 'Upper']},
                  {'name' : 'Parent Education', 'values' : ['Primary', 'Secondary', 'Tertiary']}]

    for c in config:
        if c == 1:
            for attribute in attributes:
                plot_morality_evolution(interviews, attribute)
        elif c == 2:
            shift_threshold = 0
            plot_morality_shift(interviews, shift_threshold)
        elif c == 3:
            shift_threshold = 0
            for attribute in attributes:
                plot_morality_shift_by_attribute(interviews, attribute, shift_threshold)
        elif c == 4:
            plot_ecdf(interviews)
        elif c == 5:
            plot_class_movement(interviews)
        elif c == 6:
            actions = ['Pot', 'Drink', 'Cheat']
            n_clusters = 2
            plot_action_probability(interviews, actions=actions, n_clusters=n_clusters)