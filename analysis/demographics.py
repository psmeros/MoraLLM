import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from __init__ import *
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

from preprocessing.constants import CODED_WAVES, CODERS, MORALITY_ESTIMATORS, MORALITY_ORIGIN
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
def compute_morality_shifts(interviews, estimator, attribute_name=None, attribute_value=None):
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

    return shift, N

#Plot morality shift
def plot_morality_shift(interviews):
    figs = []
    for estimator, position in zip(MORALITY_ESTIMATORS, [[0, .45], [.55, 1]]):
        shifts, _ = compute_morality_shifts(interviews, estimator)
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
    fig.update_layout(title=go.layout.Title(text='Morality Shift by Model (left) and Coders (right)', x=0.08, xanchor='left'))
    fig.write_image('data/plots/demographics-morality_shift.png')
    fig.show()

#Plot morality shift by attribute
def plot_morality_shift_by_attribute(interviews, attributes):
    #Prepare data
    shifts = []
    for estimator in MORALITY_ESTIMATORS:
        for i, attribute in enumerate(attributes):
            attributes[i]['N'] = {}
            for j, attribute_value in enumerate(attribute['values']):
                shift, N = compute_morality_shifts(interviews, estimator=estimator, attribute_name=attribute['name'], attribute_value=attribute_value)
                if not shift.empty:
                    shift['Attribute'] = attribute['name']
                    attributes[i]['N'][j] = attribute_value + ' (N = ' + str(N) + ')'
                    shift['Attribute Position'] = j
                    shift['Estimator'] = estimator
                    shifts.append(shift)
    shifts = pd.concat(shifts)
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts = shifts.groupby(['morality', 'Estimator', 'Attribute', 'Attribute Position'])['value'].sum().reset_index()
    shifts['value'] = shifts['value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(20, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='Attribute Position', orient='h', order=MORALITY_ORIGIN, col='Attribute', row='Estimator', col_order=[attribute['name'] for attribute in attributes], row_order=MORALITY_ESTIMATORS, kind='bar', legend=False, seed=42, palette=sns.color_palette('Set1'))
    g.set(xlim=(-12, 12))
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    handles, _ = ax.get_legend_handles_labels()
    g.set_titles('')
    for (j, attribute), pos in zip(enumerate(g.col_names), [(i/(len(g.col_names)+1)+.19) - (i*.01) for i in range(len(g.col_names)+1)]):
        g = g.add_legend(title=attribute, legend_data={v:h for h, v in zip(handles, attributes[j]['N'].values())}, bbox_to_anchor=(pos, 1.1), adjust_subtitles=True, loc='upper center')
    for i, label in enumerate(MORALITY_ESTIMATORS):
        g.facet_axis(i, 0).set_ylabel(label)
    plt.savefig('data/plots/demographics-morality_shift_by_attribute.png', bbox_inches='tight')
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
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=interviews, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=3, aspect=.85, palette=sns.color_palette('Set2')[:len(MORALITY_ORIGIN)])
    g.set_titles('{col_name}')
    g.legend.set_title('')
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.savefig('data/plots/demographics-morality_ecdf.png', bbox_inches='tight')
    plt.show()

def plot_class_movement(interviews):
    #Prepare data
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=CODED_WAVES)
    interviews['Household Income Change'] = (interviews[CODED_WAVES[1] + ':Income (raw)'] - interviews[CODED_WAVES[0] + ':Income (raw)'])
    interviews['Household Income Change'] = pd.to_numeric(interviews['Household Income Change'])

    interviews = pd.concat([pd.melt(interviews, id_vars=['Household Income Change'], value_vars=[CODED_WAVES[0] + ':' + mo + '_' + e for mo in MORALITY_ORIGIN], var_name='Morality Origin', value_name='Value').dropna() for e in MORALITY_ESTIMATORS])
    interviews['Estimator'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[1])
    interviews['Morality Origin'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[0].split(':')[1])
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=interviews, x='Household Income Change', y='Value', hue='Estimator', col='Morality Origin', truncate=False, x_jitter=.3, seed=42, aspect=1.2, palette=sns.color_palette('Set1'))
    g.set_ylabels('')
    g.set_titles('Morality: {col_name}')
    g.fig.subplots_adjust(wspace=0.1)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_xlim(-abs(interviews['Household Income Change']).max(), abs(interviews['Household Income Change']).max())
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
    config = [3]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)
    interviews['Race'] = interviews['Race'].apply(lambda x: x if x in ['White'] else 'Other')
    interviews['Age'] = interviews['Age'].apply(lambda x: 'Early Adolescence' if x is not pd.NA and x in ['13', '14', '15'] else 'Late Adolescence' if x is not pd.NA and x in ['16', '17', '18', '19'] else '')
    interviews['Church Attendance'] = interviews['Church Attendance'].apply(lambda x: 'Irregularly' if x is not pd.NA and x in [1,2,3] else 'Regularly' if x is not pd.NA and x in [4,5,6] else '')
    attributes = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                  {'name' : 'Race', 'values' : ['White', 'Other']},
                  {'name' : 'Income', 'values' : ['Upper', 'Lower']},
                  {'name' : 'Parent Education', 'values' : ['Tertiary', 'Secondary']},
                  {'name' : 'Age', 'values' : ['Early Adolescence', 'Late Adolescence']},
                  {'name' : 'Church Attendance', 'values' : ['Regularly', 'Irregularly']}]

    for c in config:
        if c == 1:
            for attribute in attributes:
                plot_morality_evolution(interviews, attribute)
        elif c == 2:
            plot_morality_shift(interviews)
        elif c == 3:
            plot_morality_shift_by_attribute(interviews, attributes)
        elif c == 4:
            plot_ecdf(interviews)
        elif c == 5:
            plot_class_movement(interviews)
        elif c == 6:
            actions = ['Pot', 'Drink', 'Cheat']
            n_clusters = 2
            plot_action_probability(interviews, actions=actions, n_clusters=n_clusters)