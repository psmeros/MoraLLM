import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import seaborn as sns
from __init__ import *

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings, merge_matches, merge_surveys


#Plot morality evolution
def plot_morality_evolution(interviews, attribute, wave_list=['Wave 1', 'Wave 3'], inputs=['Model', 'Coders']):
    #Prepare data for computing evolution
    interviews[[mo + '_' + inputs[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + inputs[1] for mo in MORALITY_ORIGIN]] = codings
    interviews = merge_matches(interviews, wave_list=wave_list)

    #Compute evolution for each data slice
    interviews_list = []
    for input in inputs:
        for attribute_value in attribute['values']:
            filtered_interviews = interviews[interviews[wave_list[0] + ':' + attribute['name']] == attribute_value]
            N = len(filtered_interviews)
            filtered_interviews = pd.concat([pd.DataFrame(filtered_interviews.filter(regex='^' + wave + '.*(' + input + '|Wave)$').values, columns=['Wave']+MORALITY_ORIGIN) for wave in wave_list])
            filtered_interviews['input'] = input
            filtered_interviews[attribute['name']] = attribute_value + ' (N = ' + str(N) + ')'
            interviews_list.append(filtered_interviews)
    interviews = pd.concat(interviews_list)

    #Prepare data for plotting
    interviews = pd.melt(interviews, id_vars=['Wave', 'input', attribute['name']], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Value')
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.relplot(data=interviews, y='Value', x='Wave', hue='Morality Origin', col=attribute['name'], row='input', kind='line', linewidth=4, palette='Set2')
    g.fig.subplots_adjust(wspace=0.05)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_ylabels('')
    g.set_titles('\n' + attribute['name'] + ': {col_name}\n Input: {row_name}')
    plt.xticks(interviews['Wave'].unique())
    plt.xlim(interviews['Wave'].min(), interviews['Wave'].max())
    legend = g._legend
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig('data/plots/demographics-morality_evolution_by_'+attribute['name'].lower()+'.png', bbox_inches='tight')
    plt.show()

#Compute morality shifts across waves
def compute_morality_shifts(interviews, input, shift_threshold, wave_list=['Wave 1', 'Wave 3'], attribute_name=None, attribute_value=None):
    interviews = merge_codings(interviews)
    if input == 'Coders':
        codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
        interviews[MORALITY_ORIGIN] = codings
    interviews = merge_matches(interviews, wave_list=wave_list)
    if attribute_name is not None:
        interviews = interviews[interviews[wave_list[0] + ':' + attribute_name] == attribute_value]
    N = len(interviews)

    wave_source = interviews[[wave_list[0] + ':' + mo for mo in MORALITY_ORIGIN]]
    wave_target = interviews[[wave_list[1] + ':' + mo for mo in MORALITY_ORIGIN]]
    wave_source.columns = MORALITY_ORIGIN
    wave_target.columns = MORALITY_ORIGIN

    #Combine outgoing percentage, incoming coeficients, and remaining percentage
    outgoing = (wave_source - wave_target > 0) * abs(wave_source - wave_target)
    incoming = (wave_source - wave_target < 0) * abs(wave_source - wave_target)

    outgoing_percentage = (outgoing / wave_source).fillna(0)
    coefs = incoming.div(incoming.sum(axis=1), axis=0).fillna(0)

    remaining_percentage = (1 - outgoing_percentage)
    remaining_percentage = pd.DataFrame(np.diag(remaining_percentage.sum()), index=MORALITY_ORIGIN, columns=MORALITY_ORIGIN)

    #Normalize shift
    shift = ((outgoing_percentage.T @ coefs) + remaining_percentage) / len(interviews)

    #Confirm shift from source to target
    if input == 'Model':
        assert(abs(round((wave_source @ shift - wave_target).sum(axis=1).sum(), 8)) == 0)

    #Reshape shift
    shift = pd.DataFrame(shift.values, index=[wave_list[0] + ':' + mo for mo in MORALITY_ORIGIN], columns=[wave_list[1] + ':' + mo for mo in MORALITY_ORIGIN])
    shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})

    #Compute the prior shift
    shift = shift.merge(pd.DataFrame(interviews[[wave_list[0] + ':' + mo for mo in MORALITY_ORIGIN]].mean().reset_index().values, columns=['source', 'prior']))
    shift['value'] = shift['value'] * shift['prior']

    #Apply threshold
    shift = shift[shift['value'] > shift_threshold]

    return shift, N

#Plot morality shift
def plot_morality_shift(interviews, shift_threshold, waves = ['Wave 1', 'Wave 3'], inputs = ['Model', 'Coders']):
    
    figs = []
    for input, position in zip(inputs, [[0, .45], [.55, 1]]):
        shifts, _ = compute_morality_shifts(interviews, input, shift_threshold)
        #Prepare data
        sns.set_palette("Set2")
        mapping = {wave+':'+mo:j+i*len(MORALITY_ORIGIN) for i, wave in enumerate(waves) for j, mo in enumerate(MORALITY_ORIGIN)}
        shifts = shifts.replace(mapping)
        label = pd.DataFrame([(i,j/(.69*len(MORALITY_ORIGIN))) for i, _ in enumerate(waves) for j, _ in enumerate(MORALITY_ORIGIN)], columns=['x', 'y']) + 0.001
        label['name'] = pd.Series({v:k for k, v in mapping.items()}).apply(lambda x: x.split(':')[-1])
        label['color'] = list(sns.color_palette("Set2", len(MORALITY_ORIGIN)).as_hex()) * len(waves)

        #Create Sankey
        node = dict(pad=10, thickness=30, line=dict(color='black', width=0.5), label=label['name'], color=label['color'], x=label['x'], y=label['y'])
        link = dict(source=shifts['source'], target=shifts['target'], value=shifts['value'], color=label['color'].iloc[shifts['target']])
        domain = dict(x=position)
        fig = go.Sankey(node=node, link=link, domain=domain)
        figs.append(fig)

    #Plot
    fig = go.Figure(data=figs, layout=go.Layout(height=400, width=800, font_size=14))
    fig.update_layout(title=go.layout.Title(text='Morality Shift by Coders (left) and Model (right)<br><sup>Shift Threshold: '+str(int(shift_threshold*100))+'%</sup>', x=0.08, xanchor='left'))
    fig.write_image('data/plots/demographics-morality_shift.png')
    fig.show()

#Plot morality shift by attribute
def plot_morality_shift_by_attribute(interviews, attribute, shift_threshold, inputs = ['Model', 'Coders']):
    #Prepare data
    shifts = []
    col_order = []
    for input in inputs:
        for attribute_value in attribute['values']:
            shift, N = compute_morality_shifts(interviews, input=input, shift_threshold=shift_threshold, attribute_name=attribute['name'], attribute_value=attribute_value)
            if not shift.empty:
                shift[attribute['name']] = attribute_value + ' (N = ' + str(N) + ')'
                col_order.append(shift[attribute['name']].loc[0])
                shift['input'] = input
                shifts.append(shift)
    shifts = pd.concat(shifts)
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts = shifts.groupby(['morality', 'input', attribute['name']])['value'].sum().reset_index()
    shifts['value'] = shifts['value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=1)
    plt.figure(figsize=(10, 10))
    g = sns.FacetGrid(shifts, col=attribute['name'], col_order=pd.Series(col_order).drop_duplicates())
    g.map_dataframe(sns.barplot, x='value', y='morality', hue='input', hue_order=inputs, orient='h', order=MORALITY_ORIGIN, palette=sns.color_palette('Set2'), errorbar=None)
    g.fig.subplots_adjust(wspace=0.3)
    g.add_legend()
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_titles(attribute['name'] + ': {col_name}')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/demographics-morality_shift_by_'+attribute['name'].lower()+'.png', bbox_inches='tight')
    plt.show()

def plot_distributions(interviews):
    interviews = merge_codings(interviews)
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)

    codings = pd.DataFrame(codings.values, columns=MORALITY_ORIGIN).unstack().reset_index().rename(columns={'level_0':'Morality Origin', 0:'Value'}).drop('level_1', axis=1)
    codings['Input'] = 'Coders'

    interviews = interviews[MORALITY_ORIGIN].unstack().reset_index().rename(columns={'level_0':'Morality Origin', 0:'Value'}).drop('level_1', axis=1)
    interviews['Input'] = 'Model'

    interviews = pd.concat([interviews, codings])
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))

    ax = sns.violinplot(data=interviews, x='Value', y='Morality Origin', hue='Input', orient='h', split=True, density_norm = 'area', common_norm=True, cut=0.5, bw_adjust=0, inner='quart', palette=sns.color_palette('Set2'))

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.3, 1.02))
    plt.savefig('data/plots/demographics-morality_distributions.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [4]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)

    interviews['Race'] = interviews['Race'].apply(lambda x: x if x in ['White'] else 'Other')
    attributes = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                  {'name' : 'Race', 'values' : ['White', 'Other']},
                  {'name' : 'Income', 'values' : ['Lower', 'Middle', 'Upper']},
                  {'name' : 'Parent Education', 'values' : ['Primary', 'Secondary', 'Tertiary']}]

    for c in config:
        if c == 1:
            for attribute in attributes:
                plot_morality_evolution(interviews, attribute)
        elif c == 2:
            shift_threshold = .01
            plot_morality_shift(interviews, shift_threshold)
        elif c == 3:
            shift_threshold = 0
            for attribute in attributes:
                plot_morality_shift_by_attribute(interviews, attribute, shift_threshold)
        elif c == 4:
            plot_distributions(interviews)