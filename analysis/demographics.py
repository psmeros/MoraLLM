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
def plot_morality_evolution(interviews, col_attribute, x_attribute='Wave'):
    interviews = interviews.dropna(subset=[x_attribute, col_attribute])
    interviews[x_attribute] = interviews[x_attribute].astype(int)
    if x_attribute == 'Age':
        interviews[x_attribute] = interviews[x_attribute].apply(lambda x: '13-15' if x >= 13 and x <= 15 else '16-18' if x >= 16 and x <= 18  else '19-23' if x >= 19 and x <= 23 else '')

    interviews = pd.melt(interviews, id_vars=[x_attribute, col_attribute], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Value')
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.relplot(data=interviews, y='Value', x=x_attribute, hue='Morality Origin', col=col_attribute, kind='line', linewidth=4, palette='Set2')
    g.fig.subplots_adjust(wspace=0.3)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_ylabels('')
    plt.xticks(interviews[x_attribute].unique())
    plt.xlim(interviews[x_attribute].min(), interviews[x_attribute].max())
    legend = g._legend
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig('data/plots/demographics-morality_evolution_by_'+col_attribute.lower()+'.png', bbox_inches='tight')
    plt.show()

#Compute morality shifts across waves
def compute_morality_shifts(interviews, wave_combinations, method, shift_threshold=0.01):
    wave_list = list(set([w for p in wave_combinations for w in p]))
    interviews = merge_codings(interviews)
    if method == 'Coders':
        codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
        codings = codings.div(codings.sum(axis=1), axis=0)
        interviews[MORALITY_ORIGIN] = codings    
    interviews = merge_matches(interviews, wave_list=wave_list)
    N = len(interviews)

    shifts = []
    for ws, wt in wave_combinations:
        wave_source = interviews[[ws+':'+mo for mo in MORALITY_ORIGIN]]
        wave_target = interviews[[wt+':'+mo for mo in MORALITY_ORIGIN]]
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
        assert(abs(round((wave_source @ shift - wave_target).sum(axis=1).sum(), 8)) == 0)

        #Reshape shift
        shift = pd.DataFrame(shift.values, index=[ws+':'+mo for mo in MORALITY_ORIGIN], columns=[wt+':'+mo for mo in MORALITY_ORIGIN])
        shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})
        shifts.append(shift)
    shifts = pd.concat(shifts)

    #Apply threshold
    shifts = shifts[shifts['value'] > shift_threshold]

    return shifts, N

#Plot morality shift
def plot_morality_shift(interviews, wave_combinations, shift_threshold):
    
    figs = []
    for wc, position in zip(wave_combinations.items(), [[0, 0.47], [0.53, 1.0]]):
        shifts, _ = compute_morality_shifts(interviews, wc[1], wc[0], shift_threshold)
        #Prepare data
        waves = ['Wave 1', 'Wave 2', 'Wave 3']
        sns.set_palette("Set2")
        mapping = {wave+':'+mo:j+i*len(MORALITY_ORIGIN) for i, wave in enumerate(waves) for j, mo in enumerate(MORALITY_ORIGIN)}
        shifts = shifts.replace(mapping)
        label = pd.DataFrame([(i/(len(wc[1])),j/(len(MORALITY_ORIGIN))) for i, _ in enumerate(waves) for j, _ in enumerate(MORALITY_ORIGIN)], columns=['x', 'y']) + 0.001
        label['name'] = pd.Series({v:k for k, v in mapping.items()}).apply(lambda x: x.split(':')[-1])
        label['color'] = list(sns.color_palette("Set2", len(MORALITY_ORIGIN)).as_hex()) * len(waves)

        #Create Sankey
        node = dict(pad=15, thickness=30, line=dict(color='black', width=0.5), label=label['name'], color=label['color'], x=label['x'], y=label['y'])
        link = dict(source=shifts['source'], target=shifts['target'], value=shifts['value'], color=label['color'].iloc[shifts['target']])
        domain = dict(x=position)
        fig = go.Sankey(node=node, link=link, domain=domain)
        figs.append(fig)

    #Plot
    fig = go.Figure(data=figs, layout=go.Layout(height=500, width=1200, font_size=12))
    fig.update_layout(title=go.layout.Title(text='Morality Shift by Coders (left) and Model (right)<br><sup>Shift Threshold: '+str(int(shift_threshold*100))+'%</sup>', xref='paper', x=0))
    fig.write_image('data/plots/demographics-morality_shift.png')
    fig.show()

#Plot morality shift by attribute
def plot_morality_shift_by_attribute(interviews, attribute, shift_threshold):
    #Prepare data
    shifts = []
    for method in ['Model', 'Coders']:
        for a in attribute[1]:
            shift, N = compute_morality_shifts(interviews[interviews[attribute[0]] == a], wave_combinations=[('Wave 1', 'Wave 3')], method=method, shift_threshold=shift_threshold)
            shift[attribute[0]] = a
            shift['method'] = method
            shift['N'] = str(N)
            shifts.append(shift)
    shifts = pd.concat(shifts)
    shifts['wave'] = shifts.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
    shifts['source'] = shifts['source'].apply(lambda x: x.split(':')[-1])
    shifts['target'] = shifts['target'].apply(lambda x: x.split(':')[-1])
    source_shifts = shifts.drop('target', axis=1).rename(columns={'source':'morality'})
    source_shifts['value'] = -source_shifts['value']
    target_shifts = shifts.drop('source', axis=1).rename(columns={'target':'morality'})
    shifts = pd.concat([source_shifts, target_shifts])
    shifts['value'] = shifts['value'] * 100
    shifts[attribute[0]] = shifts[attribute[0]] + ' (N = ' + shifts['N'] + ')'

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=1)
    plt.figure(figsize=(10, 10))
    g = sns.FacetGrid(shifts, col=attribute[0])
    g.map_dataframe(sns.barplot, x='value', y='morality', hue='method', orient='h', order=MORALITY_ORIGIN, palette=sns.color_palette('Set2'), errorbar=None)
    g.fig.subplots_adjust(wspace=0.3)
    g.add_legend()
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_titles(attribute[0] + ': {col_name}')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/demographics-morality_shift_by_'+attribute[0].lower()+'.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [3]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)

    for c in config:
        if c == 1:
            for attribute in ['Gender', 'Race']:
                plot_morality_evolution(interviews, attribute)
        elif c == 2:
            shift_threshold = .05
            wave_combinations = {'Coders' : [('Wave 1', 'Wave 3')], 'Model': [('Wave 1', 'Wave 3')]}
            plot_morality_shift(interviews, wave_combinations, shift_threshold)
        elif c == 3:
            shift_threshold = 0
            interviews['Race'] = interviews['Race'].apply(lambda x: x if x == 'White' else 'Other')
            attributes = {'Gender' : ['Male', 'Female'], 'Race' : ['White', 'Other'], 'Income' : ['Lower Class', 'Middle Class', 'Upper Class']}
            for attribute in attributes.items():
                plot_morality_shift_by_attribute(interviews, attribute, shift_threshold)