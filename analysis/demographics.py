import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from __init__ import *

from preprocessing.constants import CODERS, MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings, merge_matches


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

#Plot morality shift
def plot_morality_shift(interviews, source, shift_threshold=0.01):
    if source == 'Model':
        interviews = merge_matches(interviews, wave_list=['Wave 1', 'Wave 2', 'Wave 3'])
        wave_combinations = [('Wave 1', 'Wave 2'), ('Wave 2', 'Wave 3')]
    elif source == 'Coders':
        interviews = merge_codings(interviews)
        codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]]) + int(c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
        codings = codings.div(codings.sum(axis=1), axis=0)
        interviews[MORALITY_ORIGIN] = codings
        interviews = merge_matches(interviews, wave_list=['Wave 1', 'Wave 3'])
        wave_combinations = [('Wave 1', 'Wave 3')]

    #Compute shift across waves
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

    #Map nodes labels
    mapping = {wave+':'+mo:j+i*len(MORALITY_ORIGIN) for i, wave in enumerate(['Wave 1', 'Wave 2', 'Wave 3']) for j, mo in enumerate(MORALITY_ORIGIN)}
    label = pd.Series(mapping.keys()).apply(lambda x: x.split(':')[-1])
    shifts = shifts.replace(mapping)


    #Create the Sankey plot
    sns.set_palette("Set2")
    node_colors = list(sns.color_palette("Set2", len(MORALITY_ORIGIN)).as_hex())*3
    node = dict(pad=15, thickness=30, line=dict(color='black', width=0.5), label=label, color=node_colors)
    link = dict(source=shifts['source'], target=shifts['target'], value=shifts['value'], color=shifts['target'].apply(lambda x: node_colors[x]))
    fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig.update_layout(title_text='Morality Shift over Waves (' + source + ')')
    fig.write_image('data/plots/demographics-morality_shift-' + source + '.png')
    fig.show()

if __name__ == '__main__':
    #Hyperparameters
    config = [2]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')

    for c in config:
        if c == 1:
            for attribute in ['Gender', 'Race']:
                plot_morality_evolution(interviews, attribute)
        elif c == 2:
            for source in ['Model', 'Coders']:
                plot_morality_shift(interviews, source)