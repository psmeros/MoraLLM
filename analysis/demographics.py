import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from __init__ import *

from preprocessing.constants import MORALITY_ORIGIN


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


if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')

    for c in config:
        if c == 1:
            for attribute in ['Gender', 'Race']:
                plot_morality_evolution(interviews, attribute)