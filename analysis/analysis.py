import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from __init__ import *

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_matches


#Plot morality evolution over waves
def plot_morality_evolution(interviews):
    #Merge waves
    wave_1 = interviews[['Wave 1:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 1:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_2 = interviews[['Wave 2:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 2:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_3 = interviews[['Wave 3:' + mo for mo in MORALITY_ORIGIN]].rename(columns={'Wave 3:' + mo: mo for mo in MORALITY_ORIGIN})
    wave_1['Wave'] = 1
    wave_2['Wave'] = 2
    wave_3['Wave'] = 3
    interviews = pd.concat([wave_1, wave_2, wave_3])

    #Prepare for plotting
    interviews = pd.melt(interviews, id_vars=['Wave'], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Value')
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=interviews, y='Value', x='Wave', hue='Morality Origin', linewidth=4, palette='Set2')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.ylabel('')
    plt.xticks([1,2,3])
    plt.title('Morality Evolution')
    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1.03))
    for line in legend.get_lines():
        line.set_linewidth(4)
    plt.savefig('data/plots/evaluation-morality_evolution.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    interviews = pd.read_pickle('data/cache/morality_embeddings_entail-explained.pkl')

    for c in config:
        if c == 1:
            interviews = merge_matches(interviews)
            plot_morality_evolution(interviews)