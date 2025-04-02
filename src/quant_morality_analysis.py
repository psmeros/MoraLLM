import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

from __init__ import *
from src.helpers import CODED_WAVES, DEMOGRAPHICS, INCOME_RANGE, MORALITY_ORIGIN
from src.parser import prepare_data

#Compute overall morality distribution
def compute_distribution(interviews, model):
    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in CODED_WAVES]).reset_index(drop=True)
    data = data.melt(value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=data, x='Value', y='Morality', hue='Morality', hue_order=MORALITY_ORIGIN, kind='hist', legend=False, aspect=2, palette='Set2')
    g.figure.suptitle('Overall Distribution', y= 1.05, x=.5)
    g.set_ylabels('')
    g.set_xlabels('')
    g.set(xlim=(0, 100))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distro.png', bbox_inches='tight')
    plt.show()

#Show benefits of quantification by plotting ecdf
def plot_ecdf(interviews, model):
    #Prepare Data
    model = pd.DataFrame(interviews[['Wave 1:' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).assign(Estimator='NLI').assign(Estimator='NLI').dropna()
    crowd = pd.DataFrame(interviews[['Wave 1:' + mo + '_' + 'crowd' for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN).assign(Estimator='NLI').assign(Estimator='Crowd').dropna()
    data = pd.concat([model, crowd])
    data = data.melt(id_vars=['Estimator'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.displot(data=data, x='Value', hue='Morality', col='Estimator', kind='ecdf', linewidth=5, aspect=.85, palette=sns.color_palette('Set2')[:len(MORALITY_ORIGIN)])
    g.figure.suptitle('Cumulative Distribution Function', y=1.05)
    g.set_titles('{col_name}')
    g.legend.set_title('')
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y * 100:.0f}%'))
    plt.savefig('data/plots/fig-morality_ecdf.png', bbox_inches='tight')
    plt.show()

#Compute morality evolution across waves
def plot_morality_evolution(interviews, model, waves):
    #Prepare data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN] + [wave + ':' + d for d in DEMOGRAPHICS]].values, columns=MORALITY_ORIGIN + DEMOGRAPHICS).assign(Wave=int(wave.split()[1])) for wave in waves]).dropna().reset_index(drop=True)
    data = data.melt(id_vars=['Wave']+DEMOGRAPHICS, value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
    data['Value'] = pd.to_numeric(data['Value']) * 100

    data['Race'] = data['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    data['Household Income'] = data['Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    for demographic in DEMOGRAPHICS:
        data[demographic] = data[demographic].map(lambda x: x + ' (N = ' + str(int(len(data[data[demographic] == x])/len(MORALITY_ORIGIN)/len(waves))) + ')')

    #Plot overall morality evolution
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(data=data, x='Wave', y='Value', hue='Morality', hue_order=MORALITY_ORIGIN, seed=42, palette='Set2')
    ax.set_ylim(20, 100)
    ax.set_yticks([20, 60, 100])
    ax.set_xticks([int(w.split()[1]) for w in waves])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(title='', frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.suptitle('Morality Evolution over Waves', y=1)
    plt.savefig('data/plots/fig-morality_evolution_overall.png', bbox_inches='tight')
    plt.show()

    #Plot morality evolution by demographic
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=3)
    plt.figure(figsize=(20, 10))
    _, axes = plt.subplots(len(MORALITY_ORIGIN), len(DEMOGRAPHICS), figsize=(40, 10 * len(MORALITY_ORIGIN)))
    for i, morality in enumerate(MORALITY_ORIGIN):
        for j, demographic in enumerate(DEMOGRAPHICS):
            sns.lineplot(data=data[data['Morality'] == morality], x='Wave', y='Value', hue=demographic, seed=42, ax=axes[i, j], palette='Set1')
            axes[i, j].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
            axes[i, j].set_ylim(20, 100)
            axes[i, j].set_yticks([20, 60, 100])
            axes[i, j].set_xticks([int(w.split()[1]) for w in waves])
            axes[i, j].spines['top'].set_visible(False)
            axes[i, j].spines['right'].set_visible(False)
            axes[i, j].legend(title=demographic, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3) if i == 0 else axes[i, j].legend().set_visible(False)
            axes[i, j].set_xlabel('Wave')
            axes[i, j].set_ylabel(morality if j == 0 else '')
    plt.tight_layout()
    plt.suptitle('Morality Evolution over Waves', y=1.02)
    plt.savefig('data/plots/fig-morality_evolution_by_demographic.png', bbox_inches='tight')

#Compute morality shifts across waves
def plot_morality_shift(interviews, model, waves):
    data = interviews.copy()

    #Compute morality shifts across waves
    shifts = []
    if len(waves) == 3:
        slice = data[data[[wave + ':Interview Code' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]].notna().all(axis=1)]
        for from_wave, to_wave in zip(['Wave 1', 'Wave 2'], ['Wave 2', 'Wave 3']):
            shift = pd.DataFrame(slice[[to_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - slice[[from_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
            shift[DEMOGRAPHICS] = slice[[from_wave + ':' + d for d in DEMOGRAPHICS]].values
            shift ['Count'] = .5
            shift = shift.melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
            shifts.append(shift)

        slice = data[data[[wave + ':Interview Code' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]].isna().any(axis=1)]
        for from_wave, to_wave in zip(['Wave 1', 'Wave 1', 'Wave 2'], ['Wave 2', 'Wave 3', 'Wave 3']):
            shift = pd.DataFrame(slice[[to_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - slice[[from_wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
            shift[DEMOGRAPHICS] = slice[[from_wave + ':' + d for d in DEMOGRAPHICS]].values
            shift ['Count'] = 1
            shift = shift.dropna().melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
            shifts.append(shift)
    elif len(waves) == 2:
        shift = pd.DataFrame(data[[waves[1] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values - data[[waves[0] + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
        shift[DEMOGRAPHICS] = data[[waves[0] + ':' + d for d in DEMOGRAPHICS]].values
        shift ['Count'] = 1
        shift = shift.dropna().melt(id_vars=DEMOGRAPHICS+['Count'], value_vars=MORALITY_ORIGIN, var_name='Morality', value_name='Value')
        shifts.append(shift)


    #Prepare data
    shifts = pd.concat(shifts).reset_index(drop=True)
    shifts['Value'] = shifts['Value'] * 100
    shifts['Race'] = shifts['Race'].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    shifts['Household Income'] = shifts['Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    for demographic in DEMOGRAPHICS:
        shifts[demographic] = shifts[demographic].map(lambda x: x + ' (N = ' + str(int(shifts[shifts[demographic] == x]['Count'].sum()/len(MORALITY_ORIGIN))) + ')')

    #Plot overall morality shifts
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='Value', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Morality Shift over Waves', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-25, 25))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_shift_overall.png', bbox_inches='tight')
    plt.show()

    #Plot morality shifts by demographic
    plt.figure(figsize=(20, 10))
    _, axes = plt.subplots(2, 2, figsize=(20, 10))
    for i, demographic in enumerate(DEMOGRAPHICS):
        sns.barplot(data=shifts, x='Value', y='Morality', hue=demographic, order=MORALITY_ORIGIN, dodge=0.3, ax=axes[i//2,i%2], errorbar=None, palette='Set1')
        axes[i//2,i%2].axvline(x=0, color='grey', linestyle='--', linewidth=1.5)
        axes[i//2,i%2].set_xlim(-25, 25)
        axes[i//2,i%2].set_xlabel('')
        axes[i//2,i%2].set_ylabel('')
        axes[i//2,i%2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
        axes[i//2,i%2].spines['top'].set_visible(False)
        axes[i//2,i%2].spines['right'].set_visible(False)
        axes[i//2,i%2].legend(title=demographic, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=3)
    plt.tight_layout()
    plt.suptitle('Morality Shift over Waves')
    plt.savefig('data/plots/fig-morality_shift_by_demographic.png', bbox_inches='tight')

#Plot Intuitive-Consequentialist and Social-Theistic Morality Distinction
def plot_morality_distinction(interviews, model, waves):

    #Prepare Data
    data = interviews.copy()
    data = pd.concat([pd.DataFrame(data[[wave + ':' + mo + '_' + model for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN) for wave in waves]).reset_index(drop=True)
    data = data * 100

    #Compute Distinction
    data = data.apply(lambda x: pd.Series([abs(x['Intuitive'] - x['Consequentialist']), abs(x['Social'] - x['Theistic'])]), axis=1)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 10))
    g = sns.jointplot(data=data, x=0, y=1, kind='hex', color='rosybrown')
    g.figure.suptitle('Morality Distinction', y=1.03)
    ax = plt.gca()
    ax.xaxis.set_ticks([0, 50, 100])
    ax.yaxis.set_ticks([0, 50, 100])
    ax.set_xlabel('Intuitive-Consequentialist Distinction')
    ax.set_ylabel('Social-Theistic Distinction')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_distinction.png', bbox_inches='tight')
    plt.show()

def compute_decisiveness(interviews, model, waves, decisive_threshold = .5):
    #Prepare Data
    data = interviews.copy()
    decisiveness_options = ['Rigidly Decisive', 'Ambivalent', 'Rigidly Indecisive']
    decisiveness = data.apply(lambda i: pd.Series(((i[waves[0] + ':' + mo + '_' + model] >= decisive_threshold), (i[waves[1] + ':' + mo + '_' + model] >= decisive_threshold)) for mo in MORALITY_ORIGIN), axis=1).set_axis(MORALITY_ORIGIN, axis=1)
    decisiveness = decisiveness.map(lambda d: decisiveness_options[0] if d[0] and d[1] else decisiveness_options[1] if not d[0] and d[1] else decisiveness_options[1] if d[0] and not d[1] else decisiveness_options[2] if not d[0] and not d[1] else '')
    
    decisiveness = decisiveness.apply(lambda x: x.value_counts(normalize=True) * 100).T
    decisiveness = decisiveness.stack().reset_index().rename(columns={'level_0':'Morality', 'level_1':'Decisiveness', 0:'Value'})

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=3.5)
    plt.figure(figsize=(10, 10))

    sns.barplot(data=decisiveness, y='Morality', x='Value', hue='Decisiveness', order=MORALITY_ORIGIN, hue_order=decisiveness_options, palette=sns.color_palette('coolwarm', n_colors=len(decisiveness_options)))
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y :.0f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Crosswave Decisiveness')
    plt.legend(bbox_to_anchor=(1, 1.03)).set_frame_on(False)
    plt.savefig('data/plots/fig-decisiveness.png', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    #Hyperparameters
    config = [6]
    to_latex = False
    model = 'nli_sum_bin'
    interviews = prepare_data([model])

    for c in config:
        if c == 1:
            compute_distribution(interviews, model)
        elif c == 2:
            plot_ecdf(interviews, model)
        elif c == 3:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_evolution(interviews, model, waves)
        elif c == 4:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_shift(interviews, model, waves)
        elif c == 5:
            waves = ['Wave 1', 'Wave 3']
            plot_morality_distinction(interviews, model, waves)
        elif c == 6:
            compute_decisiveness(interviews, model, ['Wave 1', 'Wave 3'])