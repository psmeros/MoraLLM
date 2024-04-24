from itertools import combinations

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import spacy
from __init__ import *
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel
from scipy.spatial import ConvexHull, distance
from scipy.stats import pearsonr, zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import minmax_scale
from wordcloud import WordCloud

from preprocessing.constants import CODED_WAVES, CODERS, INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, MORALITY_ESTIMATORS, MORALITY_ORIGIN, REFINED_SECTIONS
from preprocessing.transcript_parser import merge_codings, merge_matches, merge_surveys, wave_parser

#Plot wordcloud for each morality origin
def plot_morality_wordcloud(interviews):
    nlp = spacy.load('en_core_web_lg')
    nlp.add_pipe('textrank')
    num_of_labels = int(np.ceil(interviews[[mo + '_' + c for mo in MORALITY_ORIGIN for c in CODERS]].sum(axis=1).mean()/len(CODERS)))
    lemmatize = lambda text, pos, blacklist: ' '.join([word.lemma_ for phrase in nlp(text)._.phrases[:num_of_labels] for word in nlp(phrase.text) if word.pos_ in pos and word.lemma_ not in blacklist])

    #Plot wordcloud
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 30))
    wordcloud = WordCloud(background_color='white', collocations=False, contour_width=0.1, contour_color='black',  max_font_size=150, random_state=42, colormap='Set2')

    for i, mo in enumerate(MORALITY_ORIGIN):
        text = interviews[interviews[mo + '_' + CODERS[0]] & interviews[mo + '_' + CODERS[1]]]['Morality_Origin']
        text = text.apply(lambda t: lemmatize(t, ['NOUN', 'PROPN'], ['thing', 'people', 'stuff', 'way', 'trouble', 'life'])).str.cat(sep=' ')
        plt.subplot(len(MORALITY_ORIGIN), 1, i+1)
        plt.imshow(wordcloud.generate(text), interpolation='bilinear')
        plt.axis('off')
        plt.title(mo)
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-morality_wordcloud.png', bbox_inches='tight')
    plt.show()

#Plot general wordiness statistics
def plot_general_wordiness(interviews_folder):
    #Load data
    interviews = wave_parser(interviews_folder)

    #Count words in each section
    pandarallel.initialize()
    nlp = spacy.load("en_core_web_lg")
    count = lambda section : 0 if pd.isna(section) else sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']])
    word_counts = interviews[REFINED_SECTIONS].parallel_applymap(count)

    #Split into interviewer and respondent word counts
    interviewer_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[0] + s for s in INTERVIEW_SECTIONS]]
    interviewer_word_counts.columns = INTERVIEW_SECTIONS
    respondent_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[1] + s for s in INTERVIEW_SECTIONS]]
    respondent_word_counts.columns = INTERVIEW_SECTIONS

    #Handle missing sections
    interviewer_word_counts = interviewer_word_counts.replace(0, interviewer_word_counts.median())
    respondent_word_counts = respondent_word_counts.replace(0, respondent_word_counts.median())

    #Merge dataframes
    interviewer_word_counts['Interview Participant'] = 'Interviewer'
    respondent_word_counts['Interview Participant'] = 'Respondent'
    word_counts = pd.concat([interviewer_word_counts, respondent_word_counts])
    word_counts = word_counts.join(interviews['Interview Code'])

    #Prepare data
    interviewer_counts = word_counts[word_counts['Interview Participant'] == 'Interviewer'][INTERVIEW_SECTIONS]
    respondent_counts = word_counts[word_counts['Interview Participant'] == 'Respondent'][INTERVIEW_SECTIONS]
    interview_ratio = interviewer_counts / respondent_counts
    interview_ratio = interview_ratio.dropna(axis='columns')

    interview_distribution = word_counts.melt(id_vars=['Interview Participant', 'Interview Code'], var_name='Section', value_name='Word Count')
    interview_distribution = interview_distribution[interview_distribution['Word Count'] != 0]
    
    #Plot ratio
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))
    color_palette = sns.color_palette('icefire')
    palette = {section: color_palette[-1] if median > 1 else color_palette[0] for section, median in interview_ratio.median().items()}
    ax = sns.boxplot(data=interview_ratio, orient='h', whis=[0, 100], palette=palette)
    ax.axvline(x = 1, color=color_palette[int(len(color_palette)/2)], linestyle='--', linewidth=5)
    ax.set_title('Interviewer vs Respondent Wordiness')
    ax.set_xlabel('Ratio')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.1, 1, 10], ['1:10', '1:1', '10:1'])
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-general_ratio.png', bbox_inches='tight')
    plt.show()

    #Plot distribution
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))
    color_palette = sns.color_palette('icefire')
    ax = sns.violinplot(data=interview_distribution, y='Section', x='Word Count', hue='Interview Participant', split=True, inner='quart', linewidth=1, cut=0, scale='width', scale_hue=False, palette=[color_palette[-1], color_palette[0]])
    ax.legend(title='Interview Participant', loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('Wordiness Distribution')
    ax.set_xlabel('')
    ax.set_ylabel('')
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-general_distribution.png', bbox_inches='tight')
    plt.show()

#Plot morality wordiness statistics
def plot_morality_wordiness(interviews_folder, eMFD_file):
    #Load data
    interviews = wave_parser(interviews_folder)
    interviews = interviews[['Wave', 'R:Morality']].dropna()
    interviews['Wave'] = interviews['Wave'].apply(lambda x: 'Wave ' + str(x))
    dictionary = pd.DataFrame(pd.read_pickle(eMFD_file)).T
    dictionary = dictionary.reset_index(names=['word'])['word'].tolist()

    #Lemmatize
    pandarallel.initialize()
    nlp = spacy.load("en_core_web_lg")
    lemmatize = lambda text, pos: [token.lemma_ for token in nlp(text) if token.pos_ in pos]
    interviews['Morality All Words'] = interviews['R:Morality'].parallel_apply(lambda t: lemmatize(t, ['NOUN', 'ADJ', 'VERB']))
    interviews['Morality Restricted Words'] = interviews['R:Morality'].parallel_apply(lambda t: lemmatize(t, ['NOUN', 'ADJ']))
    interviews = interviews[['Wave', 'Morality All Words', 'Morality Restricted Words']]

    #Clean
    interviews['Morality All Words'] = interviews['Morality All Words'].apply(lambda x: [word for word in x if word.isalpha()])
    interviews['Morality Restricted Words'] = interviews['Morality Restricted Words'].apply(lambda x: [word for word in x if word.isalpha()])

    #Count unique words
    interviews['Nouns & Adjectives'] = interviews['Morality Restricted Words'].apply(lambda x: len(x))
    #Count unique eMFD words
    interviews['eMFD Nouns & Adjectives'] = interviews['Morality Restricted Words'].apply(lambda x: len([w for w in x if w in dictionary]))

    #Prepare data    
    wordcount_data = interviews[['Wave', 'Nouns & Adjectives', 'eMFD Nouns & Adjectives']].melt(id_vars=['Wave'], var_name='Type', value_name='Counts')

    wordcloud_data = interviews.groupby('Wave')['Morality Restricted Words'].sum().reset_index(name='Morality Words')
    wordcloud_data['Morality Words'] = wordcloud_data['Morality Words'].apply(lambda l: ' '.join([w.strip() for w in l if w not in ['people', 'stuff', 'thing', 'lot', 'time', 'way']]))

    unique_wordcloud_data = interviews.groupby('Wave')['Morality All Words'].apply(lambda l: l.explode().tolist()).reset_index(name='All Words')
    unique_wordcloud_data['Unique Words'] = unique_wordcloud_data['All Words'].apply(set)
    unique_wordcloud_data['Unique Words'] = unique_wordcloud_data.apply(lambda i: i['Unique Words'] - set(unique_wordcloud_data['Unique Words'][list(set(range(3))-set([int(i['Wave'][-1])-1]))].explode().unique()), axis=1)
    unique_wordcloud_data['Unique Words'] = unique_wordcloud_data.apply(lambda i: [w for w in i['All Words'] if w in i['Unique Words']], axis=1)
    unique_wordcloud_data['Unique Words'] = unique_wordcloud_data['Unique Words'].apply(lambda l: ' '.join([w.strip() for w in l if w not in ['hm', 'uhm', 'mhum', 'ok', 'bowl', 'uhh', 'sorta']]))

    #Plot wordcount
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))

    color_palette = sns.color_palette('icefire')
    ax = sns.barplot(data=wordcount_data, y='Wave', x='Counts', hue='Type', palette=color_palette)
    ax.legend(title='Morality Section Counts', loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_xlabel('')
    ax.set_ylabel('')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-morality_wordcount.png', bbox_inches='tight')
    plt.show()

    #Plot wordcloud
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))
    wordcloud = WordCloud(background_color='white', collocations=False, contour_width=0.1, contour_color='black',  max_font_size=150, random_state=42, colormap='Dark2')
    for i in range (len(wordcloud_data)):
        plt.subplot(len(wordcloud_data), 1, i+1)
        wc = wordcloud.generate(wordcloud_data['Morality Words'].iloc[i])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(wordcloud_data['Wave'].iloc[i])
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-morality_wordcloud.png', bbox_inches='tight')
    plt.show()

    #Plot unique wordcloud
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))
    wordcloud = WordCloud(background_color='white', collocations=False, contour_width=0.1, contour_color='black',  max_font_size=150, random_state=42, colormap='Dark2')
    for i in range (len(unique_wordcloud_data)):
        plt.subplot(len(unique_wordcloud_data), 1, i+1)
        wc = wordcloud.generate(unique_wordcloud_data['Unique Words'].iloc[i])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(unique_wordcloud_data['Wave'].iloc[i])
    plt.tight_layout()
    plt.savefig('data/plots/deprecated-morality_unique_wordcloud.png', bbox_inches='tight')
    plt.show()

#Plot morality embeddings of all waves
def plot_morality_embeddings(interviews, dim_reduction='TSNE', perplexity=5):
    #Reduce dimensionality
    dim_reduction_alg = TSNE(n_components=2, perplexity=perplexity, random_state=42) if dim_reduction == 'TSNE' else PCA(n_components=2, random_state=42) if dim_reduction == 'PCA' else None
    interviews = interviews[['Name']].join(pd.DataFrame(dim_reduction_alg.fit_transform(interviews['Embeddings'].apply(pd.Series))))
    interviews = interviews.dropna()

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))
    color_palette = sns.color_palette('Set2')
    sns.kdeplot(data=interviews, x=0, y=1, hue='Name', fill=True, alpha=0.5, hue_order=['Wave 1', 'Wave 2', 'Wave 3'], palette=color_palette[:3])
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Morality Embeddings')
    plt.savefig('data/plots/deprecated-embeddings.png', bbox_inches='tight')
    plt.show()

#Plot moral foundations by wave
def plot_moral_foundations(interviews, moral_foundations):    
    #Compute similarity
    interviews = interviews.merge(moral_foundations, how='cross')
    interviews['Similarity'] = interviews.apply(lambda x: 1 - distance.cosine(x['Embeddings_x'], x['Embeddings_y']), axis=1)
    interviews = interviews[['Name_x', 'Name_y', 'Similarity']].rename(columns={'Name_x': 'Wave', 'Name_y': 'Anchor'})

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 20))
    ax = sns.boxplot(data=interviews, y='Anchor', x='Similarity', hue='Wave', hue_order=['Wave 1', 'Wave 2', 'Wave 3'], orient='h', palette='Set2')
    ax.legend(title='', loc='upper center', bbox_to_anchor=(0.3, -0.03), ncol=3)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Moral Foundations by Wave')
    plt.savefig('data/plots/deprecated-moral_foundations.png', bbox_inches='tight')
    plt.show()

    #Aggregate similarity
    print(interviews.groupby('Wave').mean(numeric_only=True))

#Plot semantic shift by morality origin
def plot_semantic_shift(interviews, wave_list=['Wave 1', 'Wave 2', 'Wave 3']):
    #Compute semantic shift between all pairs of waves
    compute_shift = lambda i: np.sum([distance.cosine(i[w[0] + ':R:Morality_Embeddings'], i[w[1] + ':R:Morality_Embeddings']) for w in list(combinations(wave_list, 2))])
    interviews['Shift'] = interviews.apply(compute_shift, axis=1)

    #Prepare data for plotting
    interviews = interviews[MORALITY_ORIGIN + ['Shift']]
    interviews = interviews.melt(id_vars=['Shift'], value_vars=MORALITY_ORIGIN, var_name='Morality Origin', value_name='Check')
    interviews = interviews[interviews['Check'] == True].drop(columns=['Check'])

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 10))
    ax = sns.boxplot(data=interviews, y='Morality Origin', x='Shift', orient='h', palette='Set2')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Semantic Shift by Morality Origin')
    plt.savefig('data/plots/deprecated-semantic_shift.png', bbox_inches='tight')
    plt.show()

    #Print order by median
    print(' < '.join(interviews.groupby('Morality Origin').median().sort_values(by='Shift').index.tolist()))    

#Plot silhouette score for each wave
def plot_silhouette_score(interviews):
    #Compute cosine distance between all pairs of interviews
    interviews = interviews.reset_index(names='id')
    interviews = interviews.merge(interviews, how='cross', suffixes=('', '_'))
    interviews['Distance'] = interviews.apply(lambda x: distance.cosine(x['Embeddings'], x['Embeddings_']), axis=1)
    interviews = interviews.drop(columns=['Embeddings', 'Embeddings_', 'id_'])

    #Average for each pair of interview-wave
    interviews = interviews.groupby(['id', 'Name', 'Name_']).mean().reset_index()
    interviews = interviews.pivot(index=['id', 'Name'], columns='Name_', values='Distance')
    interviews = interviews.reset_index().rename_axis(None, axis=1).drop(columns=['id'])

    #Compute Silhouette score
    interviews['b'] = interviews.apply(lambda i: min([i[w] for w in set(['Wave 1', 'Wave 3']) - set([i['Name']])]), axis=1)
    interviews['a'] = interviews.apply(lambda i: i[i['Name']], axis=1)
    interviews['Silhouette Score'] = (interviews['b'] - interviews['a']) / interviews[['a', 'b']].max(axis=1)
    interviews = interviews.sort_values(by='Name')

    #Plot
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 5))
    ax = sns.boxplot(data=interviews, y='Name', x='Silhouette Score', orient='h', palette='Set1')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Silhouette Score')
    plt.savefig('data/plots/deprecated-silhouette_score.png', bbox_inches='tight')
    plt.show()

#Plot morality evolution
def plot_morality_evolution(interviews, attributes):
    for attribute in attributes:
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
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
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
        plt.savefig('data/plots/deprecated-morality_evolution_by_'+attribute['name'].lower()+'.png', bbox_inches='tight')
        plt.show()

#Plot morality shift
def plot_sankey_morality_shift(interviews):
    figs = []
    #Compute shifts with matrix multiplication
    compute_morality_shifts = lambda _:None
    for estimator, position in zip(MORALITY_ESTIMATORS, [[0, .45], [.55, 1]]):
        shifts, _ = compute_morality_shifts(interviews, estimator)
        #Prepare data
        sns.set_palette('Set2')
        mapping = {wave+':'+mo:j+i*len(MORALITY_ORIGIN) for i, wave in enumerate(CODED_WAVES) for j, mo in enumerate(MORALITY_ORIGIN)}
        shifts['source'] = shifts['source'].map(mapping)
        shifts['target'] = shifts['target'].map(mapping)
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
    fig.write_image('data/plots/deprecated-morality_shift.png')
    fig.show()

def plot_class_movement(interviews):
    #Prepare data
    interviews['Household Income Change'] = (interviews[CODED_WAVES[1] + ':Income (raw)'] - interviews[CODED_WAVES[0] + ':Income (raw)'])
    interviews['Household Income Change'] = pd.to_numeric(interviews['Household Income Change'])

    interviews = pd.concat([pd.melt(interviews, id_vars=['Household Income Change'], value_vars=[CODED_WAVES[0] + ':' + mo + '_' + e for mo in MORALITY_ORIGIN], var_name='Morality Origin', value_name='Value').dropna() for e in MORALITY_ESTIMATORS])
    interviews['Estimator'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[1])
    interviews['Morality Origin'] = interviews['Morality Origin'].apply(lambda x: x.split('_')[0].split(':')[1])
    interviews['Value'] = interviews['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.lmplot(data=interviews, x='Household Income Change', y='Value', hue='Estimator', col='Morality Origin', truncate=False, x_jitter=.3, seed=42, aspect=1.2, palette=sns.color_palette('Set1'))
    g.set_ylabels('')
    g.set_titles('Morality: {col_name}')
    g.fig.subplots_adjust(wspace=0.1)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax.set_xlim(-abs(interviews['Household Income Change']).max(), abs(interviews['Household Income Change']).max())
    plt.savefig('data/plots/deprecated-class_movement.png', bbox_inches='tight')
    plt.show()

def plot_action_probability(interviews, n_clusters, actions):
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
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
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
    plt.savefig('data/plots/deprecated-action_probability', bbox_inches='tight')
    plt.show()

def action_prediction(interviews, actions):
    #Train Classifier
    action_prediction = []
    for action in actions:
        for estimator in MORALITY_ESTIMATORS:
            input_interviews = interviews[[CODED_WAVES[0] + ':' + mo + '_' + estimator  for mo in MORALITY_ORIGIN]+[CODED_WAVES[0] + ':' + action]].dropna()
            y = input_interviews[CODED_WAVES[0] + ':' + action].apply(lambda d: False if d == 1 else True).values
            X = input_interviews.drop([CODED_WAVES[0] + ':' + action], axis=1).values

            classifier = LogisticRegressionCV(cv=5, random_state=42, fit_intercept=False, scoring=make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')))
            classifier.fit(X, y)
            score = np.asarray(list(classifier.scores_.values())).reshape(-1)
            coefs = {mo:coef for mo, coef in zip(MORALITY_ORIGIN, classifier.coef_[0])}
            
            action_prediction.append(pd.DataFrame({'F1-Weighted Score' : score, 'Coefs' : [coefs] * len(score), 'Action' : action, 'Estimator' : estimator}))
    action_prediction = pd.concat(action_prediction)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(action_prediction, x='F1-Weighted Score', y='Action', hue='Estimator', hue_order=MORALITY_ESTIMATORS, orient='h', palette='Set1')
    ax.set_ylabel('')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Estimator')
    plt.savefig('data/plots/deprecated-action_prediction.png', bbox_inches='tight')
    plt.show()

    action_prediction = action_prediction.drop_duplicates(subset=['Action', 'Estimator', 'F1-Weighted Score']).groupby('Action')['Coefs'].apply(list).apply(lambda l: (pd.Series(l[0]) + pd.Series(l[1])).idxmax())
    action_prediction = pd.DataFrame(action_prediction.reindex(actions).reset_index().values, columns=['Action', 'Key Morality'])
    print(action_prediction)

def moral_consciousness(interviews, outlier_threshold):
    if outlier_threshold:
        outliers = pd.DataFrame([abs(zscore(interviews[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0]])) > outlier_threshold for wave in CODED_WAVES for mo in MORALITY_ORIGIN]).any()
        interviews = interviews[~outliers]
    desicion_taking = pd.get_dummies(interviews[CODED_WAVES[0] + ':' + 'Decision Taking'])
    Age = interviews[CODED_WAVES[0] + ':Age'].dropna().astype(int)
    Grades = interviews[CODED_WAVES[0] + ':Grades'].dropna().astype(int)
    Gender = pd.factorize(interviews[CODED_WAVES[0] + ':Gender'])[0]
    Race = pd.factorize(interviews[CODED_WAVES[0] + ':Race'])[0]
    Church_Attendance = interviews[CODED_WAVES[0] + ':Church Attendance'].dropna()
    Parent_Education = interviews[CODED_WAVES[0] + ':Parent Education (raw)'].dropna()
    Parent_Income = interviews[CODED_WAVES[0] + ':Income (raw)'].dropna()

    compute_correlation = lambda x: str(round(x[0], 3)).replace('0.', '.') + ('***' if float(x[1])<.005 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '')

    data = []
    correlations = pd.DataFrame(columns=[estimator + ':' + wave for estimator in MORALITY_ESTIMATORS for wave in CODED_WAVES], index=['Intuitive - Consequentialist', 'Social - Consequentialist', 'Intuitive - Social', 'Intuitive - Expressive Individualist', 'Intuitive - Utilitarian Individualist', 'Intuitive - Relational', 'Intuitive - Theistic', 'Intuitive - Age', 'Intuitive - GPA', 'Intuitive - Gender', 'Intuitive - Race', 'Intuitive - Church Attendance', 'Intuitive - Parent Education', 'Intuitive - Parent Income', 'Theistic - Church Attendance'])
    for estimator in MORALITY_ESTIMATORS:
        for wave in CODED_WAVES:
            Intuitive = interviews[wave + ':Intuitive_' + estimator]
            Consequentialist = interviews[wave + ':Consequentialist_' + estimator]
            Social = interviews[wave + ':Social_' + estimator]

            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Consequentialist.values), pd.Series(['Intuitive - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Social.values), pd.Series(Consequentialist.values), pd.Series(['Social - Consequentialist']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))
            data.append(pd.concat([pd.Series(Intuitive.values), pd.Series(Social.values), pd.Series(['Intuitive - Social']*len(interviews)), pd.Series([wave]*len(interviews)), pd.Series([estimator]*len(interviews))], axis=1))

            correlations.loc['Intuitive - Consequentialist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Consequentialist))
            correlations.loc['Social - Consequentialist', estimator + ':' + wave] = compute_correlation(pearsonr(Social, Consequentialist))
            correlations.loc['Intuitive - Social', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Social))
            
            correlations.loc['Intuitive - Expressive Individualist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Expressive Individualist']))
            correlations.loc['Intuitive - Utilitarian Individualist', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Utilitarian Individualist']))
            correlations.loc['Intuitive - Relational', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Relational']))
            correlations.loc['Intuitive - Theistic', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, desicion_taking['Theistic']))

            correlations.loc['Intuitive - Age', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Age.index], Age))
            correlations.loc['Intuitive - GPA', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Grades.index], Grades))
            correlations.loc['Intuitive - Gender', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Gender))
            correlations.loc['Intuitive - Race', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive, Race))
            correlations.loc['Intuitive - Church Attendance', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Church_Attendance.index], Church_Attendance))
            correlations.loc['Intuitive - Parent Education', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Parent_Education.index], Parent_Education))
            correlations.loc['Intuitive - Parent Income', estimator + ':' + wave] = compute_correlation(pearsonr(Intuitive.loc[Parent_Income.index], Parent_Income))
            
            correlations.loc['Theistic - Church Attendance', estimator + ':' + wave] = compute_correlation(pearsonr(interviews[wave + ':Theistic_' + estimator].loc[Church_Attendance.index], Church_Attendance))


    correlations.astype(str).to_csv('data/plots/deprecated-correlations.csv')
    print(correlations)
    data = pd.concat(data, axis=0, ignore_index=True)
    data.columns = ['x', 'y', 'Correlation', 'Wave', 'Estimator']

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 20))
    g = sns.lmplot(data=data[data['Estimator'] == 'Model'], x='x', y='y', row='Correlation', hue='Wave', seed=42, palette=sns.color_palette('Set1'))
    g.set_titles('{row_name}')
    g.set_xlabels('')
    g.set_ylabels('')
    plt.savefig('data/plots/deprecated-correlations', bbox_inches='tight')
    plt.show()

def compare_deviations(interviews):
    data = interviews[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]]
    stds = pd.Series([(np.std(interviews[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) - np.std(interviews[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]])) / np.std(interviews[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN)
    stds = stds.apply(lambda x: str(round(x * 100, 1)) + '%')
    data = data.melt(value_vars=[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    data['Wave'] = data['Morality'].apply(lambda x: x.split(':')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x + ' (σ: ' + stds[x] + ')')
    data['Value'] = data['Value'] * 100

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    g = sns.displot(data, y='Wave', x='Value', col='Morality', hue='Wave', bins=20, legend=False, palette='Set1')
    g.set_titles('{col_name}')
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/deprecated-deviation_comparison.png', bbox_inches='tight')
    plt.show()

def compare_areas(interviews, by_age):
    compute_convex_hull = lambda x: ConvexHull(np.diag(x).tolist()+[[0]*len(x)]).area
    if by_age:
        data = pd.concat([pd.DataFrame([interviews[wave + ':' + pd.Series(MORALITY_ORIGIN) + '_' + MORALITY_ESTIMATORS[0]].apply(lambda x: compute_convex_hull(x), axis=1).rename('Area'), interviews[wave + ':' + 'Age'].rename('Age')]).T for wave in CODED_WAVES])
        data = data.dropna()
        data['Age'] = data['Age'].astype(int)
        data['Area'] = data['Area'].astype(float)
    else:
        data = pd.DataFrame({wave : interviews[wave + ':' + pd.Series(MORALITY_ORIGIN) + '_' + MORALITY_ESTIMATORS[0]].apply(lambda x: compute_convex_hull(x), axis=1) for wave in CODED_WAVES})
        data = data.melt(value_vars=CODED_WAVES, var_name='Wave', value_name='Area')

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
    plt.figure(figsize=(10, 5))
    if by_age:
        ax = sns.regplot(data, y='Area', x='Age')
    else:
        ax = sns.boxplot(data, y='Wave', x='Area', hue='Wave', legend=False, orient='h', palette='Set1')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Convex Hull Area')
    plt.savefig('data/plots/deprecated-area_comparison.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #Hyperparameters
    config = [3]

    interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
    interviews = merge_surveys(interviews)
    interviews = merge_codings(interviews)
    interviews = merge_matches(interviews)

    for c in config:
        #word level
        if c == 1:
            interviews_folder='data/waves'
            plot_general_wordiness(interviews_folder)
        elif c == 2:
            interviews_folder='data/waves'
            eMFD_file='data/misc/eMFD.pkl'
            plot_morality_wordiness(interviews_folder, eMFD_file)
        elif c == 3:
            plot_morality_wordcloud(interviews)
        #vector space level
        elif c == 4:
            dim_reduction = 'TSNE'
            perplexity = 5
            plot_morality_embeddings(interviews, dim_reduction, perplexity)
        elif c == 5:
            plot_moral_foundations(interviews)
        elif c == 6:
            plot_silhouette_score(interviews)
        elif c == 7:
            plot_semantic_shift(interviews)
        #morality inference level
        elif c == 8:
            plot_morality_evolution(interviews)
        elif c == 9:
            plot_sankey_morality_shift(interviews)
        elif c == 10:
            plot_class_movement(interviews)
        elif c == 11:
            actions = ['Pot', 'Drink', 'Cheat']
            n_clusters = 2
            plot_action_probability(interviews, actions=actions, n_clusters=n_clusters)
        elif c == 12:
            actions=['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']
            action_prediction(interviews, actions=actions)
        elif c == 13:
            outlier_threshold = 2
            moral_consciousness(interviews, outlier_threshold=outlier_threshold)
        elif c == 14:
            compare_deviations(interviews)
        elif c == 15:
            by_age = False
            compare_areas(interviews, by_age=by_age)
