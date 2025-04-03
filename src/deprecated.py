import os
from itertools import combinations
import re

import matplotlib.ticker as mtick
import numpy as np
import openai
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import spacy
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel
from scipy.spatial import ConvexHull, distance
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale, normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BartModel, BartTokenizer, BertModel, BertTokenizer
from transformers import pipeline
from transformers_interpret import ZeroShotClassificationExplainer
from wordcloud import WordCloud

from src.helpers import ADOLESCENCE_RANGE, CHURCH_ATTENDANCE_RANGE, CODED_WAVES, CODERS, DEMOGRAPHICS, EDUCATION_RANGE, INCOME_RANGE, INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, MORALITY_ESTIMATORS, MORALITY_ORIGIN, MORALITY_ORIGIN_EXPLAINED, REFINED_SECTIONS
from src.parser import merge_codings, prepare_data, wave_parser


# Find k embeddings with maximum distance
def k_most_distant_embeddings(embeddings, k):

    # Calculate pairwise distances between embeddings
    distances = pdist(embeddings.tolist(), metric='cosine')
    pairwise_distances = squareform(distances)

    # Iterate through all combinations and find the one with maximum sum of distances
    combinations_k = combinations(range(len(embeddings)), k)

    max_sum_distances = -np.inf
    optimal_combination = None

    for combination in combinations_k:
        sum_distances = np.sum(pairwise_distances[np.ix_(combination, combination)])
        if sum_distances > max_sum_distances:
            max_sum_distances = sum_distances
            optimal_combination = combination

    return list(optimal_combination)

#Explain word-level attention for zero-shot models
def explain_entailment(interviews):
    pairs = [(interviews.iloc[interviews[mo + '_x'].idxmax()]['Morality_Origin'], [mo]) for mo in MORALITY_ORIGIN]

    model_name = 'cross-encoder/nli-deberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    zero_shot_explainer = ZeroShotClassificationExplainer(model, tokenizer)

    for text, labels in pairs:
        zero_shot_explainer(text=text, hypothesis_template='The morality origin is {}.',labels=labels)
        zero_shot_explainer.visualize('data/misc/zero_shot.html')

#Overfit model to codings
def inform_morality_origin_model(interviews):
    #Normalize scores
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].div(interviews[MORALITY_ORIGIN].sum(axis=1), axis=0).fillna(0.0)

    #Compute golden labels
    codings = merge_codings(interviews)[[mo + '_' + estimator for mo in MORALITY_ORIGIN for estimator in MORALITY_ESTIMATORS]].dropna()
    model_labels = pd.DataFrame(codings[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)
    golden_labels = pd.DataFrame(codings[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values, columns=MORALITY_ORIGIN)

    #Compute coefficients for more accurate morality origin estimation
    coefs = {}
    for mo in MORALITY_ORIGIN:
        regr = LinearRegression(fit_intercept=False)
        regr.fit(model_labels[mo].values.reshape(-1, 1), golden_labels[mo].values.reshape(-1, 1))
        coefs[mo] = regr.coef_[0][0]
    coefs = pd.Series(coefs)

    #Multiply with coefficients and add random gaussian noise
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN] * coefs
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN].clip(lower=0.0, upper=1.0)
    interviews[MORALITY_ORIGIN] = interviews[MORALITY_ORIGIN] - pd.DataFrame(abs(np.random.default_rng(42).normal(0, 1e-1, interviews[MORALITY_ORIGIN].shape)), columns=MORALITY_ORIGIN) * (interviews[MORALITY_ORIGIN] > .99).astype(int)

    return interviews

#Return a SpaCy, BERT, or BART vectorizer
def get_vectorizer(model='lg', filter_POS=True):
    if model in ['bert', 'bart']:
        #Load the tokenizer and model
        if model == 'bert':
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        elif model == 'bart':
            model_name = 'facebook/bart-large-mnli'
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartModel.from_pretrained(model_name)

        def extract_embeddings(text):
            #Tokenize the input text
            input = tokenizer(text, return_tensors='pt')

            #Split the input text into chunks of max_chunk_length
            num_chunks = (input['input_ids'].size(1) - 1) // tokenizer.model_max_length + 1
            chunked_input_ids = torch.chunk(input['input_ids'], num_chunks, dim=1)
            chunked_attention_mask = torch.chunk(input['attention_mask'], num_chunks, dim=1)

            #Initialize an empty tensor to store the embeddings
            all_embeddings = []

            #Forward pass through the model to get the embeddings for each chunk
            with torch.no_grad():
                for (input_ids, attention_mask) in zip(chunked_input_ids, chunked_attention_mask):

                    #Input and Output of the transformer model
                    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                    outputs = model(**inputs, output_attentions=True)

                    #Extract the embeddings from the model's output (max-pooling)
                    embeddings = torch.max(outputs.last_hidden_state[0], dim=0, keepdim=True).values
                    all_embeddings.append(embeddings)

            #Concatenate and aggegate the embeddings from all chunks (max-pooling)
            embeddings = torch.max(torch.cat(all_embeddings, dim=0), dim=0).values.numpy()

            return embeddings
    
        vectorizer = lambda x: x.apply(extract_embeddings)

    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        validate_POS = lambda w: w.pos_ in ['NOUN', 'ADJ', 'VERB'] if filter_POS else True
        mean_word_vectors = lambda s: np.mean([w.vector for w in nlp(s) if validate_POS(w)], axis=0)
        vectorizer = lambda x: x.apply(mean_word_vectors)
 
    return vectorizer

#Compute eMFD embeddings and transformation matrix
def embed_eMFD(dictionary_file, model):
    #Load data
    dictionary = pd.DataFrame(pd.read_pickle(dictionary_file)).T
    dictionary = dictionary.reset_index(names=['word'])

    #Compute global embeddings
    vectorizer = get_vectorizer(model='lg', parallel=False, filter_POS=False)
    dictionary['Embeddings'] = vectorizer(dictionary['word'].str.lower())
    dictionary = dictionary.dropna(subset=['Embeddings'])

    moral_foundations = pd.DataFrame()

    for column in dictionary.columns:
        if column not in ['word', 'Embeddings']:
            moral_foundations[column] = sum(dictionary['Embeddings']*dictionary[column])/sum(dictionary[column])

    moral_foundations = moral_foundations.T
    moral_foundations['Global Embeddings'] = moral_foundations.apply(lambda x: np.array(x), axis=1)
    moral_foundations = moral_foundations[['Global Embeddings']]
    moral_foundations = moral_foundations.reset_index(names=['Name'])

    #Average Vice and Virtue embeddings
    moral_foundations['Name'] = moral_foundations['Name'].apply(lambda x: x.split('.')[0].capitalize())
    moral_foundations = moral_foundations.groupby('Name').mean().reset_index()

    #Compute local embeddings
    vectorizer = get_vectorizer(model=model, parallel=False, filter_POS=False)
    moral_foundations['Local Embeddings'] = vectorizer(moral_foundations['Name'].str.lower())

    #Drop empty embeddings
    moral_foundations = moral_foundations[moral_foundations.apply(lambda x: (sum(x['Local Embeddings']) != 0) & (sum(x['Global Embeddings']) != 0), axis=1)]

    #Find transformation matrix
    regressor = Ridge(random_state=42)
    regressor.fit(moral_foundations['Local Embeddings'].apply(pd.Series), moral_foundations['Global Embeddings'].apply(pd.Series))
    transformation_matrix = pd.DataFrame(regressor.coef_)

    return transformation_matrix

display_notification = lambda notification: os.system("osascript -e 'display notification \"\" with title \""+notification+"\"'")


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

def compute_std_diff(interviews, attributes):
    #Prepare Data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].map(lambda x: ADOLESCENCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Household Income'] = data[CODED_WAVES[0] + ':Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].map(lambda x: CHURCH_ATTENDANCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education'].map(lambda x: EDUCATION_RANGE.get(x, None))
    data = data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES] + [CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes]]

    #Melt Data
    data = data.melt(id_vars=[CODED_WAVES[0] + ':' + attribute['name'] for attribute in attributes], value_vars=[wave + ':' + mo  + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES], var_name='Morality', value_name='Value')
    data['Wave'] = data['Morality'].apply(lambda x: x.split(':')[0])
    data['Morality'] = data['Morality'].apply(lambda x: x.split(':')[1].split('_')[0])
    data = data.rename(columns = {CODED_WAVES[0] + ':' + attribute['name'] : attribute['name'] for attribute in attributes})

    #Compute Standard Deviation
    stds = []
    for attribute in attributes:
        for j, attribute_value in enumerate(attribute['values']):
            slice = data[data[attribute['name']] == attribute_value]
            N = int(len(slice)/len(MORALITY_ORIGIN)/len(CODED_WAVES))
            slice = slice.groupby(['Wave', 'Morality'])['Value'].std().reset_index()
            slice = slice[slice['Wave'] == CODED_WAVES[0]][['Morality', 'Value']].merge(slice[slice['Wave'] == CODED_WAVES[1]][['Morality', 'Value']], on='Morality', suffixes=('_0', '_1'))
            std = round(((slice['Value_1'] - slice['Value_0'])/slice['Value_0']).mean() * 100, 1)
            std = {'Attribute Name' : attribute['name'], 'Attribute Position' : j, 'Attribute Value' : attribute_value + ' (N = ' + str(N) + ')', 'STD' : std}
            stds.append(std)
    stds = pd.DataFrame(stds)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=5.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=stds, x='STD', y='Attribute Position', hue='Attribute Position', col='Attribute Name', sharey=False, col_wrap=2, orient='h', kind='bar', seed=42, aspect=4, legend=False, palette=sns.color_palette('Set2')[-2:])
    g.set(xlim=(-30, 0))
    g.figure.subplots_adjust(wspace=0.55)
    g.figure.suptitle('Standard Deviation Crosswave Shift', y=1.03)
    g.set_titles('{col_name}')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    for j, ax in enumerate(g.axes):
        ax.set_ylabel('')
        labels = stds.iloc[2*j:2*j+2]['Attribute Value'].to_list()
        ax.set(yticks=range(len(labels)), yticklabels=labels)
    plt.savefig('data/plots/fig-std_diff.png', bbox_inches='tight')
    plt.show()

#Compute crosswave consistency
def compute_consistency(interviews, plot_type, consistency_threshold):
    data = interviews.copy()
    data = data.dropna(subset=[wave + ':Interview Code' for wave in CODED_WAVES])
    
    #Prepare Data
    data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]] = minmax_scale(data[[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN for wave in CODED_WAVES]])
    consistency = data.apply(lambda i: pd.Series(abs(i[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]] - (i[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0]]) < consistency_threshold) for mo in MORALITY_ORIGIN), axis=1).set_axis([mo for mo in MORALITY_ORIGIN], axis=1)
    consistency = (consistency.mean()) * 100
    consistency = consistency.reset_index()
    consistency.columns = ['morality', 'r']
    consistency['morality-r'] = consistency['morality'] + consistency['r'].apply(lambda r: ' (' + str(round(r, 1))) + '%)'
    consistency['angles'] = np.linspace(0, 2 * np.pi, len(consistency), endpoint=False)
    consistency.loc[len(consistency)] = consistency.iloc[0]

    #Plot
    if plot_type == 'spider':
        plt.figure(figsize=(10, 10))
        _, ax = plt.subplots(subplot_kw=dict(polar=True))
        ax.plot(consistency['angles'], consistency['r'], linewidth=2, linestyle='solid', color='rosybrown', alpha=0.8)
        ax.fill(consistency['angles'], consistency['r'], 'rosybrown', alpha=0.7)
        ax.set_theta_offset(np.pi)
        ax.set_theta_direction(-1)
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        ax.set_xticks(consistency['angles'], [])
        num_levels = 4
        for i in range(1, num_levels + 1):
            level =  100 * i / num_levels
            level_values = [level] * len(consistency)
            ax.plot(consistency['angles'], level_values, color='gray', linestyle='--', linewidth=0.7)
        for i in range(len(consistency)):
            ax.plot([consistency['angles'].iloc[i], consistency['angles'].iloc[i]], [0, 100], color='gray', linestyle='-', linewidth=0.7)
        for i, (r, horizontalalignment, verticalalignment, rotation) in enumerate(zip([105, 115, 105, 115], ['right', 'center', 'left', 'center'], ['center', 'top', 'center', 'bottom'], [90, 0, -90, 0])):
            ax.text(consistency['angles'].iloc[i], r, consistency['morality-r'].iloc[i], size=15, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, rotation=rotation)
        ax.set_rlabel_position(0)
        plt.yticks([25, 50, 75, 100], [])
        plt.title('Crosswave Interviewees Consistency', y=1.15, size=20)
        plt.savefig('data/plots/fig-morality_consistency.png', bbox_inches='tight')
        plt.show()
    
    elif plot_type == 'bar':
        sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2)
        plt.figure(figsize=(20, 10))
        g = sns.catplot(data=consistency, x='r', y='morality', hue='morality', orient='h', order=MORALITY_ORIGIN, kind='bar', seed=42, aspect=2, legend=False, palette=sns.color_palette('Set2')[:4])
        g.set_xlabels('')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.title('Crosswave Interviewees Consistency')
        plt.savefig('data/plots/fig-morality_consistency.png', bbox_inches='tight')
        plt.show()

#Plot morality shifts
def plot_morality_shifts(interviews, attributes, shift_threshold):

    #Compute morality shifts across waves
    def compute_morality_shifts(interviews, attribute_name=None, attribute_value=None):
        #Prepare data 
        if attribute_name is not None:
            interviews = interviews[interviews[CODED_WAVES[0] + ':' + attribute_name] == attribute_value]
        N = len(interviews)

        wave_source = interviews[[CODED_WAVES[0] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]]
        wave_target = interviews[[CODED_WAVES[1] + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]]
        wave_source.columns = MORALITY_ORIGIN
        wave_target.columns = MORALITY_ORIGIN

        #Compute normalized shift
        outgoing = (wave_source - wave_target).clip(lower=0)
        incoming = pd.DataFrame(normalize((wave_target - wave_source).clip(lower=0), norm='l1'))

        #Compute shifts
        shifts = []
        for i in range(N):
            shift = pd.DataFrame(outgoing.iloc[i]).values.reshape(-1, 1) @ pd.DataFrame(incoming.iloc[i]).values.reshape(1, -1)
            shift = pd.DataFrame(shift, index=[CODED_WAVES[0] + ':' + mo for mo in MORALITY_ORIGIN], columns=[CODED_WAVES[1] + ':' + mo for mo in MORALITY_ORIGIN])
            shift = shift.stack().reset_index().rename(columns={'level_0':'source', 'level_1':'target', 0:'value'})

            shift['wave'] = shift.apply(lambda x: x['source'].split(':')[0] + '->' + x['target'].split(':')[0].split()[1], axis=1)
            shift['source'] = shift['source'].apply(lambda x: x.split(':')[-1])
            shift['target'] = shift['target'].apply(lambda x: x.split(':')[-1])
            source_shift = shift.drop('target', axis=1).rename(columns={'source':'morality'})
            source_shift['value'] = -source_shift['value']
            target_shift = shift.drop('source', axis=1).rename(columns={'target':'morality'})
            shift = pd.concat([source_shift, target_shift])
            shift = shift[abs(shift['value']) > shift_threshold]
            shift['value'] = shift['value'] * 100

            shifts.append(shift)
        shifts = pd.concat(shifts)

        return shifts, N

    #Prepare data
    data = interviews.copy()
    data[CODED_WAVES[0] + ':Adolescence'] = data[CODED_WAVES[0] + ':Age'].map(lambda x: ADOLESCENCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Household Income'] = data[CODED_WAVES[0] + ':Household Income'].map(lambda x: INCOME_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Church Attendance'] = data[CODED_WAVES[0] + ':Church Attendance'].map(lambda x: CHURCH_ATTENDANCE_RANGE.get(x, None))
    data[CODED_WAVES[0] + ':Parent Education'] = data[CODED_WAVES[0] + ':Parent Education'].map(lambda x: EDUCATION_RANGE.get(x, None))
    data = data.dropna(subset=[wave + ':' + mo + '_' + MORALITY_ESTIMATORS[0] for wave in CODED_WAVES for mo in MORALITY_ORIGIN])

    shifts, _ = compute_morality_shifts(data)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Crosswave Morality Diffusion', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-10, 10))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-morality_shift.png', bbox_inches='tight')
    plt.show()

    #Prepare data
    data[[wave + ':Race' for wave in CODED_WAVES]] = data[[wave + ':Race' for wave in CODED_WAVES]].map(lambda r: {'White': 'White', 'Black': 'Other', 'Other': 'Other'}.get(r, None))
    shifts = []
    legends = {}
    symbols = ['■ ', '▼ ']

    for attribute in attributes:
        legend = []
        for attribute_value in attribute['values']:
            shift, N = compute_morality_shifts(data, attribute_name=attribute['name'], attribute_value=attribute_value)
            if not shift.empty:
                shift['Attribute'] = attribute['name']
                legend.append(symbols[attribute['values'].index(attribute_value)] + attribute_value + ' (N = ' + str(N) + ')')
                shift['order'] = str(attribute['values'].index(attribute_value)) + shift['morality'].apply(lambda mo: str(MORALITY_ORIGIN.index(mo)))
                shifts.append(shift)
        legends[attribute['name']] = attribute['name'] + '\n' + ', '.join(legend) + '\n'

    shifts = pd.concat(shifts)
    shifts = shifts.sort_values(by='order')
    shifts['Attribute'] = shifts['Attribute'].map(legends)

    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=shifts, x='value', y='morality', hue='order', orient='h', order=MORALITY_ORIGIN, col='Attribute', col_order=legends.values(), col_wrap=3, kind='point', err_kws={'linewidth': 3}, dodge=.7, markers=['s']*len(MORALITY_ORIGIN)+['v']*len(MORALITY_ORIGIN), markersize=15, legend=False, seed=42, aspect=1.5, palette=2*sns.color_palette('Set2', n_colors=len(MORALITY_ORIGIN)))
    g.figure.suptitle('Crosswave Morality Diffusion by Social Categories', x=.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-25, 25))
    g.set_xlabels('')
    g.set_ylabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    g.set_titles('{col_name}')
    plt.subplots_adjust(wspace=.3)
    plt.savefig('data/plots/fig-morality_shift_by_attribute.png', bbox_inches='tight')
    plt.show()

#Compute synthetic dataset
def compute_synthetic_data(n=25):
    chatgpt_synthetic_prompt = lambda mo: """
    You are a helpful assistant that generates interview summaries.
    These summaries describe in one sentence how people make decisions based on: 1) their intuition, 2) the consequences of their actions, 3) social influences, 4) religious reasons.
    Examples of such summaries are:
    The respondent makes decisions based on their gut feelings and the values instilled by their parents, believing that right and wrong are clear to them, and they prioritize following the advice of respected adults over personal happiness or religious guidance.
    The respondent prioritizes paying off debts and supporting loved ones over material purchases, bases their moral decisions on conscience and the potential consequences of their actions, acknowledges the influence of their parents' advice in decision-making, and expresses a desire to emulate their mother's strength and resilience.
    The respondent prioritizes using a hypothetical inheritance to pay off their mother's bills and secure their daughter's future, driven by gratitude for their mother's sacrifices, while also expressing a growing reliance on religious beliefs to guide their understanding of right and wrong, influenced by fears of moral consequences and the desire to avoid a negative legacy for their child.
    Generate pairs of summaries where the first summary gives more importance to  """ + \
    ('intuition' if mo == 'Intuitive'  else 'the consequences of actions' if mo == 'Consequentialist' else 'social influences' if mo == 'Social' else 'religious reasons' if mo == 'Theistic' else '') + \
    """ and sometimes mentions also some of the other three factors, 
    and the second summary gives less importance to """ + \
    ('intuition' if mo == 'Intuitive'  else 'the consequences of actions' if mo == 'Consequentialist' else 'social influences' if mo == 'Social' else 'religious reasons' if mo == 'Theistic' else '') + '\n' \
    """ and sometimes mentions also some of the other three factors.
    Respond strictly with each pair in a new line, separated by the special character '%'."""

    #OpenAI API
    openai.api_key = os.getenv('OPENAI_API_KEY')
    synthesizer = lambda: [openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': chatgpt_synthetic_prompt(mo)},{'role': 'user','content': 'Generate strictly ' + str(n) + ' pairs without enumerating'}], temperature=.2, max_tokens=16384, frequency_penalty=0, presence_penalty=0, seed=42) for mo in MORALITY_ORIGIN]
    aggregator = lambda r: pd.DataFrame(r, index=MORALITY_ORIGIN)['choices'].apply(lambda c: c[0]['message']['content']).str.split('\n').explode().str.split('%').apply(pd.Series).reset_index().dropna().reset_index(drop=True)
    full_pipeline = lambda: aggregator(synthesizer())

    #Generate synthetic data
    data = full_pipeline()
    data.columns = ['Morality', 'Strong Summary', 'Weak Summary']
    data.to_pickle('data/cache/synthetic_data.pkl')

#Compute synthetic morality origin
def compute_synthetic_morality():
    data = pd.read_pickle('data/cache/synthetic_data.pkl')

    #Premise and hypothesis templates
    hypothesis_template = 'The reasoning in this example is based on {}.'
    model_params = {'device':0} if torch.cuda.is_available() else {}
    morality_pipeline = pipeline('zero-shot-classification', model='roberta-large-mnli', **model_params)

    #Trasformation functions
    classifier = lambda series: pd.Series(morality_pipeline(series.tolist(), list(MORALITY_ORIGIN_EXPLAINED.keys()), hypothesis_template=hypothesis_template, multi_label=True))
    aggregator = lambda r: pd.DataFrame([{MORALITY_ORIGIN_EXPLAINED[l]:s for l, s in zip(r['labels'], r['scores'])}]).max()
    
    #Classify morality origin and join results
    morality_origin = classifier(data['Strong Summary']).apply(aggregator)
    data = data.join(morality_origin)
    morality_origin = classifier(data['Weak Summary']).apply(aggregator)
    data = data.join(morality_origin, lsuffix='_strong', rsuffix='_weak')

    data['Distinction'] = data.apply(lambda d: d[d['Morality'] + '_strong'] - d[d['Morality'] + '_weak'], axis=1)
    data = data[['Morality', 'Strong Summary', 'Weak Summary', 'Distinction']]
    data.to_pickle('data/cache/synthetic_data.pkl')

#Plot morality distinction on synthetic data
def plot_morality_distinction():
    #Prepare Data
    data = pd.read_pickle('data/cache/synthetic_data.pkl')
    data['Distinction'] = data['Distinction'] * 100
    #Plot
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    g = sns.catplot(data=data, x='Distinction', y='Morality', hue='Morality', orient='h', order=MORALITY_ORIGIN, hue_order=MORALITY_ORIGIN, kind='point', err_kws={'linewidth': 3}, markersize=10, legend=False, seed=42, aspect=2, palette='Set2')
    g.figure.suptitle('Strong-Weak Morality Distinction', x=0.5)
    g.map(plt.axvline, x=0, color='grey', linestyle='--', linewidth=1.5)
    g.set(xlim=(-100, 100))
    g.set_ylabels('')
    g.set_xlabels('')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    plt.savefig('data/plots/fig-synthetic_distinction.png', bbox_inches='tight')
    plt.show()

#Plot coders agreement using Cohen's Kappa
def plot_coders_agreement():
    #Prepare data
    codings = merge_codings(None, return_codings=True)

    #Prepare heatmap
    coder_A = codings[[mo + '_' + CODERS[0] for mo in MORALITY_ORIGIN]].astype(int).values.T
    coder_B = codings[[mo + '_' + CODERS[1] for mo in MORALITY_ORIGIN]].astype(int).values.T
    heatmap = np.zeros((len(MORALITY_ORIGIN), len(MORALITY_ORIGIN)))
    
    for mo_A in range(len(MORALITY_ORIGIN)):
        for mo_B in range(len(MORALITY_ORIGIN)):
            heatmap[mo_A, mo_B] = cohen_kappa_score(coder_A[mo_A], coder_B[mo_B])
    heatmap = pd.DataFrame(heatmap, index=['Intuitive', 'Conseq.', 'Social', 'Theistic'], columns=['Intuitive', 'Conseq.', 'Social', 'Theistic'])

    #Plot coders agreement
    sns.set_theme(context='paper', style='white', color_codes=True, font_scale=2.5)
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap = sns.color_palette('pink_r', n_colors=4), square=True, cbar_kws={'shrink': .8}, vmin=-0.2, vmax=1)
    plt.ylabel('')
    plt.xlabel('')
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([-.05, .25, .55, .85])
    colorbar.set_ticklabels(['Poor', 'Slight', 'Moderate', 'Perfect'])
    plt.title('Cohen\'s Kappa Agreement between Annotators')
    plt.savefig('data/plots/fig-coders_agreement.png', bbox_inches='tight')
    plt.show()

#Prepare data for crowd labeling
def prepare_crowd_labeling(interviews):
    morality_text = 'Wave 1:Morality Text'
    interviews[morality_text] = interviews[morality_text].replace('', pd.NA)
    interviews = interviews[['Survey Id', morality_text]].dropna(subset=[morality_text]).reset_index(drop=True)
    interviews[morality_text] = interviews[morality_text].apply(lambda t: re.sub(r'I:', '<b>I: </b>', t)).apply(lambda t: re.sub(r'R:', '<b>R: </b>', t)).apply(lambda t: re.sub(r'\n', '<br>', t))
    interviews.to_clipboard(index=False, header=False)


if __name__ == '__main__':
    #Hyperparameters
    config = []
    interviews = prepare_data([])

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
        #morality inference level
        elif c == 7:
            plot_sankey_morality_shift(interviews)
        elif c == 8:
            actions = ['Pot', 'Drink', 'Cheat']
            n_clusters = 2
            plot_action_probability(interviews, actions=actions, n_clusters=n_clusters)
        elif c == 9:
            compare_deviations(interviews)
        elif c == 10:
            by_age = False
            compare_areas(interviews, by_age=by_age)
        elif c == 11:
            attributes = DEMOGRAPHICS
            compute_std_diff(interviews, attributes)
        elif c == 12:
            consistency_threshold = .1
            plot_type = 'spider'
            compute_consistency(interviews, plot_type, consistency_threshold)            
        elif c == 13:
            attributes = DEMOGRAPHICS
            shift_threshold = 0
            plot_morality_shifts(interviews, attributes, shift_threshold)
        elif c == 14:
            compute_synthetic_data()
            compute_synthetic_morality()
            plot_morality_distinction()
        elif c == 15:
            plot_coders_agreement()
        elif c == 16:
            prepare_crowd_labeling(interviews)
