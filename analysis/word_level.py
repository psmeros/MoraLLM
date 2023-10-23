import numpy as np
import pandas as pd
import pytextrank
import seaborn as sns
import spacy
from __init__ import *
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel
from wordcloud import WordCloud

from preprocessing.constants import CODERS, INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, MORALITY_ORIGIN, REFINED_SECTIONS
from preprocessing.metadata_parser import merge_codings
from preprocessing.transcript_parser import wave_parser


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
    plt.savefig('data/plots/word_level-morality_wordcloud.png', bbox_inches='tight')
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
    plt.savefig('data/plots/wordiness-general_ratio.png', bbox_inches='tight')
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
    plt.savefig('data/plots/wordiness-general_distribution.png', bbox_inches='tight')
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
    plt.savefig('data/plots/wordiness-morality_wordcount.png', bbox_inches='tight')
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
    plt.savefig('data/plots/wordiness-morality_wordcloud.png', bbox_inches='tight')
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
    plt.savefig('data/plots/wordiness-morality_unique_wordcloud.png', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    #Hyperparameters
    config = [3]
    interviews_folder='data/waves'
    eMFD_file='data/misc/eMFD.pkl'

    for c in config:
        if c == 1:
            plot_general_wordiness(interviews_folder)
        elif c == 2:
            plot_morality_wordiness(interviews_folder, eMFD_file)
        elif c == 3:
            interviews = pd.read_pickle('data/cache/morality_model-top.pkl')
            interviews = merge_codings(interviews)
            plot_morality_wordcloud(interviews)