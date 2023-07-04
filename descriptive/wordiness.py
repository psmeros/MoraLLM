import pandas as pd
import seaborn as sns
import spacy
from __init__ import *
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel
from wordcloud import WordCloud

from preprocessing.constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, REFINED_SECTIONS
from preprocessing.transcript_parser import wave_parser

def word_count(interviews_folder, output_file, wave=1, morality_breakdown=False, pos_count=True):
    interviews = wave_parser(interviews_folder, morality_breakdown)

    interviews = interviews[interviews['Wave'] == wave].reset_index(drop=True)

    pandarallel.initialize()
    nlp = spacy.load("en_core_web_lg")

    #count words in each section
    count = lambda section : 0 if pd.isna(section) else sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']]) if pos_count else len(section.split())
    word_counts = interviews[REFINED_SECTIONS].parallel_applymap(count)

    #split into interviewer and respondent word counts
    interviewer_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[0] + s for s in INTERVIEW_SECTIONS]]
    interviewer_word_counts.columns = INTERVIEW_SECTIONS
    respondent_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[1] + s for s in INTERVIEW_SECTIONS]]
    respondent_word_counts.columns = INTERVIEW_SECTIONS

    #handle missing sections
    interviewer_word_counts = interviewer_word_counts.replace(0, interviewer_word_counts.median())
    respondent_word_counts = respondent_word_counts.replace(0, respondent_word_counts.median())

    #merge dataframes
    interviewer_word_counts['Interview Participant'] = 'Interviewer'
    respondent_word_counts['Interview Participant'] = 'Respondent'
    word_counts = pd.concat([interviewer_word_counts, respondent_word_counts])
    word_counts = word_counts.join(interviews['Interview Code'])

    #save to cache
    word_counts.to_pickle(output_file)


def morality_wordiness(interviews_folder, dictionary_file, output_file):
    interviews = wave_parser(interviews_folder)
    interviews = interviews[['Wave', 'R:Morality']].dropna()
    interviews['Wave'] = interviews['Wave'].apply(lambda x: 'Wave ' + str(x))

    pandarallel.initialize()
    nlp = spacy.load("en_core_web_lg")

    #lemmatization
    lemmatize = lambda text : [token.lemma_ for token in nlp(text) if token.pos_ in ['NOUN', 'ADJ']]
    interviews['Morality Words'] = interviews['R:Morality'].parallel_apply(lemmatize)
    interviews = interviews[['Wave', 'Morality Words']]

    #cleaning
    interviews['Morality Words'] = interviews['Morality Words'].apply(lambda x: [word for word in x if word.isalpha()])
    
    #count unique words
    interviews['Nouns & Adjectives'] = interviews['Morality Words'].apply(lambda x: len(x))

    #count unique eMFD words
    dictionary = pd.DataFrame(pd.read_pickle(dictionary_file)).T
    dictionary = dictionary.reset_index(names=['word'])['word'].tolist()
    interviews['eMFD Nouns & Adjectives'] = interviews['Morality Words'].apply(lambda x: len([w for w in x if w in dictionary]))

    #save to cache
    interviews.to_pickle(output_file)


def plot_ratio(word_counts_cache):
    word_counts = pd.read_pickle(word_counts_cache)

    interviewer_counts = word_counts[word_counts['Interview Participant'] == 'Interviewer'][INTERVIEW_SECTIONS]
    respondent_counts = word_counts[word_counts['Interview Participant'] == 'Respondent'][INTERVIEW_SECTIONS]
    interview_ratio = interviewer_counts / respondent_counts
    interview_ratio = interview_ratio.dropna(axis='columns')
    
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
    plt.savefig('data/plots/wordiness_ratio.png', bbox_inches='tight')
    plt.show()
        
def plot_distribution(word_counts_cache):
    word_counts = pd.read_pickle(word_counts_cache)

    word_counts = word_counts.melt(id_vars=['Interview Participant', 'Interview Code'], var_name='Section', value_name='Word Count')
    word_counts = word_counts[word_counts['Word Count'] != 0]

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))

    color_palette = sns.color_palette('icefire')
    ax = sns.violinplot(data=word_counts, y='Section', x='Word Count', hue='Interview Participant', split=True, inner='quart', linewidth=1, cut=0, scale='width', scale_hue=False, palette=[color_palette[-1], color_palette[0]])
    ax.legend(title='Interview Participant', loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('Wordiness Distribution')
    ax.set_xlabel('')
    ax.set_ylabel('')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/plots/wordiness_distribution.png', bbox_inches='tight')
    plt.show()

def plot_morality_wordiness(morality_wordiness_file):
    morality_wordiness = pd.read_pickle(morality_wordiness_file)
    morality_wordiness = morality_wordiness[['Wave', 'Nouns & Adjectives', 'eMFD Nouns & Adjectives']].melt(id_vars=['Wave'], var_name='Type', value_name='Counts')

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))

    color_palette = sns.color_palette('icefire')
    ax = sns.barplot(data=morality_wordiness, y='Wave', x='Counts', hue='Type', palette=color_palette)
    ax.legend(title='Morality Section Counts', loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_xlabel('')
    ax.set_ylabel('')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig('data/plots/morality_wordiness.png', bbox_inches='tight')
    plt.show()

def plot_morality_wordcloud(morality_wordiness_file):
    morality_wordiness = pd.read_pickle(morality_wordiness_file)
    words = morality_wordiness.groupby('Wave')['Morality Words'].sum().reset_index(name='Morality Words')
    words['Morality Words'] = words['Morality Words'].apply(lambda l: ' '.join([w.strip() for w in l if w not in ['people', 'stuff', 'thing', 'lot', 'time', 'way']]))

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))
    wordcloud = WordCloud(background_color='white', collocations=False, contour_width=0.1, contour_color='black',  max_font_size=150, random_state=42, colormap='Dark2')
    for i in range (len(words)):
        plt.subplot(len(words), 1, i+1)
        wc = wordcloud.generate(words['Morality Words'].iloc[i])
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(words['Wave'].iloc[i])
    plt.tight_layout()
    plt.savefig('data/plots/morality_wordcloud.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    word_count(interviews_folder='data/waves', output_file='data/cache/word_counts.pkl')
    plot_ratio(word_counts_cache='data/cache/word_counts.pkl')
    plot_distribution(word_counts_cache='data/cache/word_counts.pkl')

    morality_wordiness(interviews_folder='data/waves', dictionary_file='data/misc/eMFD.pkl', output_file='data/cache/morality_wordiness.pkl')
    plot_morality_wordiness(morality_wordiness_file='data/cache/morality_wordiness.pkl')
    plot_morality_wordcloud(morality_wordiness_file='data/cache/morality_wordiness.pkl')