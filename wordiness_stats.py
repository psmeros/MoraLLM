import pandas as pd
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel

from constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, REFINED_SECTIONS
from transcript_parser import wave_parser

pd.set_option('mode.chained_assignment', None)

def word_count(interviews_folder=None, morality_breakdown=False, pos_count=True, save_to_cache=True, word_counts_cache=None):
    if interviews_folder:
        interviews = wave_parser(interviews_folder, morality_breakdown)
        
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
        if save_to_cache:
            word_counts.to_pickle('data/cache/word_counts.pkl')

    elif word_counts_cache:
        word_counts = pd.read_pickle(word_counts_cache)
        
    return word_counts


def plot_ratio(word_counts, save=False):
    interview_ratio = word_counts[word_counts['Interview Participant'] == 'Interviewer'][INTERVIEW_SECTIONS] / word_counts[word_counts['Interview Participant'] == 'Respondent'][INTERVIEW_SECTIONS]
    
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
    if save:
        plt.savefig('data/plots/wordiness_ratio.png', bbox_inches='tight')
    plt.show()
        
def plot_distribution(word_counts, save=False):
    word_counts = word_counts.melt(id_vars=['Interview Participant', 'Interview Code'], var_name='Section', value_name='Word Count')

    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 15))

    color_palette = sns.color_palette('icefire')
    ax = sns.violinplot(data=word_counts, y='Section', x='Word Count', hue='Interview Participant', split=True, inner='quart', linewidth=1, cut=0, scale='width', scale_hue=False, palette=[color_palette[-1], color_palette[0]])
    ax.legend(title='Interview Participant', loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel('Wordiness Distribution')
    ax.set_ylabel('')

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if save:
        plt.savefig('data/plots/wordiness_distribution.png', bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    # word_counts = word_count(interviews_folder='data/wave_1')
    word_counts = word_count(word_counts_cache='data/cache/word_counts.pkl')
    plot_ratio(word_counts, save=True)
    plot_distribution(word_counts, save=True)