import pandas as pd
import seaborn as sns
import spacy
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from pandarallel import pandarallel

from constants import (INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS,
                       MORALITY_QUESTIONS, REFINED_INTERVIEW_SECTIONS)
from transcript_parser import wave_parser

pd.set_option('mode.chained_assignment', None)

def word_count(interviews_folder=None, word_counts_cache=None, pos_count=True, save_to_cache=True):
    if interviews_folder:
        interviews = wave_parser(interviews_folder)
        
        pandarallel.initialize()
        nlp = spacy.load("en_core_web_lg")

        def word_count(section):
            count = 0
            if not pd.isna(section):
                if pos_count:
                    count = sum([1 for token in nlp(section) if token.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']])
                else:
                    count = len(section.split())
            return count

        word_counts = interviews[REFINED_INTERVIEW_SECTIONS].parallel_applymap(word_count)

        for participant in INTERVIEW_PARTICIPANTS:
            word_counts[participant+'Morality'] = word_counts[[participant+'Morality:'+question[:-1] for question in MORALITY_QUESTIONS]].sum(axis=1)

        word_counts = word_counts[[participant + section for section in INTERVIEW_SECTIONS for participant in INTERVIEW_PARTICIPANTS]]
        word_counts = word_counts.join(interviews['Interview Code'])

        if save_to_cache:
            word_counts.to_pickle('data/cache/word_counts.pkl')

    elif word_counts_cache:
        word_counts = pd.read_pickle(word_counts_cache)
        
    return word_counts



word_counts = word_count(word_counts_cache='data/cache/word_counts.pkl')
# word_counts = word_count(interviews_folder='data/wave_1')

#split into interviewer and respondent word counts
interviewer_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[0] + s for s in INTERVIEW_SECTIONS] + ['Interview Code']]
interviewer_word_counts.columns = INTERVIEW_SECTIONS + ['Interview Code']
interviewer_word_counts['Interview Participant'] = 'Interviewer'
respondent_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[1] + s for s in INTERVIEW_SECTIONS] + ['Interview Code']]
respondent_word_counts.columns = INTERVIEW_SECTIONS + ['Interview Code']
respondent_word_counts['Interview Participant'] = 'Respondent'

#handle missing sections
interviewer_word_counts[INTERVIEW_SECTIONS] = interviewer_word_counts[INTERVIEW_SECTIONS].replace(0, interviewer_word_counts[INTERVIEW_SECTIONS].median())
respondent_word_counts[INTERVIEW_SECTIONS] = respondent_word_counts[INTERVIEW_SECTIONS].replace(0, respondent_word_counts[INTERVIEW_SECTIONS].median())


word_counts = pd.concat([interviewer_word_counts, respondent_word_counts])



def plot_ratio(word_counts):
    interview_ratio = word_counts[word_counts['Interview Participant'] == 'Interviewer'][INTERVIEW_SECTIONS] / word_counts[word_counts['Interview Participant'] == 'Respondent'][INTERVIEW_SECTIONS]
    
    sns.set(context='paper', style='white', color_codes=True, font_scale=4)
    plt.figure(figsize=(20, 20))

    color_palette = sns.color_palette('vlag')
    color_palette = {section: color_palette[-1] if median > 1 else color_palette[0] for section, median in interview_ratio.median().items()}
    ax = sns.boxplot(data=interview_ratio, orient='h', whis=[0, 100], palette=color_palette)
    ax.set_title('Interviewer to Respondent Wordiness Ratio')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xticks([0.1, 1, 10], ['0.1', '1', '10'])

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    #plt.savefig('ratio.png', bbox_inches='tight')
    plt.show()

plot_ratio(word_counts)

#word_counts = word_counts.melt(id_vars=['Interview Participant', 'Interview Code'], var_name='Section', value_name='Word Count')



# word_counts[word_counts['Word Count'] > 2000]

#interview_ratio[(interview_ratio>100).any(axis=1)]
#interviews.iloc[interview_ratio[(interview_ratio>100).any(axis=1)].index]


# word_counts['Word Count'].plot(kind='hist', bins=30, logx=True)

# word_counts['Word Count'] = word_counts['Word Count'].astype(int)


# ax = sns.violinplot(data=word_counts, y='Section', x='Word Count', hue='Interview Participant', split=True, inner='quart', linewidth=1)
# ax.set_title('Word Count by Interview Section')




# if __name__ == '__main__':
#     word_count(interviews_folder='data/wave_1')