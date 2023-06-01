from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from transcript_parser import wave_parser
from constants import INTERVIEW_PARTICIPANTS, INTERVIEW_SECTIONS, MORALITY_QUESTIONS, REFINED_INTERVIEW_SECTIONS

interviews = wave_parser('data/wave_1')

def word_count(section):
    count = 0
    if not pd.isna(section):
        count = len(section.split())
    return count
# Get number of words per section

word_counts = interviews[REFINED_INTERVIEW_SECTIONS].applymap(word_count)

for participant in INTERVIEW_PARTICIPANTS:
    word_counts[participant+'Morality'] = word_counts[[participant+'Morality:'+question[:-1] for question in MORALITY_QUESTIONS]].sum(axis=1)

word_counts = word_counts[(word_counts[[participant + section for section in INTERVIEW_SECTIONS for participant in INTERVIEW_PARTICIPANTS]] > 5).all(axis=1)]

interviewer_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[0] + s for s in INTERVIEW_SECTIONS]]
interviewer_word_counts.columns = INTERVIEW_SECTIONS
respondent_word_counts = word_counts[[INTERVIEW_PARTICIPANTS[1] + s for s in INTERVIEW_SECTIONS]]
respondent_word_counts.columns = INTERVIEW_SECTIONS

interview_ratio = interviewer_word_counts / respondent_word_counts

#interview_ratio[(interview_ratio>100).any(axis=1)]
#interviews.iloc[interview_ratio[(interview_ratio>100).any(axis=1)].index]

sns.set(context='paper', style='white', color_codes=True, font_scale=1)
ax = sns.boxplot(data=interview_ratio, orient='h', whis=[0, 100], palette='vlag')
ax.set_title('Interviewer to Respondent Word Count Ratio')
#ax.set_xscale('log')
ax.set_xlabel('Ratio')
sns.despine(left=True, bottom=True)
plt.tight_layout()
#plt.savefig('ratio.png', bbox_inches='tight')
plt.show()

