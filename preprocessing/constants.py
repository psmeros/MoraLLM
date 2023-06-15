TRANSCRIPT_ENCODING='utf-8'

INTERVIEW_SECTIONS = [  #wave_1
                        'Household',
                        'Friends',
                        'Family Relationships',
                        'Adult Involvements',
                        'Morality',
                        'Wellbeing',
                        'Religion',
                        'Religious Experience',
                        'Religious Practices',
                        'Individualization/De-Institutionalization',
                        'Evaluate Church',
                        'School',
                        'Volunteering',
                        'Dating',
                        'Sex',
                        'Media',
                        'Future Prospects',
                        #wave_2
                        'General Update',
                        'Substance',
                        'Institutional Evaluation',
                        'Organized Activities and Work',
                        'Romantic Relationships',
                        'Conclusion',
                        #wave_3
                        'General Orientation',
                        'Politics, Poverty, Policy',
                        'Consumerism',
                        'Childbearing']


INTERVIEW_METADATA = [  #wave_1
                        'M',
                        'W',
                        'Name of Interviewer',
                        'Date of Interview',
                        'Interview Location',
                        'Interview Code',
                        'Age',
                        'Gender',
                        'Race',
                        'Survey Religion',
                        'Survey Denom',
                        'Religious Affiliation',
                        'Physical Description',
                        #wave_2
                        'S',
                        #wave_3
                        'P']

INTERVIEW_COMMENTS =[   #all waves
                        '#IC:',
                        '#ICM:',
                        '#C:',
                        '#OC:',
                        '#I:',
                        '#NC:',
                        '#IN:',
                        '#FIELD NOTES',
                        '#END',
                        #wave_3
                        '#Interview File Name:',
                        '#NOTES']

INTERVIEW_PARTICIPANTS = ['I:', 'R:']

INTERVIEW_MARKERS_MAPPING = { #all waves
                              '#X:': 'R:',
                              '#IX:': 'I:',
                              '#R:': 'R:',
                              r'\[.*\]': '',                              
                              #wave_1
                              r':W\d+:': ':',
                              #wave_2
                              r':S\d+:': ':',
                              #wave_3
                              '#IV Code:': '#Interview Code:',
                              r':P\d+:': ':',
                              r':C\d+:': ':',
                              '#ORGANIZED ACTIVITIES' : '#ORGANIZED ACTIVITIES AND WORK',
                              '#WORK' : '#ORGANIZED ACTIVITIES AND WORK',
                              }

MORALITY_QUESTIONS = ['M' + str(i) + ':' for i in list(range(17))+['X']]

REFINED_SECTIONS_WITH_MORALITY_BREAKDOWN = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in [s for s in INTERVIEW_SECTIONS if s not in ['Morality']] + ['Morality:'+q[:-1] for q in MORALITY_QUESTIONS]]
REFINED_SECTIONS = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in INTERVIEW_SECTIONS]

# MORALITY_ENTITIES = {'Deontological Morality' : ['Duty', 'Moral obligations', 'Categorical imperative', 'Kantian ethics', 'Divine command theory', 'Universalizability', 'Intrinsic value', 'Non-consequentialism', 'Moral rules', 'Intention', 'Moral absolutes', 'Moral rights', 'Moral wrongs', 'Good will', 'Moral duty', 'Moral responsibility', 'Moral worth', 'Ethical principles', 'Moral decision-making', 'Prima facie duties'],
#                      'Consequential Morality' : ['Utilitarianism', 'Maximization of overall happiness', 'Ends justify the means', 'Consequences-based ethics', 'Greatest good for the greatest number', 'Hedonistic calculus', 'Cost-benefit analysis', 'Moral calculation', 'Teleological ethics', 'Ethical egoism', 'Act consequentialism', 'Rule consequentialism', 'Positive consequentialism', 'Negative consequentialism', 'Proportional consequentialism', 'Non-maximizing consequentialism', 'Moral impartiality', 'Intentions vs. outcomes', 'Consequentialist reasoning', 'Morally right actions']}

# MORALITY_ENTITIES = {'Deontological Morality' : ['Duty', 'Obligations', 'Intrinsic Value', 'Rules', 'Intention', 'Right', 'Wrong', 'Responsibility'],
#                      'Consequential Morality' : ['Utilitarianism', 'Consequences', 'Hedonistic Calculus', 'Cost Benefit Analysis', 'Teleological', 'Egoism', 'Consequentialism', 'Impartiality']}

# MORALITY_ENTITIES = {'Intrinsic Values' : ['Values', 'Intrinsic', 'Morality', 'Importance', 'Meaning', 'Ends', 'Ethics', 'Beliefs', 'Autonomy', 'Dignity', 'Rights', 'Humanity', 'Inherent', 'Goodness', 'Essence', 'Nature', 'Characteristics', 'Inner', 'Unconditional', 'Worth'],
#                      'Consequentialism' : ['Ethics', 'Utilitarianism', 'Morality', 'Actions', 'Outcomes', 'Consequences', 'Maximization', 'Principles', 'Happiness', 'Consequentialist', 'Value', 'Moral', 'Philosophy', 'Decisions', 'Ethical', 'Altruism', 'Conduct', 'Evaluating', 'Results', 'Utility'],
#                      'Trust' : ['Trust', 'Reliability', 'Confidence', 'Faith', 'Belief', 'Credibility', 'Integrity', 'Honesty', 'Dependability', 'Loyalty', 'Transparency', 'Accountability', 'Fidelity', 'Assurance', 'Openness', 'Sincerity', 'Consistency', 'Goodwill', 'Reputation', 'Commitment']}

MORALITY_ENTITIES = {'Care': ['kindness', 'compassion', 'nurture', 'empathy', 'suffer', 'cruel', 'hurt', 'harm'],
                     'Equality': ['equality', 'egalitarian', 'justice', 'nondiscriminatory', 'prejudice', 'inequality', 'discrimination', 'biased'],
                     'Proportionality': ['proportional', 'merit', 'deserving', 'reciprocal', 'disproportionate', 'cheating', 'favoritism', 'recognition'],
                     'Loyalty': ['loyal', 'solidarity', 'patriot', 'fidelity', 'betray', 'treason', 'disloyal', 'traitor'],
                     'Authority': ['authority', 'obey', 'respect', 'tradition', 'subversion', 'disobey', 'disrespect', 'chaos'],
                     'Purity': ['purity', 'sanctity', 'sacred', 'wholesome', 'impurity', 'depravity', 'degradation', 'unnatural']}
