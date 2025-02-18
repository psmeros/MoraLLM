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

INTERVIEW_SINGLELINE_COMMENTS =[ #all waves
                                '#ICM:',
                                '#C:',
                                '#OC:',
                                #wave_3
                                '#Interview File Name:']

INTERVIEW_MULTILINE_COMMENTS =[ #all waves
                                '#IC:',
                                '#I:',
                                '#NC:',
                                '#IN:',
                                '#FIELD NOTES',
                                '#END',
                                #wave_3
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

REFINED_SECTIONS = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in [s for s in INTERVIEW_SECTIONS if s not in ['Morality']] + ['Morality:'+q[:-1] for q in MORALITY_QUESTIONS]]

METADATA_GENDER_MAP = {'Female':'Female',
                       'Girls':'Female',
                       'GIRL':'Female',
                       'Girl':'Female',
                       'BOY':'Male',
                       'boy':'Male',
                       'Boy':'Male'}

METADATA_RACE_MAP = {'Caucasian':'White',
                     'White':'White',
                     'white':'White',
                     'BL':'Black',
                     'W':'White',
                     'Black':'Black',
                     'Hispanic':'Hispanic',
                     'MIXED BLACK/WHITE':'Other',
                     'wh':'White',
                     'Other – Iranian descent':'Other',
                     'B':'Black',
                     'Asian':'Asian',
                     'Filipino ancestry, Asian':'Asian',
                     'other (mother is Mexican, father is Egyptian)':'Other',
                     'black':'Black',
                     'WHITE':'White',
                     'Hispanic/Columbian':'Hispanic',
                     'Egyptian': 'Other',
                     'White/Jewish':'White',
                     'Refused':'Other',
                     'Middle Eastern':'Other',
                     'Indian (Tamil)':'Other',
                     'Hispanic (although looked African-American)':'Hispanic',
                     'Native American? (that’s what sheet says, but she looks white in person)':'Native American',
                     'white (half Latina)':'White',
                     'Native American':'Native American',
                     'Black (Creole)':'Black',
                     'East Indian':'Other',
                     'Asian (?)':'Asian',
                     'Korean':'Asian',
                     'HISPANIC (Black)':'Hispanic',
                     'BLACK':'Black',
                     'MIDDLE EASTERN/IRANIAN':'Other',
                     'HISPANIC':'Hispanic',
                     'BLACK/CREOLE':'Black',
                     'BLACK (Originally from Trinidad/Tobago)':'Black',
                     'NATIVE AMERICAN':'Native American',
                     'ASIAN':'Asian',
                     'ASIAN / KOREAN':'Asian',
                     'WHITE / HISPANIC':'Other',
                     'HISPANIC / ARABIC':'Other',
                     'MIXED HISPANIC / ITALIAN':'Other',
                     'asian':'Asian',
                     'OTHER / INDIAN':'Other',
                     'MIXED HISPANIC/WHITE':'Other',
                     'Hispanic / Venezuelan':'Hispanic',
                     'Hispanic (per teen, Mexican & Puerto Rican)':'Hispanic',
                     'Hispanic / Colombian':'Hispanic',
                     'White (South-African)':'White',
                     'Hispanic / Peruvian':'Hispanic',
                     'Islander':'Other',
                     'Other (Middle Eastern, Iranian)':'Other',
                     'HISPANIC RACE: White':'Other',
                     'Hispanic (Mexico)':'Hispanic',
                     'Arabic & Islamic':'Other',
                     'Asian – may be part Caucasian (father who brought him was Caucasian)':'Asian',
                     'White #RACE: Asian mixed':'Other',
                     'Hispanic / Argentinian':'Hispanic',
                     'Black-Trinidadian':'Black',
                     'Other/East Indian':'Other',
                     'Asian (Chinese)':'Asian',
                     'Don’t know':None}

MORALITY_ORIGIN = ['Intuitive', 'Consequentialist', 'Social', 'Theistic']

MORALITY_ORIGIN_EXPLAINED = {**dict.fromkeys(['intuition'], 'Intuitive'),
                             **dict.fromkeys(['consequences'], 'Consequentialist'),
                             **dict.fromkeys(['social'], 'Social'),
                             **dict.fromkeys(['religion'], 'Theistic')}

MORALITY_VOCAB = {mo:vocab for mo, vocab in zip(MORALITY_ORIGIN, [['intuition', 'intuitive', 'gut', 'feel', 'instinct'], ['consequence', 'outcome', 'affect', 'impact', 'future', 'cost', 'benefit', 'harm', 'help', 'maximize', 'minimize'], ['parent', 'mother', 'father', 'brother', 'sister', 'mom', 'dad', 'friend', 'school', 'teacher', 'society', 'social'], ['god', 'devil', 'faith', 'prayer', 'pray', 'church', 'islam', 'bible', 'religion', 'religious', 'belief', 'commandment', 'heaven', 'hell']])}

CODERS = ['Leke', 'Tomas']

CHATGPT_SUMMARY_PROMPT = """You are a helpful assistant that summarizes interview transcripts.
In these interviews, respondents are asked by the interviewers how they make decisions.
Interviewers are marked with "I:" and respondents are marked with "R:" in the transcripts.
Your job is to summarize each transcript strictly in one sentence.
Make sure to include any references of the respondents to intuition, the consequences of their actions, social influences such as parents or friends, or religious reasons.
If there are more than one references, order them in the summary according to their importance."""

chatgpt_prompt = lambda mo, r: """You are a helpful assistant that classifies interview transcripts.
In these interviews, respondents are asked by the interviewers how they make decisions.
Interviewers are marked with "I:" and respondents are marked with "R:" in the transcripts.
Your job is to detect whether respondents refer to """ + \
('intuition.' if mo == 'Intuitive'  else 'the consequences of their actions.' if mo == 'Consequentialist' else 'social influences such as parents or friends.' if mo == 'Social' else 'religious reasons.' if mo == 'Theistic' else '') + '\n'+ \
('Response strictly with 1 if they refer and with 0 if they do not refer.' if r == 'bin' else 'Response strictly on a Likert scale from 0 to 4, depending on the emphasis the respondents give to ' + ('intuition.' if mo == 'Intuitive'  else 'the consequences of their actions.' if mo == 'Consequentialist' else 'social influences such as parents or friends.' if mo == 'Social' else 'religious reasons.' if mo == 'Theistic' else '') if r == 'quant' else '')

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

SURVEY_ATTRIBUTES = {'Wave 1':{'IDS':'Survey Id',
                               'PINCOME':'Household Income',
                               'PDADEDUC':'Father Education',
                               'PMOMEDUC':'Mother Education',
                               'POT':'Pot',
                               'DRINK':'Drink',
                               'CHEATED':'Cheat',
                               'CUTCLASS':'Cutclass',
                               'SECRET':'Secret',
                               'VOLUNTER':'Volunteer',
                               'HELPED':'Help',
                               'HOWDECID':'Moral Schemas',
                               'GRADES':'GPA',
                               'ATTEND':'Church Attendance',
                               'RELTRAD':'Religion',
                               'BNREGSO':'Region'},
                     'Wave 2':{'IDS':'Survey Id',
                               'POT':'Pot',
                               'DRINK':'Drink',
                               'CHEATED':'Cheat',
                               'CUTCLASS':'Cutclass',
                               'SECRET':'Secret',
                               'VOLUNTER':'Volunteer',
                               'HELPED':'Help',
                               'HOWDECID':'Moral Schemas',
                               'I_ATTEND':'Church Attendance',
                               'RELAFF':'Religion',
                               'BNREGSO':'Region',
                               'FRNDS':'Number of friends'},
                     'Wave 3':{'IDS':'Survey Id',
                               'EARNINGS':'Household Income',
                               'POT':'Pot',
                               'DRINK':'Drink',
                               'VOLUNTER':'Volunteer',
                               'HELPED':'Help',
                               'HOWDECID':'Moral Schemas',
                               'ATTEND':'Church Attendance',
                               'RELIGION':'Religion',
                               'BNREGSO':'Region',
                               'NUMFRIEN':'Number of friends'},
                     'Wave 4':{'IDS':'Survey Id',
                               'POTNEVER_W4':'Pot',
                               'DRINK_W4':'Drink',
                               'VOLUNTER_W4':'Volunteer',
                               'HELPED_W4':'Help'}}

NETWORK_ATTRIBUTES = {'ids': 'Survey Id',
                      'frnds1': 'Wave 1:Number of friends',
                      'frnds2': 'Wave 1:Number of friends',
                      'y187y_01': 'Wave 1:Regular volunteers',
                      'y187y_02': 'Wave 1:Regular volunteers',
                      'y187y_03': 'Wave 1:Regular volunteers',
                      'y187y_04': 'Wave 1:Regular volunteers',
                      'y187y_05': 'Wave 1:Regular volunteers',
                      'y187p_01': 'Wave 1:Use drugs',
                      'y187p_02': 'Wave 1:Use drugs',
                      'y187p_03': 'Wave 1:Use drugs',
                      'y187p_04': 'Wave 1:Use drugs',
                      'y187p_05': 'Wave 1:Use drugs',
                      'y187f_01': 'Wave 1:Similar beliefs',
                      'y187f_02': 'Wave 1:Similar beliefs',
                      'y187f_03': 'Wave 1:Similar beliefs',
                      'y187f_04': 'Wave 1:Similar beliefs',
                      'y187f_05': 'Wave 1:Similar beliefs',
                      'frndvol_w2': 'Wave 2:Regular volunteers',
                      'frnddrgs_w2': 'Wave 2:Use drugs',
                      'frndrelblf_w2': 'Wave 2:Similar beliefs',
                      'sfrndvol_w2': 'Wave 2:Regular volunteers',
                      'sfrnddrgs_w2': 'Wave 2:Use drugs',
                      'sfrndrelblf_w2': 'Wave 2:Similar beliefs',
                      'frndvol_w3': 'Wave 3:Regular volunteers',
                      'frnddrgs_w3': 'Wave 3:Use drugs',
                      'frndrelblf_w3': 'Wave 3:Similar beliefs',
                      'sfrndrelblf_w2': 'Wave 2:Similar beliefs',
                      'sfrndvol_w3': 'Wave 3:Regular volunteers',
                      'sfrnddrgs_w3': 'Wave 3:Use drugs',
                      'sfrndrelblf_w3': 'Wave 3:Similar beliefs'}

#Income: Dichotomy at 50k
INCOME_RANGE = {**dict.fromkeys(range(1, 5), 'Low'), **dict.fromkeys(range(5, 12), 'High')}

#Education: Dichotomy at College
EDUCATION_RANGE = {**dict.fromkeys(range(0, 8), 'Less Than College'), **dict.fromkeys(range(8, 15), 'College or More')}

#Church Attendance: Irregular vs Regular
CHURCH_ATTENDANCE_RANGE = {**dict.fromkeys(range(0, 5), 'Irregular'), **dict.fromkeys(range(5, 7), 'Regular')}

#Adolescence: Early vs Late
ADOLESCENCE_RANGE = {**dict.fromkeys(range(13, 16), 'Early'), **dict.fromkeys(range(16, 20), 'Late')}

#Race: White vs Other
RACE_RANGE = {**dict.fromkeys(['White'], 'White'), **dict.fromkeys(['Black'], 'Black'), **dict.fromkeys(['Asian', 'Hispanic', 'Native American' 'Other'], 'Other')}

#Moral Schemas map
MORAL_SCHEMAS = {1:'Expressive Individualist',
                   2:'Utilitarian Individualist',
                   3:'Relational',
                   4:'Theistic'}

#Religion map
RELIGION = {'Wave 1':{1:'Evangelical Protestant',
                      2:'Mainline Protestant',
                      3:'Black Protestant',
                      4:'Catholic',
                      5:'Jewish',
                      6:'Mormon',
                      7:'Not Religious',
                      **dict.fromkeys(range(8, 10), 'Indeterminate')},
            'Wave 2':{1:'Evangelical Protestant',
                      2:'Mainline Protestant',
                      **dict.fromkeys(range(9, 11), 'Black Protestant'),
                      3:'Catholic',
                      4:'Jewish',
                      6:'Mormon',
                      5:'Not Religious',
                      **dict.fromkeys(range(7, 9), 'Indeterminate')},
            'Wave 3':{1:'Evangelical Protestant',
                      2:'Mainline Protestant',
                      **dict.fromkeys(range(3, 5), 'Black Protestant'),
                      5:'Catholic',
                      6:'Jewish',
                      7:'Mormon',
                      8:'Not Religious',
                      **dict.fromkeys(range(9, 13), 'Indeterminate')}}

#Region map
REGION = {0:'Not South', 1:'South'}

DEMOGRAPHICS = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                {'name' : 'Race', 'values' : ['White', 'Other']},
                {'name' : 'Household Income', 'values' : ['High', 'Low']},
                {'name' : 'Parent Education', 'values' : ['Tertiary', 'Secondary']},
                {'name' : 'Adolescence', 'values' : ['Early', 'Late']},
                {'name' : 'Church Attendance', 'values' : ['Regular', 'Irregular']}]

CODED_WAVES = ['Wave 1', 'Wave 3']
MORALITY_ESTIMATORS = ['Model', 'Coders']

UNCERTAINT_TERMS = ['hypothetically speaking', 'possibly', 'potentially', 'must', 'equivocal', 'it looks like', 'is likely to', 'unclear', 'could be', 'it is conceivable', 'reportedly', 'could', 'allegedly', 'seemingly', 'tends to', 'conceivably', 'apparently', 'likely', 'probably', 'there is a chance', 'will', 'unsure', 'there is a possibility', 'supposedly', 'feasibly', 'suggests that', 'it is feasible', 'is unlikely to', 'may', 'arguably', 'might', 'is probable', 'perhaps', 'might be', 'vague', 'it is possible', 'maybe', 'presumably', 'uncertain', 'ambiguous', 'it appears', 'hypothetically', 'would', 'is improbable', 'doubtful', 'imaginably', 'it seems', 'can', 'ostensibly', 'should']

format_pvalue = lambda x: '-' if x is None else (('{:.2f}'.format(x[0]).replace('0.', '.') if abs(x[0]) < 1 else '{:.2f}'.format(x[0])) + ('' if x[1] == None else '***' if float(x[1])<.001 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '†' if float(x[1])<.1 else ''))