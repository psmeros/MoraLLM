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

llm_prompt = lambda mo, r: """You are a helpful assistant that classifies interview transcripts.
In these interviews, respondents are asked by the interviewers how they make decisions.
Interviewers are marked with "I:" and respondents are marked with "R:" in the transcripts.
Your job is to detect whether respondents refer to """ + \
('intuition.' if mo == 'Intuitive'  else 'the consequences of their actions.' if mo == 'Consequentialist' else 'social influences such as parents or friends.' if mo == 'Social' else 'religious reasons.' if mo == 'Theistic' else 'four dimensions: i) intuition, ii) the consequences of their actions, iii) social influences such as parents or friends, and iv) religious reasons.' if mo == 'all' else '') + '\n'+ \
('Response strictly with 1 if they refer and with 0 if they do not refer.' if r == 'bin' else 'Response strictly with four digits, each one being 1 if they refer to this dimension and 0 if they do not refer to this dimension.' if r == 'bin_all' else 'Response strictly on a Likert scale from 0 to 4, depending on the emphasis the respondents give to ' + ('intuition.' if mo == 'Intuitive'  else 'the consequences of their actions.' if mo == 'Consequentialist' else 'social influences such as parents or friends.' if mo == 'Social' else 'religious reasons.' if mo == 'Theistic' else '') if r == 'quant' else '')

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

#Income: Dichotomy at 90k
INCOME_RANGE = {**dict.fromkeys(range(1, 9), 'Low'), **dict.fromkeys(range(9, 12), 'High')}

#Education: Dichotomy at College
EDUCATION_RANGE = {**dict.fromkeys(range(0, 8), '< College'), **dict.fromkeys(range(8, 15), '≥ College')}

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

DEMOGRAPHICS = ['Gender', 'Race', 'Household Income', 'Parent Education']

CODED_WAVES = ['Wave 1', 'Wave 3']

UNCERTAINT_TERMS = ['hypothetically speaking', 'possibly', 'potentially', 'must', 'equivocal', 'it looks like', 'is likely to', 'unclear', 'could be', 'it is conceivable', 'reportedly', 'could', 'allegedly', 'seemingly', 'tends to', 'conceivably', 'apparently', 'likely', 'probably', 'there is a chance', 'will', 'unsure', 'there is a possibility', 'supposedly', 'feasibly', 'suggests that', 'it is feasible', 'is unlikely to', 'may', 'arguably', 'might', 'is probable', 'perhaps', 'might be', 'vague', 'it is possible', 'maybe', 'presumably', 'uncertain', 'ambiguous', 'it appears', 'hypothetically', 'would', 'is improbable', 'doubtful', 'imaginably', 'it seems', 'can', 'ostensibly', 'should']

MORALITY_MODELS = ['deepseek_bin', 'deepseek_resp_bin', 'deepseek_sum_bin', 'chatgpt_bin', 'chatgpt_bin_3.5', 'chatgpt_resp_bin', 'chatgpt_sum_bin', 'deepseek_bin_dto1', 'deepseek_bin_cto1', 'deepseek_bin_rto1', 'deepseek_bin_to1', 'deepseek_bin_toa', 'chatgpt_bin_dto1', 'chatgpt_bin_cto1', 'chatgpt_bin_rto1', 'chatgpt_bin_to1', 'chatgpt_bin_toa', 'deepseek_bin_ar', 'deepseek_bin_nt', 'chatgpt_bin_ar', 'chatgpt_bin_nt', 'nli_bin', 'nli_resp_bin', 'nli_sum_bin', 'sbert_bin', 'sbert_resp_bin', 'sbert_sum_bin', 'lda_bin', 'lda_resp_bin', 'lda_sum_bin', 'wc_bin', 'wc_resp_bin', 'wc_sum_bin', 'nli_quant', 'nli_resp_quant', 'nli_sum_quant', 'chatgpt_quant']

format_pvalue = lambda x: '-' if x is None else (('{:.2f}'.format(x[0]).replace('0.', '.') if abs(x[0]) < 1 else '{:.2f}'.format(x[0])) + ('' if x[1] == None else '***' if float(x[1])<.001 else '**' if float(x[1])<.01 else '*' if float(x[1])<.05 else '†' if float(x[1])<.1 else ''))