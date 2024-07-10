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

REFINED_SECTIONS_WITH_MORALITY_BREAKDOWN = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in [s for s in INTERVIEW_SECTIONS if s not in ['Morality']] + ['Morality:'+q[:-1] for q in MORALITY_QUESTIONS]]
REFINED_SECTIONS = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in INTERVIEW_SECTIONS]

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

MORALITY_ORIGIN =   ['Experience',
                     'Consequences',
                     'Family',
                     'Community',
                     'Friends',
                     'Media',
                     'Laws',
                     'Holy Scripture']

MORALITY_ORIGIN_EXPLAINED = {'Experience and Instinct and Feeling':'Experience',
                             'Consequences':'Consequences',
                             'Family and Parents':'Family',
                             'Community and Society and School':'Community',
                             'Friends':'Friends',
                             'Media and TV and Books':'Media',
                             'Laws and Rules':'Laws',
                             'Holy Scripture and God and Bible':'Holy Scripture'}

NEWLINE = '\n\n\n'

CODERS = ['Leke', 'Tomas']

CHATGPT_PROMPT = 'Estimate the probability of the following text belonging to the categories: Experience, Consequences, Family, Community, Friends, Media, Laws, Holy Scripture. Response in the form \"Category:Probability\".'

SURVEY_ATTRIBUTES = {'Wave 1':{'IDS':'Survey Id',
                               'PINCOME':'Income (raw)',
                               'PDADEDUC':'Father Education',
                               'PMOMEDUC':'Mother Education',
                               'POT':'Pot',
                               'DRINK':'Drink',
                               'CHEATED':'Cheat',
                               'CUTCLASS':'Cutclass',
                               'SECRET':'Secret',
                               'VOLUNTER':'Volunteer',
                               'HELPED':'Help',
                               'HOWDECID':'Decision Taking',
                               'GRADES':'Grades',
                               'ATTEND':'Church Attendance'},
                     'Wave 2':{'IDS':'Survey Id'},
                     'Wave 3':{'IDS':'Survey Id',
                               'EARNINGS':'Income (raw)'}}

#Income: Lower Class < 70K < Upper Class
HOUSEHOLD_CLASS = {1:'Low',
                   2:'Low',
                   3:'Low',
                   4:'Low',
                   5:'Low',
                   6:'Low',
                   7:'Low',
                   8:'High',
                   9:'High',
                   10:'High',
                   11:'High'}

#Education: Basic < College < Higher < PhD < Advanced
EDUCATION = {0:'Primary',
             1:'Primary',
             2:'Primary',
             3:'Secondary',
             4:'Secondary',
             5:'Secondary',
             6:'Secondary',
             7:'Secondary',
             8:'Tertiary',
             9:'Tertiary',
             10:'Tertiary',
             11:'Tertiary',
             12:'Tertiary',
             13:'Tertiary',
             14:'Tertiary'}

#Survey max values
MAX_POT_VALUE = 4
MAX_DRINK_VALUE = 7
MAX_CHEAT_VALUE = 6
MAX_CUTCLASS_VALUE = 4
MAX_SECRET_VALUE = 6
MAX_VOLUNTEER_VALUE = 4
MAX_HELP_VALUE = 4
MAX_GRADES_VALUE = 10
MAX_CHURCH_ATTENDANCE_VALUE = 6

#Decision taking map
DECISION_TAKING = {1:'Expressive Individualist',
                   2:'Utilitarian Individualist',
                   3:'Relational',
                   4:'Theistic'}

DEMOGRAPHICS = [{'name' : 'Gender', 'values' : ['Male', 'Female']},
                {'name' : 'Race', 'values' : ['White', 'Other']},
                {'name' : 'Household Income', 'values' : ['High', 'Low']},
                {'name' : 'Parent Education', 'values' : ['Tertiary', 'Secondary']},
                {'name' : 'Age', 'values' : ['Early Adolescence', 'Late Adolescence']},
                {'name' : 'Church Attendance', 'values' : ['Regular', 'Irregular']}]

CODED_WAVES = ['Wave 1', 'Wave 3']
MORALITY_ESTIMATORS = ['Model', 'Coders']

MERGE_MORALITY_ORIGINS = True
MORALITY_ORIGIN = ['Intuitive', 'Consequentialist', 'Social', 'Theistic'] if MERGE_MORALITY_ORIGINS else MORALITY_ORIGIN

UNCERTAINT_TERMS = ['hypothetically speaking', 'possibly', 'potentially', 'must', 'equivocal', 'it looks like', 'is likely to', 'unclear', 'could be', 'it is conceivable', 'reportedly', 'could', 'allegedly', 'seemingly', 'tends to', 'conceivably', 'apparently', 'likely', 'probably', 'there is a chance', 'will', 'unsure', 'there is a possibility', 'supposedly', 'feasibly', 'suggests that', 'it is feasible', 'is unlikely to', 'may', 'arguably', 'might', 'is probable', 'perhaps', 'might be', 'vague', 'it is possible', 'maybe', 'presumably', 'uncertain', 'ambiguous', 'it appears', 'hypothetically', 'would', 'is improbable', 'doubtful', 'imaginably', 'it seems', 'can', 'ostensibly', 'should']