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

SURVEY_ATTRIBUTES = {'Wave 1':{'IDS':'Survey Id', 'PINCOME':'Income'},
                     'Wave 2':{'IDS':'Survey Id'},
                     'Wave 3':{'IDS':'Survey Id', 'EARNINGS':'Income'}}

#Income: Lower Class < 40K < Upper Class
HOUSEHOLD_CLASS = {1:'Lower Class',
                   2:'Lower Class',
                   3:'Lower Class',
                   4:'Lower Class',
                   5:'Upper Class',
                   6:'Upper Class',
                   7:'Upper Class',
                   8:'Upper Class',
                   9:'Upper Class',
                   10:'Upper Class',
                   11:'Upper Class'}