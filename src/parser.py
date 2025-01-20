import os
import re

import openai
import pandas as pd
from __init__ import *
from striprtf.striprtf import rtf_to_text

from src.helpers import CHATGPT_SUMMARY_PROMPT, CHURCH_ATTENDANCE_RANGE, CODERS, MORAL_SCHEMAS, EDUCATION_RANGE, INCOME_RANGE, INTERVIEW_SINGLELINE_COMMENTS, INTERVIEW_MULTILINE_COMMENTS, INTERVIEW_SECTIONS, INTERVIEW_PARTICIPANTS, INTERVIEW_METADATA, INTERVIEW_MARKERS_MAPPING, METADATA_GENDER_MAP, METADATA_RACE_MAP, MORALITY_ESTIMATORS, MORALITY_ORIGIN, MORALITY_QUESTIONS, NETWORK_ATTRIBUTES, RACE_RANGE, REFINED_SECTIONS, REGION, RELIGION, SURVEY_ATTRIBUTES, TRANSCRIPT_ENCODING


#Convert encoding of files in a folder
def convert_encoding(folder_path, from_encoding, to_encoding):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            if from_encoding == 'rtf':
                with open(file_path, 'r') as file:
                    file_contents = file.read()
                    file_contents = rtf_to_text(file_contents, encoding = to_encoding)
                    file_path = file_path[:-len(from_encoding)] + 'txt'
            else:
                with open(file_path, 'r', encoding = from_encoding) as file:
                    file_contents = file.read()
            with open(file_path, 'w', encoding = to_encoding) as file:
                file.write(file_contents)
            print('Converted file:', filename)

#Print error message and file with line number
def error_handling(filename, target_line, error_message, print_line=False):
    filename = os.path.abspath(filename)
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, 1):
            if target_line in line:
                print(error_message, '\n', filename+':'+str(line_number))
                if print_line:
                    print(target_line)
                return
    print(error_message, '\n', filename, target_line)

#Metadata normalization
def normalize_metadata(line, filename):
    key, value = None, None
    try:
        line_key, line_value = line.split(':', 1)
    except:
        line_key, line_value = line, ''

    line_key = re.sub(r'[ ]+', '', line_key.strip().lower()[1:])
    for m in INTERVIEW_METADATA:
        if re.sub(r'[ ]+', '', m.lower()) == line_key:
            key = m
            value = line_value.strip()
            break
    
    if not key and not value:
        error_handling(filename, line, 'Metadata Not Found!')
        key, value = '', ''

    return key, value

#Section name normalization
def normalize_section_name(line, filename):
    section = None
    line = re.sub(r'[\d: -]+', '', line[1:]).strip().lower()
    for s in INTERVIEW_SECTIONS:
        if re.sub(r'[ -]+', '', s).lower() in line:
            section = s
            break
    
    if not section:
        error_handling(filename, line, 'Section Not Found!')
        section = ''

    return section

#Morality questions name normalization
def normalize_morality_question_name(line, filename):
    questions = []
    search = True
    while search:
        search = False
        for question in MORALITY_QUESTIONS:
            if line.startswith(question):
                questions.append(question[:-1])
                line = line[len(question):].strip()
                search = True
                break
        
    if not questions:
        error_handling(filename, line, 'Morality Question Not Found!')
        questions, line = [], ''

    return questions, line

def interview_parser(filename):
    with open(filename, 'r', encoding = TRANSCRIPT_ENCODING) as f:
        text = f.read()

        #apply markers mapping
        for k, v in INTERVIEW_MARKERS_MAPPING.items():
            text = re.sub(k, v, text)

        lines = text.split('\n')
        interview = {'Filename': filename} | {field : '' for field in INTERVIEW_METADATA + INTERVIEW_SECTIONS}
        section = ''
        metadata_lines = True
        comment_lines = False

        for line in lines:

            #End of interview metadata
            if line.startswith('#START'):
                metadata_lines = False

            #Skip empty lines
            elif line.strip() == '':
                continue

            #Skip mutliline comments
            elif any(re.sub(r'[\s]+', '', line).lower().startswith(re.sub(r'[\s]+', '', comment).lower()) for comment in INTERVIEW_MULTILINE_COMMENTS):
                comment_lines = True

            #Skip singleline comments
            elif any(re.sub(r'[\s]+', '', line).lower().startswith(re.sub(r'[\s]+', '', comment).lower()) for comment in INTERVIEW_SINGLELINE_COMMENTS):
                continue

            #Interview metadata
            elif line.startswith('#') and metadata_lines:
                key, value = normalize_metadata(line, filename)
                interview[key] = value
            
            #Section headers
            elif line.startswith('#'):
                comment_lines = False
                #Section name normalization
                section = normalize_section_name(line, filename)
                               
            #Section content
            elif not line.startswith('#') and not comment_lines:
                    interview[section] += line + '\n'
        
        return interview

#Get raw text for each section and participant
def get_raw_text(interview):
    raw_text = {section : '' for section in REFINED_SECTIONS + ['Morality_Full_Text']}

    for section in INTERVIEW_SECTIONS:
  
        if not interview[section] or interview[section].strip() == '':
            continue
        else:
            lines = interview[section].strip().split('\n')

        participant = None
        morality_questions = []

        for line in lines:
            for p in INTERVIEW_PARTICIPANTS:
                if line.startswith(p):
                    participant = p
                    line = line[len(participant):].strip()
                    if section == 'Morality':
                        morality_questions, line = normalize_morality_question_name(line, interview['Filename'])
                    break
          
            if not participant:
                error_handling(interview['Filename'], line, 'Participant Not Found!')
                continue
            else:
                if section == 'Morality':
                    for question in morality_questions:
                        raw_text[participant + section + ':' + question] += line.strip() + ' '
                    raw_text['Morality_Full_Text'] += participant + ':'.join(morality_questions) + ':' + line.strip() + '\n'
                else:
                    raw_text[participant + section] += line.strip() + ' '

    raw_text = pd.Series(raw_text)
    return raw_text

#parse folder of transcripts
def wave_parser(waves_folder='data/interviews/waves'):
    
    waves = []
    for foldername in os.listdir(waves_folder):
        foldername = os.path.join(waves_folder, foldername)
        if os.path.isdir(foldername):
            interviews = []

            for filename in os.listdir(foldername):
                filename = os.path.join(foldername, filename)
                if os.path.isfile(filename) and filename.endswith('.txt'):
                    interview = interview_parser(filename)
                    interviews.append(interview)
            interviews = pd.DataFrame(interviews)

            #clean interviews
            interviews = interviews[INTERVIEW_METADATA].join(interviews[INTERVIEW_SECTIONS+['Filename']].apply(lambda i: get_raw_text(i), axis = 1))
            interviews = interviews.replace('', pd.NA)
            
            #add wave
            interviews = pd.concat([interviews, pd.Series([int(foldername[-1])] * len(interviews), name='Wave')], axis=1)
            waves.append(interviews)

    waves = pd.concat(waves, ignore_index=True)
    
    #Clean Metadata
    waves['Gender'] = waves['Gender'].map(METADATA_GENDER_MAP)
    waves['Race'] = waves['Race'].map(METADATA_RACE_MAP)
    waves['Race'] = waves['Race'].map(RACE_RANGE)
    waves['Age'] = waves['Age'].astype('Int64')
    
    return waves

#Merge matched interviews from different waves
def merge_matches(interviews, extend_dataset, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'], matches_file = 'data/interviews/alignments/crosswave.csv'):
    matches = pd.read_csv(matches_file)[wave_list]

    for wave in wave_list:
        wave_interviews = interviews[interviews['Wave'] == int(wave.split()[-1])]
        wave_interviews = wave_interviews.add_prefix(wave + ':')
        matches = matches.merge(wave_interviews, left_on = wave, right_on = wave + ':Interview Code', how = ('left' if extend_dataset else 'inner'))

    matches = matches.drop(wave_list, axis=1)

    return matches

#Merge codings from two coders for wave 1 and wave 3 of interviews
def merge_codings(interviews, return_codings = False, codings_folder = 'data/interviews/codings', gold_file = 'data/interviews/alignments/gold_standard.csv'):
    #Parse codings
    codings_wave_1 = []
    codings_wave_3 = []
    for file in os.listdir(codings_folder):
        file = os.path.join(codings_folder, file)
        if os.path.isfile(file) and file.endswith('.csv'):
            coding = pd.read_csv(file)
            coding['Wave'] = int(file.split('_')[1])
            coding.attrs['Wave'] = int(file.split('_')[1])
            coding.attrs['Coder'] = file.split('_')[2].split('.')[0].capitalize()
            if coding.attrs['Wave'] == 1:
                coding['Interview Code'] = coding['Interview Code'].str.split('-').apply(lambda x: x[0]+'-'+x[1])

            coding = coding.set_index(['Interview Code', 'Wave'])
            coding = coding.map(lambda x: not pd.isnull(x))
            coding['Experience'] = coding['Experience'] | coding['Intrinsic']
            coding['Family'] = coding['Family'] | coding['Parents']
            coding = coding.drop(['Intrinsic', 'Parents'], axis=1)

            coding['Intuitive'] = coding['Experience']
            coding['Consequentialist'] = coding['Consequences']
            coding['Social'] = coding[['Family', 'Community', 'Friends']].any(axis=1)
            coding['Theistic'] = coding['Holy Scripture']
            coding = coding.drop(['Experience', 'Consequences', 'Family', 'Community', 'Friends', 'Media', 'Laws', 'Holy Scripture'], axis=1)

            if coding.attrs['Wave'] == 1:
                codings_wave_1.append(coding)
            elif coding.attrs['Wave'] == 3:
                codings_wave_3.append(coding)

    #Merge codings
    codings_wave_1 = codings_wave_1[0].join(codings_wave_1[1], lsuffix='_'+codings_wave_1[0].attrs['Coder'], rsuffix='_'+codings_wave_1[1].attrs['Coder'])
    codings_wave_3 = codings_wave_3[0].join(codings_wave_3[1], lsuffix='_'+codings_wave_3[0].attrs['Coder'], rsuffix='_'+codings_wave_3[1].attrs['Coder'])
    codings = pd.concat([codings_wave_1, codings_wave_3])
    codings = codings.reset_index()

    if return_codings:
        interviews = codings
    else:
        codings = pd.concat([codings[['Wave', 'Interview Code']], codings.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN], index=MORALITY_ORIGIN), axis=1)], axis=1)
        
        gold = pd.read_csv(gold_file)
        gold = codings.merge(gold, on=['Wave', 'Interview Code'], suffixes=('', '_gold'), how = 'left')
        gold[[mo + '_gold' for mo in MORALITY_ORIGIN]] = pd.concat([gold[mo + '_gold'].fillna(gold[mo]) for mo in MORALITY_ORIGIN], axis=1)
        gold = gold.drop(MORALITY_ORIGIN, axis=1)
        interviews = interviews.merge(gold, on=['Wave', 'Interview Code'], how = 'left', validate = '1:1')
        #Hybrid morality estimation
        if len(MORALITY_ESTIMATORS) == 3:
            interviews[[mo + '_' + MORALITY_ESTIMATORS[2] for mo in MORALITY_ORIGIN]] = interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]].values * interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]].values
    
    return interviews

#Merge interviews and surveys
def merge_surveys(interviews, surveys_folder = 'data/interviews/surveys', alignment_file = 'data/interviews/alignments/interview-survey.csv'):
    surveys = pd.read_csv(alignment_file)
    wave_1_surveys = surveys[surveys['Wave'] == 1].drop('Wave', axis=1)
    wave_3_surveys = surveys[surveys['Wave'] == 3].drop('Wave', axis=1)

    for file in os.listdir(surveys_folder):
        file = os.path.join(surveys_folder, file)
        if os.path.isfile(file) and file.endswith('.csv'):
            wave = int(file.split('_')[1].split('.')[0])
            survey = pd.read_csv(file, usecols=SURVEY_ATTRIBUTES['Wave ' + str(wave)].keys()).rename(columns=SURVEY_ATTRIBUTES['Wave ' + str(wave)])
            if wave == 1:
                survey['Pot'] = survey['Pot'].apply(lambda x: x - 1 if x in range(1, 5) else None)
                survey['Drink'] = survey['Drink'].apply(lambda x: 7 - x if x in range(1, 8) else None)
                survey['Cheat'] = survey['Cheat'].apply(lambda x: 6 - x if x in range(1, 7) else None)
                survey['Cutclass'] = survey['Cutclass'].apply(lambda x: x - 1 if x in range(1, 5) else None)
                survey['Secret'] = survey['Secret'].apply(lambda x: 6 - x if x in range(1, 7) else None)
                survey['Volunteer'] = survey['Volunteer'].apply(lambda x: x - 1 if x in range(1, 5) else None)
                survey['Help'] = survey['Help'].apply(lambda x: 4 - x if x in range(1, 5) else None)
                survey['GPA'] = survey['GPA'].apply(lambda x: x if x in range(1, 11) else None)
                survey['GPA'] = survey['GPA'].map(lambda x: 6 if x >= 6 else x)
                survey['Moral Schemas'] = survey['Moral Schemas'].map(lambda x: MORAL_SCHEMAS.get(x, None))
                survey['Parent Education'] = survey[['Father Education', 'Mother Education']].apply(lambda x: max(x.iloc[0], x.iloc[1]) if (x.iloc[0] <= max(EDUCATION_RANGE.keys())) and (x.iloc[1] <= max(EDUCATION_RANGE.keys())) else min(x.iloc[0], x.iloc[1]), axis=1)
                survey['Parent Education'] = survey['Parent Education'].map(lambda x: 5 if x <= 5 else x)
                survey['Church Attendance'] = survey['Church Attendance'].apply(lambda x: x if x in CHURCH_ATTENDANCE_RANGE.keys() else None)
                survey['Household Income'] = survey['Household Income'].apply(lambda i: i if i in INCOME_RANGE.keys() else None)
                survey['Religion'] = survey['Religion'].map(lambda x: RELIGION['Wave 1'].get(x, None))
                survey['Region'] = survey['Region'].map(lambda x: REGION.get(x, None))
            elif wave == 2:
                survey['Pot'] = survey['Pot'].apply(lambda x: 7 - x if x in range(1, 8) else None)
                survey['Drink'] = survey['Drink'].apply(lambda x: 7 - x if x in range(1, 8) else None)
                survey['Cheat'] = survey['Cheat'].apply(lambda x: 6 - x if x in range(1, 7) else None)
                survey['Cutclass'] = survey['Cutclass'].apply(lambda x: x - 1 if x in range(1, 5) else None)
                survey['Secret'] = survey['Secret'].apply(lambda x: 6 - x if x in range(1, 7) else None)
                survey['Volunteer'] = survey['Volunteer'].apply(lambda x: x - 1 if x in range(1, 5) else None)
                survey['Help'] = survey['Help'].apply(lambda x: 4 - x if x in range(1, 5) else None)
                survey['Moral Schemas'] = survey['Moral Schemas'].map(lambda x: MORAL_SCHEMAS.get(x, None))
                survey['Church Attendance'] = survey['Church Attendance'].apply(lambda x: x-1 if x-1 in CHURCH_ATTENDANCE_RANGE.keys() else None)
                survey['Religion'] = survey['Religion'].map(lambda x: RELIGION['Wave 2'].get(x, None))
                survey['Region'] = survey['Region'].map(lambda x: REGION.get(x, None))
            elif wave == 3:
                survey['Pot'] = survey['Pot'].apply(lambda x: 7 - x if x in range(1, 8) else None)
                survey['Drink'] = survey['Drink'].apply(lambda x: 7 - x if x in range(1, 8) else None)
                survey['Volunteer'] = survey['Volunteer'].apply(lambda x: x if x in range(0, 2) else None)
                survey['Help'] = survey['Help'].apply(lambda x: 4 - x if x in range(1, 5) else None)
                survey['Moral Schemas'] = survey['Moral Schemas'].map(lambda x: MORAL_SCHEMAS.get(x, None))
                survey['Church Attendance'] = survey['Church Attendance'].apply(lambda x: x if x in CHURCH_ATTENDANCE_RANGE.keys() else None)
                survey['Household Income'] = survey['Household Income'].apply(lambda i: i if i in INCOME_RANGE.keys() else None)
                survey['Religion'] = survey['Religion'].map(lambda x: RELIGION['Wave 3'].get(x, None))
                survey['Region'] = survey['Region'].map(lambda x: REGION.get(x, None))
            elif wave == 4:
                survey['Pot'] = survey['Pot'].apply(lambda x: 1 - x if x in range(0, 2) else None)
                survey['Drink'] = survey['Drink'].apply(lambda x: 0 if x in range(7, 9) else 7 - x if x in range (1, 7) else None)
                survey['Volunteer'] = survey['Volunteer'].apply(lambda x: x if x in range(0, 2) else None)
                survey['Help'] = survey['Help'].apply(lambda x: 4 - x if x in range(1, 5) else None)

            survey.columns = [survey.columns[0]] + ['Wave ' + str(wave) + ':' + c for c in survey.columns[1:]]
            if wave in [1, 2]:
                wave_1_surveys = wave_1_surveys.merge(survey, on = 'Survey Id', how = 'left')
            elif wave in [3, 4]:
                wave_3_surveys = wave_3_surveys.merge(survey, on = 'Survey Id', how = 'left')

    interviews = interviews.merge(wave_1_surveys, left_on = 'Wave 1:Interview Code',  right_on = 'Interview Code', how = 'left').drop(['Interview Code'], axis=1)
    interviews = interviews.merge(wave_3_surveys, left_on = 'Wave 3:Interview Code',  right_on = 'Interview Code', how = 'left').drop(['Interview Code'], axis=1)
    interviews['Survey Id'] = interviews['Survey Id_x'].fillna(interviews['Survey Id_y'])
    interviews = interviews.drop(['Survey Id_x', 'Survey Id_y'], axis=1).dropna(subset=['Survey Id']).drop_duplicates(subset=['Survey Id'], keep='first').reset_index(drop=True)
    interviews['Survey Id'] = interviews['Survey Id'].astype(int)

    #Missing data
    interviews['Wave 3:Age'] = interviews['Wave 3:Age'].fillna(interviews['Wave 1:Age'] + int((interviews['Wave 3:Age'] - interviews['Wave 1:Age']).mean()))
    interviews['Wave 1:Age'] = interviews['Wave 1:Age'].fillna(interviews['Wave 3:Age'] - int((interviews['Wave 3:Age'] - interviews['Wave 1:Age']).mean()))
    interviews[['Wave 2:' + demographic for demographic in['Parent Education', 'GPA', 'Household Income']]] = interviews[['Wave 1:' + demographic for demographic in['Parent Education', 'GPA', 'Household Income']]]
    interviews[['Wave 3:' + demographic for demographic in['Parent Education', 'GPA']]] = interviews[['Wave 1:' + demographic for demographic in['Parent Education', 'GPA']]]
    interviews[['Wave 2:' + mo + '_gold' for mo in MORALITY_ORIGIN]] = pd.NA
    interviews[[wave + ':' + action for wave in ['Wave 3', 'Wave 4'] for action in ['Cheat', 'Cutclass', 'Secret']]] = pd.NA

    return interviews

#Merge network variables
def merge_network(interviews, file = 'data/interviews/network/net_vars.dta'):
    network = pd.read_stata(file)[NETWORK_ATTRIBUTES.keys()].rename(columns=NETWORK_ATTRIBUTES)
    network = network.map(lambda x: None if x in ['DON\'T KNOW', 'LEGITIMATE SKIP'] else x)
    network = network.map(lambda x: 4 if x in [4,5] else x)
    interviews = interviews.merge(network, on='Survey Id', how='left')
    return interviews

#Merge all different types of data
def prepare_data(models, extend_dataset):

    interviews = pd.read_pickle('data/cache/morality_model-'+models[0]+'.pkl')
    interviews[[mo + '_' + models[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    for model in models[1:]:
        interviews = pd.merge(interviews, pd.read_pickle('data/cache/morality_model-'+model+'.pkl')[MORALITY_ORIGIN + ['Interview Code', 'Wave']], on=['Interview Code', 'Wave'], how='left', suffixes=('', '_'+model))

    interviews = merge_codings(interviews)
    interviews = merge_matches(interviews, extend_dataset)
    interviews = merge_surveys(interviews)
    interviews = merge_network(interviews)

    columns = ['Survey Id'] + [wave + ':' + 'Interview Code' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]

    columns += [wave + ':' + mo + '_' + estimatior for wave in ['Wave 1', 'Wave 2', 'Wave 3'] for estimatior in ['gold'] + models for mo in MORALITY_ORIGIN]

    columns += [wave + ':' + demographic for wave in ['Wave 1', 'Wave 2', 'Wave 3'] for demographic in ['Age', 'Gender', 'Race', 'Household Income', 'Parent Education', 'Church Attendance', 'GPA', 'Moral Schemas', 'Religion', 'Region']]

    columns += [wave + ':' + network for wave in ['Wave 1', 'Wave 2', 'Wave 3'] for network in ['Number of Friends']]

    columns += [wave + ':' + covariate for wave in ['Wave 1', 'Wave 2', 'Wave 3'] for covariate in ['Verbosity', 'Uncertainty', 'Complexity', 'Sentiment']]

    columns += [wave + ':' + action for wave in ['Wave 1', 'Wave 2', 'Wave 3', 'Wave 4'] for action in ['Pot', 'Drink', 'Cheat', 'Cutclass', 'Secret', 'Volunteer', 'Help']]

    columns += [wave + ':' + 'Morality Text' for wave in ['Wave 1', 'Wave 2', 'Wave 3']]

    interviews = interviews[columns]
    return interviews

#Compute summary of morality text
def compute_morality_summary():
    interviews = wave_parser()
    interviews['Morality_Full_Text'] = interviews['Morality_Full_Text'].replace('', pd.NA)
    interviews = interviews.dropna(subset=['Morality_Full_Text']).reset_index(drop=True)
    #OpenAI API
    openai.api_key = os.getenv('OPENAI_API_KEY')
    summarizer = lambda text: openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': CHATGPT_SUMMARY_PROMPT},{'role': 'user','content': text}], temperature=.2, max_tokens=256, frequency_penalty=0, presence_penalty=0, seed=42)
    aggregator = lambda r: r['choices'][0]['message']['content']
    full_pipeline = lambda text: aggregator(summarizer(text))
    interviews['Morality Summary'] = interviews['Morality_Full_Text'].apply(full_pipeline)
    interviews.to_pickle('data/cache/interviews.pkl')

if __name__ == '__main__':
    #Hyperparameters
    config = [2]

    for c in config:
        if c == 1:
            compute_morality_summary()
        elif c == 2:
            models = ['chatgpt_bin', 'chatgpt_quant', 'chatgpt_sum_bin', 'chatgpt_sum_quant', 'nli_bin', 'nli_quant', 'nli_sum_bin', 'nli_sum_quant']
            extend_dataset = True
            interviews = prepare_data(models, extend_dataset=extend_dataset)
            interviews.sort_values(by='Survey Id').to_clipboard(index=False)