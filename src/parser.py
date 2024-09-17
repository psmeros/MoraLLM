import os
import re

import pandas as pd
from __init__ import *
from striprtf.striprtf import rtf_to_text

from src.helpers import ADOLESCENCE_RANGE, CHURCH_ATTENDANCE_RANGE, CODED_WAVES, CODERS, DECISION_TAKING, EDUCATION_RANGE, INCOME_RANGE, INTERVIEW_SINGLELINE_COMMENTS, INTERVIEW_MULTILINE_COMMENTS, INTERVIEW_SECTIONS, INTERVIEW_PARTICIPANTS, INTERVIEW_METADATA, INTERVIEW_MARKERS_MAPPING, MAX_CHEAT_VALUE, MAX_CHURCH_ATTENDANCE_VALUE, MAX_CUTCLASS_VALUE, MAX_DRINK_VALUE, MAX_GRADES_VALUE, MAX_HELP_VALUE, MAX_POT_VALUE, MAX_SECRET_VALUE, MAX_VOLUNTEER_VALUE, MERGE_MORALITY_ORIGINS, METADATA_GENDER_MAP, METADATA_RACE_MAP, MORALITY_ESTIMATORS, MORALITY_ORIGIN, MORALITY_QUESTIONS, RACE_RANGE, REFINED_SECTIONS, REFINED_SECTIONS_WITH_MORALITY_BREAKDOWN, SURVEY_ATTRIBUTES, TRANSCRIPT_ENCODING


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
def get_raw_text(interview, morality_breakdown):
    if morality_breakdown:
        raw_text = {section : '' for section in REFINED_SECTIONS_WITH_MORALITY_BREAKDOWN}
    else:
        raw_text = {section : '' for section in REFINED_SECTIONS}

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
                if section == 'Morality' and morality_breakdown:
                    for question in morality_questions:
                        raw_text[participant + section + ':' + question] += line.strip() + ' '
                else:
                    raw_text[participant + section] += line.strip() + ' '

    raw_text = pd.Series(raw_text)
    return raw_text

#parse folder of transcripts
def wave_parser(waves_folder='data/interviews/waves', morality_breakdown=False):
    
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
            interviews = interviews[INTERVIEW_METADATA].join(interviews[INTERVIEW_SECTIONS+['Filename']].apply(lambda i: get_raw_text(i, morality_breakdown), axis = 1))
            interviews = interviews.replace('', pd.NA)
            
            #add wave
            interviews = pd.concat([interviews, pd.Series([int(foldername[-1])] * len(interviews), name='Wave')], axis=1)
            waves.append(interviews)

    waves = pd.concat(waves, ignore_index=True)
    
    #Clean Gender/Race Metadata
    waves['Gender'] = waves['Gender'].map(METADATA_GENDER_MAP)
    waves['Race'] = waves['Race'].map(METADATA_RACE_MAP)
    
    return waves

#Merge matched interviews from different waves
def merge_matches(interviews, wave_list = CODED_WAVES, matches_file = 'data/interviews/alignments/crosswave.csv'):
    matches = pd.read_csv(matches_file)[wave_list].dropna()

    for wave in wave_list:
        wave_interviews = interviews[interviews['Wave'] == int(wave.split()[-1])]
        wave_interviews = wave_interviews.add_prefix(wave + ':')
        matches = matches.merge(wave_interviews, left_on = wave, right_on = wave + ':Interview Code', how = 'inner')

    matches = matches.drop(wave_list, axis=1)

    return matches

#Merge codings from two coders for wave 1 and wave 3 of interviews
def merge_codings(interviews, codings_folder = 'data/interviews/codings'):
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

            if MERGE_MORALITY_ORIGINS:
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
    codings = codings[~(~codings).all(axis=1)]
    codings = codings.reset_index()

    interviews = interviews.merge(codings, on=['Wave', 'Interview Code'], how = 'inner', validate = '1:1')
    codings = interviews.apply(lambda c: pd.Series([int(c[mo + '_' + CODERS[0]] & c[mo + '_' + CODERS[1]]) for mo in MORALITY_ORIGIN]), axis=1)
    interviews[[mo + '_' + MORALITY_ESTIMATORS[0] for mo in MORALITY_ORIGIN]] = interviews[MORALITY_ORIGIN]
    interviews[[mo + '_' + MORALITY_ESTIMATORS[1] for mo in MORALITY_ORIGIN]] = codings

    return interviews

#Merge interviews and surveys
def merge_surveys(interviews, surveys_folder = 'data/interviews/surveys', alignment_file = 'data/interviews/alignments/interview-survey.csv'):
    surveys = []
    for file in os.listdir(surveys_folder):
        file = os.path.join(surveys_folder, file)
        if os.path.isfile(file) and file.endswith('.csv'):
            wave = int(file.split('_')[1].split('.')[0])
            survey = pd.read_csv(file)[SURVEY_ATTRIBUTES['Wave ' + str(wave)].keys()].rename(columns=SURVEY_ATTRIBUTES['Wave ' + str(wave)])
            survey['Wave'] = wave
            surveys.append(survey)
    surveys = pd.concat(surveys)

    alignment = pd.read_csv(alignment_file)
    surveys = surveys.merge(alignment, on = ['Wave', 'Survey Id'], how = 'inner')

    surveys['Pot'] = surveys['Pot'].apply(lambda x: x if x in range(1, MAX_POT_VALUE + 1) else pd.NA)
    surveys['Drink'] = surveys['Drink'].apply(lambda x: MAX_DRINK_VALUE + 1 - x if x in range(1, MAX_DRINK_VALUE + 1) else pd.NA)
    surveys['Cheat'] = surveys['Cheat'].apply(lambda x: MAX_CHEAT_VALUE + 1 - x if x in range(1, MAX_CHEAT_VALUE + 1) else pd.NA)
    surveys['Cutclass'] = surveys['Cutclass'].apply(lambda x: x if x in range(1, MAX_CUTCLASS_VALUE + 1) else pd.NA)
    surveys['Secret'] = surveys['Secret'].apply(lambda x: MAX_SECRET_VALUE + 1 - x if x in range(1, MAX_SECRET_VALUE + 1) else pd.NA)
    surveys['Volunteer'] = surveys['Volunteer'].apply(lambda x: x if x in range(1, MAX_VOLUNTEER_VALUE + 1) else pd.NA)
    surveys['Help'] = surveys['Help'].apply(lambda x: MAX_HELP_VALUE + 1 - x if x in range(1, MAX_HELP_VALUE + 1) else pd.NA)
    surveys['Decision Taking'] = surveys['Decision Taking'].apply(lambda x: DECISION_TAKING.get(x, pd.NA))
    surveys['Grades'] = surveys['Grades'].apply(lambda x: x if x in range(1, MAX_GRADES_VALUE + 1) else pd.NA)

    surveys['Parent Education (raw)'] = surveys[['Father Education', 'Mother Education']].apply(lambda x: max(x.iloc[0], x.iloc[1]) if (x.iloc[0] <= max(EDUCATION_RANGE.keys())) and (x.iloc[1] <= max(EDUCATION_RANGE.keys())) else min(x.iloc[0], x.iloc[1]), axis=1)
    surveys['Parent Education'] = surveys['Parent Education (raw)'].map(EDUCATION_RANGE)

    surveys['Church Attendance (raw)'] = surveys['Church Attendance (raw)'].apply(lambda x: x if x in range(1, MAX_CHURCH_ATTENDANCE_VALUE + 1) else pd.NA)
    surveys['Church Attendance'] = surveys['Church Attendance (raw)'].map(CHURCH_ATTENDANCE_RANGE)

    surveys['Income (raw)'] = surveys['Income (raw)'].apply(lambda i: i if i in INCOME_RANGE.keys() else pd.NA)
    surveys['Household Income'] = surveys['Income (raw)'].map(INCOME_RANGE)

    interviews = interviews.merge(surveys, on = ['Wave', 'Interview Code'], how = 'inner', validate = '1:1')
    
    interviews['Age'] = interviews['Age'].astype('Int64')
    interviews['Adolescence'] = interviews['Age'].map(ADOLESCENCE_RANGE)

    interviews['Race'] = interviews['Race'].map(RACE_RANGE)

    return interviews

if __name__ == '__main__':
    #Hyperparameters
    config = []
    interviews = wave_parser()

    for c in config:
        if c == 1:            
            interviews = merge_codings(interviews)
        elif c == 2:
            interviews = merge_matches(interviews)
        elif c == 3:
            interviews = merge_surveys(interviews)