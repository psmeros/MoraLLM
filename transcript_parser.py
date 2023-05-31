import os
import re

import pandas as pd
import numpy as np

from constants import INTERVIEW_SECTIONS, INTERVIEW_PARTICIPANTS, INTERVIEW_METADATA, INTERVIEW_COMMENTS, INTERVIEW_MARKERS_MAPPING, MORALITY_QUESTIONS, REFINED_INTERVIEW_SECTIONS, TRANSCRIPT_ENCODING
from helpers import error_handling


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
        if line.startswith(re.sub(r'[ -]+', '', s).lower()):
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
        interview = {'Filename': filename}
        section = ''
        metadata_lines = True

        for line in lines:

            #End of interview metadata
            if line.startswith('#START'):
                metadata_lines = False

            #Skip comments and empty lines
            elif line.strip() == '' or any(re.sub(r'[\s]+', '', line).lower().startswith(re.sub(r'[\s]+', '', comment).lower()) for comment in INTERVIEW_COMMENTS):
                continue

            #Interview metadata
            elif line.startswith('#') and  metadata_lines:
                key, value = normalize_metadata(line, filename)
                interview[key] = value
            
            #Section headers
            elif line.startswith('#'):
                
                #Section name normalization
                section = normalize_section_name(line, filename)
                
                #Initialize section
                interview[section] = interview.get(section, '') + '\n'

                               
            #Section content
            elif not line.startswith('#') and section != '':
                interview[section] += line + '\n'
        
        return interview

#Get raw text for each section and participant
def get_raw_text(interview):
    raw_text = {section : '' for section in REFINED_INTERVIEW_SECTIONS}

    for section in INTERVIEW_SECTIONS:
  
        if not interview[section]:
            error_handling(interview['Filename'], section, 'Section None!', print_line=True)
            continue
        elif interview[section].strip() == '':
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
                else:
                    raw_text[participant + section] += line.strip() + ' '

    raw_text = pd.Series(raw_text)
    return raw_text

#parse folder of transcripts
def wave_parser(folder):
    interviews = []
    for filename in os.listdir(folder):
        filename = os.path.join(folder, filename)
        interview = interview_parser(filename)
        interviews.append(interview)
    interviews = pd.DataFrame(interviews)

    #get raw text for each interview
    interviews = interviews[INTERVIEW_METADATA].join(interviews[INTERVIEW_SECTIONS+['Filename']].apply(get_raw_text, axis = 1))
    
    #cleaning
    interviews = interviews.replace('', pd.NA)

    return interviews


if __name__ == '__main__':
    interviews = wave_parser('data/wave_1')