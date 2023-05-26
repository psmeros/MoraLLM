import os
import re

import pandas as pd
import numpy as np

from constants import INTERVIEW_SECTIONS, INTERVIEW_PARTICIPANTS, INTERVIEW_METADATA, INTERVIEW_COMMENTS, INTERVIEW_MARKERS_MAPPING, MORALITY_SECTIONS, TRANSCRIPT_ENCODING


#Metadata normalization
def normalize_metadata(line, filename):
    try:
        key, value = line.split(':', 1)
    except:
        key, value = line, ''

    key = re.sub(r'[ ]+', '', key.strip().lower()[1:])
    for m in INTERVIEW_METADATA:
        if re.sub(r'[ ]+', '', m.lower()) == key:
            return m, value.strip()
    
    print(os.path.abspath(filename), line, '(Metadata Not Found!)')
    return None, None


#Section name normalization
def normalize_section_name(line, filename):
    line = re.sub(r'[\d: -]+', '', line[1:]).strip().lower()
    for section in INTERVIEW_SECTIONS:
        if line.startswith(re.sub(r'[ -]+', '', section).lower()):
            return section
    
    print(os.path.abspath(filename), line, '(Section Not Found!)')
    return None


def normalize_morality_section_name(line, filename):
    #TODO: lines can belong to multiple sections
    for section in MORALITY_SECTIONS:
        if line.startswith(section):
            return section[:-1], line[len(section):].strip()
    
    print(os.path.abspath(filename), line, '(Morality Section Not Found!)')
    return None, None

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

            #Skip comments
            elif any(re.sub(r'[\s]+', '', line).lower().startswith(re.sub(r'[\s]+', '', comment).lower()) for comment in INTERVIEW_COMMENTS):
                section = ''

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
    raw_text = {participant + section : '' for section in INTERVIEW_SECTIONS for participant in INTERVIEW_PARTICIPANTS}

    
    for section in INTERVIEW_SECTIONS:
        try:
            lines = interview[section].split('\n')
        except:
            lines = []
        insert = [False] * len(INTERVIEW_PARTICIPANTS)

        for line in lines:

            for index, participant in enumerate(INTERVIEW_PARTICIPANTS):
                if line.startswith(participant):
                    insert = [False] * len(INTERVIEW_PARTICIPANTS)
                    insert[index] = True
                    line = line[len(participant):].strip()
                    if section in ['Morality']:
                        normalize_morality_section_name(line, interview['Filename'])
                    break

            for index, participant in enumerate(INTERVIEW_PARTICIPANTS):
                if insert[index] == True:
                    raw_text[participant + section] += line.strip() + ' '
                    break

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