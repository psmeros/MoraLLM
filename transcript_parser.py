import os
import re

import pandas as pd
import numpy as np

from constants import INTERVIEW_SECTIONS, INTERVIEW_PARTICIPANTS, INTERVIEW_METADATA, INTERVIEW_COMMENTS, INTERVIEW_MARKERS_MAPPING, TRANSCRIPT_ENCODING

#Convert encoding of files in a folder
def convert_encoding(folder_path, from_encoding, to_encoding):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)        
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding = from_encoding) as file:
                file_contents = file.read()
            with open(file_path, 'w', encoding = to_encoding) as file:
                file.write(file_contents)
            print('Converted file:', filename)


#Metadata normalization
def normalize_metadata(line):
    try:
        key, value = line.split(':', 1)
    except:
        key, value = line, ''

    key = re.sub(r'[ ]+', '', key.strip().lower()[1:])
    for m in INTERVIEW_METADATA:
        if re.sub(r'[ ]+', '', m.lower()) == key:
            return m, value.strip()
    
    print('Metadata not found: ' + key)
    return None, None


#Section name normalization
def normalize_section_name(section):
    section = re.sub(r'[\d: -]+', '', section[1:]).strip().lower()
    for s in INTERVIEW_SECTIONS:
        if section.startswith(re.sub(r'[ -]+', '', s).lower()):
            return s
    
    print('Section not found: ' + section)
    return None

def interview_parser(filename):
    with open(filename, 'r', encoding = TRANSCRIPT_ENCODING) as f:
        text = f.read()

        #apply markers mapping
        for k, v in INTERVIEW_MARKERS_MAPPING.items():
            text = re.sub(k, v, text)

        lines = text.split('\n')
        interview = {}
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
                key, value = normalize_metadata(line)
                interview[key] = value
            
            #Section headers
            elif line.startswith('#'):
                
                #Section name normalization
                section = normalize_section_name(line)
                
                #Initialize section
                interview[section] = interview.get(section, '') + '\n'

                               
            #Section content
            elif not line.startswith('#') and section != '':
                interview[section] += line + '\n'
        
        return interview

#Get raw text for an interview participant (Interviewer or Respondent)
def get_raw_section_text(section_text, interview_participant):
    raw_text = ''
    try:
        lines = section_text.split('\n')
        insert = False

        for line in lines:
            if line.startswith(interview_participant):
                insert = True
                line = line[len(interview_participant):].strip()
            elif any(line.startswith(i) for i in INTERVIEW_PARTICIPANTS):
                insert = False

            if insert:
                raw_text = ' '.join([raw_text.strip(), line])

        raw_text = raw_text.strip()
    except:
        pass

    return raw_text

#parse folder of transcripts
def wave_parser(folder):
    interviews = []
    for filename in os.listdir(folder):
        filename = os.path.join(folder, filename)
        interview = interview_parser(filename)
        interviews.append(interview)
    interviews = pd.DataFrame(interviews)

    #Get raw text for each interview section and participant
    for section in INTERVIEW_SECTIONS:
        for participant in INTERVIEW_PARTICIPANTS:
            interviews[participant + ' ' + section] = interviews[section].apply(lambda s: get_raw_section_text(s, participant))

    #cleaning
    interviews[INTERVIEW_SECTIONS] = interviews[INTERVIEW_SECTIONS].applymap(lambda x: x.strip() if not pd.isna(x) else x)
    interviews = interviews.replace('', pd.NA)

    return interviews


if __name__ == '__main__':
    interviews = wave_parser('downloads/wave_1')