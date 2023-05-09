
import os
import re

import pandas as pd

TRANSCRIPT_ENCODING='windows-1252'

interview_sections = ['Household',
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
            'Home School',
            'Volunteering & Organized Activities',
            'Dating',
            'Sexuality',
            'The Media',
            'Future Prospects']

interview_metadata = ['M',
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
                      'Physical Description']

skipped_comments =['#IC',
                   '#ICM',
                   '#C',
                   '#OC',
                   '#I',
                   '#NC',
                   '#X',
                   '#IN',
                   '#IX',
                   '#R',
                   '#FIELD NOTES',
                   '#END']


#Metadata normalization
def normalize_metadata(line):
    try:
        key, value = line.split(':', 1)
    except:
        key, value = line, ''

    key = re.sub(r'[ ]+', '', key.strip().lower()[1:])
    for m in interview_metadata:
        if re.sub(r'[ ]+', '', m.lower()) == key:
            return m, value.strip()
    
    print('Metadata not found: ' + key)
    return None, None


#Section name normalization
def normalize_section_name(section):
    section = re.sub(r'[\d: -]+', '', section[1:]).strip().lower()
    for s in interview_sections:
        if section.startswith(re.sub(r'[ -]+', '', s).lower()):
            return s
    
    print('Section not found: ' + section)
    return None

def interview_parser(filename):
    with open(filename, 'r', encoding=TRANSCRIPT_ENCODING) as f:
        text = f.read()
        lines = text.split('\n')
        interview = {}
        section = ''
        metadata_lines = True

        for line in lines:

            #End of interview metadata
            if line.startswith('#START'):
                metadata_lines = False

            #Skip comments
            elif any(re.sub(r'[\s]+', '', line).lower().startswith(re.sub(r'[\s]+', '', comment).lower()) for comment in skipped_comments):
                section = ''

            #Interview metadata
            elif line.startswith('#') and  metadata_lines:
                key, value = normalize_metadata(line)
                interview[key] = value
            
            #Section headers
            elif line.startswith('#'):
                # if section != '':
                #     interview[section] = interview[section].strip()
                
                #Section name normalization
                section = normalize_section_name(line)
                
                if section != '' and section in interview:
                    interview[section] += '\n'
                elif section != '' and section not in interview:
                    interview[section] = ''
                               
            #Section content
            elif not line.startswith('#') and section != '':
                interview[section] += line + '\n'
        
        return interview


#parse folder of transcripts
def wave_parser(folder):
    interviews = []
    for filename in os.listdir(folder):
        filename = os.path.join(folder, filename)
        interview = interview_parser(filename)
        interviews.append(interview)
    interviews = pd.DataFrame(interviews)
    return interviews


#interview = interview_parser('downloads/wave_1/TC-14180-13-B-W-RC-RG-SD_S.txt')

interviews = wave_parser('downloads/wave_1')


#interview['DATING']


#def fiter_interview(person):


