
import os
import re

import pandas as pd

TRANSCRIPT_ENCODING='iso-8859-1'

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

interview_metadata = ['IC',
                      'M',
                      'W',
                      'ICM',
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
                      'Field Notes']

#M: 0
#W: 0
#ICM:

#Name of Interviewer:	Chris Smith
#Date of Interview:	4/29/03
#Interview Location:	Lansdale Public Library, Lansdale, PA
#Interview Code:	CS-07216

#AGE: 14
#GENDER: Boy
#RACE: White
#SURVEY RELIGION: Christian	
#SURVEY DENOM: Bible Church/ Bible Believe
#RELIGIOUS AFFILIATION: Bible Church
 
#PHYSICAL DESCRIPTION: Bleached blond hair, wrestler, acne on face, braces, talked a little funny

#FIELD NOTES

#Section name normalization
def normalize_section_name(section):
    section = re.sub(r'[\d: -]+', '', section[1:]).strip().lower()
    for s in interview_sections:
        if re.sub(r'[ -]+', '', s).lower() in section:
            return s
    if section not in ['start', 'end', 'fieldnotes']:
        print('Section not found: ' + section)


def interview_parser(filename):
    with open(filename, 'r', encoding=TRANSCRIPT_ENCODING) as f:
        text = f.read()
        lines = text.split('\n')
        interview = {}
        section = ''


        for line in lines:

            #Initial key-value pairs
            if line.startswith('#') and  re.search(r':[ \t]', line):
                key, value = line.split(':', 1)
                interview[key[1:]] = value.strip()
            
            #Section headers
            elif line.startswith('#'):
                if section != '':
                    interview[section] = interview[section].strip()
                
                #Section name normalization
                section = normalize_section_name(line)
                
                if section in ['START', 'END']:
                    section = ''
                else:
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


interview = interview_parser('downloads/wave_1/TC-14180-13-B-W-RC-RG-SD_S.txt')

interviews = wave_parser('downloads/wave_1')


#interview['DATING']


#def fiter_interview(person):


