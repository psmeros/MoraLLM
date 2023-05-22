TRANSCRIPT_ENCODING='utf-8'

INTERVIEW_SECTIONS = ['Household',
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
            'Volunteering & Organized Activities',
            'Dating',
            'Sexuality',
            'The Media',
            'Future Prospects']

INTERVIEW_METADATA = ['M',
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

INTERVIEW_COMMENTS =['#IC:',
                   '#ICM:',
                   '#C:',
                   '#OC:',
                   '#I:',
                   '#NC:',
                   '#IN:',
                   '#FIELD NOTES',
                   '#END']

INTERVIEW_PARTICIPANTS = ['I:',
                          'R:']

INTERVIEW_MARKERS_MAPPING = {'#X:': 'R:',
                   '#IX:': 'I:',
                   '#R:': 'R:',
                   r':M\d+:': ':',
                   r':W\d+:': ':'}
