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
                #    r':M\d+:': ':',
                #    r':W\d+:': ':'
                   }

MORALITY_QUESTIONS = ['M' + str(i) + ':' for i in list(range(17))+['X']]

REFINED_INTERVIEW_SECTIONS = [participant + section for participant in INTERVIEW_PARTICIPANTS for section in [s for s in INTERVIEW_SECTIONS if s not in ['Morality']] + ['Morality:'+q[:-1] for q in MORALITY_QUESTIONS]]
