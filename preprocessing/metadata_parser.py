from __init__ import *

from preprocessing.transcript_parser import wave_parser


#Merge matched interviews from different waves
def merge_matches(interviews, wave_list, matches_file = 'data/waves/interview_matches.csv'):
    matches = pd.read_csv(matches_file)[wave_list].dropna()

    for wave in wave_list:
        wave_interviews = interviews[interviews['Wave'] == int(wave.split()[-1])]
        wave_interviews = wave_interviews.add_prefix(wave + ':')
        matches = matches.merge(wave_interviews, left_on = wave, right_on = wave + ':Interview Code', how = 'inner')

    matches = matches.drop(wave_list, axis=1)

    return matches

#Merge codings for wave 1 of interviews
def merge_codings(interviews, codings_file = 'data/waves/interview_codings.csv'):
    codings = pd.read_csv(codings_file)

    codings['Interview Code'] = codings['Interview Code'].str.split('-').apply(lambda x: x[0]+'-'+x[1])
    codings = codings.set_index('Interview Code')
    codings = codings.applymap(lambda x: not pd.isnull(x))
    codings['Family'] = codings['Family'] | codings['Parents'] 
    codings['Community'] = codings['Community'] | codings['Friends']
    codings = codings.drop(['Parents', 'Friends'], axis=1)
    codings = codings.reset_index()

    interviews = interviews.merge(codings, left_on='Wave 1:Interview Code', right_on='Interview Code', how = 'left')
    
    return interviews


if __name__ == '__main__':
    interviews = wave_parser()
    interviews = merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])
    interviews = merge_codings(interviews)