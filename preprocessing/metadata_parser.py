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

#Merge codings for wave 1 and wave 3 of interviews
def merge_codings(interviews, codings_file_1 = 'data/waves/interview_codings-wave_1.csv', codings_file_3 = 'data/waves/interview_codings-wave_3.csv'):
    codings_1 = pd.read_csv(codings_file_1)
    codings_3 = pd.read_csv(codings_file_3)
    codings_1['Interview Code'] = codings_1['Interview Code'].str.split('-').apply(lambda x: x[0]+'-'+x[1])
    codings_1['Wave'] = 1
    codings_3['Wave'] = 3
    codings = pd.concat([codings_1, codings_3])

    codings = codings.set_index(['Interview Code', 'Wave'])
    codings = codings.applymap(lambda x: not pd.isnull(x))
    codings['Experience'] = codings['Experience'] | codings['Intrinsic']
    codings['Family'] = codings['Family'] | codings['Parents']
    codings = codings.drop(['Intrinsic', 'Parents'], axis=1)
    codings = codings.reset_index()

    interviews = interviews.merge(codings, on=['Wave', 'Interview Code'], how = 'left', validate = '1:1')
    
    return interviews


if __name__ == '__main__':
    interviews = wave_parser()
    interviews = merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])
    interviews = merge_codings(interviews)