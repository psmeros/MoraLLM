from __init__ import *

from preprocessing.transcript_parser import wave_parser


#merge matched interviews from different waves
def merge_matches(wave_list, matches_file = 'data/waves/interview_matches.csv'):

    interviews = wave_parser()
    matches = pd.read_csv(matches_file)[wave_list].dropna()

    for wave in wave_list:
        wave_interviews = interviews[interviews['Wave'] == int(wave.split()[-1])]
        wave_interviews = wave_interviews.add_prefix(wave + ':')
        matches = matches.merge(wave_interviews, left_on = wave, right_on = wave + ':Interview Code', how = 'left', validate = '1:1')

    matches = matches.drop(wave_list, axis=1)

    return matches


if __name__ == '__main__':
    matches = merge_matches(wave_list = ['Wave 1', 'Wave 2', 'Wave 3'])