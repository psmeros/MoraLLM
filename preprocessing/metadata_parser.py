from __init__ import *
from preprocessing.constants import HOUSEHOLD_CLASS, SURVEY_ATTRIBUTES

from preprocessing.transcript_parser import wave_parser


#Merge matched interviews from different waves
def merge_matches(interviews, wave_list = ['Wave 1', 'Wave 2', 'Wave 3'], matches_file = 'data/interviews/alignments/crosswave.csv'):
    matches = pd.read_csv(matches_file)[wave_list].dropna()

    for wave in wave_list:
        wave_interviews = interviews[interviews['Wave'] == int(wave.split()[-1])]
        wave_interviews = wave_interviews.add_prefix(wave + ':')
        matches = matches.merge(wave_interviews, left_on = wave, right_on = wave + ':Interview Code', how = 'inner')

    matches = matches.drop(wave_list, axis=1)

    return matches

#Merge codings from two coders for wave 1 and wave 3 of interviews
def merge_codings(interviews, codings_folder = 'data/interviews/codings'):
    #Parse codings
    codings_wave_1 = []
    codings_wave_3 = []
    for file in os.listdir(codings_folder):
        file = os.path.join(codings_folder, file)
        if os.path.isfile(file) and file.endswith('.csv'):
            coding = pd.read_csv(file)
            coding['Wave'] = int(file.split('_')[1])
            coding.attrs['Wave'] = int(file.split('_')[1])
            coding.attrs['Coder'] = file.split('_')[2].split('.')[0].capitalize()
            if coding.attrs['Wave'] == 1:
                coding['Interview Code'] = coding['Interview Code'].str.split('-').apply(lambda x: x[0]+'-'+x[1])

            coding = coding.set_index(['Interview Code', 'Wave'])
            coding = coding.applymap(lambda x: not pd.isnull(x))
            coding['Experience'] = coding['Experience'] | coding['Intrinsic']
            coding['Family'] = coding['Family'] | coding['Parents']
            coding = coding.drop(['Intrinsic', 'Parents'], axis=1)

            if coding.attrs['Wave'] == 1:
                codings_wave_1.append(coding)
            elif coding.attrs['Wave'] == 3:
                codings_wave_3.append(coding)
    
    #Merge codings
    codings_wave_1 = codings_wave_1[0].join(codings_wave_1[1], lsuffix='_'+codings_wave_1[0].attrs['Coder'], rsuffix='_'+codings_wave_1[1].attrs['Coder'])
    codings_wave_3 = codings_wave_3[0].join(codings_wave_3[1], lsuffix='_'+codings_wave_3[0].attrs['Coder'], rsuffix='_'+codings_wave_3[1].attrs['Coder'])
    codings = pd.concat([codings_wave_1, codings_wave_3])
    codings = codings[~(~codings).all(axis=1)]
    codings = codings.reset_index()

    interviews = interviews.merge(codings, on=['Wave', 'Interview Code'], how = 'inner', validate = '1:1')
    
    return interviews

#Merge interviews and surveys
def merge_surveys(interviews, surveys_folder = 'data/interviews/surveys', alignment_file = 'data/interviews/alignments/interview-survey.csv'):

    surveys = []
    for file in os.listdir(surveys_folder):
        file = os.path.join(surveys_folder, file)
        if os.path.isfile(file) and file.endswith('.csv'):
            wave = int(file.split('_')[1].split('.')[0])
            survey = pd.read_csv(file)[SURVEY_ATTRIBUTES['Wave ' + str(wave)].keys()].rename(columns=SURVEY_ATTRIBUTES['Wave ' + str(wave)])
            survey['Wave'] = wave
            surveys.append(survey)
    surveys = pd.concat(surveys)

    alignment = pd.read_csv(alignment_file)
    surveys = surveys.merge(alignment, on = ['Wave', 'Survey Id'], how = 'inner')

    surveys['Income'] = surveys['Income'].apply(lambda x: HOUSEHOLD_CLASS.get(x, pd.NA))
    surveys = surveys.dropna()

    interviews = interviews.merge(surveys, on = ['Wave', 'Interview Code'], how = 'inner', validate = '1:1')

    return interviews

if __name__ == '__main__':
    #Hyperparameters
    config = [3]
    interviews = wave_parser()

    for c in config:
        if c == 1:
            interviews = merge_codings(interviews)
        elif c == 2:
            interviews = merge_matches(interviews)
        elif c == 3:
            interviews = merge_surveys(interviews)