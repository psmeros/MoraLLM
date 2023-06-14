import pandas as pd
import spacy
from pandarallel import pandarallel
from simpletransformers.language_representation import RepresentationModel

from __init__ import *
from preprocessing.transcript_parser import wave_parser

#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section, morality_breakdown=False, keep_POS=True, model='lg'):

    interviews = wave_parser(wave_folder, morality_breakdown=morality_breakdown)

    #Keep only POS of interest (was ['VERB', 'NOUN', 'ADJ', 'ADV'], now just ['NOUN'])
    if keep_POS:
        nlp = spacy.load('en_core_web_lg')
        pandarallel.initialize()
        interviews[section] = interviews[section].parallel_apply(lambda s: ' '.join(set([w.text for w in nlp(s.lower()) if w.pos_ in ['NOUN']])).strip() if not pd.isna(s) else pd.NA)
        interviews = interviews.dropna(subset=[section])

    #Compute embeddings
    if model == 'trf':
        transformer = RepresentationModel(model_type='bert', model_name='bert-base-uncased', use_cuda=False)
        vectorizer = lambda x: pd.Series(transformer.encode_sentences(x, combine_strategy='mean').tolist())
    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        pandarallel.initialize()
        vectorizer = lambda x: x.parallel_apply(lambda y: nlp(y).vector)

    interviews[section + '_Embeddings'] = vectorizer(interviews[section])
    interviews = interviews.dropna(subset=[section + '_Embeddings'])
    interviews.to_pickle(output_file)


if __name__ == '__main__':
    model = 'lg'
    compute_embeddings(wave_folder='data/waves', output_file='data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality:M1', morality_breakdown=True, keep_POS=True)