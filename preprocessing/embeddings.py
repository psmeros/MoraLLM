import pandas as pd
import spacy
from pandarallel import pandarallel
from simpletransformers.language_representation import RepresentationModel

from preprocessing.transcript_parser import wave_parser

#Compute embeddings for a folder of transcripts
def compute_embeddings(wave_folder, output_file, section, keep_POS=True, model='lg'):
                                                   
    interviews = wave_parser(wave_folder)

    #Keep only POS of interest
    if keep_POS:
        nlp = spacy.load('en_core_web_lg')
        interviews[section] = interviews[section].apply(lambda s: ' '.join([w.text for w in nlp(s) if w.pos_ in ['VERB', 'NOUN', 'ADJ', 'ADV']]) if not pd.isna(s) else pd.NA)
        interviews[section] = interviews[section].str.strip()
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
    compute_embeddings('data/waves', 'data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality')
