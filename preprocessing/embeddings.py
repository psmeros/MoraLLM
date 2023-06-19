import numpy as np
import pandas as pd
import spacy
from __init__ import *
from pandarallel import pandarallel
from simpletransformers.language_representation import RepresentationModel
from sklearn.linear_model import LinearRegression

from preprocessing.transcript_parser import wave_parser

def get_vectorizer(model='lg'):
    if model == 'trf':
        transformer = RepresentationModel(model_type='bert', model_name='bert-base-uncased', use_cuda=False)
        vectorizer = lambda x: pd.Series(transformer.encode_sentences(x, combine_strategy='mean').tolist())
    elif model in ['lg', 'md']:
        nlp = spacy.load('en_core_web_'+model)
        pandarallel.initialize()
        vectorizer = lambda x: x.parallel_apply(lambda y: nlp(y).vector)
    return vectorizer

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
    vectorizer = get_vectorizer(model=model)
    interviews[section + '_Embeddings'] = vectorizer(interviews[section])
    
    #Drop interviews with no embeddings
    interviews = interviews.dropna(subset=[section + '_Embeddings'])
    interviews = interviews[interviews[section + '_Embeddings'].apply(lambda x: sum(x) != 0)]

    interviews.to_pickle(output_file)


def transform_embeddings(embeddings, anchors, model='lg'):

    #Compute anchor embeddings
    anchors = pd.DataFrame(anchors).melt(var_name='Name', value_name='Embeddings')
    vectorizer = get_vectorizer(model=model)
    anchors['Embeddings'] = vectorizer(anchors['Embeddings'].str.lower())
    anchors = anchors.groupby('Name').mean().reset_index()

    #Normalize anchor embeddings
    np.random.seed(42)
    anchors['Normalized Embeddings'] = [np.random.randn(anchors['Embeddings'].iloc[0].shape[0]) for _ in range(len(anchors))]
    displacement_vector = anchors['Normalized Embeddings'].apply(pd.Series).to_numpy() - np.array(np.mean(anchors['Normalized Embeddings'], axis=0))
    anchors['Normalized Embeddings'] = (displacement_vector / np.linalg.norm(displacement_vector)).tolist()

    # Find transformation matrix
    regressor = LinearRegression()
    regressor.fit(anchors['Embeddings'].apply(pd.Series), anchors['Normalized Embeddings'].apply(pd.Series))
    transformation_matrix = regressor.coef_.T

    #Transform embeddings
    embeddings = embeddings.apply(pd.Series).apply(lambda x: np.dot(x, transformation_matrix), axis=1)
    anchors = anchors.drop(columns=['Embeddings']).rename(columns={'Normalized Embeddings': 'Embeddings'})

    return embeddings, anchors


if __name__ == '__main__':
    model = 'lg'
    compute_embeddings(wave_folder='data/waves', output_file='data/cache/morality_embeddings_'+model+'.pkl', model=model, section='R:Morality', morality_breakdown=False, keep_POS=True)