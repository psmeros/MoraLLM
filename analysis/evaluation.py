import pandas as pd
from __init__ import *
from sklearn.metrics import log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import ZeroShotClassificationExplainer

from preprocessing.constants import MORALITY_ORIGIN
from preprocessing.metadata_parser import merge_codings


def cross_entropy_loss(interviews):
    #Keep interviews with codings and merge 
    interviews = interviews[interviews['Wave'].isin([1,3])]
    interviews = merge_codings(interviews)
    interviews[[mo + '_y' for mo in MORALITY_ORIGIN]] = interviews[[mo + '_y' for mo in MORALITY_ORIGIN]].applymap(int)
    
    #Compute loss
    weight = interviews[[mo + '_y' for mo in MORALITY_ORIGIN]].sum()/interviews[[mo + '_y' for mo in MORALITY_ORIGIN]].sum().sum()
    loss = pd.Series({mo: log_loss(interviews[mo + '_y'], interviews[mo + '_x']) for mo in MORALITY_ORIGIN})
    loss = (weight.reset_index(drop=True) * loss.reset_index(drop=True))
    loss.index = MORALITY_ORIGIN
    
    return loss

#Explain word-level attention for zero-shot models
def explain_entailment(interviews):

    pairs = [(interviews.iloc[interviews[mo + '_x'].idxmax()]['Morality_Origin'], [mo]) for mo in MORALITY_ORIGIN]

    model_name = 'cross-encoder/nli-deberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    zero_shot_explainer = ZeroShotClassificationExplainer(model, tokenizer)

    for text, labels in pairs:
        zero_shot_explainer(text=text, hypothesis_template='The morality origin is {}.',labels=labels)
        zero_shot_explainer.visualize('data/misc/zero_shot.html')

if __name__ == '__main__':
    #Hyperparameters
    config = [1]
    models = ['md', 'lg', 'bert', 'bart', 'entail']

    for c in config:
        if c == 1:
            for model in models:
                interviews = pd.read_pickle('data/cache/morality_embeddings_'+model+'.pkl')
                loss = cross_entropy_loss(interviews)
                print(model, 'model loss:', loss.sum())
                # print(model, 'max loss:', loss.max(), loss.idxmax())
                # print(model, 'max loss:', loss.min(), loss.idxmin())
        if c == 2:
            interviews = pd.read_pickle('data/cache/morality_embeddings_entail.pkl')
            interviews = interviews[interviews['Wave'] == 1]
            interviews = merge_codings(interviews)
            explain_entailment(interviews)

