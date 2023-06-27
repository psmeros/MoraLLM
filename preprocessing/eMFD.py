import pandas as pd
import os
from __init__ import *


dictionary = pd.DataFrame(pd.read_pickle('data/misc/eMFD.pkl')).T
dictionary = dictionary.reset_index(names=['word'])

