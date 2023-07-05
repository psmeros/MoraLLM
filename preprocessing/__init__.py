import os
import sys
import pandas as pd

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)
os.chdir(PARENT_DIR)

pd.set_option('display.max_columns', None)
pd.set_option('mode.chained_assignment', None)
