# import pdb
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from scipy.io import mmread, mmwrite
tqdm.pandas()

# Load files
encoding = json.load(open("/Users/darvesh/antibody-design/data/encoding.json", "r"))
tcells_path = "/Users/darvesh/antibody-design/data/tcell_receptor_table_export_1667353882.csv"
symbol2int = encoding['symbol']['one-character']
tcells = pd.read_csv(tcells_path, low_memory=False)

# Define apply functions
def convert(x):
    """Convert amino acid symbols into integer"""
    return np.array([symbol2int[char] for char in x])

def pad_array(x, max_size):
    """Pad unequal numpy arrays with 0"""
    return np.pad(x, (0, max_size - len(x)), "constant", constant_values=0)

# Subset to full sequences
full_chain_cols = ['Chain 1 Full Sequence', 'Chain 2 Full Sequence']
full_chain_df = tcells[full_chain_cols]
full_chain_df = full_chain_df.dropna()

# Turn chain 1 and 2 sequences into 1 column
full_chain_df = full_chain_df.melt(value_vars=['Chain 1 Full Sequence', 'Chain 2 Full Sequence'], value_name='full_chain')
full_chain_df = full_chain_df[['full_chain']]

full_chain_df['encoded'] = full_chain_df['full_chain'].progress_apply(convert)

max_size = max([len(x) for x in full_chain_df['encoded']])

full_chain_df['padded'] = full_chain_df['encoded'].progress_apply(pad_array, convert_dtype=True, args=(max_size,))
padded_arrays = full_chain_df['padded'].to_numpy(dtype='object')
padded_2darray = np.row_stack(padded_arrays)
padded_matrix = np.matrix(padded_2darray)

mmwrite("/Users/darvesh/antibody-design/data/encoded_antibodies.mtx", padded_matrix)
