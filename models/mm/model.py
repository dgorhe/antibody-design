import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import defaultdict

class MarkovModel():
    def __init__(self, filename):
        self.transition_matrix = self.generate_transition_matrix(filename)

    # Load primary sequence data
    def generate_transition_matrix(self, filename):
        data = self.load_data(filename)
        pairwise_freq, symbols = self.find_pairwise_frequencies(data)
        
        matrix = np.zeros((len(symbols), len(symbols)))
        matrix = pd.DataFrame(matrix)
        matrix.columns = [s for s in sorted(symbols)]
        matrix.index = [s for s in sorted(symbols)]

        for pair, value in tqdm(pairwise_freq.items()):
            matrix.loc[pair[0], pair[1]] = value
        
        return matrix
    
    # Load primary sequence data
    def load_data(self, filename):
        try:
            if filename.endswith("csv"):
                df = pd.read_csv(filename)
            elif filename.endswith("tsv"):
                df = pd.read_csv(filename, sep='\t')
            else:
                print("Please use one of the following file extensions:")
                print("csv, tsv")
        except FileNotFoundError:
            print(f"Looks like the file: {filename} doesn't exist :(")
            print("Re-run the program with the correct filepath")
            sys.exit(0)

        return df[df.columns[0]].tolist()

    # Find p_ij and p_ji for each amino acid i --> amino acid j transition
    def find_pairwise_frequencies(self, strings: list[str]):
        symbols = set()
        pairs = defaultdict(lambda: 0)

        for s in strings:
            for i in range(1, len(s)):
                # sliding window of size 2
                pair = (s[i-1], s[i])
                symbols.add(s[i-1])
                pairs[pair] += 1
        
        # Normalize by counts
        total = sum(pairs.values())
        normalized = {key: value/total for key, value in pairs.items()}

        return normalized, symbols

    def generate_sequence(self, length, start=None):
        if start is None:
            start = np.random.choice(self.transition_matrix.columns)
            sequence = [start]
        else:
            sequence = [s for s in start]
            
        for _ in range(length - 1):
            transition_probs = self.transition_matrix.loc[sequence[-1]]
            normalized_probs = transition_probs / transition_probs.sum()
            next_aa = np.random.choice(
                self.transition_matrix.columns, 
                p = normalized_probs
            )
            sequence.append(next_aa)
            
        return ''.join(sequence)

    def save_transition_matrix(self, extension='parquet'):
        if extension == 'parquet':
            self.transition_matrix.to_parquet(f'transition-matrix.{extension}')
        if extension == 'csv':
            self.transition_matrix.to_csv(f'transition-matrix.{extension}', sep=',')
        if extension == 'tsv':
            self.transition_matrix.to_tsv(f'transition-matrix.{extension}', sep='\t')


if __name__ == "__main__":
    tcr_path = os.path.abspath("../../data/tcr.csv")
    mm = MarkovModel(tcr_path)
    print(mm.transition_matrix)
