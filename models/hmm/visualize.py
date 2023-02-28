import sys
import os

import model as hmm
import numpy as np
import matplotlib.pyplot as plt
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from utils import DIRS, FILES

# Font for the labels
font = {'family': 'monospace',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

# One color per amino acid
palette = [
    '#C8C8C8', '#145AFF', '#00DCDC', '#E60A0A', '#E6E600',
    '#00DCDC', '#E60A0A', '#EBEBEB', '#8282D2', '#0F820F', 
    '#0F820F', '#145AFF', '#E6E600', '#3232AA', '#DC9682', 
    '#FA9600', '#FA9600', '#B45AB4', '#3232AA', '#0F820F', 
    '#FFFFFF'
]

# All amino acids + gap
aa = 'ARNDCQEGHILKMFPSTWYV-'

def find_consensus_sequence():
    pass

def alignment_visualization(
    sequence_length=25,
    number_of_sequences=10, 
    start_sequence="A",
    save=True,
    output="alignment.png") -> None:
    
    model = hmm.HiddenMarkovModel(FILES["TCR"])
            
    msa = []
    for i in range(number_of_sequences):
        seq = model.generate_sequence(sequence_length, start=start_sequence)
        msa.append(SeqRecord(Seq(seq), id=f"seq{i}"))
    
    L = len(msa[0].seq)
    N = len(msa)

    # Find the consensus sequence
    freq = np.zeros((L, sequence_length))
    consensus = np.zeros(L)
    for i in range(N):
        for j in range(L):
            j_aa = aa.find(msa[i].seq[j])
            freq[j,j_aa] = freq[j,j_aa] + 1

    for i in range(0, L):
        consensus[i] = freq[i].argmax()

    # Calculate the conservation
    conservation = np.sqrt(np.sum((np.square(freq/N - 0.05)),axis=1))

    # Plot the alignment
    figure = plt.figure(figsize=(10,2))
    axes = figure.add_axes((0,0,1,1))
    axes.bar(range(0,L),conservation, align='edge', linewidth = 0, color = 'red')
    axes.set_ylabel('Conservation')

    spacing_scale = axes.get_ylim()[1]/4
    spacing = spacing_scale*2
    seq_display = np.linspace(start = 0, stop = N - 1, num = N, dtype=np.int8)

    # For each sequence generated, plot the characters in the sequence
    for j in seq_display:
        # Label and initial positioning of the sequence
        posit = -float(np.where(seq_display == j)[0].item()) * spacing_scale - spacing
        axes.text(x = -5, y = posit, s = f"Seq {str(j+1)}")
        
        # Each character
        for i in range(0, L):
            axes.text(
                x = float(i),
                y = posit,
                s = msa[j].seq[i],
                bbox=dict(facecolor=palette[aa.find(msa[j].seq[i])], 
                alpha=0.5),
                fontdict=font
            )
            
    posit = posit - spacing
    axes.text(-5, posit, "Consensus")
    for i in range(0, L):
        axes.text(float(i),posit, 'ARNDCQEGHILKMFPSTWYV-'[int(consensus[i])] ,
                    bbox=dict(facecolor=palette[int(consensus[i])], 
                    alpha=0.5),fontdict=font)
    
    if save:
        figure.savefig(output, dpi=500, bbox_inches='tight')
    
if __name__ == "__main__":
    alignment_visualization(
        sequence_length=25,
        number_of_sequences=10, 
        start_sequence="A",
        save=True,
        output=os.path.join(DIRS["FIGURES"], "mm-alignment.png")
    )