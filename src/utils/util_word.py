import itertools

import numpy as np

from utils.util_fasta import get_seqs

ALPHABET = {'DNA': 'ACGT', 'RNA': 'ACGU', 'Protein': 'ACDEFGHIKLMNPQRSTVWY'}
ALPHABET_X = {'DNA': 'ACGTX', 'RNA': 'ACGUX', 'Protein': 'ACDEFGHIKLMNPQRSTVWYX'}


def make_km_list(k, alphabet):
    try:
        return ["".join(e) for e in itertools.product(alphabet, repeat=k)]
    except TypeError:
        print("TypeError: k must be an inter and larger than 0, alphabet must be a string.")
        raise TypeError
    except ValueError:
        print("TypeError: k must be an inter and larger than 0")
        raise ValueError


def seq_length_fixed(seq_list, fixed_len):
    sequence_list = []
    for seq in seq_list:
        seq_len = len(seq)
        if seq_len <= fixed_len:
            for i in range(fixed_len - seq_len):
                seq += 'X'
        else:
            seq = seq[:fixed_len]
        sequence_list.append(seq)

    return sequence_list


def km_words(input_file, category, fixed_len, word_size):
    """ convert sequence to corpus """
    with open(input_file, 'r') as f:
        seq_list = get_seqs(f, ALPHABET[category])
    seq_list = seq_length_fixed(seq_list, fixed_len)
    km_list = make_km_list(word_size, ALPHABET_X[category])
    corpus = []
    for sequence in seq_list:
        word_list = []
        # windows slide along sequence to generate gene/protein words
        for i in range(len(sequence)):
            if i < len(sequence) - word_size + 1:
                word = sequence[i: i + word_size]
            else:
                word_lst = list(sequence[i: len(sequence)]) + ['X'] * (word_size - (len(sequence) - i))
                word = ''.join(word_lst)
                # print(word)
            word_list.append(km_list.index(word))
        corpus.append(word_list)
    return np.array(corpus, dtype=int)
