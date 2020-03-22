import numpy as np
from collections import Counter
import string
from scipy.special import logsumexp
from copy import deepcopy
import re

def get_transitions(s, alphabet):
    s = s.lower()    
    deletions = ''.join([c for c in string.printable if c not in alphabet])
    s = s.translate(None, deletions)
    transitions = zip(s[:-1], s[1:])
    return transitions


def get_trans_prob(alphabet):
    '''
    Get letter transition probabilities by reading a standard text
    '''

    with open('war_and_peace.txt') as f:
        wap = f.read()
    
    M = len(alphabet)
    transitions = get_transitions(wap, alphabet)
    freqs = Counter(transitions)

    Tmtx = np.zeros(shape=(M, M), dtype=np.float)
    for i, frm in enumerate(alphabet):
        for j, to in enumerate(alphabet):
            val = freqs.get((frm, to), 1.0e-6)            
            Tmtx[i, j] = val

    Pmtx = np.log(Tmtx / Tmtx.sum(axis=1).reshape((-1, 1)))
    assert np.allclose(logsumexp(Pmtx, axis=1), 0)

    log_prob = {}
    for i, frm in enumerate(alphabet):
        for j, to in enumerate(alphabet):
            log_prob[(frm, to)] = Pmtx[i, j]

    return log_prob

def score(str_, alphabet, log_prob, beta):
    transitions = get_transitions(str_, alphabet)
    log_scores = np.array([log_prob[t] for t in transitions])
    return np.exp(beta * log_scores.sum())
    

def main():
    alphabet = string.ascii_lowercase + ' '
    M = len(alphabet)

    permutation = ''.join(np.random.permutation(list(alphabet)))    
    substitution = string.maketrans(alphabet, permutation)

    # true_text = '''the moon is a barren rocky world without air water. 
    # it has dark lava plain on its surface. the moon is filled with craters.
    # it has no light of its own. it gets its light from
    # the sun. The Moon keeps changing its shape as it moves round the
    # Earth. It spins on its axis in 27.3 days stars were named after
    # the Edwin Aldrin were the first ones to set their foot on the Moon
    # on 21 July 1969 They reached the Moon in their space craft named
    # '''

    true_text = '''A portmanteau is a
    linguistic blend of words, in which parts of multiple words or
    their phones (sounds) are combined into a new word, as in smog,
    coined by blending smoke and fog, or motel, from motor and
    hotel. In linguistics, a portmanteau is defined as a single
    morph that represents two or more morphemes.

    The definition overlaps with the grammatical term contraction, but
    contractions are formed from words that would otherwise appear
    together in sequence, such as do and not to make don't, whereas a
    portmanteau word is formed by combining two or more existing words
    that all relate to a singular concept. A portmanteau also differs from
    a compound, which does not involve the truncation of parts of the
    stems of the blended words.'''




    true_text = re.sub('[ \n]+', ' ', true_text).strip().lower()

    coded = true_text.translate(substitution)
    print coded    

    
    log_prob = get_trans_prob(alphabet)

    # MCMC calculation
    ulist = list(alphabet)
    beta = 0.1
    

    for itr in range(10000):
        i, j = np.random.randint(0, M, 2)
        vlist = deepcopy(ulist)
        vlist[i], vlist[j] = vlist[j], vlist[i]
        v = ''.join(vlist)
        u = ''.join(ulist)

        decoded_old = coded.translate(string.maketrans(u, alphabet))
        decoded_new = coded.translate(string.maketrans(v, alphabet))
        args = alphabet, log_prob, beta
        a = min(1.0, score(decoded_new, *args) / score(decoded_old, *args))
        if np.random.random() < a:
            ulist = vlist
           
        if itr % 100 == 0: 
            print beta, decoded_new + '\n'
    



    

if __name__ == '__main__':
    main()