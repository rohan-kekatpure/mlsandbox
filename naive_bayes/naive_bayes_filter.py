import glob
import os
import string 
from collections import defaultdict
from math import log, exp
import string
import random


ALLOWED = " " + string.ascii_lowercase + string.digits + string.punctuation
def standardize(str_):
    s1 = str_.lower().replace("\n", " ").replace("\r", " ")    
    return "".join([c for c in s1 if c in ALLOWED])


def std_bag(str_):
    return [w[:4] for w in standardize(str_).split()]

def get_word_prob(dir_):
    paths = glob.glob(os.path.join(dir_, "*"))
    word_count = defaultdict(int)
    for fn in paths:
        with open(fn, "r") as ef:
            bag = std_bag(ef.read())
            for w in bag:                 
                word_count[w] += 1

    # Compute log probabilities by normalizing with total_count_
    MIN_FREQ, MAX_FREQ = 0, 10
    filtered_word_count = {
        k: v for k, v in word_count.items() 
        if (MIN_FREQ < v < MAX_FREQ) 
    }

    total_count = float(sum(filtered_word_count.values()))
    return {k: log(v / total_count) for k, v in filtered_word_count.items()}


def spam_proba(email_str, wpl_spam, wpl_ham, lprior=0.0):
    bag = std_bag(email_str)
    S = float(lprior) + sum([wpl_spam.get(w, 0.0) - wpl_ham.get(w, 0.0) for w in bag])
    return float(S >= 0.0)


def main():

    # Get probabilities from the Training data
    TRAIN_DIR = "../data/train"
    wpl_ham = get_word_prob(os.path.join(TRAIN_DIR, "ham"))  
    wpl_spam = get_word_prob(os.path.join(TRAIN_DIR, "spam"))

    # Generate predictions for test set
    TEST_DIR = "../data/test"
    test_email_files = glob.glob(os.path.join(TEST_DIR, "ham/*")) \
                     + glob.glob(os.path.join(TEST_DIR, "spam/*")) 
    
    scores_dct = dict()
    for email_file in test_email_files:
        with open(email_file, "r") as ef:
             scores_dct[os.path.basename(email_file)] = \
                spam_proba(ef.read(), wpl_spam, wpl_ham, lprior=0.0)

    # Write scores to output file
    with open("predicted.csv", "w") as pf:
        for fname, scr in scores_dct.items():
            label = "spam" if scr > 0 else "ham"
            pf.write("%s,%s\n" % (fname, label))

if __name__ == "__main__":
    main()
