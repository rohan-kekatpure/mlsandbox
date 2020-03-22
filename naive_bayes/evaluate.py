import sys
import pandas as pd

if len(sys.argv) < 3:
    print "Usage:"
    print "python evaluate.py <predicted_labels_file> <true_labels_file>"
    sys.exit(1)

fname_pred, fname_true = sys.argv[1], sys.argv[2]
df_pred = pd.read_csv(fname_pred, names=["fname", "label"])
df_true = pd.read_csv(fname_true, names=["fname", "label"])
df_merged = pd.DataFrame.merge(df_pred, df_true, on="fname", 
                               how="outer", suffixes=["_pred", "_true"])

N_spam_true = len(df_true[df_true.label=="spam"])
N_ham_true = len(df_true[df_true.label=="ham"])

N_true_pos = len(df_merged[(df_merged.label_pred=="spam") & (df_merged.label_true=="spam")])
N_true_neg = len(df_merged[(df_merged.label_pred=="ham") & (df_merged.label_true=="ham")])
N_false_pos = len(df_merged[(df_merged.label_pred=="spam") & (df_merged.label_true=="ham")])
N_false_neg = len(df_merged[(df_merged.label_pred=="ham") & (df_merged.label_true=="spam")])

assert N_true_pos + N_true_neg + N_false_pos + N_false_neg == len(df_pred)
print "Accuracy: %0.2f" % (100.0 * float(N_true_pos + N_true_neg) / len(df_pred))
print "True positive rate: %0.2f" % (100.0 * float(N_true_pos) / float(N_spam_true))
print "False positive rate: %0.2f" % (100.0 * float(N_false_pos) / float(N_ham_true))





    
