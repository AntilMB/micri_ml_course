import sys
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

y_true = np.array(['B', 'B', 'B', 'B', 'B', 'B', 'B', 'M', 'B', 'M', 'B', 'B', 'B',
		   'B', 'M', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B',
       		   'B', 'B', 'B', 'M', 'B', 'M', 'B', 'M', 'B', 'B', 'B', 'B', 'B',
                   'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'M',
                   'B', 'B', 'M', 'B', 'M', 'M', 'B', 'M', 'M', 'M', 'M', 'B', 'B',
       'M', 'M', 'B', 'B', 'B', 'M', 'B', 'M', 'M', 'B', 'B', 'B', 'M',
       'M', 'B', 'B', 'M', 'B', 'B', 'B', 'M', 'M', 'B', 'M', 'B', 'M',
       'M', 'B', 'B', 'B', 'M', 'M', 'M', 'M', 'M', 'B', 'B', 'M', 'B',
       'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'B', 'B',
       'M', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'B', 'B', 'B', 'M',
       'M', 'B', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'B', 'M', 'M', 'M'])

print(sys.argv)
df = pd.read_csv(sys.argv[1])

score = roc_auc_score(y_true, df['predicted_prob'].values)

msg = 'score {}\nreversed score {}'
print(msg.format(score, 1 - score))

