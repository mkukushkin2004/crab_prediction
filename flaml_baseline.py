import pandas as pd
import numpy as np
from flaml import AutoML

import warnings
warnings.filterwarnings('ignore')
SEED = 46
np.random.seed(SEED)

# noinspection PyShadowingNames
def read_data():
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/test.csv')
    original = pd.read_csv('input/CrabAgePrediction.csv')
    extended = pd.read_csv('input/train_extended.csv')
    synthetic = pd.read_csv('input/train_synthetic.csv')

    train = train.drop(columns=['id'])
    test = test.drop(columns=['id'])
    extended = extended.drop(columns=['id'])
    synthetic = synthetic.drop(columns=['id'])

    train['original'] = 0
    test['original'] = 0
    original['original'] = 1
    extended['original'] = 1
    synthetic['original'] = 1

    for df in [train, test, original, extended, synthetic]:
        df['Sex'] = df['Sex'].replace('F', 0).replace('I', 1).replace('M', 2).replace('0.025', 0).astype('int')

    train = pd.concat([train, original, extended, synthetic], axis=0)
    train.reset_index(inplace=True, drop=True)
    return train, test, original, extended, synthetic


train, test, original, extended, synthetic = read_data()
X, y = train.drop(columns=['Age']), train['Age']


automl = AutoML()
automl.fit(X, y,
           task='regression',
           metric='mae',
           time_budget=600,
           verbose=0)

print(automl.best_estimator)
print(automl.best_config_per_estimator)
pred = automl.predict(test)
submission = pd.read_csv('input/sample_submission.csv')
submission['Age'] = pred.round().astype(int)
submission.to_csv('output/FLAML_Baseline_Submission_60.csv', index=False)
