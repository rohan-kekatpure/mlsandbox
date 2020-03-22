import pandas as pd
import matplotlib.pyplot as pl
from rusboost import RusBoost, trace
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, get_scorer
from sklearn.model_selection import GridSearchCV
import numpy as np


def evaluate(estimator, X_train, y_train, X_test, y_test):
    estimator.fit(X_train, y_train)
    y_train_proba = estimator.predict_proba(X_train)[:, 1]
    y_test_proba = estimator.predict_proba(X_test)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba, pos_label=1)
    auc_train = roc_auc_score(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba, pos_label=1)
    auc_test = roc_auc_score(y_test, y_test_proba)
    return fpr_train, tpr_train, auc_train, fpr_test, tpr_test, auc_test


def get_staimage(sampl_pct):

    df_train = pd.read_csv('./data/satimg_train.csv', sep=' ')
    df_test = pd.read_csv('./data/satimg_test.csv', sep=' ')
    xcols = df_train.columns[:-1]
    PERCENT = 0.01

    df_train['y1'] = 1
    df_test['y1'] = 1

    df_train.loc[df_train['y'] == 4, 'y1'] = -1
    df_test.loc[df_test['y'] == 4, 'y1'] = -1

    # Sampling of negative examples
    pos_idx = df_train['y1'] == 1
    neg_idx = ~ pos_idx
    df_train = df_train[pos_idx].append(df_train[neg_idx].sample(frac=sampl_pct * PERCENT))
    print df_train['y1'].value_counts()

    X_train = df_train[xcols].values
    y_train = df_train['y1'].values
    X_test = df_test[xcols].values
    y_test = df_test['y1'].values

    return X_train, y_train, X_test, y_test


def tune(X, y, estimator, param_grid):
    gcv = GridSearchCV(estimator, param_grid, refit=True, scoring=get_scorer('roc_auc'),
                       n_jobs=-1, verbose=5)
    gcv.fit(X, y)
    return gcv.best_estimator_


def test():
    # Estimators
    rusb = RusBoost()
    rf = RandomForestClassifier(class_weight='balanced')
    gbm = GradientBoostingClassifier()

    # Parameter grids
    rf_param_grid = {'max_depth': range(5, 10), 'n_estimators': [400, 500, 600, 700]}
    rusb_param_grid = {'base_estimator': [DecisionTreeClassifier(max_depth=i) for i in range(1, 5)],
                       'n_estimators': [200, 300, 400, 500, 600, 700], 'learning_rate': np.linspace(0.1, 2.0, 10)}
    gbm_param_grid = {'max_depth': range(1, 5), 'n_estimators': range(100, 800, 100),
                      'learning_rate': np.linspace(0.1, 2.0, 10)}

    estimator_objs = [
        ('RUSB', rusb, rusb_param_grid),
        ('GBM', gbm, gbm_param_grid),
        ('RF', rf, rf_param_grid)
    ]

    # Choose class imbalance ratios. Note that the dataset we test on is already imbalanced at 10:1 ratio.
    # So a minority class sampling percent of 50% means an imbalance of 5%. And so on.
    imbalances = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2]
    fig, axes = pl.subplots(len(imbalances), 4, figsize=(12, 10))
    largs = dict(loc=4, prop={'size': 6})

    for k, ip in enumerate(imbalances):
        # Get data
        tdata = X_train, y_train, X_test, y_test = get_staimage(ip * 10.0)

        # Tune each estimator and evaluate performance
        for i, (est_name, est_base, est_params) in enumerate(estimator_objs):
            est_best = tune(X_train, y_train, est_base, est_params)
            fpr_train, tpr_train, auc_train, fpr_test, tpr_test, auc_test = evaluate(est_best, *tdata)
            axes[k, i].plot(fpr_train, tpr_train, 'g-', label='{} auc_train = {:0.4f}'.format(est_name, auc_train))
            axes[k, i].plot(fpr_test, tpr_test, 'r-', label='{} auc_test = {:0.4f}'.format(est_name, auc_test))
            axes[k, i].legend(**largs)
            axes[k, 3].plot(fpr_test, tpr_test, '-', label=est_name)
            axes[k, 3].legend(**largs)
            if i == 2:
                axes[k, 3].text(0.75, 0.75, 'imbal {:0.2f}%'.format(ip), fontsize=6)


    pl.tight_layout()
    pl.savefig('img/results.png')
    pl.show()

    # trace()


if __name__ == '__main__':
    test()

