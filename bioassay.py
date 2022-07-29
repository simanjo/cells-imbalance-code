#%%
from email.mime import base
import os
from random import random

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import randomized_svd
from sklearn.linear_model import RANSACRegressor
plt.style.use('default')

script_dir = os.path.dirname(os.path.abspath("bioassay.py"))
data_dir = os.path.join(script_dir, "VirtualScreeningData")

AIDS = [456, 1608, 439, 1284, 644]

RANDOM_STATE = 5164826

SCORING = {
    'acc': 'accuracy',
    'bacc': 'balanced_accuracy',
    'auc': 'roc_auc',
    'AP': 'average_precision',
    'MCC': skmetrics.make_scorer(skmetrics.matthews_corrcoef)
}

# auxillary functions

# loading of training and test set
def _load_data(AID, type):
    if AID not in AIDS:
        print("Please specify a valid AID")
        return None

    if type not in ["train", "test"]:
        print(
            "Please specify a valid data type, 'train' or 'test'."
        )
        return None

    def _convert_outcome(x):
        if x == 'Active':
            return 1
        if x == 'Inactive':
            return 0
        if x == 'Inconc':
            return 0
        raise ValueError(
            "The data contains an outcome different from Active/Inactive"
        )
    data = pd.read_csv(os.path.join(data_dir, f"AID{AID}red_{type}.csv"))
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1].apply(_convert_outcome)
    return X, y

# custom ranking functions
def _refit_metric(cv_results):
    # initialize as array of zeroes
    rank = np.zeros(len(cv_results[f"rank_test_acc"]))
    for score_name in SCORING.keys():
        if "score_name" == 'acc':
            # skip accuracy as it is highly unreliable
            continue
        rank += cv_results[f"rank_test_{score_name}"]
    return np.argmin(rank)

def _get_params(test, data, verbose=False):
    model, grid = test
    train_X, train_y = data

    grid_search = model_selection.GridSearchCV(
            model, grid, scoring=SCORING, refit=_refit_metric
    )
    grid_search.fit(train_X, train_y)
    scores = {}
    if verbose:
        print(f"Best parameters set found on development set for {model}:")
        print()
        print(grid_search.best_params_)
    for score_name in SCORING.keys():
        if verbose:
            print()
            print(f"Grid scores on development set for {score_name}:")
            print()
        means = grid_search.cv_results_[f"mean_test_{score_name}"]
        stds = grid_search.cv_results_[f"std_test_{score_name}"]
        score_res = []
        for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
            if verbose:
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            score_res.append((mean, std, params))
        scores[score_name] = score_res
    return grid_search.best_params_, scores, grid_search.cv_results_


def perform_cv_training(AID, test_set):
    if AID not in AIDS:
        print("Please specify a valid AID")
        return None

    params = {}
    scores = {}
    full = {}
    data = _load_data(AID, 'train')
    for (name, test) in test_set.items():
        print(f"Started CV for {name}")
        params[name], scores[name], full[name] = _get_params(test, data)
        print(f"Finished CV for {name}")
    best_scores = {}
    for name, param in params.items():
        best_scores[name] = {}
        score_tuples = scores[name]
        print(f"Best scores for {name}:")
        for score, vals in score_tuples.items():
            best_score = (0, -1)
            for mean, std, pars in vals:
                if pars == param:
                    best_score = (mean, std)
                    break
            best_scores[name][score] = best_score
            print(f"{score}: {best_score[0]} (+/- {2*best_score[1]})")
    return params, scores, full, best_scores


def fit_and_evaluate_model(AID, model, latex=False):
    if AID not in AIDS:
        print("Please specify a valid AID")
        return None

    X_train, y_train = _load_data(AID, 'train')
    X_test, y_test = _load_data(AID, 'test')
    print(model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if latex:
        output = ""
        output += f"& {round(skmetrics.accuracy_score(y_test, y_pred), 3)} "
        output += f"& {round(skmetrics.balanced_accuracy_score(y_test, y_pred), 3)} "
        output += f"& {round(skmetrics.RocCurveDisplay.from_estimator(model, X_test, y_test).roc_auc, 3)} "
        output += f"& {round(skmetrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test).average_precision, 3)} "
        output += f"& {round(skmetrics.matthews_corrcoef(y_test, y_pred), 3)}\\"
        return output
    print(f"Accuracy: {round(skmetrics.accuracy_score(y_test, y_pred), 3)}")
    print(f"Balanced Accuracy: {round(skmetrics.balanced_accuracy_score(y_test, y_pred), 3)}")
    print(f"ROC AUC: {round(skmetrics.RocCurveDisplay.from_estimator(model, X_test, y_test).roc_auc, 3)}")
    print(f"PR AUC: {round(skmetrics.PrecisionRecallDisplay.from_estimator(model, X_test, y_test).average_precision, 3)}")
    print(f"MCC: {round(skmetrics.matthews_corrcoef(y_test, y_pred), 3)}")

#%%

# normal training
from sklearn import metrics as skmetrics
from sklearn import model_selection, naive_bayes, svm, neighbors, tree
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import feature_selection

# setting up grids

bayes_grid = {'bayes__var_smoothing': [1e-09, 1e-06, 1e-03, 1e-02]}

svm_grid = {
    'svm__class_weight': [None],
    'svm__C': [0.1, 1, 10, 100, 1000],
    'svm__gamma': [0.01, 0.1, 'auto', 'scale']
}

svm_bal_grid = {
    'svm__class_weight': ['balanced'],
    'svm__C': [0.1, 1, 10, 100, 1000],
    'svm__gamma': [0.01, 0.1, 'auto', 'scale']
}

knn_grid = {
    'knn__weights': ['uniform', 'distance'],
    'knn__n_neighbors': [2, 5, 10, 15, 20]
}

tree_grid = {
    'dtree__class_weight': [None],
    'dtree__random_state': [RANDOM_STATE],
    'dtree__criterion' : ["gini", "log_loss"]
}

tree_bal_grid = {
    'dtree__class_weight': ['balanced'],
    'dtree__random_state': [RANDOM_STATE],
    'dtree__criterion' : ["gini", "log_loss"]
}

fa_grid = {
    'kbest__k': [10, 20, 30, 40, 50, 75, 100]
}

# setting up tests

normal_tests = {
    'bayes': (
        Pipeline([
            (
                'kbest',
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('bayes', naive_bayes.GaussianNB())
        ]), bayes_grid | fa_grid
    ),
    'svm': (
        Pipeline([
            ('kbest', 
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('svm', svm.SVC(kernel='rbf', random_state=RANDOM_STATE))
        ]), svm_grid | fa_grid
    ),
    'svm_bal': (
        Pipeline([
            ('kbest', 
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('svm', svm.SVC(kernel='rbf', random_state=RANDOM_STATE))
        ]), svm_bal_grid | fa_grid
    ),
    'knn': (
        Pipeline([
            ('kbest',
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('knn', neighbors.KNeighborsClassifier(n_jobs=-1))
        ]), knn_grid | fa_grid
    ),
    'tree': (
        Pipeline([
            ('kbest',
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('dtree', tree.DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]), tree_grid | fa_grid
    ),
    'tree_bal': (
        Pipeline([
            ('kbest',
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('dtree', tree.DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]), tree_bal_grid | fa_grid
    ),
}

linear_svm_tests = {
    'svm_lin': (
        Pipeline([
            ('kbest', 
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('svm', svm.SVC(kernel='linear', random_state=RANDOM_STATE))
        ]), {'svm__C': [1, 10, 100, 1000]} | fa_grid
    ),
    'svm_lin_bal': (
        Pipeline([
            ('kbest', 
                feature_selection.SelectKBest(
                    score_func=feature_selection.f_classif
                )
            ),
            ('svm', svm.SVC(
                kernel='linear', random_state=RANDOM_STATE,
                class_weight='balanced'
            ))
        ]), {'svm__C': [1, 10, 100, 1000]} | fa_grid
    ),
}

#%%
#balanced ensemble training

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

# brf = BalancedRandomForestClassifier(
#     n_estimators=100, random_state=RANDOM_STATE, max_features=None
# )

# setting up grids

bag_grid = {
    'bag__random_state': [RANDOM_STATE],
    'bag__sampler': [
        RandomUnderSampler(random_state=RANDOM_STATE),
        TomekLinks(),
        SMOTE(random_state=RANDOM_STATE),
        ADASYN(random_state=RANDOM_STATE),
        RandomOverSampler(random_state=RANDOM_STATE, shrinkage=0.5)
    ]
}

bag_bayes_grid = bag_grid | {
    'bag__base_estimator': [
        naive_bayes.GaussianNB(var_smoothing=0.001),
        naive_bayes.GaussianNB(var_smoothing=1e-06),
        naive_bayes.GaussianNB(var_smoothing=1e-09)
    ],
    'bag__n_estimators': [10, 20, 50]
}

bag_svm_grid = bag_grid | {
    'bag__base_estimator': [
        svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE)
    ],
    'bag__n_estimators': [10]
}

bag_knn_grid = bag_grid | {
    'bag__base_estimator': [
        neighbors.KNeighborsClassifier(n_neighbors=2, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance'),
        neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance'),
        neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance'),
    ],
    'bag__n_estimators': [10, 20, 50]
}

bag_forest_grid = bag_grid | {
    'bag__base_estimator': [
        RandomForestClassifier(random_state=RANDOM_STATE, max_features=None),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    ],
    'bag__n_estimators': [10, 20]
}


eec_grid = {
    'eec__random_state': [RANDOM_STATE],
}

eec_bayes_grid = eec_grid | {
    'eec__base_estimator': [
        naive_bayes.GaussianNB(var_smoothing=0.001),
        naive_bayes.GaussianNB(var_smoothing=1e-06),
        naive_bayes.GaussianNB(var_smoothing=1e-09)
    ],
    'eec__n_estimators': [10, 20, 50]
}

eec_svm_grid = eec_grid | {
    'eec__base_estimator': [
        svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE)
    ],
    'eec__n_estimators': [10, 20]
}

eec_knn_grid = eec_grid | {
    'eec__base_estimator': [
        neighbors.KNeighborsClassifier(n_neighbors=2, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform'),
        neighbors.KNeighborsClassifier(n_neighbors=2, weights='distance'),
        neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance'),
        neighbors.KNeighborsClassifier(n_neighbors=10, weights='distance'),
    ],
    'eec__n_estimators': [10, 20, 50]
}

eec_forest_grid = eec_grid | {
    'eec__base_estimator': [
        RandomForestClassifier(random_state=RANDOM_STATE, max_features=None),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    ],
    'eec__n_estimators': [10, 20, 50]
}


rus_grid = {
    'rus__random_state': [RANDOM_STATE],
}

rus_bayes_grid = rus_grid | {
    'rus__base_estimator': [
        naive_bayes.GaussianNB(var_smoothing=0.001),
        naive_bayes.GaussianNB(var_smoothing=1e-06),
        naive_bayes.GaussianNB(var_smoothing=1e-09)
    ],
    'rus__n_estimators': [10, 20, 50]
}

rus_svm_grid = rus_grid | {
    'rus__base_estimator': [
        svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=10, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='auto', class_weight='balanced',
                random_state=RANDOM_STATE),
        svm.SVC(C=1, gamma='scale', class_weight='balanced',
                random_state=RANDOM_STATE)
    ],
    'rus__n_estimators': [10, 20, 50],
    'rus__algorithm': ['SAMME']
}

rus_forest_grid = rus_grid | {
    'rus__base_estimator': [
        RandomForestClassifier(random_state=RANDOM_STATE, max_features=None),
        BalancedRandomForestClassifier(random_state=RANDOM_STATE, max_features=None)
    ],
    'rus__n_estimators': [10, 20, 50]
}


# setting up test pipelines

fa_grid_small = {
    'kbest__k': [10, 20, 30, 50]
}


bagging_tests = {
    _name: (
        Pipeline([
            ('kbest', feature_selection.SelectKBest()),
            ('bag', BalancedBaggingClassifier())
        ]), fa_grid_small | _grid
    ) for _name,_grid in [
        ("Bag", bag_grid), ("Bag+Bayes", bag_bayes_grid),
        ("Bag+SVM", bag_svm_grid), ("Bag+KNN", bag_knn_grid),
        ("Bag+Forest", bag_forest_grid)
    ]
}

eec_tests = {
    _name: (
        Pipeline([
            ('kbest', feature_selection.SelectKBest()),
            ('eec', EasyEnsembleClassifier())
        ]), fa_grid_small | _grid
    ) for _name,_grid in [
        ("ECC", eec_grid), ("ECC+Bayes", eec_bayes_grid),
        ("ECC+SVM", eec_svm_grid), ("ECC+KNN", eec_knn_grid),
        ("ECC+Forest", eec_forest_grid)
    ]
}

rus_tests = {
    _name: (
        Pipeline([
            ('kbest', feature_selection.SelectKBest()),
            ('rus', RUSBoostClassifier())
        ]), fa_grid_small | _grid
    ) for _name,_grid in [
        ("RUS", rus_grid), ("RUS+Bayes", rus_bayes_grid),
        ("RUS+SVM", rus_svm_grid), ("RUS+Forest", rus_forest_grid)
    ]
}

def perform_balanced_cv_training(AID, test_set):
    if AID not in AIDS:
        print("Please specify a valid AID")
        return None

    params = {}
    scores = {}
    full = {}
    data = _load_data(AID, 'train')
    for (name, test) in test_set.items():
        print(f"Started CV for {name}")
        params[name], scores[name], full[name] = _get_params(test, data)
        print(f"Finished CV for {name}")
    best_scores = {}
    for name, param in params.items():
        best_scores[name] = {}
        score_tuples = scores[name]
        print(f"Best scores for {name}:")
        for score, vals in score_tuples.items():
            best_score = (0, -1)
            for mean, std, pars in vals:
                if pars == param:
                    best_score = (mean, std)
                    break
            best_scores[name][score] = best_score
            print(f"{score}: {best_score[0]} (+/- {2*best_score[1]})")
    return params, scores, full, best_scores


#%%
# models for AID439 from hyperparamtertuning
from sklearn.pipeline import make_pipeline
bayes_439 = make_pipeline(
    feature_selection.SelectKBest(), naive_bayes.GaussianNB()
)
svm_439 = make_pipeline(
    feature_selection.SelectKBest(),
    svm.SVC(gamma=0.1, random_state=RANDOM_STATE)
)
svm_balanced_439 = make_pipeline(
    feature_selection.SelectKBest(),
    svm.SVC(
        C=10, class_weight='balanced',
        gamma=0.1, random_state=RANDOM_STATE
    )
)
knn_439 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    neighbors.KNeighborsClassifier(n_neighbors=2)
)
tree_439 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
)
tree_balanced_439 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    tree.DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_STATE
    )
)

normal_models_439 = [bayes_439, svm_439, svm_balanced_439, knn_439, tree_439, tree_balanced_439]

#####################################################
bag_439 = make_pipeline(
    feature_selection.SelectKBest(k=30),
    BalancedBaggingClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=RandomOverSampler(random_state=RANDOM_STATE, shrinkage=0.5)
        )
)
bag_bayes_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=naive_bayes.GaussianNB(var_smoothing=1e-06),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=RandomUnderSampler(random_state=RANDOM_STATE)
        )
)
bag_svm_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=RandomUnderSampler(random_state=RANDOM_STATE)
        )
)
bag_knn_439 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        base_estimator=neighbors.KNeighborsClassifier(n_neighbors=2),
        n_estimators=50,
        random_state=RANDOM_STATE,
        sampler=TomekLinks()
        )
)
bag_forest_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=RandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        n_estimators=20,
        random_state=RANDOM_STATE,
        sampler=RandomOverSampler(random_state=RANDOM_STATE, shrinkage=0.5)
    )
)

bagging_models_439 = [bag_439, bag_bayes_439, bag_svm_439, bag_knn_439, bag_forest_439]

#####################################################
eec_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE
    )
)
eec_bayes_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=10,
        base_estimator=naive_bayes.GaussianNB(var_smoothing=1e-06),
        random_state=RANDOM_STATE
    )
)
eec_svm_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=10,
        base_estimator=svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
eec_knn_439 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=neighbors.KNeighborsClassifier(
            n_neighbors=2, weights='distance'
        ),
        random_state=RANDOM_STATE
    )
)
eec_forest_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=10,
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

eec_models_439 = [eec_439, eec_bayes_439, eec_svm_439, eec_knn_439, eec_forest_439]

#####################################################

rus_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    RUSBoostClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    )
)
rus_bayes_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    RUSBoostClassifier(
        n_estimators=10,
        base_estimator=naive_bayes.GaussianNB(),
        random_state=RANDOM_STATE
    )
)
rus_svm_439 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    RUSBoostClassifier(
        n_estimators=10,
        algorithm='SAMME',
        base_estimator=svm.SVC(C=10, gamma='auto', random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
rus_forest_439 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    RUSBoostClassifier(
        n_estimators=50,
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

rus_models_439 = [rus_439, rus_bayes_439, rus_svm_439, rus_forest_439]

#%%
# models for AID1608 from hyperparamtertuning
bayes_1608 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    naive_bayes.GaussianNB(var_smoothing=0.01)
)
svm_1608 = make_pipeline(
    feature_selection.SelectKBest(k=30),
    svm.SVC(C=100, gamma='scale', random_state=RANDOM_STATE)
)
svm_balanced_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    svm.SVC(
        C=1, class_weight='balanced',
        gamma='scale', random_state=RANDOM_STATE
    )
)
knn_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    neighbors.KNeighborsClassifier(n_neighbors=20)
)
tree_1608 = make_pipeline(
    feature_selection.SelectKBest(k=40),
    tree.DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        criterion='log_loss'
    )
)
tree_balanced_1608 = make_pipeline(
    feature_selection.SelectKBest(k=40),
    tree.DecisionTreeClassifier(
        class_weight='balanced', random_state=RANDOM_STATE
    )
)

normal_models_1608 = [bayes_1608, svm_1608, svm_balanced_1608, knn_1608, tree_1608, tree_balanced_1608]

#####################################################
bag_1608 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=SMOTE(random_state=RANDOM_STATE)
        )
)
bag_bayes_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=naive_bayes.GaussianNB(var_smoothing=1e-06),
        n_estimators=20,
        random_state=RANDOM_STATE,
        sampler=RandomUnderSampler(random_state=RANDOM_STATE)
        )
)
bag_svm_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=svm.SVC(
            C=1, class_weight='balanced',
            random_state=RANDOM_STATE),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=TomekLinks()
        )
)
bag_knn_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    BalancedBaggingClassifier(
        base_estimator=neighbors.KNeighborsClassifier(n_neighbors=10),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=RandomUnderSampler(random_state=RANDOM_STATE)
        )
)
bag_forest_1608 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=SMOTE(random_state=RANDOM_STATE)
    )
)

bagging_models_1608 = [bag_1608, bag_bayes_1608, bag_svm_1608, bag_knn_1608, bag_forest_1608]

#####################################################
eec_1608 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    EasyEnsembleClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE
    )
)
eec_bayes_1608 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=naive_bayes.GaussianNB(var_smoothing=1e-06),
        random_state=RANDOM_STATE
    )
)
eec_svm_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=20,
        base_estimator=svm.SVC(C=10, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
eec_knn_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=neighbors.KNeighborsClassifier(),
        random_state=RANDOM_STATE
    )
)
eec_forest_1608 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

eec_models_1608 = [eec_1608, eec_bayes_1608, eec_svm_1608, eec_knn_1608, eec_forest_1608]

#####################################################

rus_1608 = make_pipeline(
    feature_selection.SelectKBest(k=10),
    RUSBoostClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    )
)
rus_bayes_1608 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    RUSBoostClassifier(
        n_estimators=10,
        base_estimator=naive_bayes.GaussianNB(var_smoothing=0.01),
        random_state=RANDOM_STATE
    )
)
rus_svm_1608 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    RUSBoostClassifier(
        n_estimators=10,
        algorithm='SAMME',
        base_estimator=svm.SVC(C=10, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
rus_forest_1608 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    RUSBoostClassifier(
        n_estimators=20,
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

rus_models_1608 = [rus_1608, rus_bayes_1608, rus_svm_1608, rus_forest_1608]

#%%
# models for AID1284 from hyperparamtertuning
bayes_1284 = make_pipeline(
    feature_selection.SelectKBest(k=100),
    naive_bayes.GaussianNB(var_smoothing=0.01)
)
svm_1284 = make_pipeline(
    feature_selection.SelectKBest(k=100),
    svm.SVC(C=0.1, gamma=0.01, random_state=RANDOM_STATE)
)
svm_balanced_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    svm.SVC(
        C=1, class_weight='balanced',
        gamma='scale', random_state=RANDOM_STATE
    )
)
knn_1284 = make_pipeline(
    feature_selection.SelectKBest(k=75),
    neighbors.KNeighborsClassifier(n_neighbors=20, weights='distance')
)
tree_1284 = make_pipeline(
    feature_selection.SelectKBest(k=30),
    tree.DecisionTreeClassifier(
        random_state=RANDOM_STATE
    )
)
tree_balanced_1284 = make_pipeline(
    feature_selection.SelectKBest(k=40),
    tree.DecisionTreeClassifier(
        class_weight='balanced', 
        criterion='log_loss',
        random_state=RANDOM_STATE
    )
)

normal_models_1284 = [bayes_1284, svm_1284, svm_balanced_1284, knn_1284, tree_1284, tree_balanced_1284]

#####################################################
bag_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=SMOTE(random_state=RANDOM_STATE)
        )
)
bag_bayes_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        base_estimator=naive_bayes.GaussianNB(var_smoothing=0.001),
        n_estimators=20,
        random_state=RANDOM_STATE,
        sampler=TomekLinks()
        )
)
bag_svm_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    BalancedBaggingClassifier(
        base_estimator=svm.SVC(
            C=1, class_weight='balanced',
            random_state=RANDOM_STATE),
        n_estimators=10,
        random_state=RANDOM_STATE,
        sampler=ADASYN(random_state=RANDOM_STATE)
        )
)
bag_knn_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        base_estimator=neighbors.KNeighborsClassifier(n_neighbors=10),
        n_estimators=50,
        random_state=RANDOM_STATE,
        sampler=RandomOverSampler(random_state=RANDOM_STATE, shrinkage=0.5)
    )
)
bag_forest_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    BalancedBaggingClassifier(
        base_estimator=RandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        n_estimators=20,
        random_state=RANDOM_STATE,
        sampler=RandomUnderSampler(random_state=RANDOM_STATE)
    )
)

bagging_models_1284 = [bag_1284, bag_bayes_1284, bag_svm_1284, bag_knn_1284, bag_forest_1284]

#####################################################
eec_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    EasyEnsembleClassifier(
        n_estimators=10,
        random_state=RANDOM_STATE
    )
)
eec_bayes_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    EasyEnsembleClassifier(
        n_estimators=10,
        base_estimator=naive_bayes.GaussianNB(var_smoothing=0.001),
        random_state=RANDOM_STATE
    )
)
eec_svm_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    EasyEnsembleClassifier(
        n_estimators=10,
        base_estimator=svm.SVC(C=10, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
eec_knn_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=neighbors.KNeighborsClassifier(),
        random_state=RANDOM_STATE
    )
)
eec_forest_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    EasyEnsembleClassifier(
        n_estimators=50,
        base_estimator=BalancedRandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

eec_models_1284 = [eec_1284, eec_bayes_1284, eec_svm_1284, eec_knn_1284, eec_forest_1284]

#####################################################

rus_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    RUSBoostClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE
    )
)
rus_bayes_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    RUSBoostClassifier(
        n_estimators=10,
        base_estimator=naive_bayes.GaussianNB(var_smoothing=0.001),
        random_state=RANDOM_STATE
    )
)
rus_svm_1284 = make_pipeline(
    feature_selection.SelectKBest(k=20),
    RUSBoostClassifier(
        n_estimators=10,
        algorithm='SAMME',
        base_estimator=svm.SVC(C=10, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE
    )
)
rus_forest_1284 = make_pipeline(
    feature_selection.SelectKBest(k=50),
    RUSBoostClassifier(
        n_estimators=50,
        base_estimator=RandomForestClassifier(
            max_features=None, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE
    )
)

rus_models_1284 = [rus_1284, rus_bayes_1284, rus_svm_1284, rus_forest_1284]
# %%

samples = [439, 1284, 644]

results = {}
for AID in samples:
    results[AID] = {}
    results[AID]['normal'] = perform_cv_training(AID, normal_tests)
    results[AID]['bag'] = perform_cv_training(AID, bagging_tests)
    results[AID]['eec'] = perform_cv_training(AID, eec_tests)
    results[AID]['rus'] = perform_cv_training(AID, rus_tests)
    

# %%

for AID in [456]:
    results[AID] = {}
    results[AID]['normal'] = perform_cv_training(AID, normal_tests)
    results[AID]['bag'] = perform_cv_training(AID, bagging_tests)
    results[AID]['eec'] = perform_cv_training(AID, eec_tests)
    results[AID]['rus'] = perform_cv_training(AID, rus_tests)
    
# %%
# evaluation results
normal_ids = ["Bayes", "SVM", "bal. SVM", "kNN", "Tree", "bal. Tree"]
bag_ids = ["Bag", "Bag+Bayes", "Bag+SVM", "Bag+kNN", "Bag+Forest"]
eec_ids = ["EEC", "EEC+Bayes", "EEC+SVM", "EEC+kNN", "EEC+Forest"]
rus_ids = ["RUS", "RUS+Bayes", "RUS+SVM", "RUS+Forest"]

output = ""

for id, model in zip(normal_ids, normal_models_1284):
    output += id + " "
    output += fit_and_evaluate_model(1284, model, latex=True)
    output += "\n"
for id, model in zip(bag_ids, bagging_models_1284):
    output += id + " "
    output += fit_and_evaluate_model(1284, model, latex=True)
    output += "\n"
for id, model in zip(eec_ids, eec_models_1284):
    output += id + " "
    output += fit_and_evaluate_model(1284, model, latex=True)
    output += "\n"
for id, model in zip(rus_ids, rus_models_1284):
    output += id + " "
    output += fit_and_evaluate_model(1284, model, latex=True)
    output += "\n"



#%%
# cv scores

cv_output = ""
normal_cv_params1284 = results[1284]['normal'][0].values()
normal_cv_scores1284 = results[1284]['normal'][1].values()

for id, param, score in zip(normal_ids, normal_cv_params1284, normal_cv_scores1284):
    cv_output += id + " "
    for vals in score.values():
        for mean, std, pars in vals:
            if pars == param:
                cv_output += f"& {round(mean, 3)} (+/-{round(2*std, 3)}) "
                break
    cv_output += "\n"

bag_cv_params1284 = results[1284]['bag'][0].values()
bag_cv_scores1284 = results[1284]['bag'][1].values()
for id, param, score in zip(bag_ids, bag_cv_params1284, bag_cv_scores1284):
    cv_output += id + " "
    for vals in score.values():
        for mean, std, pars in vals:
            if pars == param:
                cv_output += f"& {round(mean, 3)} (+/-{round(2*std, 3)}) "
                break
    cv_output += "\n"

eec_cv_params1284 = results[1284]['eec'][0].values()
eec_cv_scores1284 = results[1284]['eec'][1].values()
for id, param, score in zip(eec_ids, eec_cv_params1284, eec_cv_scores1284):
    cv_output += id + " "
    for vals in score.values():
        for mean, std, pars in vals:
            if pars == param:
                cv_output += f"& {round(mean, 3)} (+/-{round(2*std, 3)}) "
                break
    cv_output += "\n"

rus_cv_params1284 = results[1284]['rus'][0].values()
rus_cv_scores1284 = results[1284]['rus'][1].values()
for id, param, score in zip(rus_ids, rus_cv_params1284, rus_cv_scores1284):
    cv_output += id + " "
    for vals in score.values():
        for mean, std, pars in vals:
            if pars == param:
                cv_output += f"& {round(mean, 3)} (+/-{round(2*std, 3)}) "
                break
    cv_output += "\n"

# %%
# from sklearn.model_selection import KFold
data_X, data_y = _load_data(1284, 'train')
data_X1, data_y1 = _load_data(1284, 'test')
data_X = pd.concat([data_X, data_X1])
data_y = pd.concat([data_y, data_y1])

extra_cv_results = {'normal': {}, 'bag': {}, 'eec': {}, 'rus': {}}
extra_cv_output = ""

for id, test in zip(normal_ids, normal_tests.values()):

    model, grid = test
    inner_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    outer_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    grid_search = model_selection.GridSearchCV(
        estimator=model, param_grid=grid,
        scoring=SCORING, refit=_refit_metric, cv=inner_cv
    )
    scores = model_selection.cross_validate(
        grid_search, X=data_X, y=data_y,
        scoring=SCORING, cv=outer_cv
    )
    extra_cv_results['normal'][id] = scores
    extra_cv_output += id + " "
    for score_name in SCORING.keys():
        mean = scores[f"test_{score_name}"].mean()
        std = scores[f"test_{score_name}"].std()
        median = np.median(scores[f"test_{score_name}"])

        extra_cv_output += f"& {round(mean, 3)}/{round(median, 3)} (+/-{round(2*std, 3)}) "
    extra_cv_output += "\n"
print(extra_cv_output)

for id, test in zip(bag_ids, bagging_tests.values()):

    model, grid = test
    inner_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    outer_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    grid_search = model_selection.GridSearchCV(
        estimator=model, param_grid=grid,
        scoring=SCORING, refit=_refit_metric, cv=inner_cv
    )
    scores = model_selection.cross_validate(
        grid_search, X=data_X, y=data_y,
        scoring=SCORING, cv=outer_cv
    )
    extra_cv_results['bag'][id] = scores
    extra_cv_output += id + " "
    for score_name in SCORING.keys():
        mean = scores[f"test_{score_name}"].mean()
        std = scores[f"test_{score_name}"].std()
        median = np.median(scores[f"test_{score_name}"])

        extra_cv_output += f"& {round(mean, 3)}/{round(median, 3)} (+/-{round(2*std, 3)}) "
    extra_cv_output += "\n"

print(extra_cv_output)
for id, test in zip(eec_ids, eec_tests.values()):

    model, grid = test
    inner_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    outer_cv = model_selection.KFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )

    grid_search = model_selection.GridSearchCV(
        estimator=model, param_grid=grid,
        scoring=SCORING, refit=_refit_metric, cv=inner_cv
    )
    scores = model_selection.cross_validate(
        grid_search, X=data_X, y=data_y,
        scoring=SCORING, cv=outer_cv
    )
    extra_cv_results['eec'][id] = scores
    extra_cv_output += id + " "
    for score_name in SCORING.keys():
        mean = scores[f"test_{score_name}"].mean()
        std = scores[f"test_{score_name}"].std()
        median = np.median(scores[f"test_{score_name}"])

        extra_cv_output += f"& {round(mean, 3)}/{round(median, 3)} (+/-{round(2*std, 3)}) "
    extra_cv_output += "\n"

print(extra_cv_output)
# for id, test in zip(normal_ids, normal_tests.values()):

#     model, grid = test
#     inner_cv = model_selection.KFold(
#         n_splits=5, shuffle=True, random_state=RANDOM_STATE
#     )
#     outer_cv = model_selection.KFold(
#         n_splits=5, shuffle=True, random_state=RANDOM_STATE
#     )

#     grid_search = model_selection.GridSearchCV(
#         estimator=model, param_grid=grid,
#         scoring=SCORING, refit=_refit_metric, cv=inner_cv
#     )
#     scores = model_selection.cross_validate(
#         grid_search, X=data_X, y=data_y,
#         scoring=SCORING, cv=outer_cv
#     )
#     extra_cv_results['normal'][id] = scores
#     extra_cv_output += id + " "
#     for score_name in SCORING.keys():
#         mean = scores[f"test_{score_name}"].mean()
#         std = scores[f"test_{score_name}"].std()
#         median = np.median(scores[f"test_{score_name}"])

#         extra_cv_output += f"& {round(mean, 3)}/{round(median, 3)} (+/-{round(2*std, 3)}) "
#     extra_cv_output += "\n"

# %%
