""" This file holds functions to train and verify support vector machines, given various features. 

Functions: train(df, feature_f, team, params)
           validate(df, feature_f, team, model)
           grid_search(df, feature_f, team, params)
           test(df, feature_f, team, model)
"""

import numpy as np
from sklearn.svm import SVC

from src import pull


def train(df, feature_f, team, params):
    """ Return a dictionary of models, trained using the given feature function.
    
    :param df: Pandas data-frame to work with. Should be the original.
    :param feature_f: Feature function to use (what we are testing).
    :param team: Name of the team whose model this is for.
    :param params: Hyperparameter dictionary to use for our SVM.
    :return: Dictionary of support vector machines.
    """
    # Collect the game IDs for our training data.
    training_map = pull.weeks_to_ids(df, [1, 3, 5, 7, 9, 11, 13, 15])

    # Some weird funkiness for LA and JAX.
    t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)

    # Construct our model, and train it.
    model, f, ell = SVC(C=params['C'], gamma=params['gamma']), [], []
    for game in training_map[team]:
        f.append(feature_f(pull.first_quarter_stats(df, game), t))
        ell.append(pull.is_team_leading_qtr(df, game, t, 2))
    model.fit(f, ell)

    return model


def validate(df, feature_f, team, model):
    """ Return a list of results using the given model and the feature function. 
    
    :param df: Pandas data-frame to work with. Should be the original.
    :param feature_f: Feature function used to train the given models.
    :param team: Name of the team whose model this is for.
    :param model: Model to test.
    :return: List of results, existing in the space [0, 1]. 
    """
    # Collect the game IDs for our verification data.
    validate_map = pull.weeks_to_ids(df, [2, 6, 10, 14])

    # Some weird funkiness for LA and JAX.
    t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)

    # Predict each game in our validation map.
    results = []
    for game in validate_map[team]:
        r = model.predict(np.array([feature_f(pull.first_quarter_stats(df, game), t)]))[0]
        results.append(1 if r == pull.is_team_leading_qtr(df, game, t, 2) else 0)

    return results


def grid_search(df, feature_f, team, params):
    """ For the given sets in 'params', iterate through all combinations to find the best 
    hyperparameters for our model.
    
    :param df: Panda's data-frame to work with. Should be the original.
    :param feature_f: Feature function used to train the given model. 
    :param team: Name of the team whose model this is for.
    :param params: Dictionary of hyperparameter sets to search through.
    :return: The most optimal parameters 
    """
    model_a, lead_r = 0, 0

    # Here, we are only iterating through our C and gamma parameters.
    for c in params['C']:
        for gamma in params['gamma']:
            model_b = train(df, feature_f, team, {'gamma': gamma, 'C': c})
            r = np.mean(validate(df, feature_f, team, model_b))

            # We have found a new leader. Replace the old one.
            model_a = model_a if r < lead_r else model_b
            lead_r = lead_r if r < lead_r else r

    return model_a


def test(df, feature_f, team, model):
    """ Return a list of results, using the given model and feature function.

    :param df: Pandas data-frame to work with. Should be the original.
    :param feature_f: Feature function used to train the given model.
    :param team: Name of the team whose model this is for.
    :param model: Model to test.
    :return: List of results, existing in the space [0, 1].
    """
    # Collect the game IDs for our testing data.
    test_map = pull.weeks_to_ids(df, [4, 8, 12, 16])

    # Some weird funkiness for LA and JAX.
    t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)

    # If this is a correct prediction, we append a 1. Otherwise, append a 0.
    results = []
    for game in test_map[team]:
        r = model.predict(np.array([feature_f(pull.first_quarter_stats(df, game), t)]))[0]
        results.append(1 if r == pull.is_team_leading_qtr(df, game, t, 2) else 0)

    return results
