""" This file holds functions to train and verify support vector machines, given various features. 

Functions: train(df, feature_f, kernel='rbf', c=1.0)
           verify(df, feature_f, models)
"""

import numpy as np
from sklearn.svm import SVC

from src import pull


def train(df, feature_f, kernel='rbf', c=1.0):
    """ Return a dictionary of models, trained using the given feature function.
    
    :param df: Pandas data-frame to work with. Should be the original.
    :param feature_f: Feature function to use (what we are testing).
    :param kernel: Kernel for the SVM to use.
    :param c: Complexity term for the SVM to use. Lower => less complexity. 
    :return: Dictionary of support vector machines.
    """
    # Collect the game IDs for our training data.
    training_map = pull.weeks_to_ids(df, [1, 3, 5, 7, 9, 11, 13, 15])

    # We will have a different model for every team. Keep the same keys as our training map.
    models = {}
    list(map(lambda a: models.update({a: SVC(kernel=kernel, C=c)}), list(training_map.keys())))

    # Every game is a data point to train our model on.
    for team in list(training_map.keys()):
        t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)
        f, ell = [], []

        # We label each point with the halftime lead of that game.
        for game in training_map[team]:
            f.append(feature_f(pull.first_quarter_stats(game, df), t))
            ell.append(pull.is_team_leading_half(game, t, df))
        models[team].fit(f, ell)

    return models


def verify(df, feature_f, models):
    """ Return a dictionary of results, using the given models and the feature function. 
    
    :param df: Pandas data-frame to work with. Should be the original.
    :param feature_f: Feature function used to train the given models.
    :param models: Models to verify.
    :return: Dictionary of lists of results, existing in the space [0, 1]. 
    """

    # Collect the game IDs for our verification data.
    verification_map = pull.weeks_to_ids(df, [2, 6, 10, 14])

    # Store the results in a dictionary of lists.
    results = {}
    list(map(lambda b: results.update({b: []}), list(verification_map.keys())))

    # Predict each game in our verification map.
    for team in list(verification_map.keys()):
        t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)

        # If this is a correct prediction, we append a 1. Otherwise, append a 0.
        for game in verification_map[team]:
            r = models[team].predict(np.array(
                [feature_f(pull.first_quarter_stats(game, df), t)]))[0]
            results[team].append(1 if r == pull.is_team_leading_half(game, t, df) else 0)

    return results
