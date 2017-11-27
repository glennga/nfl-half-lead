from sklearn.svm import SVC

import pull


def train(df, feature_f):
    """ Return a dictionary of models, trained using the given feature function.
    
    :param df: Pandas dataframe to work with. Should be the original.
    :param feature_f: Feature function to use (what we are testing).
    :return: Dictionary of support vector machines.
    """
    # Collect the game IDs for our training data.
    training_map = pull.weeks_to_ids(df, [1, 3, 5, 7, 9, 11, 13, 15])

    # We will have a different model for every team. Keep the same keys as our training map.
    models = {}
    list(map(lambda a: models.update({a: SVC()}), list(training_map.keys())))

    # Every game is a data point to train our model on.
    for team in list(training_map.keys()):
        t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)
        f, ell = [], []

        # We label each point with the halftime lead of that game.
        for game in training_map[team]:
            f.append(feature_f(pull.first_quarter_stats(game, df)).as_list(t))
            ell.append(pull.is_team_leading_half(game, t, df))
        models[team].fit(f, ell)

    return models


def verify(df, feature_f, models):
    """ Return a dictionary of results, using the given models and the feature function. 
    
    :param df: Pandas dataframe to work with. Should be the original.
    :param feature_f: Feature function used to train the given models.
    :param models: Models to verify.
    :return: Dictionary of lists of results, existing in the space [0, 1]. 
    """

    # Collect the game IDs for our verification data.
    verification_map = pull.weeks_to_ids(df, [2, 6, 10, 14])

    # Store the results in dictionary of lists.
    results = {}
    list(map(lambda b: results.update({b: []}), list(verification_map.keys())))

    # Predict each game in our verification map.
    for team in list(verification_map.keys()):
        t = ['LA', 'STL'] if team == 'STL-LA' else (['JAC', 'JAX'] if team == 'JAC-JAX' else team)

        # If this is a correct prediction, we append a 1. Otherwise, append a 0.
        for game in verification_map[team]:
            r = models[team].predict(feature_f(pull.first_quarter_stats(game, df)).as_list(t))[0]
            results[team].append(1 if r == pull.is_team_leading_half(game, t, df) else 0)

    return results
