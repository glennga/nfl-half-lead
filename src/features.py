""" This file holds a collection of various features (i.e. descriptions) to build each model with. 

Functions: offense_focus(df, team)
           defense_focus(df, team):
           special_teams_focus(df, team):
           general_performance_focus(df, team):
"""

import numpy as np


class __Features:
    """ Generic feature class for half-time lead prediction given first-quarter performance. The 
    specific features are defined by the function passed during instantiation. 
    """

    # Data-frame containing first-quarter data of a specific game, for each team.
    df_a, df_b = None, None

    # String corresponding to which team is A, and which is B.
    a_team, b_team = "", ""

    # Dictionary of features.
    features_a, features_b = {}, {}

    def __init__(self, df, parse_features, parse_order):
        """ Constructor. Given a data-frame for the **first quarter of a specific game**, parse 
        the relevant features using the given function.
        
        :param df: Pandas data-frame for a specific game's first quarter.
        :param parse_features: Function to parse features.
        :param parse_order: Order with which to parse features for. Exists in space ['forward', 
            'reverse', 'forward-permute'].
        """
        self.a_team, self.b_team = df['HomeTeam'].iloc[0], df['AwayTeam'].iloc[0]
        self.df_a, self.df_b = df[df['posteam'] == self.a_team], df[df['posteam'] == self.b_team]
        self.features_a.clear(), self.features_b.clear()

        if parse_order == 'forward':
            list(map(parse_features, [self.df_a, self.df_b], [self.features_a, self.features_b]))
        elif parse_order == 'reverse':
            list(map(parse_features, [self.df_b, self.df_a], [self.features_a, self.features_b]))
        elif parse_order == 'forward-permute':
            list(map(parse_features, [[self.df_a, self.df_b], [self.df_b, self.df_a]],
                     [self.features_a, self.features_b]))
        else:
            raise Exception('ParseOrder does not exist in space [forward, reverse, '
                            'forward-permute]')

    def as_list(self, team):
        """ Return all features as a numpy list. 
        
        :param team: Which team to get features from. For teams that are dual-indexed, the user can 
            enter a two-element list to check from. 
        :return: Features as a numpy list.
        """
        # We handle the cases where teams might be addressed by multiple keys.
        is_team_a = (not isinstance(team, list) and team == self.a_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     (team[0] == self.a_team or team[1] == self.a_team))
        is_team_b = (not isinstance(team, list) and team == self.b_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     team[0] == self.b_team or team[1] == self.b_team)

        if is_team_a:
            return np.nan_to_num(list(self.features_a.values()))
        elif is_team_b:
            return np.nan_to_num(list(self.features_b.values()))
        else:
            raise Exception('Team does not exist in this instance.')


def offense_focus(df, team):
    """ Return features that describe a team's first quarter offensive performance: 
    
     ydstogo -> Distance to go for first down.
     GoalToGo -> Is the play a goal down situation?
     FirstDown -> Is the play a first down conversion?
     Yards.Gained -> Amount of yards gained per play.
     Touchdown -> Is the play a touchdown?
     PlayType -> Type of play => [Kickoff, Punt, Pass, Sack, Run, Field Goal,
                                  Extra Point Quarter End, Two Minute Warning,
                                  Half End, End of Game No Play, QB Kneel, Spike, Timeout].
     PassOutcome -> Result of pass => [Complete, Incomplete Pass].
    
    :param df: Pandas data-frame for a specific game's first quarter.
    :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
    :return: Features that describe a team's first quarter offensive performance as a numpy list.
    """
    return __Features(df, lambda x, features_x: features_x.update({
        'a_ydstogo': x['ydstogo'].mean(),
        's_GoalToGo': x['GoalToGo'].sum(),
        's_FirstDown': x['FirstDown'].sum(),
        'a_Yards.Gained': x['Yards.Gained'].mean(),
        's_Touchdown': x['Touchdown'].sum(),
        's_PlayType_Pass': len(x[x['PlayType'] == 'Pass']),
        's_PlayType_Run': len(x[x['PlayType'] == 'Run']),
        's_PassOutcome_Complete': len(x[x['PassOutcome'] == 'Complete'])}), 'forward').as_list(team)


def defense_focus(df, team):
    """ Return features that describe a team's first quarter defensive performance. 

    QBHit -> Is the quarterback hit?
    Safety -> Is the play a safety (against offense)?
    PassOutcome -> Result of pass => [Complete, Incomplete Pass].
    InterceptionThrown -> Is pass intercepted?
    Fumble -> Did fumble occur (against offense)?
    Sack -> Is the play a sack (against offense)?

    :param df: Pandas data-frame for a specific game's first quarter.
    :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
    :return: Features that describe a team's first quarter defensive performance as a numpy list.
    """
    return __Features(df, lambda x, features_x: features_x.update({
        's_QBHit': x['QBHit'].sum(),
        's_Safety': x['Safety'].sum(),
        's_PassOutcome_Incomplete_Pass': len(x[x['PassOutcome'] == 'Incomplete Pass']),
        's_InterceptionThrown': x['InterceptionThrown'].sum(),
        's_Fumble': x['Fumble'].sum(),
        's_Sack': x['Sack'].sum()}), 'reverse').as_list(team)


def special_teams_focus(df, team):
    """ Return features that describe a team's first quarter special team's performance. 

    ExPointResult -> Result of extra-point => [NA, Made, Missed, Blocked, Aborted].
    TwoPointConv -> Result of two point conversion => [NA, Success, Failure].
    FieldGoalResult -> Result of field goal => [No Good, Good, Blocked].
    FieldGoalDistance -> Field goal length of attempt.

    :param df: Pandas data-frame for a specific game's first quarter.
    :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
    :return: Features that describe a team's first quarter special team's performance as a numpy 
        list.
    """
    return __Features(df, lambda y, features_y: features_y.update({
        's_ExPointResult_Made': len(y[0][y[0]['ExPointResult'] == 'Made']),
        's_ExPointResult_Missed': len(y[1][y[1]['ExPointResult'] == 'Missed']),
        's_ExPointResult_Blocked': len(y[1][y[1]['ExPointResult'] == 'Blocked']),
        's_TwoPointConv_Success': len(y[0][y[0]['TwoPointConv'] == 'Success']),
        's_TwoPointConv_Failure': len(y[1][y[1]['TwoPointConv'] == 'Failure']),
        's_FieldGoalResult_Good': len(y[0][y[0]['FieldGoalResult'] == 'Good']),
        's_FieldGoalResult_Blocked': len(y[1][y[1]['FieldGoalResult'] == 'Blocked']),
        'a_FieldGoalDistance': y[0]['FieldGoalDistance'].mean()}), 'forward-permute').as_list(team)


def general_performance_focus(df, team):
    """ Return features that describe a team's first quarter overall performance. This is less 
    specific than the other features, and is more of a summary of the other three.

    FirstDown -> Is the play a first down conversion?
    Touchdown -> Is the play a touchdown?
    InterceptionThrown -> Is pass intercepted?
    Reception -> Is reception recorded?
    Fumble -> Did fumble occur (against offense)?
    Sack -> Is the play a sack (against offense)?
    FieldGoalResult -> Result of field goal => [No Good, Good, Blocked].

    :param df: Pandas data-frame for a specific game's first quarter.
    :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
    :return: Features that describe a team's first quarter general performance as a numpy list.
    """
    return __Features(df, lambda y, features_y: features_y.update({
        's_FirstDown': y[0]['FirstDown'].sum(),
        's_Touchdown': y[0]['Touchdown'].sum(),
        's_InterceptionThrown_Opposite': y[1]['InterceptionThrown'].sum(),
        's_Fumble': y[1]['Fumble'].sum(),
        's_Sack': y[1]['Sack'].sum(),
        's_FieldGoalResult_Good': len(y[0][y[0]['FieldGoalResult'] == 'Good'])}),
                      'forward-permute').as_list(team)
