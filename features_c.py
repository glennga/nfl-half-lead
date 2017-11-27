""" This file holds features C for halftime lead prediction. This focuses solely on the team's 
special team's performance.

Areas of Interest: ExPointResult -> Result of extra-point => [NA, Made, Missed, Blocked, Aborted].
                   TwoPointConv -> Result of two point conversion => [NA, Success, Failure].
                   FieldGoalResult -> Result of field goal => [No Good, Good, Blocked].
                   FieldGoalDistance -> Field goal length of attempt.
"""

import numpy as np


class FeaturesC:
    """ Features C for half-time lead prediction given first-quarter performance. These features 
    emphasize a team's performance in special teams. 
    """

    # Data-frame containing first-quarter data of a specific game, for each team.
    df_a, df_b = None, None

    # String corresponding to which team is A, and which is B.
    a_team, b_team = "", ""

    # Dictionary of special team's features.
    features_sta, features_stb = {}, {}

    def __parse_special_teams_features(self):
        """ Parse features: sum(ExPointResult == Made) for each team.
                            sum(ExPointResult == Missed) for each team (opposite).
                            sum(ExPointResult == Blocked) for each team (opposite).
                            sum(TwoPointConv == Success) for each team.
                            sum(TwoPointConv == Failure) for each team (opposite).
                            sum(FieldGoalResult == No Good) for each team (opposite).
                            sum(FieldGoalResult == Good) for each team.
                            sum(FieldGoalResult == Blocked) for each team (opposite). 
                            avg(FieldGoalDistance) for each team.
        :return: None.
        """
        list(map(lambda y, features_y: features_y.update({
            's_ExPointResult_Made': len(y[0][y[0]['ExPointResult'] == 'Made']),
            's_ExPointResult_Missed': len(y[1][y[1]['ExPointResult'] == 'Missed']),
            's_ExPointResult_Blocked': len(y[1][y[1]['ExPointResult'] == 'Blocked']),
            's_TwoPointConv_Success': len(y[0][y[0]['TwoPointConv'] == 'Success']),
            's_TwoPointConv_Failure': len(y[1][y[1]['TwoPointConv'] == 'Failure']),
            's_FieldGoalResult_No_Good': len(y[1][y[1]['FieldGoalResult'] == 'No Good']),
            's_FieldGoalResult_Good': len(y[0][y[0]['FieldGoalResult'] == 'Good']),
            's_FieldGoalResult_Blocked': len(y[1][y[1]['FieldGoalResult'] == 'Blocked']),
            'a_FieldGoalDistance': y[0]['FieldGoalDistance'].mean()}),
                 [[self.df_a, self.df_b], [self.df_b, self.df_a]],
                 [self.features_sta, self.features_stb]))

    def __init__(self, df):
        """ Constructor. Parses the necessary features given the data-frame for a the **first 
        quarter of a specific game**. 
        
        :param df: Pandas data-frame for a specific game. 
        """
        self.a_team, self.b_team = df['HomeTeam'].iloc[0], df['AwayTeam'].iloc[0]
        self.df_a, self.df_b = df[df['posteam'] == self.a_team], df[df['posteam'] == self.b_team]

        self.__parse_special_teams_features()

    def as_list(self, team):
        """ Return all features as a numpy list. 

        :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
        :return Special teams features as a numpy list.
        """
        # We handle the cases where teams might be addressed by multiple keys.
        is_team_a = (not isinstance(team, list) and team == self.a_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     (team[0] == self.a_team or team[1] == self.a_team))
        is_team_b = (not isinstance(team, list) and team == self.b_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     team[0] == self.b_team or team[1] == self.b_team)

        if is_team_a:
            return np.nan_to_num(list(self.features_sta.values()))
        elif is_team_b:
            return np.nan_to_num(list(self.features_stb.values()))
        else:
            raise Exception('Team does not exist in this instance.')
