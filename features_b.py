""" This file holds features B for halftime lead prediction. This focuses solely on the team's 
defensive performance.

Areas of Interest: QBHit -> Is the quarterback hit?
                   Safety -> Is the play a safety (against offense)?
                   PassOutcome -> Result of pass => [Complete, Incomplete Pass].
                   InterceptionThrown -> Is pass intercepted?
                   Fumble -> Did fumble occur (against offense)?
                   Sack -> Is the play a sack (against offense)?
"""

import numpy as np


class FeaturesB:
    """ Features B for half-time lead prediction given first-quarter performance. These features 
    emphasize a team's performance in defense. 
    """

    # Data-frame containing first-quarter data of a specific game, for each team.
    df_a, df_b = None, None

    # String corresponding to which team is A, and which is B.
    a_team, b_team = "", ""

    # Dictionary of defensive features.
    features_da, features_db = {}, {}

    def __parse_defense_features(self):
        """ Parse features: sum(QBHit) for each team. 
                            sum(Safety) for each team.
                            sum(PassOutcome == Incomplete Pass) for each team.
                            sum(InterceptionThrown) for each team.
                            sum(Fumble) for each team.
                            sum(Sack) for each team. 
        :return: None.
        """
        # Note that we feed the opposite team into each map.
        list(map(lambda x, features_x: features_x.update({
            's_QBHit': x['QBHit'].sum(),
            's_Safetfy': x['Safety'].sum(),
            's_PassOutcome_Incomplete_Pass': len(x[x['PassOutcome'] == 'Incomplete Pass']),
            's_InterceptionThrown': x['InterceptionThrown'].sum(),
            's_Fumble': x['Fumble'].sum(),
            's_Sack': x['Sack'].sum()}), [self.df_b, self.df_a],
                 [self.features_da, self.features_db]))

    def __init__(self, df):
        """ Constructor. Parses the necessary features given the data-frame for a the **first 
        quarter of a specific game**. 
        
        :param df: Pandas data-frame for a specific game. 
        """
        self.a_team, self.b_team = df['HomeTeam'].iloc[0], df['AwayTeam'].iloc[0]
        self.df_a, self.df_b = df[df['posteam'] == self.a_team], df[df['posteam'] == self.b_team]

        self.__parse_defense_features()

    def as_list(self, team):
        """ Return all features as a numpy list. 

        :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
        :return Defensive features as a numpy list.
        """
        # We handle the cases where teams might be addressed by multiple keys.
        is_team_a = (not isinstance(team, list) and team == self.a_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     (team[0] == self.a_team or team[1] == self.a_team))
        is_team_b = (not isinstance(team, list) and team == self.b_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     team[0] == self.b_team or team[1] == self.b_team)

        if is_team_a:
            return np.nan_to_num(list(self.features_da.values()))
        elif is_team_b:
            return np.nan_to_num(list(self.features_db.values()))
        else:
            raise Exception('Team does not exist in this instance.')