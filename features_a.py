""" This file holds features A for halftime lead prediction. This focuses solely on the team's 
offensive performance.

Areas of Interest: ydstogo -> Distance to go for first down.
                   YardsAfterCatch -> Yards gained after catch.
                   yrdline100 -> Distance to opponents's end zone.
                   GoalToGo -> Is the play a goal down situation?
                   FirstDown -> Is the play a first down conversion?
                   Yards.Gained -> Amount of yards gained per play.
                   Touchdown -> Is the play a touchdown?
                   PlayType -> Type of play => [Kickoff, Punt, Pass, Sack, Run, Field Goal,
                                                Extra Point Quarter End, Two Minute Warning,
                                                Half End, End of Game No Play, QB Kneel, Spike,
                                                Timeout].
                   PassOutcome -> Result of pass => [Complete, Incomplete Pass].
                   InterceptionThrown -> Is pass intercepted?
                   Reception -> Is reception recorded?
"""

import numpy as np


class FeaturesA:
    """ Features A for half-time lead prediction given first-quarter performance. These features 
    emphasize a team's performance in offense.
    """

    # Data-frame containing first-quarter data of a specific game, for each team.
    df_a, df_b = None, None

    # String corresponding to which team is A, and which is B.
    a_team, b_team = "", ""

    # Dictionary of offensive features.
    features_oa, features_ob = {}, {}

    def __parse_offense_features(self):
        """ Parse features: avg(ydstogo) for each team.
                            avg(YardsAfterCatch) for each team.
                            avg(yrdline100) for each team.
                            sum(GoalToGo) for each team.
                            sum(FirstDown) for each team.
                            avg(Yards.Gained) for each team.
                            sum(Touchdown) for each team.
                            sum(PlayType == Pass) for each team.
                            sum(PlayType == Run) for each team.
                            sum(PassOutcome == Complete) for each team.
                            sum(Reception) for each team.
        :return: None.
        """
        list(map(lambda x, features_x: features_x.update({
            'a_ydstogo': x['ydstogo'].mean(),
            'a_YardsAfterCatch': x['YardsAfterCatch'].mean(),
            'a_yrdline100': x['yrdline100'].mean(),
            's_GoalToGo': x['GoalToGo'].sum(),
            's_FirstDown': x['FirstDown'].sum(),
            'a_Yards.Gained': x['Yards.Gained'].mean(),
            's_Touchdown': x['Touchdown'].sum(),
            's_PlayType_Pass': len(x[x['PlayType'] == 'Pass']),
            's_PlayType_Run': len(x[x['PlayType'] == 'Run']),
            's_PassOutcome_Complete': len(x[x['PassOutcome'] == 'Complete']),
            's_Recption': x['Reception'].sum()}), [self.df_a, self.df_b],
                 [self.features_oa, self.features_ob]))

    def __init__(self, df):
        """ Constructor. Parses the necessary features given the data-frame for a the **first 
        quarter of a specific game**. 
        
        :param df: Pandas data-frame for a specific game. 
        """
        self.a_team, self.b_team = df['HomeTeam'].iloc[0], df['AwayTeam'].iloc[0]
        self.df_a, self.df_b = df[df['posteam'] == self.a_team], df[df['posteam'] == self.b_team]

        self.__parse_offense_features()

    def as_list(self, team):
        """ Return all features as a numpy list. 
        
        :param team: Which team to get features from. For teams that are dual-indexed, the user can 
        enter a two-element list to check from. 
        :return Offensive features as a numpy list.
        """
        # We handle the cases where teams might be addressed by multiple keys.
        is_team_a = (not isinstance(team, list) and team == self.a_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     (team[0] == self.a_team or team[1] == self.a_team))
        is_team_b = (not isinstance(team, list) and team == self.b_team) or \
                    ((isinstance(team, list) and len(team) == 2) and
                     team[0] == self.b_team or team[1] == self.b_team)

        if is_team_a:
            return np.nan_to_num(list(self.features_oa.values()))
        elif is_team_b:
            return np.nan_to_num(list(self.features_ob.values()))
        else:
            raise Exception('Team does not exist in this instance.')
