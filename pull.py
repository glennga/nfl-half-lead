""" This file holds functions to pull useful data from the PBP data-set. """

import numpy as np


def is_team_leading_half(game_id, team, df):
    """ Given the data frame and the game ID, return 1 if the given team is leading at the 
    half, otherwise 0. 

    :param game_id: ID of the game to return lead for.
    :param team: Team to return lead for. For teams that are dual-indexed, the user can enter a 
    two-element list to check from. 
    :param df: Pandas data frame to work with.
    :return: 1 if the team is leading at the half. 0 otherwise. 
    """
    half_f, u = df[(df['GameID'] == game_id) & (df['qtr'] == 2)], 1
    score = half_f.tail(u)[['posteam', 'DefensiveTeam', 'PosTeamScore', 'DefTeamScore']]

    # We find the first non-zero tail entry.
    while all(score.isnull().iloc[0]):
        u += 1
        score = half_f.tail(u)[['posteam', 'DefensiveTeam', 'PosTeamScore', 'DefTeamScore']]

    # Handle both cases, where teams may be addressed with multiple keys.
    is_pos_team = (not isinstance(team, list) and score['posteam'].iloc[0] == team) or \
                  ((isinstance(team, list) and len(team) == 2) and
                   (score['posteam'].iloc[0] == team[0] or score['posteam'].iloc[0] == team[1]))
    is_def_team = (not isinstance(team, list) and score['DefensiveTeam'].iloc[0] == team) or \
                  ((isinstance(team, list) and len(team) == 2) and
                   (score['DefensiveTeam'].iloc[0] == team[0] or
                    score['DefensiveTeam'].iloc[0] == team[1]))

    if is_pos_team:
        return 1 if score['PosTeamScore'].iloc[0] > score['DefTeamScore'].iloc[0] else 0
    elif is_def_team:
        return 1 if score['DefTeamScore'].iloc[0] > score['PosTeamScore'].iloc[0] else 0
    else:
        raise Exception('Team does not exist in game.')


def halftime_score(game_id, df):
    """ Given the data frame and the game ID, return the halftime score as another data frame.

    :param game_id: ID of the game to return the halftime score for.
    :param df: Pandas data frame to work with. 
    :return: The half time score as another data frame, with attributes for the team and their 
    score in terms of possession.  
    """
    half_f = df[(df['GameID'] == game_id) & (df['qtr'] == 2)]
    return half_f.tail(1)[['posteam', 'DefensiveTeam', 'PosTeamScore', 'DefTeamScore']]


def first_quarter_stats(game_id, df):
    """ Given the data frame and the game ID, return the first quarter data as another data 
    frame.

    :param game_id: ID of the game to return the first quarter score.
    :param df: Pandas data frame to work with.
    :return: All data related to the first quarter of this game. 
    """
    return df[(df['GameID'] == game_id) & (df['qtr'] == 1)]


def weeks_to_ids(df, week_space):
    """ For the entire PBP data-set, we return the game IDs for each team whose "week" lies in 
    the given week space for all seasons.

    :param df: Pandas data frame to work with. Should hold the original CSV.
    :param week_space: List holding week space for each season (1-indexed).
    :return: Dictionary of each team and their respective game IDs. 
    """
    # We handle LA, STL, JAC, and JAX separately. Relocation and fights about abbreviations.
    team_map = {'ARI': [], 'ATL': [], 'BAL': [], 'BUF': [], 'CAR': [], 'CHI': [], 'CIN': [],
                'CLE': [], 'DAL': [], 'DEN': [], 'DET': [], 'GB': [], 'HOU': [], 'IND': [],
                'KC': [], 'MIA': [], 'MIN': [], 'NE': [], 'NO': [], 'NYG': [], 'NYJ': [],
                'OAK': [], 'PHI': [], 'PIT': [], 'SD': [], 'SEA': [], 'SF': [], 'TB': [],
                'TEN': [], 'WAS': []}

    sid_f = df[['GameID', 'Season', 'HomeTeam', 'AwayTeam']].drop_duplicates()

    # Find the games involving a specific team.
    for team in list(team_map.keys()):
        t_f = sid_f[(sid_f['HomeTeam'] == team) | (sid_f['AwayTeam'] == team)]

        # With these team games, find the games of a specific year.
        for season in np.arange(2009, 2017):
            st_f = t_f[t_f['Season'] == season]

            # Given our week space, insert all game IDs into the appropriate dictionary entry.
            list(map(lambda x: team_map[team].append(st_f['GameID'].iloc[x - 1]), week_space))

    # We store the dual-indexed teams with the following keys:
    team_map.update({'STL-LA': [], 'JAC-JAX': []})

    # Repeat for our dual-indexed teams.
    for team in [['LA', 'STL', 'STL-LA'], ['JAC', 'JAX', 'JAC-JAX']]:
        t_f = sid_f[(sid_f['HomeTeam'] == team[0]) | (sid_f['HomeTeam'] == team[1]) |
                    (sid_f['AwayTeam'] == team[0]) | (sid_f['AwayTeam'] == team[1])]

        for season in np.arange(2009, 2017):
            st_f = t_f[t_f['Season'] == season]
            list(map(lambda x: team_map[team[2]].append(st_f['GameID'].iloc[x - 1]), week_space))

    # Flatten our dictionary, and return all games that exist here.
    # return df[df['GameID'].isin(list(np.array(team_map.values()).flatten()))]
    return team_map
