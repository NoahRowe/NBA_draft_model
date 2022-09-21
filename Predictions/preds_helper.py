import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.wrappers.scikit_learn import KerasRegressor


trainingCols = ['FT%', '3P%', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'OWS', 'DWS', 'FTA', '3PA', 'PTS', 'PF', 'MP_per_PF', 'FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA', 'C', 'F', 'G', 'PPM', 'PPG', 'HEIGHT', 'WEIGHT', 'gamesPlayed', 'minutes', 'SOS', 'PER', 'FGA', 'MP', 'AST_per_TOV', 'ORtg', 'DRtg', 'awards', 'RSCI', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'SHUTTLE_RUN', 'THREE_QUARTER_SPRINT', 'STANDING_VERTICAL', 'MAX_VERTICAL', 'BENCH_PRESS', 'BODY_FAT', 'HAND_LENGTH', 'HAND_WIDTH', 'didCombine', 'HEIGHT_W_SHOES', 'REACH', 'WINGSPAN', 'dist_avg', 'dist_std', 'dist_dot_min', 'dist_dot_WS', 'min_dist', 'label_count']

target = 'WM'


# Function to load in the data to train the model
def load_training_data():
    data = pd.read_pickle("../Model/team_features_new.df")
    return data[trainingCols], data[target]

def load_draftee_data():
    # GET THE DRAFT DATA
    data = pd.read_pickle("../Data/PredictionData/prediction_data_1.df")

    data.rename(columns={"Name":"Player", "Guard":"G", "Center":"C", "Forward":"F", 
                         "awardCount":"awards", "mock1":"m1", "mock2":"m2", "mock3":"m3", "mock4":"m4",
                         "mock5":"m5", "mock6":"m6", "THREE_Q_SPRINT":"THREE_QUARTER_SPRINT", 
                         "MAX_VERTICAL_LEAP":"MAX_VERTICAL", "STANDING_LEAP":"STANDING_VERTICAL"}, inplace=True)
    data = data.drop_duplicates(subset=["Player"])
    data.reset_index(drop=True, inplace=True)
    draftData = data.copy()
    return draftData

def getNBAdata():
    # Get NBA data
    playerData = pd.read_csv('../Model/NBAPlayerData/Players.csv')
    playerStats = pd.read_csv('../Model/NBAPlayerData/Seasons_Stats.csv')

    playerStats.rename(columns={'G':"gamesPlayed"}, inplace=True)

    # Get the players that play since 2007
    playerStats = playerStats[playerStats['Year']>=2006]
    # Only look at players who have played more than 10 games
    playerStats = playerStats[playerStats["gamesPlayed"]>10]
    # Only look at players who have played more than minutes
    playerStats = playerStats[playerStats["MP"]>300]
    playerData = playerData[playerData['Player'].isin(playerStats['Player'])]
    # Turn years into int
    playerStats = playerStats.astype({"Year": int})

    # Merge the datasets
    playerStats = playerStats.merge(playerData, on="Player")

    # Make some of the columns we need
    playerStats['PPG'] = playerStats['PTS']/playerStats['gamesPlayed']
    playerStats['PPM'] = playerStats['PTS']/playerStats['MP']
    playerStats["AST_per_TOV"] = np.nan
    playerStats['MP_per_PF'] = np.nan
    playerStats['FTA_per_FGA'] = np.nan
    playerStats['MP_per_3PA'] = np.nan
    playerStats['PTS_per_FGA'] = np.nan
    for i in range(len(playerStats)):
        ast, tov = playerStats['AST'].iloc[i], playerStats['TOV'].iloc[i]
        playerStats.iloc[i, playerStats.columns.get_loc("AST_per_TOV")] = ast/tov if tov!=0 else np.nan
        
        pf, mp = playerStats["PF"].iloc[i], playerStats["MP"].iloc[i]
        playerStats.iloc[i, playerStats.columns.get_loc("MP_per_PF")] =  mp/pf if pf!=0 else np.nan
        
        fta, fga = playerStats["FTA"].iloc[i], playerStats["FGA"].iloc[i]
        playerStats.iloc[i, playerStats.columns.get_loc('FTA_per_FGA')] =  fta/fga if fga!=0 else np.nan
        
        pa3 = playerStats["3PA"].iloc[i]
        playerStats.iloc[i, playerStats.columns.get_loc('MP_per_3PA')] =  mp/pa3 if pa3!=0 else np.nan

        pts = playerStats["PTS"].iloc[i]
        playerStats.iloc[i, playerStats.columns.get_loc('PTS_per_FGA')] = pts/fga if fga!=0 else np.nan

    # Rename some columns
    playerStats.rename(columns={"weight":"WEIGHT", "height":"HEIGHT"}, inplace=True)

    # Fix some team names
    playerStats.replace("NJN", "BRK", inplace=True)
    playerStats.replace("NOH", "NOP", inplace=True)
    playerStats.replace("CHA", "CHO", inplace=True)
    playerStats.replace("CHH", "CHO", inplace=True)
    playerStats.replace("VAN", "MEM", inplace=True)
    playerStats.replace("SEA", "OKC", inplace=True)
    playerStats.replace("NOK", "NOP", inplace=True)

    # Drop players that got traded that season
    playerStats = playerStats[playerStats.Tm!='TOT']

    # Create the position columns
    playerStats["G"] = [1 if "G" in playerStats['Pos'].iloc[i] else 0 for i in range(len(playerStats))]
    playerStats["F"] = [1 if "F" in playerStats['Pos'].iloc[i] else 0 for i in range(len(playerStats))]
    playerStats["C"] = [1 if "C" in playerStats['Pos'].iloc[i] else 0 for i in range(len(playerStats))]

    # Fill in missing values
    playerStats["3P%"] = playerStats['3P%'].fillna(0)
    playerStats["FT%"] = playerStats['FT%'].fillna(0)
    playerStats["MP_per_3PA"] = playerStats['MP_per_3PA'].fillna(np.mean(playerStats['MP_per_3PA']))
    impute_mean_cols = ['AST_per_TOV', 'MP_per_PF', 'FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA']
    # for col in impute_mean_cols:
    #     playerStats[col] = playerStats.fillna(np.mean(playerStats.loc[playerStats[col].isnull()==False]))[col]
    
    return playerStats.copy()

## NEURAL NETWORK MODEL
def create_NN():
    N, epochs, batch_size = 62, 20, 20
    def create_model():
        model = Sequential()
        model.add(Dense(N, input_dim=N, kernel_initializer='normal', activation='sigmoid'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer="Adam")
        return model
    return KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)