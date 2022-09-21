import numpy as np
import pandas as pd

def getData():
    nbaData = getNBAdata()
    draftData = getDraftData()
    
    # Fix the missing year issue
    dropCols = []
    for i in range(len(draftData)):
        draftee = draftData.iloc[i]
        draftYear, draftTeam = draftee["Year"], draftee["Tm"]
        nbaTeamSize = len(nbaData[(nbaData["Tm"]==draftTeam) & (nbaData["Year"]==draftYear-1)])
        if nbaTeamSize==0:
            dropCols.append(i)
            
    draftData = draftData.drop(dropCols, axis=0)
    draftData.reset_index(drop=True, inplace=True)
    nbaData.reset_index(drop=True, inplace=True)
    return nbaData, draftData

def getNBAdata():
    # Get NBA data
    playerData = pd.read_csv('NBAPlayerData/Players.csv')
    playerStats = pd.read_csv('NBAPlayerData/Seasons_Stats.csv')

    playerStats.rename(columns={'G':"gamesPlayed"}, inplace=True)

    # Get the players that play since 2000
    playerStats = playerStats[playerStats['Year']>=1999]
    # Only look at players who have played more than 20 games
    playerStats = playerStats[playerStats["gamesPlayed"]>30]
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


def getDraftData():
    # GET THE DRAFT DATA
    data = pd.read_pickle("../Data/final_data1.df")
    data.replace("New Jersey Nets", "Brooklyn Nets", inplace=True)
    data.replace("New Orleans Hornets", "New Orleans Pelicans", inplace=True)
    data.replace("Charlotte Bobcats", "Charlotte Hornets", inplace=True)
    data.replace("LA Clippers", "Los Angeles Clippers", inplace=True)
    data.replace("NJN", "BRK", inplace=True)
    data.replace("NOH", "NOP", inplace=True)
    data.replace("CHA", "CHO", inplace=True)
    data.replace("CHH", "CHO", inplace=True)
    data.replace("VAN", "MEM", inplace=True)
    data.replace("SEA", "OKC", inplace=True)
    data.replace("NOK", "NOP", inplace=True)

    data.rename(columns={"Name":"Player", "Guard":"G", "Center":"C", "Forward":"F", 
                         "awardCount":"awards", "mock1":"m1", "mock2":"m2", "mock3":"m3", "mock4":"m4",
                         "mock5":"m5", "mock6":"m6"}, inplace=True)
    data = data.drop_duplicates(subset=["Player"])
    data["WM"] = [(data["EWA"].iloc[i]+data["WP"].iloc[i] + data["WS"].iloc[i])/3. for i in range(len(data))]
    data.dropna(subset=["WM"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    draftData = data.copy()
    return draftData