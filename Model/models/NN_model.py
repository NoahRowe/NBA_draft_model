#!/usr/bin/env python3
'''
This is the base function for the neural network version of the code. 
PARAMETERS: {scaler type, n_clusters, distance_metric, n_players, 
OUTPUT: {MSE, added_wins, added_wins_ratio, nba_features}

This is a good file to look into the distance metrics, clusters, ...
but not really for feature selection or fundamental model changes

'''

#################################################
# Imports
#################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helper as my

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # DO NOT WANT TO USE THE GPU

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import pairwise
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error


##################################################################################################
# START THE FUNCTION
##################################################################################################
def simulation(scaler_type, n_clusters, distance_metric):
    OS_size, OS_val = 0, 0

    #################################################
    # Get the data
    #################################################
    NBAdata, draftData = my.getData()

    ### define the columns we want
    clusteringCols = ['FT%', '3P%', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'OWS', 'DWS', 
                  'FTA', '3PA', 'PTS', 'PF', 'MP_per_PF', 'FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA', 
                  'C', 'F', 'G', 'PPM', 'PPG', 'HEIGHT', 'WEIGHT']
    
    scalingCols = ['FT%', '3P%', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 'STL%', 'BLK%', 'USG%', 'OWS', 'DWS', 
                  'FTA', '3PA', 'PTS', 'PF', 'MP_per_PF', 'FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA', 
                  'PPM', 'PPG', 'HEIGHT', 'WEIGHT']

    x_cols = ['gamesPlayed', 'minutes', 'FT%', '3P%', 'SOS', 'PER', 'eFG%', 'ORB%', 'DRB%', 'AST%', 'TOV%', 
          'STL%', 'BLK%', 'USG%','OWS', 'DWS', 'FTA', 'FGA', 'MP', '3PA', 'PTS', 'PF', 'MP_per_PF', 'PPG', 
          'PPM','FTA_per_FGA', 'MP_per_3PA', 'PTS_per_FGA', "AST_per_TOV", 'ORtg', 'DRtg','awards','RSCI', 
          'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'SHUTTLE_RUN','THREE_QUARTER_SPRINT', 'STANDING_VERTICAL', 
          'MAX_VERTICAL','BENCH_PRESS', 'BODY_FAT', 'HAND_LENGTH', 'HAND_WIDTH', "didCombine", 
          'HEIGHT_W_SHOES', 'REACH', 'WEIGHT', 'WINGSPAN', 'C', 'F', 'G']

    target = "WM"
    allCols = list(dict.fromkeys(clusteringCols + x_cols)) # removes duplicates
    draftOnlyCols = [col for col in allCols if col not in clusteringCols + ['C', 'F', 'G']]

    #################################################
    # Scaling and Clustering
    #################################################
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()

    scaledNBA, scaledDraft = NBAdata.copy(), draftData.copy()
    scaledNBA[scalingCols] = scaler.fit_transform(scaledNBA[scalingCols])
    scaledDraft[scalingCols] = scaler.transform(scaledDraft[scalingCols])
    scaledDraft[draftOnlyCols] = scaler.fit_transform(scaledDraft[draftOnlyCols])

    ### Do the clustering
    fittedCluster = KMeans(n_clusters=n_clusters).fit(scaledNBA[clusteringCols].values)
    scaledNBA['label'] = fittedCluster.labels_
    scaledDraft['label'] = fittedCluster.predict(scaledDraft[clusteringCols].values, sample_weight=None)

    #################################################
    # Define cluster-based features
    #################################################
    metricCols = clusteringCols
    metric_function = pairwise.distance_metrics()[distance_metric]
    nba_features = ['dist_avg', "dist_dot_min", "dist_dot_WS", "min_dist", "min_dist_WS", "label_count",
                    'label_dist_avg', "label_dist_dot_min", "label_dist_dot_WS", "label_min_dist", "label_min_dist_WS"
                   ]

    #################################################
    # Cluster feature creation function
    #################################################
    def getTeamFeatures(draftee):
        n_players = 5
        
        draftYear = draftee["Year"]
        nbaYear = scaledNBA[scaledNBA["Year"]==draftYear-1]
        thisYearsTeams = nbaYear["Tm"].unique()
        preds = pd.DataFrame({"Team":thisYearsTeams})
        for col in nba_features: preds[col] = np.nan

        for i in range(len(preds)):
            nbaTeamName = preds["Team"].iloc[i]
            fullnbaTeam = nbaYear[nbaYear["Tm"]==nbaTeamName].sort_values(by=["MP"])
            labelTeam = fullnbaTeam[fullnbaTeam['label']==draftee["label"]]
            teamSize, labelSize = len(fullnbaTeam), len(labelTeam)

            # Grabs the n_players with the highest playing time
            nbaTeam = fullnbaTeam.iloc[:n_players] if teamSize>=n_players else fullnbaTeam

            distances = np.array([metric_function(draftee[metricCols].to_numpy().reshape(1,-1), 
                                         nbaTeam[metricCols].iloc[i][metricCols].to_numpy().reshape(1,-1)).item()
                         for i in range(len(nbaTeam))])
            label_distances = np.array([metric_function(draftee[metricCols].to_numpy().reshape(1,-1), 
                                         labelTeam[metricCols].iloc[i][metricCols].to_numpy().reshape(1,-1)).item()
                         for i in range(len(labelTeam))])
            # Turn them into a feature vector
            preds.iloc[i,preds.columns.get_loc("dist_avg")] = np.mean(distances)
            preds.iloc[i,preds.columns.get_loc("dist_dot_min")] = np.dot(1./distances, nbaTeam["MP"].values)
            preds.iloc[i,preds.columns.get_loc("dist_dot_WS")] = np.dot(1./distances, nbaTeam["WS"].values)
            preds.iloc[i,preds.columns.get_loc("min_dist")] = np.min(distances)
            preds.iloc[i,preds.columns.get_loc("min_dist_WS")] = nbaTeam["WS"].values[np.argmin(distances)]
            preds.iloc[i,preds.columns.get_loc("label_count")] = labelSize/teamSize
            
            if len(label_distances)>0:
                preds.iloc[i,preds.columns.get_loc("label_dist_avg")] = np.mean(label_distances)
                preds.iloc[i,preds.columns.get_loc("label_min_dist")] = np.min(label_distances)
                preds.iloc[i,preds.columns.get_loc("label_dist_dot_min")] = np.dot(1./label_distances, labelTeam["MP"].values)
                preds.iloc[i,preds.columns.get_loc("label_dist_dot_WS")] = np.dot(1./label_distances, labelTeam["WS"].values)
                preds.iloc[i,preds.columns.get_loc("label_min_dist_WS")] = labelTeam["WS"].values[np.argmin(label_distances)]

        # Do some quick mean imputation if there are missing values
        for col in nba_features:
            preds[col] = preds[col].fillna(np.mean(preds[col]))

        return preds

    #################################################
    # Implement the clustering features
    #################################################
    if distance_metric=="euclidean":
        teamFeatures = np.load("/home/nrowe/Thesis/Model/teamFeatures_2.npy", allow_pickle=True).tolist()
    elif distance_metric=="manhattan":
        teamFeatures = np.load("/home/nrowe/Thesis/Model/teamFeatures_1.npy", allow_pickle=True).tolist()
    else:
        print("Features not loaded correctly") ## WILL ERROR 
    
#     teamFeatures = {}
#     for i in range(len(scaledDraft)):
#         draftee = scaledDraft.iloc[i]
#         drafteeName = draftee["Player"]
#         teamFeatures[drafteeName] = getTeamFeatures(draftee)

#         # Do some mean imputation on the missing values
#         if i+1 == len(scaledDraft):
#             print("{}/{} completed! - {}".format(i+1, len(scaledDraft), filename))

#     np.save("/home/nrowe/Thesis/Model/teamFeatures_1.npy", teamFeatures)

    print("Team Features Loaded! - {}".format(filename))
    #################################################
    # Scale the clustering features
    #################################################
    allTeamFeatures = [0 for i in range(len(teamFeatures.keys()))]
    for i, key in enumerate(teamFeatures.keys()): 
        allTeamFeatures[i] = teamFeatures[key]

    allTeamFeatures = pd.concat(allTeamFeatures)
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    scaler.fit(allTeamFeatures[nba_features])

    for i, key in enumerate(teamFeatures.keys()): 
        teamFeatures[key][nba_features] = scaler.transform(teamFeatures[key][nba_features])

    # Change what our training subset looks like 
    allCols = list(dict.fromkeys(allCols)) # removes duplicate
    trainingCols = allCols + nba_features


    #################################################
    # Combine them to create a final dataset
    #################################################
    for col in nba_features: scaledDraft[col] = np.nan
    for i in range(len(scaledDraft)):
        key = scaledDraft.iloc[i]["Player"]
        tm = scaledDraft.iloc[i]["Tm"]
        featuresDF = teamFeatures[key]
        for col in nba_features:
            value = featuresDF[featuresDF["Team"]==tm][col]
            scaledDraft.iloc[i, scaledDraft.columns.get_loc(col)] = featuresDF[featuresDF["Team"]==tm][col].values[0]


    #################################################
    # Define the actual neural network model
    #################################################
    def create_NN():
        N, epochs, batch_size = len(trainingCols), 20, 20
        def create_model():
            model = Sequential()
            model.add(Dense(N, input_dim=N, kernel_initializer='normal', activation='sigmoid'))
            model.add(Dense(1, kernel_initializer='normal')) # currently has no activation (linear i guess...)
            # Compile model
            model.compile(loss="mean_squared_error", optimizer="Adam")
            return model
        return KerasRegressor(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

    
    #################################################
    # Set up oversampling
    #################################################
    ## Split the data into X and Y
    draftX, draftY = scaledDraft[trainingCols], scaledDraft[target]

    ## Set up oversampling (only affects draft data I think)
    oversampledX, oversampledY = draftX.copy(), draftY.copy()
    highDraftX, highDraftY = oversampledX[oversampledY > OS_val], oversampledY[oversampledY > OS_val]
    
    for i in range(OS_size):
        addRow = np.random.randint(len(highDraftX))
        oversampledX = oversampledX.append(highDraftX.iloc[addRow])
        oversampledY = oversampledY.append(pd.Series(highDraftY.iloc[addRow], index=[highDraftY.index[addRow]]))

#     lowValue = US_val
#     lowDraftX, lowDraftY = oversampledX[oversampledY < lowValue], oversampledY[oversampledY < lowValue]
#     for i in range(US_size):
#         addRow = np.random.randint(len(lowDraftX))
#         oversampledX = oversampledX.append(lowDraftX.iloc[addRow])
#         oversampledY = oversampledY.append(pd.Series(lowDraftY.iloc[addRow], index=[lowDraftY.index[addRow]]))  
        
    #################################################
    # Perform the actual predictions
    #################################################
    loo = LeaveOneOut()
    preds = pd.DataFrame(index=scaledDraft.index, columns=["WM_pred"])

    model = create_NN()
    for _, test_index in loo.split(draftX):

        # Split the data according to the oversampling
        X_test, Y_test = draftX.iloc[test_index[0]:test_index[0]+1], draftY.iloc[test_index[0]:test_index[0]+1]
        X_train = oversampledX.drop([test_index[0]], axis=0, inplace=False)
        Y_train = oversampledY.drop([test_index[0]], axis=0, inplace=False)

        # Make sure no occurances of test in training
        X_train = X_train[X_train.index!=X_test.index.values[0]]
        Y_train = Y_train[X_train.index!=X_test.index.values[0]]

        # Convert the input to tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        Y_train = tf.convert_to_tensor(Y_train, dtype=tf.float32)
        old_X_test = X_test
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

        # Generate the model
        model.fit(X_train, Y_train)
        preds["WM_pred"].iloc[test_index] = model.predict(X_test)

        # Now we use that trained model to make predictions on all of the team combinations
        key = scaledDraft.loc[test_index]["Player"].values[0]
        teamFeaturesDic = teamFeatures[key]
        teamFeaturesDic['WM_pred'] = np.nan

        targetColLoc = teamFeaturesDic.columns.get_loc("WM_pred")
        for i in range(len(teamFeaturesDic)):
            team = teamFeaturesDic.iloc[i]["Team"]
            teamData = teamFeaturesDic.iloc[i:i+1][nba_features]
            # Merge it with the other features
            for col in allCols: 
                teamData[col] = old_X_test[col].values[0]

            # Do the predictions
            teamData = tf.convert_to_tensor(teamData, dtype=tf.float32)
            teamFeaturesDic.iloc[i, targetColLoc] = model.predict(teamData)

        teamFeatures[key] = teamFeaturesDic

    print("Model Training Complete! - {}".format(filename))
    scaledDraft = pd.concat([scaledDraft, preds], axis=1, sort=False)
    datatypes = {"WM_pred":float}
    scaledDraft = scaledDraft.astype(datatypes)

    #################################################
    # Calculate MSE
    #################################################
    MSE = mean_squared_error(scaledDraft.loc[scaledDraft["WM"].isnull()==False]["WM"], 
                                                 preds.loc[scaledDraft["WM"].isnull()==False]["WM_pred"], 
                                                 squared=True)

    #################################################
    # Function to get best avaliable pick for a given team
    #################################################
    def getHighestPred(data, team):
        playerNames = data['Player']
        bestName, bestProj = None, -100
        for name in playerNames:
            playerProjs = teamFeatures[name]
            proj = playerProjs[playerProjs["Team"]==team]["WM_pred"].values[0]

            if proj > bestProj:
                bestName = name

        return bestName

    #################################################
    # Simulate a mock draft
    #################################################
    scaledDraft.rename(columns={"Pk":"overallPick"}, inplace=True)
    simData = pd.DataFrame(columns=["team", "oldPick", "newPick", "year"])
    ALL_TEAMS = scaledDraft["Tm"].unique()
    for year in np.unique(scaledDraft["Year"]):
        if year > 2016:
            continue
        yearDraftData = scaledDraft[scaledDraft["Year"]==year].copy()
        yearNBAData = scaledNBA[scaledNBA["Year"]==year]
        yearDraftData.sort_values(by=["overallPick"], inplace=True)
        picks, teams = yearDraftData["overallPick"].to_numpy(), yearDraftData["Tm"].to_numpy()
        picks, teams = picks[~pd.isnull(picks)], teams[~pd.isnull(picks)]

        for myTeam in ALL_TEAMS:
            #print("YEAR: {}, TEAM: {}".format(year, myTeam))
            oldPicks = yearDraftData.loc[yearDraftData["Tm"]==myTeam]["Player"].to_numpy() # These are players
            myPicks = yearDraftData.loc[yearDraftData["Tm"]==myTeam]['overallPick'].to_numpy() # These are numbers
            alreadyPicked = []
            myActualPicks = []

            # Figure out what the new picks will be
            for pick in picks:
                avalPicks = yearDraftData[~yearDraftData.Player.isin(alreadyPicked)]
                if pick in myPicks:
                    # Choose highest remaining target value
                    # myPick = getHighestPred(avalPicks, myTeam)
                    myPick = avalPicks.loc[avalPicks["WM_pred"]==max(avalPicks["WM_pred"])]["Player"].iloc[0]

                    alreadyPicked.append(myPick)
                    myActualPicks.append(myPick)
                else:
                    minPick = min(avalPicks['overallPick'])
                    theirPick = avalPicks.loc[avalPicks['overallPick']==minPick]["Player"].iloc[0]
                    alreadyPicked.append(theirPick)

            if len(np.unique(myActualPicks))!=len(myActualPicks):
                print("ERROR")

            for i in range(len(myPicks)):
                teamSeries = pd.Series(index=["team", "oldPick", "newPick", "year", "overallPick"], dtype=np.int16)
                teamSeries["year"] = year
                teamSeries["team"] = myTeam
                teamSeries["oldPick"] = oldPicks[i]
                teamSeries["newPick"] = myActualPicks[i]
                teamSeries['overallPick'] = myPicks[i]

                simData = simData.append(teamSeries, ignore_index=True)

    print("Simulation Completed! -", filename)
    ################################################# 
    # Find average wins from simulated draft
    #################################################
    resultCols = ["team", "oldWins", "newWins", "addedWins", "numPicks"]
    resultsData = pd.DataFrame(columns=resultCols)
    for team in np.unique(simData["team"]):
        teamData = simData[simData["team"]==team]
        oldWins, newWins = [], []
        for i in range(len(teamData)):
            oldPlayer = teamData["oldPick"].iloc[i]
            newPlayer = teamData["newPick"].iloc[i]
            oW = scaledDraft[scaledDraft["Player"]==oldPlayer]["WM"].iloc[0]
            nW = scaledDraft[scaledDraft["Player"]==newPlayer]["WM"].iloc[0]
            if pd.isnull(oW) or pd.isnull(nW):
                continue
            oldWins.append(oW)
            newWins.append(nW)

        resultSeries = pd.Series(index=resultCols, dtype=np.int16)
        resultSeries["team"] = team
        resultSeries["oldWins"] = np.mean(oldWins)
        resultSeries["newWins"] = np.mean(newWins)
        resultSeries["addedWins"] = np.mean([newWins[i] - oldWins[i] for i in range(len(newWins))])
        resultSeries["numPicks"] = len(teamData)
        resultsData = resultsData.append(resultSeries, ignore_index=True)

    added_wins_per_team = resultsData.addedWins.to_numpy()

    #################################################
    # Get the ratio of new wins to old wins
    #################################################
    oldWMs, newWMs = [], []
    for i in range(len(simData)):
        oldPlayer = simData["oldPick"].iloc[i]
        newPlayer = simData["newPick"].iloc[i]

        oldWM = scaledDraft[scaledDraft["Player"]==oldPlayer].iloc[0]["WM"]
        newWM = scaledDraft[scaledDraft["Player"]==newPlayer].iloc[0]["WM"]

        if pd.isnull(oldWM) or pd.isnull(newWM):
            continue

        oldWMs.append(oldWM)
        newWMs.append(newWM)

    new_added_wins = [newWMs[i] - oldWMs[i] for i in range(len(newWMs))]
    
    
    #################################################
    # Save the results in a pickle
    #################################################
    results = {"MSE":MSE, "added_wins_per_team":added_wins_per_team, "new_added_wins":new_added_wins, "new_picks":newWMs, 
               "old_picks":oldWMs, 
               "actual_values":scaledDraft.loc[scaledDraft["WM"].isnull()==False]["WM"].values, 
               "predictions":preds.loc[scaledDraft["WM"].isnull()==False]["WM_pred"].values}
    
    np.save("./Results/NN_results/"+filename, results)
    print("SAVED:", filename)
    
    return True

# sort out the arguments
import sys
scaler_type = sys.argv[1]
n_clusters = int(sys.argv[2])
distance_metric = sys.argv[3]
file_count = int(sys.argv[4])

filenamevals = [str(i) for i in [scaler_type, n_clusters, distance_metric, file_count]]
filename = '_'.join(filenamevals)+".npy"
print(filename)

simulation(scaler_type, n_clusters, distance_metric)
