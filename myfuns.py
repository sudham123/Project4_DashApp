import pandas as pd
import requests
import numpy as np
# Define the URL for movie data
smatrix2 = pd.read_csv('https://raw.githubusercontent.com/sudham123/Project4_App/refs/heads/main/output.csv')

# print(smatrix2)


def myIBCF(newuser):
  #all the movies the user has previously rated
  previously_rated = np.where(~np.isnan(newuser))[0]
  # print(previously_rated)

  unrated = np.where(np.isnan(newuser))[0]
  # print(unrated)
  ratings = newuser.iloc[previously_rated]

  preds = []

  for i in unrated:
    # print(i)
    similarity = smatrix2.iloc[i, previously_rated]
    weighted_ratings =  similarity * ratings
    # print(weighted_ratings)


    sim_sum = similarity.sum()

    if sim_sum == 0:
      prediction = np.nan
    else:
      prediction = weighted_ratings.sum() / sim_sum
    preds.append(prediction)
    # newuser.iloc[i] = prediction
    # print(similarity)
  preds_series = pd.Series(preds)
  preds_series_clean = preds_series.dropna()
  # print(preds_series_clean)
  top_10_indices = preds_series.nlargest(10).index

  # print(top_10_indices)
  un2 = pd.Series(unrated).iloc[top_10_indices]
  # print(un2)
  return newuser.iloc[un2].index



  # map_indicies = unrated.index[top_10_indices]
  # print(top_10_movies)

  # known_ratings = new
  # print(unrated)
  # print(previously_rated)

  # similarity = smatrix2.loc[previously_rated.index]

