# ecs-171-project Abstract

Group Members:
1. Xiaoling Huang
2. Jeff Lee
3. Donna Moon
4. Harshil Patel
5. Cenny Rangel
6. Leela Srinivasan

<Introductory Sentence> For our term project, we'd like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several Machine Learning Techniques. We will begin by preprocessing the data and creating preliminary data visualizations, following which we will delve into Supervised Learning Modeling. The goal is to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. 

We have 114,00 observations without a set data distribution, so we opted to normalize the data with a default MinMaxScaler() from 0 to 1.  The dataset was already cleaned; there was no null data, but we checked to make sure. The main attributes include but are not limited to Track ID,	Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceablity, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, Track Genre
  
In our data preprocessing steps, we dropped columns with redundant information and encoded the album name and track genre to allow us to compute correlations through a Seaborn heatmap. 
