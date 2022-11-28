# ecs-171-project Abstract

Group Members:
1. Xiaoling Huang
2. Jeff Lee
3. Donna Moon
4. Harshil Patel
5. Cenny Rangel
6. Leela Srinivasan

<Introductory Sentence> For our term project, we'd like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several Machine Learning Techniques. We will begin by preprocessing the data and creating preliminary data visualizations, following which we will delve into Supervised Learning Modeling. The goal is to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. 

We have 114,000 observations without a set data distribution, from which we extracted about 80,000 entries. The main attributes include but are not limited to Track ID, Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceablity, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, and Track Genre.

In our data preprocessing steps, we began by checking for null data, dropping columns with redundant information and splitting songs with multiple artists into individual rows in order to link attributes to artists. We then used a label encoder to encode categorical data such as album name, artist, and track genre. However, we quickly realized that we needed to backtrack, as our vision to predict popularity across all genres was flawed — two songs might hold similar popularity scores, but if they belong in starkly different genres, it's impossible to draw correlations between musical attributes that fluctuate with genre changes. For example, ‘I’m Yours’ by Jason Mraz is an acoustic hit with a popularity score above 80, but a low energy score, whereas ‘The Motto’ by Tiesto has high energy and a similar popularity score, stemming from the trance/EDM genre. We decided to focus in on a single genre, ‘party’, for the remainder of our project, to ensure that the general musical makeup would be similar enough to draw comparisons.

Popularity was initially a metric between 0-100, but within the ‘party’ genre, we divided the popularity indices into 5 classes as follows:

0: 0-24  
1: 25-49  
2: 50-74  
3: 75-99  
4: 100  
 
We found that the dataset entries in the party genre exist exclusively in classes 0-2. We computed log transforms of various features for feature expansion, and opted to normalize the dataset with a default MinMaxScaler() from 0 to 1, and at this stage, our data was fit for modeling.
 
