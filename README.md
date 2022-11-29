# ecs-171-project Abstract

### Group Members:
1. Xiaoling Huang
2. Jeff Lee
3. Donna Moon
4. Harshil Patel
5. Cenny Rangel
6. Leela Srinivasan

### Abstract
<Introductory Sentence> For our term project, we'd like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several Machine Learning Techniques. We will begin by preprocessing the data and creating preliminary data visualizations, following which we will delve into Supervised Learning Modeling. The goal is to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. 

### Data encoding and preprocessing
We have 114,000 observations without a set data distribution, from which we extracted about 8,000 entries. The main attributes include but are not limited to Track ID, Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceablity, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, and Track Genre.

In our data preprocessing steps, we began by checking for null data, dropping columns with redundant information such as track_id and track_name. We also split songs with multiple artists into individual rows in order to link attributes to artists. We then used a label encoder to encode categorical data such as album name, artist, and track genre. However, we quickly realized that we needed to backtrack, as our vision to predict popularity across all genres was flawed — two songs might hold similar popularity scores, but if they belong in starkly different genres, it's impossible to draw correlations between musical attributes that fluctuate with genre changes. For example, ‘I’m Yours’ by Jason Mraz is an acoustic hit with a popularity score above 80, but a low energy score, whereas ‘The Motto’ by Tiesto has high energy and a similar popularity score, stemming from the trance/EDM genre. We decided to focus in on a single genre, ‘party’, for the remainder of our project, to ensure that the general musical makeup would be similar enough to draw comparisons. For the new dataset we just extracted all the song with 'party' genre from the original dataset which equals a dataset with 1,000 entries. After selecting 'party' we also dropped the genre column as it would be same for all the songs. We also decided to drop the artists columns as it was causing redundancy in the data. The reason for that is because artist names were very weakly corelated to popularity but due to spliting same song with multiple artists into different rows, it creates duplicates in all other features.

We also changed the Popularity which was initially a metric between 0-100, into 5 classes as follows:

0: 0-24  
1: 25-49  
2: 50-74  
3: 75-99  
4: 100  

We decided to do this because our dataset with all song with 'party' genre was 1000 entries and it is not effective to train a model with 100 classes with such a small dataset, so we split popularity into the above ranges. For the 'party' genre the popularity is the classes [0,1,2]. We computed log transforms of various features for feature expansion, and opted to normalize the dataset with a default MinMaxScaler() from 0 to 1, and at this stage, our data was fit for modeling.
 
### Model 1: Neural Network

We initially planned to make a logistic regression model, but we quickly ran into an issue. The problem with using a logistic regression model is that it only does binary classification so it can not be used to classify data with multiple classes, 3 in our case. We still ran the model but instead of splitting the data into 2 by brawing a line at 0.5 in the yhat, we decided to devide it by 3 using 0.33 and 0.66 as threshholds. Though this result was incorrect we got a baseline accuracy and an idea of how accuract our model should be by minimun.

###screenshot here

Next we decided to make a Neural Network, 
