<p align="center">
### ECS 171 Term Project
</p>

### I: Group Members:
1. Xiaoling Huang
2. Jeff Lee
3. Donna Moon
4. Harshil Patel
5. Cenny Rangel
6. Leela Srinivasan

Jupyter Notebook (Google Colab link): https://colab.research.google.com/drive/1c0VEUOjGAMLicl0ULKlNQ5caiShoe-jn?usp=sharing <br>
The notebook is also available in the GitHub repository.

### II: Introduction
<Introductory Sentence> For our term project, we'd like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several Machine Learning Techniques. We will begin by preprocessing the data and creating preliminary data visualizations, following which we will delve into Supervised Learning Modeling. The goal is to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. 

### III: Figures
 
### IV: Methods
We have 114,000 observations without a set data distribution, from which we extracted about 8,000 entries. The main attributes include but are not limited to Track ID, Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, and Track Genre.

In our data preprocessing steps, we began by checking for null data, dropping columns with redundant information such as track_id and track_name. We also split songs with multiple artists into individual rows in order to link attributes to artists. We then used a label encoder to encode categorical data such as album name, artist, and track genre. However, we quickly realized that we needed to backtrack, as our vision to predict popularity across all genres was flawed — two songs might hold similar popularity scores, but if they belong in starkly different genres, it's impossible to draw correlations between musical attributes that fluctuate with genre changes. For example, ‘I’m Yours’ by Jason Mraz is an acoustic hit with a popularity score above 80 but a low energy score, whereas ‘The Motto’ by Tiesto has high energy and a similar popularity score, stemming from the trance/EDM genre. We decided to focus in on a single genre, ‘party’, for the remainder of our project, to ensure that the general musical makeup would be similar enough to draw comparisons. When we extracted all the songs from the 'party' genre from the original dataset, our new dataset had 1,000 entries, with the 'genre' column dropped. We also decided to drop the 'artists' column because our correlation matrix revealed that the artist name was very weakly correlated to popularity, and splitting multi-artist songs into rows was causing redundancy in the data.

Popularity was initially a metric between 0-100, but we split it into 5 classes as follows:

0: 0-24  
1: 25-49  
2: 50-74  
3: 75-99  
4: 100  

We decided on this division because our working dataset consists of 1,000 songs from the 'party' genre, which is not enough to train a model with 100 individual classes. In our sample dataset, the 'party' genre only contains songs in the classes 0, 1, and 2, indicating that no song had a popularity index above 74.  After computing the log transforms of various features for feature expansion, we opted to normalize the dataset with a default MinMaxScaler() from 0 to 1, and at this stage, our data was fit for modeling.
 
### Model 1: Neural Network

We initially planned to make a logistic regression model, but we quickly ran into an issue. A logistic regression model only does binary classification so it cannot be used to classify data with more than 2 classes; 3 in our case. We still ran the model because though this result was incorrect, it gave us a baseline accuracy to improve from.

![image](https://user-images.githubusercontent.com/91860903/204436487-c5299271-365f-4272-afbe-4383d0627a70.png)

Next, we decided to make a Neural Network. We designed our initial neural network using relu layers as our hidden layers and a sigmoid layer for our output. We also used 'rmsprop' as our optimizer and 'binary_crossentropy' as our loss function. We later realized that this has the same problem as logistic regression.  Since we ended with a sigmoid layer and used binary_crossentropy, our model could only predict 2 classes.

![image](https://user-images.githubusercontent.com/91860903/204427530-4382e0b3-2f96-4358-a6ac-55709eda9449.png)

The above image is a summary of our first model. Since we were trying to predict 3 classes, we didn't use 0.5 as our threshold.  We instead set 2 thresholds, 0.33 and 0.67. As expected for a flawed model, the accuracy was low.

![image](https://user-images.githubusercontent.com/91860903/204436550-97c5997a-95e7-4f2d-b432-21069428f487.png)

To solve these issues, we decided to use the categorical loss function and using a softmax layer as output. To implement this, we transformed our y_train set by one-hot encoding it into a data set with 3 columns each representing a class. We also decided to use 'selu' layers instead of 'relu' layers as activation and hidden layers as they handle negative values better. The image below is the summary of our model.

![image](https://user-images.githubusercontent.com/91860903/204428513-780fc3c2-e6bb-4fd5-bce7-25ac6045b7b0.png)

This model was trained for 200 epochs with a batch size of 5 and with a validation set of 10%. This model is more accurate because it handles the categorical data better. To change the output from 3 columns back to one column, we chose to take the column with the highest value at each row. We do this because the model outputs in each row is how strong the model think the input is of the class represented by the column. The image below shows the classification report of our model, and as we can see, the accuracy is 73%.

![image](https://user-images.githubusercontent.com/91860903/204428993-33105d30-acf7-47d0-8862-dc2c77b31ae1.png)

The accuracy of the model is also similar to the accuracy with the training data, as shown in the image below.

![image](https://user-images.githubusercontent.com/91860903/204436637-ad362fea-06a8-4466-9180-89c94d88afac.png)

We use accuracy as a metric to base the strength of our model as our goal is to know that we can predict the popularity of a song based on its various features to a reasonable accuracy. We think that our model falls within the right range in the fitting graph as the accuracy and loss of the model does not start decreasing and increasing respectively as it would if it was overfitted. We also do not think that the model is not underfitting as it is trained for a high number of epochs and the accuracy is showing an increasing trend, as shown in the graph below.

![image](https://user-images.githubusercontent.com/91860903/204436693-702f1373-8d05-480f-af90-87a2490f26b2.png)

### V: Results
 
### VI: Discussion
 
### VII: Conclusion
 
### VIII: Contributions
 
We did not define rigid roles for our group project, but rather collaborated on general logistics as a full group and then split into two groups for the model completion. Harshil, Cenny and Leela worked on the Neural Network, while Xiaoling, Jeff and Donna worked on the SVM, and then we came back together as a group to complete the writeup collectively. Individual contributions are as follows:
 
 Harshil:  
 Cenny:  
 Leela:  
 Xiaoling:  
 Jeff:  
 Donna:  
 
### IX: Summary
 
 
