# ECS 171 Term Project: ML Predictions of Track Popularity

## I. Group Members
1. Xiaoling Huang
2. Jeff Lee
3. Donna Moon
4. Harshil Patel
5. Cenny Rangel
6. Leela Srinivasan

Jupyter Notebook (Google Colab link): https://colab.research.google.com/drive/1c0VEUOjGAMLicl0ULKlNQ5caiShoe-jn?usp=sharing <br>

## II. Introduction
<Introductory Sentence> Music is an integral part of peoples lives, regardless of their background or culture, and has been for at least 35,000 years. It can improve our mood on a daily basis and allow people to express their emotions in ways that words cannot. Music is very accessible in our current time, and currently, one of the most popular music streaming services is Spotify. Because of the relatibility of the Spotify software and popular music, for our term project, we would like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several predictive Machine Learning Techniques and Models. In doing so, we hope to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. Having a good predictive model would allow artists to determine what patterns in music increase popularity within a specific genre, and help people determine what qualities attract them to a song. 

## III. Figures
 
## IV. Methods
The complete dataset imported from Kaggle contains 114,000 observations without a set data distribution, from which we extracted a subset of 8,000 entries. The main attributes include but are not limited to Track ID, Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, and Track Genre.

```
df = df.sample(n= 80000, random_state=21)
```
 
### Data Preprocessing
In our data preprocessing steps, we began by checking for null data, dropping columns with redundant information such as track_id and track_name. 
 
```
substring = 'None'
df_rem[df_rem.apply(lambda row: row.astype(str).str.contains(substring, case=False).any(), axis=1)]
df_rem = df.drop(columns=['track_id', 'track_name'])
```
 
We extracted a subset containings songs exclusively from the 'party' genre, yielding a dataset with 1,000 entries, with the 'genre' column dropped. We used a label encoder to encode album name, which is categorical data, and decided to drop the 'artists' column because our correlation matrix revealed that the artist name was very weakly correlated to popularity.
 
```
label_encoder = LabelEncoder()
df_rem['album_name'] = label_encoder.fit_transform(df_rem['album_name'])
df_rem = df_rem.drop(columns=['track_genre','artists'])
```

The popularity feature in the original dataset is a value between 0-100, but in order to prepare for our model classification, we split the values into 5 classes as follows:

0: 0-24  
1: 25-49  
2: 50-74  
3: 75-99  
4: 100  

```
modified_df = df_rem.drop(columns=['album_name', 'popularity'])
df_norm['album_name'] = df_rem['album_name'].to_numpy()
df_norm['popularity'] = df_rem['popularity'].to_numpy()/25
df_norm.popularity = df_norm.popularity.astype(int)
```
 
We decided on this division because our working dataset consists of 1,000 songs from the 'party' genre, which is not enough to train a model with 100 individual classes. In our sample dataset, the 'party' genre only contains songs in the classes 0, 1, and 2, indicating that no song had a popularity index above 74.  After computing the log transforms of various features for feature expansion, we opted to normalize the dataset with a default MinMaxScaler() from 0 to 1, and at this stage, our data was fit for modeling.

```
df_rem['danceability_log2'] = np.log2(df_rem['danceability'])
df_rem['danceability_log10'] = np.log10(df_rem['danceability'])
```
 
```
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(modified_df), columns=modified_df.columns)
```
 
### Data Exploration

To make preliminary predictions from our processed dataset, we used the Seaborn module to compute a heatmap and pairplot, and proceeded to scatterplot various features against popularity that showed promising correlations. 
 
```
corr = df_norm.corr()
fig, ax = plt.subplots(figsize=(14, 14))
_ = sns.heatmap(corr, vmin=-1, vmax=1, center=0, annot=True, fmt='.2', cmap= 'coolwarm')
```
 
```
_=sns.pairplot(data=df_norm)
```
 
```
_=sns.scatterplot(data=df_rem, x='danceability', y='popularity') 
```
 
### Model 1: Neural Network

We began by splitting our dataset into the training and testing set.

```
X_train, X_test, y_train, y_test = train_test_split(df_norm.drop(['popularity'], axis=1), df_norm.popularity, test_size=0.2, random_state=21)
```

We designed our initial neural network with relu hidden layers and a sigmoid output layer, using 'rmsprop' as our optimizer and 'binary_crossentropy' as our loss function. Since we ended with a sigmoid layer and used binary_crossentropy, our model could only predict 2 classes.

```
model = Sequential()
model.add(Dense(units=14, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=7, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=9, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=7, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=1, activation='sigmoid', input_dim=X_train.shape[1]))
model.summary()
```

```
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
his = model.fit(X_train.astype('float'), y_train, validation_split=0.1, batch_size=5, epochs=200)
```

We computed the classification report as follows.
 
```
yhat_test = model.predict(X_test.astype(float))
yhat = []
 
for y in yhat_test:
    if y <= 0.33:
        yhat.append(0)
    elif y <= 0.67:
        yhat.append(1)
    else:
        yhat.append(2)
```

``` 
print('Model Classification Report:')
print(classification_report(y_test, yhat))
```

Our second iteration of the neural network utilizes the categorical loss function and a softmax output layer. To implement this, we transformed our y_train set by one-hot encoding it into a data set with 3 columns each representing a class. 

```
one_hot_encoding = pd.get_dummies(y_train)
y_train = one_hot_encoding
```

We also decided to use 'selu' layers instead of 'relu' layers as activation and hidden layers as they handle negative values better. 

```
model = Sequential()
model.add(Dense(units=30, activation = 'selu', input_dim = X_train.shape[1]))
model.add(Dense(units=15, activation = 'selu'))
model.add(Dense(units=10, activation = 'selu'))
model.add(Dense(units=10, activation = 'selu'))
model.add(Dense(units=5, activation = 'selu'))
model.add(Dense(units = 3, activation = 'softmax'))
model.summary()
```

```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
his = model.fit(X_train.astype('float'), y_train, validation_split=0.1, batch_size=5, epochs=200)
```

Once again, we computed the classification report.
 
```
yhat_test = model.predict(X_test.astype(float))
yhat = []
for y in yhat_test:
    yhat.append(np.argmax(y))
yhat = np.array(yhat)
```

```
print(classification_report(y_test, yhat))
```
 
### Model 2: SVM
For our second model, we constructed a SVM, designating the kernel to be rbf and utilizing the same testing and training split as previously computed.
 
```
rbf = svm.SVC(kernel='rbf', gamma=1, decision_function_shape='ovo').fit(X_train, y_train)
```
 
We predicted yhat(rbf_pred) using the SVM model and X_test, and printed out the corresponding classification report.
 
```
rbf_pred = rbf.predict(X_test)
cm_rbf = classification_report(y_test, rbf_pred)
print(cm_rbf)
```

## V. Results

(First two images are for Neural Net V1)
![image](https://user-images.githubusercontent.com/91860903/204427530-4382e0b3-2f96-4358-a6ac-55709eda9449.png)

![image](https://user-images.githubusercontent.com/91860903/204436550-97c5997a-95e7-4f2d-b432-21069428f487.png)
 
 (Neural Net V2)
 
![image](https://user-images.githubusercontent.com/91860903/204428513-780fc3c2-e6bb-4fd5-bce7-25ac6045b7b0.png)

This model was trained for 200 epochs with a batch size of 5 and with a validation set of 10%. This model is more accurate because it handles the categorical data better. To change the output from 3 columns back to one column, we chose to take the column with the highest value at each row. We do this because the model outputs in each row is how strong the model think the input is of the class represented by the column. The image below shows the classification report of our model, and as we can see, the accuracy is 73%.

![image](https://user-images.githubusercontent.com/91860903/204428993-33105d30-acf7-47d0-8862-dc2c77b31ae1.png)

The accuracy of the model is also similar to the accuracy with the training data, as shown in the image below.

![image](https://user-images.githubusercontent.com/91860903/204436637-ad362fea-06a8-4466-9180-89c94d88afac.png)

We use accuracy as a metric to base the strength of our model as our goal is to know that we can predict the popularity of a song based on its various features to a reasonable accuracy. We think that our model falls within the right range in the fitting graph as the accuracy and loss of the model does not start decreasing and increasing respectively as it would if it was overfitted. We also do not think that the model is not underfitting as it is trained for a high number of epochs and the accuracy is showing an increasing trend, as shown in the graph below.

![image](https://user-images.githubusercontent.com/91860903/204436693-702f1373-8d05-480f-af90-87a2490f26b2.png)
 
## VI. Discussion
 
- We also split songs with multiple artists into individual rows in order to link attributes to artists.
- We then used a label encoder to encode categorical data such as album name, artist, and track genre. However, we quickly realized that we needed to backtrack, as our vision to predict popularity across all genres was flawed — two songs might hold similar popularity scores, but if they belong in starkly different genres, it's impossible to draw correlations between musical attributes that fluctuate with genre changes. For example, ‘I’m Yours’ by Jason Mraz is an acoustic hit with a popularity score above 80 but a low energy score, whereas ‘The Motto’ by Tiesto has high energy and a similar popularity score, stemming from the trance/EDM genre. We decided to focus in on a single genre, ‘party’, for the remainder of our project, to ensure that the general musical makeup would be similar enough to draw comparisons.
 
We initially planned to make a logistic regression model, but we quickly ran into an issue. A logistic regression model only does binary classification so it cannot be used to classify data with more than 2 classes; 3 in our case. We still ran the model because though this result was incorrect, it gave us a baseline accuracy to improve from.

![image](https://user-images.githubusercontent.com/91860903/204436487-c5299271-365f-4272-afbe-4383d0627a70.png)
 
## VII. Conclusion


 
## VIII. Collaboration
 
We did not define rigid roles for our group project, but rather collaborated on general logistics as a full group and then split into two groups for the model completion. Harshil, Cenny and Leela worked on the Neural Network, while Xiaoling, Jeff and Donna worked on the SVM, and then we came back together as a group to complete the writeup collectively. Individual contributions are as follows:
 
 Harshil:  
 Cenny:  
 Leela:  
 Xiaoling:  
 Jeff:  
 Donna:  
 
## IX. Summary
 
 
