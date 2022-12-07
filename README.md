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
<Introductory Sentence> Music is an integral part of people's lives, regardless of background or culture, and has been for at least 35,000 years. It can improve our mood on a daily basis and allow people to express their emotions in ways that words cannot. Music is very accessible in our current time, and currently, one of the most popular music streaming services is Spotify. Because of the relatability of the Spotify software and popular music, for our term project, we would like to explore the following dataset that holds information on popular Spotify tracks: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset. The aim is to explore the correlation between popularity and various auditory features with the use of several predictive Machine Learning Techniques and Models. In doing so, we hope to reevaluate our conception of likeability within music based on what appeals to Spotify users, the popularity metric being derived from the total number of plays and how recent those plays are. Having a good predictive model would allow artists to determine what patterns in music increase popularity within a specific genre, and help people determine what qualities attract them to a song. 

## III. Figures
 
## IV. Methods
The complete dataset imported from Kaggle contains 114,000 observations without a set data distribution. The data attributes are Track ID, Artists, Album Name, Track Name, Popularity, Duration (ms), Explicit, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Time Signature, and Track Genre.

We used the wget command to easily allow people to download the dataset. The dataset is also present in the GitHub repository.
```
import platform
mysystem = platform.system()
file_id = '10PSeKeL3aUA56faRhr4ZfkEPcVtKjlry'
file_download_link = "https://docs.google.com/uc?export=download&id=" + file_id
# Check if system is Windows
if mysystem != 'Windows':
    !wget -O dataset.csv --no-check-certificate "$file_download_link"
    # !unzip data.zip
print('Please download the data using the following link:', file_download_link)
```
We imported the dataset and extracted a subset of 8,000 songs.

```
df = pd.read_csv('dataset.csv')
df = df.sample(n= 80000, random_state=21)
```
The following libraries were imported for the project.
```
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn import svm
```
 
### Data Preprocessing
In our data preprocessing steps, we began by checking for null data, dropping columns with redundant information such as track_id and track_name. 
 
```
substring = 'None'
df_rem[df_rem.apply(lambda row: row.astype(str).str.contains(substring, case=False).any(), axis=1)]
df_rem = df.drop(columns=['track_id', 'track_name'])
```

We encoded ‘album_name’ and extracted a new subset containing all songs from the 'party' genre.  This yielded a dataset with 1,000 entries.  We then dropped the 'genre' and ‘artist’ columns.
 
```
label_encoder = LabelEncoder()
df_rem['album_name'] = label_encoder.fit_transform(df_rem['album_name'])
df_rem = df_rem.drop(columns=['track_genre','artists'])
```

The popularity feature in the original dataset is a value between 0-100, but to prepare for our model classification, we split the values into 5 classes as follows:

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
 
We took the logarithm of danceability to see how this affected correlation.

```
df_rem['danceability_log2'] = np.log2(df_rem['danceability'])
df_rem['danceability_log10'] = np.log10(df_rem['danceability'])
```

We opted to normalize our data using MinMaxScaler().
```
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(modified_df), columns=modified_df.columns)
```
 
### Data Exploration

To make preliminary predictions from our processed dataset, we used the Seaborn module to compute a heatmap and pair plot, and proceeded to scatterplot various features against popularity that showed promising correlations. 
 
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
We decided to go with the untouched danceability metric instead of the logarithm.
```
df_norm = df_norm.drop(columns=['danceability_log2','danceability_log10'])
```
 
### Model 1: Neural Network

We began by splitting our dataset into the training and testing set.

```
X_train, X_test, y_train, y_test = train_test_split(df_norm.drop(['popularity'], axis=1), df_norm.popularity, test_size=0.2, random_state=21)
```

We designed our initial neural network with 'relu' hidden layers and a sigmoid output layer, using 'rmsprop' as our optimizer and 'binary_crossentropy' as our loss function. Since we ended with a sigmoid layer and used binary_crossentropy, our model could only predict 2 classes.

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

![image](https://user-images.githubusercontent.com/91860903/204427530-4382e0b3-2f96-4358-a6ac-55709eda9449.png)
 
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

Our second iteration of the neural network uses the categorical loss function and a softmax output layer. To implement this, we transformed our y_train with one-hot encoding into a data set with 3 columns each representing a class. 

```
one_hot_encoding = pd.get_dummies(y_train)
y_train = one_hot_encoding
```

We also decided to replace our 'relu' layers with 'selu' layers as they handle negative values better. Our second version of the Neural Net was trained for 200 epochs with a batch size of 5 and validation set of 10%.

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
For our second model, we tried SVM with an 'rbf' kernel.
 
```
rbf = svm.SVC(kernel='rbf', gamma=1, decision_function_shape='ovo').fit(X_train, y_train)
```
 
We predicted yhat(rbf_pred) and printed out the corresponding classification report.
 
```
rbf_pred = rbf.predict(X_test)
cm_rbf = classification_report(y_test, rbf_pred)
print(cm_rbf)
```

## V. Results

### Neural Network
 
For the first iteration of our Neural Net using binary cross entropy, the classification report was as follows.
 
![image](https://user-images.githubusercontent.com/91860903/204436550-97c5997a-95e7-4f2d-b432-21069428f487.png)

The second Neural Net had an accuracy of 73%, shown in the classification report below. The accuracy of the model is also similar to the accuracy with the training data.

![image](https://user-images.githubusercontent.com/91860903/204436637-ad362fea-06a8-4466-9180-89c94d88afac.png)

The graph below shows the loss and accuracy for the training and validation data.
 
![image](https://user-images.githubusercontent.com/91860903/204436693-702f1373-8d05-480f-af90-87a2490f26b2.png)
 
### Support Vector Machine 
 
 The image below shows the classification report for our SVM model. The accuracy for our model is 81%. 
 
 <img width="483" alt="Screen Shot 2022-12-05 at 4 30 29 PM" src="https://user-images.githubusercontent.com/51987755/205776317-3436e74e-dc78-46bc-8e5f-ddff25250652.png">
 
 We plotted our SVM classifier and considered two features, energy and loudness, as you can see in the graph below. We decided on this based on which features had the highest correlation.
 
 <img width="437" alt="Screen Shot 2022-12-05 at 4 24 08 PM" src="https://user-images.githubusercontent.com/51987755/205777543-f37eb729-7321-4288-b7a6-6135eedcb7a9.png">
 
## VI. Discussion

### Data Preprocessing

We began with a dataset of 114,000 songs and decided to cut it down to 8,000 randomly sampled ones.  We did this to reduce computation time while keeping all the genres.  Later this became unnecessary because we decided to limit our scope to the ‘party’ genre, which reduced our dataset to 1,000 songs.  We did to prevent genre from hindering our model. The problem is that different attributes are important for different genres.  For example, ‘I’m Yours’ by Jason Mraz and ‘Radioactive’ by Imagine Dragons both have a popularity score of 80 but are completely different in their other features.  This is largely because one is an acoustic track and one is rock. Musical attributes fluctuate with depending on the genre, so when two songs belong to starkly different genres, it is impossible to see which attributes are important. We chose the ‘party’ genre specifically because it was one of the largest we could find.  There are over 100 genres, so 1,000 songs is relatively large.

Also, we also assumed that artist name would be a big determiner of popularity.  To accurately measure the effect, we split songs with multiple artists into multiple entries.  This was not very helpful, though, so we reverted the change and later dropped the artist attribute altogether.  This improved our model’s accuracy.

The most important change we made was compressing popularity to 5 classes (0-24, 25-49, 50-74, 100).  We decided on this division because our working dataset was only 1,000 songs.  This would not be enough training data to choose among 101 classifications.  We were then able to go a step further since 'party' only contains songs in the classes 0-24, 25-49, and 50-74.  

We considered taking the log transform of various features, but this had a negative effect.  We simply normalized our numeric data with MinMaxScaler().


### Data Exploration

Data exploration was disheartening because we found that all our attributes had very low correlation with popularity.  The best correlations were loudness and energy, which both had -0.15 correlation.  Despite this, we still ended up with very good results.  This is because the model’s ability to predict was not dependent on any single correlation value.  It used the interplay of all the different attributes to predict behavior.

We finished our exploration by throwing our data into a logistic regression model.  This demonstrated our first major issue.  Logistic regression does binary classification, so it cannot predict more than 2 classes, in our case 3.  The model had 56% accuracy as shown, so this was our baseline to improve from.
 
![image](https://user-images.githubusercontent.com/91860903/204436487-c5299271-365f-4272-afbe-4383d0627a70.png)

From here, we split into two teams.  One to build a neural net model and one to build an SVM model.
 
### Model 1: Neural Net
When we built our first neural net, we had not realized the issue with our logistic regression.  We used a sigmoid layer for our output, which also did binary classification.  When we discovered this, we did some research to adapt the neural net.  This is where we replaced sigmoid with softmax and ‘binary_crossentropy’ with ‘categorical_crossentropy’.  We were also able to get advice from a friend in industry who recommended using ‘selu’ as our activation function instead of ‘relu’ and ‘adam’ as our optimizer.  These changes gave us a working model.  Then, it was a matter of tuning our hyperparameters.  We were hoping for 68% accuracy and ended up achieving 73%.  Furthermore, we knew the neural net was not overfitting because our graph of accuracy and loss showed that they were both improving.

![image](https://user-images.githubusercontent.com/91860903/204436693-702f1373-8d05-480f-af90-87a2490f26b2.png)

### Model 2: SVM
Our SVM model also started with binary classification, but it still got 67% accuracy.  We then adjusted it to predict all classes and tested different kernels.  We settled on ‘rbf’, and this gave us an accuracy of 81%.

### Shortcomings

Our main shortcoming is that our model is not generalizable.  Since ‘party’ only had 3 classes of popularity, we designed our model to only predict those 3.  It cannot be used for genres with more or fewer than 3 classes.  Our model might also be improved by dropping more attributes with low correlation.

## VII. Conclusion

For this project we only considered the songs with a genre of “party”. We could choose different genres and plot our correlation matrix on them and see how these correlation matrices are different or similar to each other. 
Also, since we dropped the feature “genre”, we are not taking into account the effect of genre on a song’s popularity. We could also explore a way to address how genre impacts popularity and build new models accordingly. 
During the data processing step, we used label encoder to encode the categorical features. We could also try different encoding methods and see if they impact the accuracy of our models.
 
## VIII. Collaboration
 
We did not define rigid roles for our group project, but rather collaborated on general logistics as a full group and then split into two groups for the model completion. Harshil, Cenny and Leela worked on the Neural Network, while Xiaoling, Jeff and Donna worked on the SVM, and then we came back together as a group to complete the writeup collectively. Individual contributions are as follows:
 
 Harshil: Neural Network team member
 * Worked on data exploration.
 * Worked on write-up for second milestone and figures section of final write-up.
 * Worked on the Neural Network, including training and trying different models.
 * Communicate with professor through Office hours and piazza.
 
 Cenny: Neural Network Team Member:
 * Made initial neural net model and experimented with hyperparameters
 * Annotated Jupyter notebook and arranged into logical flow
 * Contributed to Abstract and First Milestone write-up and wrote Discussion section of final write-up

 Leela:  
 
 Xiaoling: SVM Team Member
 * One-hot encoded categorical features for SVM uses (not used in the final SVM model).
 * Worked on the SVM model write-up and Conclusion write-up.
 
 Jeff:  
 Donna:  
 
## IX. Summary
 
 
