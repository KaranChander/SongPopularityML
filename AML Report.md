

**Report: Music Popularity Prediction using Machine Learning**

By: Karan Lnu (02074358)

**Abstract:**

Today, the music industry has grown significantly with the emergence of smartphones and streaming services. In the

past, most of the revenue was from the album’s sales and concerts. However, these days, streaming services on the

web or smartphones have become a huge part of the music industry. The phenomenon of popularity in music can be

highly subjective so predicting popularity from audio features may potentially be more difficult than one perceives.

Using audio data from Spotify, I have attempted to predict the popularity of music.

**Intro:**

Music is a large part of just about any culture on the planet and is an emotional method of communication between

human beings. The digital music platform Spotify grants listeners access to millions of songs for free and gives

people greater access to music than ever before. With the digitization of music and the creation of music streaming

platforms, the social implications and lucrativeness of music have grown exponentially in recent history. Because of

the large societal and financial impacts music has on world culture, I decided to center my deep learning project on

music.

Music has always been an integral part of my life. I was always into music and I play guitar as well. My musical

tastes branched out widely from just alt-rock to jazz, funk, progressive rock, bluegrass, and many other genres.

However, my personal musical tastes often differ quite drastically from what is popular in the mainstream. As such,

I have always been fascinated by why certain songs are popular. i.e., what is it about certain songs that cause them to

have billions of listens?

Spotify is a digital music service that enables users to remotely source millions of different songs on various record

labels from a laptop, smartphone or other device. To recommend new music to users, and to be able to internally

classify songs, Spotify assigns each song values from 12 different attributes/features. These features are mostly

numerical values. Spotify also assigns each song a popularity score, based on total number of clicks/listens.

**Data Exploration:**

**About the Data:**

I have collected my data from <http://organizeyourmusic.playlistmachinery.com>. They use the Spotify developer API

to fetch the songs and its features from any playlist. Inorder to use the most optimal data set I used an assumption.

Initially I found all the tracks from 2019 and decided to train my model on 230k songs of 2019. But, on exploring

the data I found that my model might not learn in the best possible ways because there were some songs with similar

features but they had a big difference in their popularity score.

Example:

What could be the reason for this? Maybe the song with 0 popularity was not advertised well or didn’t reach the

audience.

Maybe the song with more popularity score was signed by a big record label company and it was promoted well

whereas the song with low popularity was made by an individual creator.

This lead me to make a decision on fetching the music tracks that I will be using. Instead of using all of the song

from a year I decided to use record labels. These record labels publish all of their songs in similar ways and the

chances of similar songs having gap in popularity will be less.





I have chosen 3 of the biggest record label in the United States; **Atlantic Records**, **Republic Records** and **Capitol**

**Records**.

**Data Overview:**

In this section, the following represents top 5 songs from the songs gathered from the record labels. It illustrates

title, artist, top genre, year, added, bpm, energy, dancability, loudness, liveliness, valence, duration acousticness,

speechiness and popularity.

**Features:**

**Attribute**

**Description**

BPM

The overall estimated tempo of a track in beats per minute (BPM). In musical terminology,

tempo is the speed or pace of a given piece and derives directly from the average beat

duration.

Energy (nrgy)

Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and

activity. Typically, energetic tracks feel fast, loud, and noisy.

Danceability (dnce)

describes how suitable a track is for dancing based on a combination of musical elements

including tempo, rhythm stability, beat strength, and overall regularity.

Loudness (dB)

Liveness (live)

The overall loudness of a track in decibels (dB)

Detects the presence of an audience in the recording. Higher liveness values represent an

increased probability that the track was performed live

Valence (val)

A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks

with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with

low valence sound more negative (e.g. sad, depressed, angry)

Duration (dur)

An estimated overall time duration of a track.

Acousticness (acous)

Speechiness (spch)

A confidence measure from 0.0 to 1.0 of whether the track is acoustic.

Speechiness detects the presence of spoken words in a track. The more exclusively

speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute

value

Popularity (pop)

The popularity of the track. The value will be between 0 and 100, with 100 being the most

popular. The popularity of a track is a value between 0 and 100, with 100 being the most

popular. The popularity is calculated by algorithm and is based, in the most part, on the total

number of plays the track has had and how recent those plays are.





\-

Data distribution of song popularity against number of songs for the 1,352 sound tracks I have within my

database.

As I continued looking into the dataset as a whole, I realized that this was probably going to be a difficult problem

for linear regression to solve, simply due to the fact that many of the features do not appear to have much correlation

with the target variable. Here are some selected scatter plots of features vs. popularity.





**Data Cleaning:**

I encountered a problem with the top genre field. I found that there were 153 unique genres in my data set and most

of them just occurred once. It was not possible to categorize these genres and also most of the genres were missing

from the data set. I decided to discard this field and work with the remaining features.

**-**

**90 null values of top**

**genre**

After removing unnecessary features (i.e. title, artist, top genre, year, and added) I finalized my 9 features.

Also, transformed the popularity column with binary values 0s and 1s. Considering the mean of the popularity score,

I decided to assign all the popularity scores greater than 46 to be 1 and others as 0.





**Correlation of factors:**

I also wanted to take a look at the correlation coefficient within my independent variables, and used sns to

make a nice heatmap of those, shown above. Overall, there aren't too many independent variables with high

correlation values, but energy/loudness could potentially cause problems but 0.7 is still not a very high number to be

considered highly correlated. One thing potentially worrying about this though: not much correlation b/w

independent variables and popularity.

**Feature Scaling:**

**Before Scaling**

I observed that some of the values like bpm, duration, valence and acousticness have high varying magnitudes and I

decided to perform standard scaling on all of the features.





**After Scaling**

**MODEL IMPLEMENTATION:**

**LOGISTIC REGRESSION**

I decided to see if I could create a logistic regression model, in order to predict 'Popular' or 'Not Popular', using

samples taken from my dataset. Basically, all songs that had a popularity score at or above the popularity score

cutoff were flagged as having a popularity binary value of 1, and all songs below the cutoff were flagged as having a

\0. The process of setting the cutoff, sampling, running the logistic regression, and then evaluating the outputs of the

model was repeated for several different cutoff points.

I used the Sklearn library to perform logistic regression for my model. I started by dividing my data into training and

test set.

20% of my data is used for test set and the remaining is used for training and validation set.

**Hyper Parameter tuning:**

The logistic model takes the following parameter:

**C (float, default=1.0) -**

Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify

stronger regularization.

**Penalty{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’ -**

A regression model that uses L1 regularization technique is called Lasso Regression and model which uses L2 is

called Ridge Regression.

**Solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’ -**

Algorithm to use in the optimization problem. Default is ‘lbfgs’.

After tuning the parameters and using different combinations of C, penalty and solver, I plotted a learning curve

demonstrating the accuracy and cost of the model with the following values.

C=0.01, penalty='l2', solver='liblinear'





**ROC curve with different C value**

C = 0.001

C = 0.01

C = 0.1

C = 1.0

C = 10





**Learning Curves:**

\- Accuracy Graph

\- Cost Graph

**Precision and Recall:**

Output

precision

0.65

recall

0.70

0.49

f1-score

0 (Not popular)

1 (popular)

0.67

0.51

0.54

The precision and recall for non-popular songs was fairly well but for popular songs, it was poor. I decided to further

investigate on the results using confusion matrix.

**Confusion matrix:**

After observing the precision and recall and confusion matrix I concluded that the model is performing well to

identify non popular songs but the output to predict popular song was poor.





**ROC Curve:**

**AUC: 0.59**

Overall I observed that the model was not able to fairly distinguish among the popular and non-popular songs, also

the AUC score was just fair as well. I further decided to train my model using SVM.

**SVM (Support Vector Machine):**

Results with logistic regression were not good enough to be accepted, therefore, I decided to use SVM to train my

model and compared it with Logistic Regression

**Hyperparameter Tuning:**

SVM model takes the following parameters:

**C (float, default=1.0) -**

Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify

stronger regularization.

**Kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’**

Specifies the kernel type to be used in the algorithm.

**Gamma {‘scale’, ‘auto’} or float, default=’scale’**

The gamma parameter defines how far the influence of a single training example reaches.

●

●

●

if gamma='scale' (default) is passed then it uses 1 / (n\_features \* X.var()) as value of gamma,

if ‘auto’, uses 1 / n\_features

if float, must be non-negative.

After tuning the parameters and using different combinations of C, gamma and kernel, I plotted a learning curve

demonstrating the accuracy and cost of the model with the following values.

gamma=0.016, C=12, kernel = 'sigmoid'





**ROC curve with different Gamma values**

gamma = 0.001

gamma = 0.01

gamma = 0.1

gamma = 0.016

**Learning Curves:**

\- Accuracy Graph

\- Cost Graph





**Precision and Recall:**

Output

precision

0.66

recall

0.69

0.53

f1-score

0.68

0 (Not popular)

1 (Popular)

0.55

0.54

The precision and recall results for SVM were a bit better than Logistic Regression but there is no significant

improvement.

**Confusion Matrix:**

After observing the precision and recall and confusion matrix I concluded that the model is performing similar to

Logistic Regression and there was only a little bit improvement with SVM.

**ROC Curve:**

**AUC = 0.61**





**ROC Curve of Logistic Regression and SVM(Comparison):**

Both of the training algorithms performed in a very similar way.

**Conclusion:**

It is quite difficult to determine if a song will be a popular or not, and there appear to be other factors at play that are

not necessarily included in this dataset. Although, I do believe that song popularity model can perform a lot better if

we could fetch more features which highly correlates with popularity and also we should consider what the current

trend of music genre is.

Other factors that influence if a song will be popular or not could potentially be:

Does a particular artist have any current name recognition?

Has this artist had any previous hits?

What is this artist's genre of music?

Has this artist collaborated with other popular artists?

**References:**

[1] Music Database - <http://organizeyourmusic.playlistmachinery.com/index.html>

[2] Spotify Data Analysis- <https://developer.spotify.com/community/showcase/spotify-audio-analysis/>

[3] SKLearn functionalities and modules - <https://scikit-learn.org/stable/>

[4] Information on Logistic Regression using SKlearn

<https://www.youngwonks.com/blog/What-is-sklearn-Logistic-Regression>

[5]Information on SVM using SKlearn

https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python

