
# Report: Music Popularity Prediction using Machine Learning

Today, the music industry has grown tremendously with the emergence of smartphones and streaming services. In
the past, most of the revenue was from the album’s sales and concerts. However, these days, streaming services on
the web or smartphones have become a huge part of the music industry. The phenomenon of popularity in music can
be highly subjective so predicting popularity from audio features may potentially be more difficult than one
perceives. Using audio data from Spotify, I have attempted to predict the popularity of music.
## Data Exploration:

I have collected my data from http://organizeyourmusic.playlistmachinery.com. They use the Spotify developer API
to fetch the songs and its features from any playlist. Inorder to use the most optimal data set I used an assumption.
Initially I found all the tracks from 2019 and decided to train my model on 230k songs of 2019. But, on exploring
the data I found that my model might not learn in the best possible ways because there were some songs with similar
features but they had a big difference in their popularity score.


![App Screenshot](https://drive.google.com/uc?id=1VgLeS8EH7uiXZS47j2CHVnr2eDGtKkTU)

