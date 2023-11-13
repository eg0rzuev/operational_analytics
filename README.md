# Operational analytics
My project for DYM Unibo Operational Analytics course. It predicts songs' popularity with different statistical and machine learning models.

# 1. Choose songs

I've chosen the songs that are "periodically" popular in the United States (Christmas, National holidays songs). I get songs' popularity from trends.google.com and teach my models to predict their popularity based on their previous popularity and exougenous factors such as artists' or corresponding holiday popularity. 

Selected Christmas songs:
- Santa Tell Me
- Last Christmas
- Jingle Bells
- Jingle Bell Rock
  
Selected songs associated with national holidays:
- America The Beautiful
- God Bless America
- USA Anthem 

# 2. Select models
The models I've used are Linear regression, Linear regression with binarization​, SVR​, SVR with exogenous data​, keras.Sequential​, auto_arima SARIMA​ and SARIMAX.

# 3. Obtained results

You can see the results in plo​t directory and results.txt files.

Usually auto SARIMA or SARIMAX models perform the best since they are the most complex, run through a grid of parameters and take the most time to learn.

Sequential model has one of the worst performances since neural networks usually require significantly more data to learn on. 

It is easier to predict simple sequences, like Christmas songs with one peak a year, than more complex ones: songs that are associated with several celebrations during the year with many peaks.
