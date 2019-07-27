
# Project 3: Subreddit Classification using NLP

> ### Project by Chris Birch



## Executive Summary



### Table of Contents

- [Acknowledgments](#acknowledgments)
- [Problem Statement](#problem-statement)
- [Data Collection](#data-collection)
- [Data Cleaning / EDA](#data-cleaningeda)
- [Preprocessing and Modeling](#preprocessing-and-modeling)
- [Evaluation](#evaluation)
- [Conclusion / Summary / Recommendations](#conclusion-summary-recommendations)
- [Object Oriented Programming](#object-oriented-programming)
- [Learning Points](#learning-points)
- [Moving Forward](#moving-forward)


### Acknowledgments

Matt Brems - Histogram Plots - GA DSI  
J Beightol - GA DSI - Word Count Plots - GA DSI  
https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/  


### Problem Statement

We're trying to classify Reddit posts into the proper subreddit based on the post (title) alone.  
An important part of this process is the decision whether we care about a particular positive class, or just want to get the overall best accuracy possible.
Time to fit, bias, variance, and overal generalization are important in model selection as well.

### Data Collection

Collecting data for this project is relatively easy since it is easy to get hold of using the Reddit API.  A simple for loop allows us to run through and get a (maybe) unlimited amount of data if it is available.  Taking care to include a sleep time so as not to bombard the site with requests.
I collected approximately 10,000 posts from each of my chosen Subreddits.  At first I had only collected 1000 posts and my results were abismal.  Accuracy below 50% for one model I remember, which was worse than the baseline score of the 50% distribution of posts from each subreddit.  
Collecting 10,000 posts made all the difference.
**Note** Subreddit selection also plays a key in the modeling process.  Choosing two subreddits that are vastly different will make any model perform outstanding since it is not a hard task.  Picking very similar subreddits allows for better testing of models and tuning of hyperparameters.

### Data Cleaning / EDA

This was the easiest part of this project.  Thanks to my close friends Tf-idf and CountVectorizer, the cleaning and preprocessing was handled swimmingly for me.  
I chose not to experiment with lemmatizing and stemming on my own using NLTK since I spent most of my time on other parts of the project, namely creating functions and practicing object oriented programming skills.

### Preprocessing and Modeling

CountVectorizer does a rather good job of cleaning up text and making it avaialbe for modeling.  CountVectorizer does tokenizing and stripping of punctuation for us.  Tf-idf does more or less the same, but the way they classify features is slightly different.  CountVectorizer only uses word frequency as the defining feature whereas Tf-idf also gives each word a particular weight based on how frequently it appears across documents in the corpus.  Less frequent words are viewed as having more predictive power between classes.

Another option is using stop words to get rid of the most common words that don't add any information to the text.  I also experimented with taking the top 10 words that appeared in each subreddit, and the top 10 words that had the highest coefficients, and adding them to the stop words list and re-running the modeling.  This proved to have little effect on my results, less than 0.1% each time.

Logistic Regression is the tried and true workhorse of the modeling world. (Fact not checked.)  It is fast and easy to implement, and has a higher interpretability than other models.
Indeed, fitting and exploring logistic regression modeling was quick and easy.

Naive Bayes assumes that all features are independent of each others, which is indeed a naive assumption.  The category with the highest probability will be chosen.  These models are known to be particularly useful for Natural Language Processing.
They are fast and accurate, and tend to have generalized results over varied applications.  

### Evaluation

Turns out that the subreddits chosen, *'science''* and *'technology'* weren't the best candidates to see how my model performed.  
In short, there were many defining features (words) in each subreddit that easily identified it from the other.  Most of my models were at 99% accuracy, sensitivity, and specificity.  My most recent logistic regression model was actually at 100%.  
While not complaing about this fact, I do believe that using subreddits that are more closely related would test the true capabilites of the models and the effects of tuning parameters on them.  

### Conclusion / Summary / Recommendations

Even though almost every model I tried peformed very well, I believe logistic regression would be my choice for this problem at this point.  
Logistic regression is easier to understand, runs fast, performs well over a wide variety of scenarios, and as I found, has lots of documentation and sources to learn about it and understand the model and modeling process better.  

### Object Oriented Programming

By far the largest part of this project wasn't actually the project itself.  I found myself copy-pasting so many models in my notebook that became very cumbersome, tedious, error-prone, and almost manageable.  Obviously, if I decided to change one part of the model or modeling process in one model, I had to go back and manually edit every copy of the same code.  
My OOP journey for this project started with making a few simple functions to more easily make the (lol) word clouds and then the ROC curve.  Once I had these functions working properly, I then decided to make one GIGANTIC function that went from scraping the subreddits all the way to modeling.  This completed a few times but obviously I didn't want to re-scrape every time I wanted to test a new model.  
The next problem I faced was that I had some functions taking in variables, but not returning them, or creating generic internal variables which could not be returned without overwriting them the next time.  All of these problems OOP is designed to alleviate.  Go figure.  
Creating a class to hold all of the methods and attributes about the object, from the scraped dataframe to the model and model results, proved to be extremely useful and will most likely be used in the future.  If nothing else, this is a great example and template for being able to run through something many times with minor changes but without having to write code each time for every instance.  

### Learning Points

 - I feel I'm much more confident in selecting a model in general, and then selecting tuning parameters and how they will affect the various aspects of the model.  
 - I learned the true value of a properly set up pipeline and gridsearch, and how much it can improve a base model with a minimal amount of effort (maybe not minimal time)
 - OOP concepts I believe are extremely important to the overall process of building models, applications, programs, or anything that makes life easier using python.
 - I am now much more familiar with using objects, methods, and attributes of other pre-built python modules and libraries.
 - Web scraping can be a rich source of data.  Implementing programmatic methods to gather large amounts of data will absolutely be a key skill to have in the future.

### Moving Forward

[ ] Finish the functions / methods for the class - debug the ones not working  
[ ] do_it_all method that includes all functions  
[ ] Research how to make multiple models with copy, variable assignment, etc.  
[ ] passing arguments and variables between functions  
[ ] Test out saving the class and function block as a .py file and import it  
[ ] Think about / research a good method of storing / editing the preprocessor and model dictionaries.  
[ ] Try: Except: blocks for catching errors of using the wrong preprocessors with the wrong models.  
[ ] Add more models to the mix!  
[ ] Use functions to print / display text and dataframes better.  
[ ] Add more scores to the scoring section, denoted below.  
[ ] UNDERSTAND MORE MATH AND STATISTICS  


**Questions to be able to answer about every model and in general**  
> Name and briefly explain several evaluation metrics that are useful for classification problems.  
>
>1. Accuracy - measures the percentage of the time you correctly classify samples: (true positive + true negative) / all samples
>2. Precision - measures the percentage of the predicted members that were correctly classified: true positives / (true positives + false positives)
>3. Recall - measures the percentage of true members that were correctly classified by the algorithm: true positives / (true positives + false negative)
>4. F1 - measurement that balances accuracy and precision (or you can think of it as balancing Type I and Type II error)
>5. AUC - describes the probability that a classifier will rank a randomly chosen positive instance higher than a randomly chosen negative one





