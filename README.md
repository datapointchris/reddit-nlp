
<h1 align="center">
  Subreddit Classification Using NLP
  <br>
</h1>

<h4 align="center">Program for scraping subbreddits and storing them in various user-chosen ways, later loading data from multiple sources into a standard format to perform analysis and NLP classification.<br />
Saves and extracts data from multiple sources.<br />
Using configuration files and class definitions to seperate structure and function.</h4>

<p align="center">
	<a href="#description">Description</a> •
	<a href="#features">Features</a> •
	<a href="#future-features">Future Features</a> •
	<a href="#file-descriptions">File Descriptions</a> •
	<a href="#how-to-use">How To Use</a> •
	<a href="#requirements">Requirements</a> •
	<a href="#credits">Credits</a> •
	<a href="#license">License</a>
<br />
<br />
<img src='images/nlp.jpg' height=400>
</p>


## Description

#### Project Purpose:
Scrape subreddits with a simple script, allowing the user to easily configure the sort and save method across various methods and databases.
Create a data loader that can easily load data from multiple sources into a standard format for analysis, visualization, or NLP.
Import modules and classes across many files, use JSON configuration files to abstract the parameters from the main code.


## Features

* Scrapes a list of subreddits
* Saves data as CSV, SQLite, Postgres, MongoDB, MySQL, S3 (Framework but not fully functioning)
* Option to run scraper from command line with config file.
* Loading in of data in a standard format ready for modeling.
* NLP to classify subreddit by post title and description.
* Class and functions to run multiple models and compare results.
* Visualizations of most and least common words between subreddits.


## Future Features

* Add S3 option to save and load functions
* Advanced functions for running multiple models across selected subreddits
* Logging of actions
* Advanced Visualizations
* Automate the NLP / machine learning process.
* Config file for full project, able to run start to finish.
* Selection of models and preprocessors in simple config file to run on data.
* Web Interface with more options and prettier output.


## File Descriptions

`scraping.ipynb` - this file explains how the scraper works, with examples for running the scraper in Jupyter or the .py equivalent file on the command line.
`scraping.py` - scraper without interface, set and run
`databases.py` - handles all of the database classes for saving and extracting data (work in progress)
`ds_workflow.ipynb` - data science workflow from loading the data to running NLP and generating visualizations **(Current work in progress, messy file, apologies)**
`05_Functions.ipynb` - file with class and functions from original version of project to use as starter




## How To Use

Currently, only `scraping.ipynb`, `scraping.py`, and `ds_workflow.ipynb` are fully functioning.
Instructions to run the scraper for both the Jupyter and CLI are included in `scraping.ipynb`



## Requirements

Ever increasing!
[list of requirements once completed]


## Credits

- [Matt Brems - Histogram Plots - GA DSI  
- J Beightol - GA DSI - Word Count Plots - GA DSI  
- https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
- https://www.datacamp.com/community/tutorials/scraping-reddit-python-scrapy
- https://www.osrsbox.com/blog/2019/03/18/watercooler-scraping-an-entire-subreddit-2007scape/
- https://data-flair.training/blogs/python-switch-case/
- https://jaxenter.com/implement-switch-case-statement-python-138315.html
- https://www.journaldev.com/15642/python-switch-case


## License

[MIT](https://tldrlegal.com/license/mit-license)