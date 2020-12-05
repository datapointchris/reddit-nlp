
<h1 align="center">
  Subreddit Classification Using NLP
  <br>
</h1>

<h4>Program for scraping subbreddits and storing them in various user-chosen ways, later loading data from multiple sources into a standard format to perform analysis and NLP classification.<br />
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
Load data from multiple sources into a standard format for analysis, visualization, or NLP.


## Features

* Scrapes a list of subreddits
* Saves data as CSV or SQLite to either local folder or S3
* Option to run scraper from command line with config file.
* Loading in of data in a standard format ready for modeling.
* NLP to classify subreddit by post title and description.
* Class and functions to run multiple models and compare results.
* Visualizations of most and least common words between subreddits.


## License

[MIT](https://tldrlegal.com/license/mit-license)