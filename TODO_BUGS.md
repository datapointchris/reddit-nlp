# To - Do
---

## scraping.ipynb

- [x]  Add multiple DBs to scrape save
- [x]  Create main function for CLI
- [x]  Figure out argparse
- [ ]  Do I actually need this file?



## scraping.py

- [x] Add logging
- [ ] unittest
- [x]  Set Linux Cron job for auto scrape


## scraper_config.json

- [x]  Simple format for only scraper config
- [x]  Update to include other config parameters, possibly preprocessor and model dictionaries
        - dictionary is in `grid_models.py`
- [x]  Most likely rename file to generic config file.


## ds_workflow.ipynb

- [x]  finish data loader
- [x]  create better functions using `05_Functions` as template
- [x]  Functions need to use combinatorics (?) to compare only two subreddits at a time for certain analyses
- [x]  This file should eventually become a `dataloader.py` file and be imported into DS workflows



## Project Files

- [x]  Create multiple DS workflow notebooks for use cases
- [ ]  Create and install multiple DBs to make functions, ummm function.
- [x]  Create logging file(s)
- [x] Take code out of `code` folder and restructure project


## Additions

- [ ]  Dump data into MongoDB also?
- [ ]  Find more sources of data
- [ ]  SQL Stats


## Problems / Issues / Lack of Knowledge

- [x] Functions for engineering / bare code for EDA and analysis?
- [x] Copy of a slice
- [ ] UNIT TESTING
- [x] .ipynb vs .py - Keep them both? `demo.ipynb`?
        - I think this is solved with using Sphinx for documentation instead of having a demo jupyter notebook for every .py file.
        - https://www.sphinx-doc.org/en/master/


# Bugs

## scraping.py

- [x] scraper returns less than 10 posts when sort is set to anything other than 'new'
        - fixed this with using for loop set to 40 pages, since limit is 1000 anyway
- [ ] 

