from pathlib import Path, PurePath

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / 'data'
LOGS_DIR = ROOT_DIR / 'logs'
TESTS_DIR = ROOT_DIR / 'tests'
IMAGES_DIR = ROOT_DIR / 'images'
MODEL_COMPARE_DIR = DATA_DIR / 'compare_df'
SCRAPED_SUBREDDITS_DIR = DATA_DIR / 'scraped_subreddits'

# print(ROOT_DIR)
# print(MODEL_COMPARE_DIR)
# mod = 'yes'
# print(f'{SCRAPED_SUBREDDITS_DIR}/{mod}')
# print(DATA_DIR / 'models')

# ========================= USER DEFINED ========================= #

# All paths should be relative to the ROOT_DIR
# MODEL_COMPARE_DIR = 'data/compare_df/*.csv'
