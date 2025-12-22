PROFILE_TABLE_SAMPLE_LIMIT = 3  # Sample up to three tables for data comparison
PROFILE_MIN_ROWS = 10  # Require at least X rows to get sufficient data
PROFILE_SCORE_THRESHOLD = 0.70  # If table score drops below 0.70, the variant cannot be trusted; removed from running (raised from 0.60 to improve data quality)

# Feature toggles / flags
FLAG_DISSECT_ALL = False  # option to use sqlite_dissect on all corrupt DBs, even those without exemplar matches
