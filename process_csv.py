import pandas as pd 
import numpy as np

import os

in_dir = "0229_Experiment/csv/"
out_dir = "0229_Experiment/processed_csv/"

# add the composer column to the data
# files is an array, sorted by composer 
def process_csv(files):
	for idx, file in enumerate(files):
		if not os.path.isfile(in_dir + file):
			continue
		a = pd.read_csv(in_dir + file)

		row_count, col_count = a.shape

		# Glue back to originaal data
		a.insert(1, "Composer", pd.DataFrame((idx+1) * np.ones((row_count, 1))))

		a.to_csv(out_dir + file)

composers = ["dandrieu", "soler",
			 "dvorak", "schumann",
			 "buxehude", "faure",
			 "scriabin", "byrd",
			 "shostakovich", "brahms",
			 "chopin", "debussy",
			 "schubert", "alkan",
			 "handel", "mozart",
			 "haydn", "beethoven",
			 "scarlatti", "bach"]

files = [c + "_all.csv" for c in composers]

process_csv(files)
