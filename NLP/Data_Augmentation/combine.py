import os
import glob
import pandas as pd
extension = 'csv'
all_filenames = [i for i in glob.glob('data_augmented_folder/*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ], ignore_index=True)
combined_csv = combined_csv.drop("Unnamed: 0", axis=1)
print(combined_csv.columns)

#export to csv
combined_csv.to_csv( "combined_csv.csv", index=True, encoding='utf-8')