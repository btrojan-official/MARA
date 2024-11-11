import os
import shutil

input_path = "data/imdb_mlh"
output_path = "data/MARA_imdb_mlh"

os.makedirs(output_path, exist_ok=True)
shutil.copy(os.path.join(input_path, "features1000.txt"), os.path.join(output_path))
shutil.copy(os.path.join(input_path, "positives.txt"), os.path.join(output_path))
shutil.copy(os.path.join(input_path, "classes.txt"), os.path.join(output_path))