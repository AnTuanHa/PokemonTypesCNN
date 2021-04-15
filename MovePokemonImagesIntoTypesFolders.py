import csv
import sys

from pathlib import Path

IMAGES_FOLDER_NAME = "PokemonData"
TYPES_CSV_FILE_NAME = "pokemon.csv"
DATASET_FOLDER_NAME = "dataset"

if not Path(IMAGES_FOLDER_NAME).exists():
    print(f"'{IMAGES_FOLDER_NAME}' folder does not exist!")
    print("Please download from: https://www.kaggle.com/lantian773030/pokemonclassification")
    print("Exiting...")
    sys.exit()

# Dictionary to hold
# Key: Pokemon name
# Value: Type1
pokemon_types = {}
with open(TYPES_CSV_FILE_NAME, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        # Each row is a list of ["POKEMON_NAME", "TYPE_1", "TYPE_2"]
        # Note: TYPE_2 may be null.
        # This does not matter though since we are only concerned with their primary type
        pokemon_types[row[0]] = row[1]

dataset_folder = Path(DATASET_FOLDER_NAME)
dataset_folder.mkdir(parents=True, exist_ok=True)
for entry in Path(IMAGES_FOLDER_NAME).iterdir():
    pokemon_name = entry.name.replace(" ", "").lower()
    if pokemon_name in pokemon_types:
        type_folder = Path(dataset_folder / pokemon_types[pokemon_name])
        type_folder.mkdir(parents=True, exist_ok=True)

        for image_file in Path(entry.resolve()).glob("*.*"):
            image_file.rename(type_folder / image_file.name)
