# PokemonTypesCNN

A Convolutional Neural Network (CNN) that classifies Pokemon types based on Pokemon images.

We use the datasets provided from Kaggle:
* [7,000 Labeled Pokemon](https://www.kaggle.com/lantian773030/pokemonclassification)
* [Pokemon Image Dataset](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)

We modified the `pokemon.csv` file from `Pokemon Image Dataset` so that it includes the `Alalon
Sandslash` Pokemon, as well as renamed the row `mr-mime` to `mrmime`.

## Setup

1. Install `python 3.8.8`

2. Install python dependencies in a virtual environment.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirementx.txt
```

3. Install [ImageMagick](https://imagemagick.org/index.php)
```bash
sudo apt install imagemagick
```

## Prepare Data

1. Download the [7,000 Labeled Pokemon](https://www.kaggle.com/lantian773030/pokemonclassification)
  dataset which will be called `archive.zip`
2. Extract `archive.zip` into the `PokemonTypesCNN` project root directory which will create a
  `PokemonData` folder with subfolders containing the Pokemon images sorted by their name
3. Run `python MovePokemonImagesIntoTypesFolder.py` which will create a `dataset` folder with
  subfolders containing the Pokemon images sorted by their types
4. Run `./convertDatasetToJPG.sh` which will recursively convert all the images found in the
  `dataset` folder to `.jpg` since some of them are not in the `.jpg` file format
5. Run `python TrainTestDatasetGenerator.py` which will create a `train_dataset` and `test_dataset`
  folder of an 85-15 train/test split from the `dataset` folder

## Train the Model
Run `python TrainModel.py` which will save the trained model as `TrainModel_trained` and create a
`TrainModel_results` folder containing test results and accuracy/loss graphs.

## Generate Confusion Matrix Heatmap From Trained Model
After training the model, you can run `python confusion_matrix.py` which will generate a confusion
matrix heatmap saved as `confusion_matrix_heatmap.png`

## Using Our Trained Model
We have trained and saved our model with an NVIDIA RTX 2080 Ti under the `saved_results` folder. The
model is saved as `TrainModel_trained`. We have also attached a `TestModel.py` that loads the model
and tests the performance against the test set. You will need to copy `test_dataset` into this
folder.

## Report
We have attached our report to the repository with our findings and discussion saved as
[report.pdf](report.pdf) in the project root directory.

### Notes
The `saved_results` folder contains the trained architecture and weights of our best performing
model. The directory `test_dataset` contains a random sample of 15% of the images from our original
dataset. It contains one subdirectory for each Pokemon type. Each subdirectory contains images of
Pokemon of that type. For example, all Pokemon in `test_dataset/Water` are `Water` types. Thus, the
subdirectory's name is the label for all the images it contains. The python script `TestModel.py`
contains code for loading the saved model and evaluating it on `test_dataset`, outputting accuracy
and loss. It requires an environment with the modules listed under the `Setup` section. Also make
sure to follow the `Prepare Data` section above.

```bash
# NOTE: test_dataset should be generated with RANDOM_STATE=42 when running
# the TrainTestDatasetGenerator.py script

cp -r test_dataset saved_results  # Move test_dataset into saved_results folder
cd saved_results
python TestModel.py
```

You should see an accuracy of about 78.9%. The reason this is lower than the accuracy of 83%
mentioned in our report is that we used a random train/testing split each time we trained our
models, to prevent overfitting to our testing set, which can happen if we keep reusing the same
testing set and make changes to our model based on its results. However, we didn't save those sets
(they only existed during runtime). So when we created a `test_dataset` directory after, it was
different from the one that gave an accuracy of 83%. We also made sure that `TrainModel_trained` was
trained on a training set which doesn't overlap with `test_dataset`.
