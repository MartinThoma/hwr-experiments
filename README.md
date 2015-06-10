hwr-experiments
===============

Experiments for handwriting recognition

## Update an existing classifier

If you have new data, but don't want to change the classifier (hence: no
difference in preprocessing, features or model and no new symbols), you should
do the following:

1. `$ backup.py`:  30 minutes
2. Update preprocessing data source to new file which was just created by
   backup.py
3. Delete files in preprocessing, features and model folder
4. `$ train.py`: 30 minutes (eventually multiple times for SLP)

## Add new symbols

See "update existing classifier", but with modifications:

1. adjust the model files (number of output neurons)
2. Adjust the data source (look in hwr-experiments/raw-datasets)

## After new training

1. Run `$ test.py -n 3` (should be less than 5%)
