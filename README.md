hwr-experiments
===============

Experiments for handwriting recognition


## Notice

This repository is mainly for internal usage. If you want to get the data,
please have a view at [write-math.com/data](http://write-math.com/data)


## Update an existing classifier

If you have new data, but don't want to change the classifier (hence: no
difference in preprocessing, features or model and no new symbols), you should
do the following:

1. `$ backup.py`:  30 minutes
2. Update `hwrt/misc/latex2writemathindex.csv`
3. Update preprocessing data source to new file which was just created by
   backup.py
4. `hwrt filter_dataset -s ~/GitHub/hwrt/hwrt/misc/symbols.yml -r ~/GitHub/hwr-experiments/raw-datasets/2015-11-14-17-06-handwriting_datasets-all-raw.pickle -d ~/GitHub/hwr-experiments/raw-datasets/2015-11-15-filtered-raw.pickle`
5. Delete files in preprocessing, features and model folder
6. `$ train.py`: 30 minutes (eventually multiple times for SLP)

## Add new symbols

See "update existing classifier", but with modifications:

1. Adjust the model files (number of output neurons)
2. Adjust the data source (look in hwr-experiments/raw-datasets)

## After new training

1. Run `$ test.py -n 3` (should be less than 5%)
