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