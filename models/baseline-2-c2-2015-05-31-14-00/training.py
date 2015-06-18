#!/usr/bin/env python

"""Train this model with supervised layer-wise pretraining."""

import os
import shutil
import json

import logging
import sys

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

# Make model with one hidden layer
logging.info("Make model with one hidden layer")
os.chdir("../baseline-1-c2-2015-05-31-14-00")
if not os.path.isfile("model-1.json"):
    os.system("train.py")
shutil.copyfile("model-1.json", "../baseline-2-c2-2015-05-31-14-00/model-1.json")
os.chdir("../baseline-2-c2-2015-05-31-14-00")

# Remove last layer
logging.info("Remove last layer")
with open("model-1.json") as f:
    model = json.load(f)
model['layers'].pop()
with open("model-1.json", "w") as f:
    json.dump(model, f)

# Make new layer
os.system("detl make mlp 500:500:377 > layer2.json")
os.system("detl stack model-1.json layer2.json > model-2.json")
os.system("rm layer2.json")

logging.info("Train it")
os.system("train.py")
