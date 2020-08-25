#!/usr/bin/env python

"""Train this model with supervised layer-wise pretraining."""

# Core Library
import json
import logging
import os
import shutil
import sys

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)

# Make model with one hidden layer
logging.info("Make model with one hidden layer")
os.chdir("../baseline-1-c2")
if not os.path.isfile("model-1.json"):
    os.system("hwrt train")
shutil.copyfile("model-1.json", "../baseline-2-c2/model-1.json")

# Remove last layer
logging.info("Remove last layer")
with open("model-1.json") as f:
    model = json.load(f)
model["layers"].pop()

# Make new layer
os.chdir("../baseline-2-c2")
with open("model-1mod.json", "w") as f:
    json.dump(model, f)
os.system("detl make mlp 500:500:378 > layer2.json")
os.system("detl stack model-1mod.json layer2.json > model-2.json")
os.system("rm layer2.json")

logging.info("Train it")
os.system("hwrt train")
