This model is based on `baseline-2-crohme`. Only another 500 hidden layer is added.

## How it was created

1. Copy `model-1.json` from `archive/models/baseline-1-crohme` to this folder.
2. Manually remove the last layer from `model-1.json`.
3. `$ nntoolkit make mlp 500:500:56 > layer2.json`.
4. `$ nntoolkit stack model-1.json layer2.json > model-2.json`.
5. `$ rm layer2.json`
