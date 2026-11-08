# Clockwise Order Classifier

This project trains a binary image classifier. It predicts whether the clockwise order of labels on a coloured loop is `ABC` (`+`) or `ACB` (`-`). The prediction comes from a trained CNN, not from a hand-written clockwise rule.

The dataset is arranged by fixed image ID, not random splitting:

- `training/`: IDs 0 through 139
- `validation/`: IDs 140 through 169
- `testing/`: IDs 170 through 199

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py --training-dir training --validation-dir validation --testing-dir testing --output-dir artifacts
```

This command reads the three pre-arranged folders, rejects duplicate image content across splits, trains a Logistic Regression baseline and the CNN, saves the selected CNN checkpoint, and writes metrics, a loss/F1 chart, and a test confusion matrix to `artifacts/`. The checkpoint records the best epoch and validation loss.

## Predict a new image

```bash
python inference.py path/to/new_figure.png --checkpoint artifacts/clockwise_cnn.pt
```

The inference script applies the checkpoint's saved greyscale normalization, prints the predicted class, prints confidence for that selected class, and prints the probability for `+`.
