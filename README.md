# 李凱 / Daffa Khairul Ammar - F1151035
# SE Midterm

# Clockwise Order Classifier

This project trains a neural network to classify a figure from the clockwise order of its labels. It predicts one of two classes:

> **Appendix reference.** The complete, labelled source code is maintained in the [GitHub repository](https://github.com/khairuonwork/NFU-SE-midterm). This README is the project's user manual: it explains how to install the dependencies, train a model, reproduce the model comparison, and run the prediction system.

| Clockwise order | Class symbol | Binary target |
|---|---:|---:|
| `A → B → C` | `+` | `1` |
| `A → C → B` | `-` | `0` |

The model is a learned CNN. It receives image pixels, not the text sequence `ABC` or `ACB`, and learns the visual patterns associated with each training target.

## The class rule in code

The following mapping in [labels.py](labels.py) defines the target convention used for supervised training:

```python
CLOCKWISE_ORDER_TO_CLASS_ID = {
    "ABC": 1,  # A → B → C clockwise is positive (+)
    "ACB": 0,  # A → C → B clockwise is negative (-)
}
```

This mapping does not make the program deterministic. It does not inspect an image and manually trace the loop from A to B or C. Instead, it tells the training program which numeric answer belongs to each labelled image:

```text
0_+.png  →  +  →  target 1  →  clockwise ABC
1_-.png  →  -  →  target 0  →  clockwise ACB
```

During training, the CNN compares its output with those targets and adjusts its weights through backpropagation. At inference time, it sees only the image and produces a learned probability for class `+`.

## Dataset splits

The dataset has fixed splits. The program does not randomly split images.

| Folder | Image IDs | Purpose |
|---|---|---|
| `training/` | 0 to 139 | Updates the CNN weights. |
| `validation/` | 140 to 169 | Selects the best checkpoint and controls early stopping. |
| `testing/` | 170 to 199 | Provides the final held-out evaluation. |

The training script also checks SHA-256 hashes and stops if the same image content appears in more than one split.

## User manual

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Train a model

The interactive menu is the normal workflow for training and testing:

```bash
python program.py
```

The menu offers two actions:

1. **Test image**: choose a saved model from `artifacts/`, review its held-out test metrics, then choose a numbered image from `testing/`. The program prints the prediction, confidence, expected label, and whether that prediction is correct.
2. **Train new model**: enter a new model name. Training saves named artifacts, then the program exits. Run `program.py` again to select the newly trained model.

The test-image menu hides dataset filenames before selection because filename suffixes contain the true class label.

You can also train directly from the command line:

```bash
python train.py \
  --training-dir training \
  --validation-dir validation \
  --testing-dir testing \
  --output-dir artifacts \
  --model-name clockwise_cnn
```

The command trains two models:

- A Logistic Regression baseline on flattened 64 × 64 grayscale pixels.
- `ClockwiseCNN`, the selected convolutional neural network, on 128 × 128 grayscale images.

For the selected CNN checkpoint, the program saves these files under `artifacts/`:

```text
<model-name>.pt
<model-name>_metrics.json
<model-name>_training_curves.png
<model-name>_test_confusion_matrix.png
```

### 3. Recalculate the model comparison

Run the comparison script after training to generate a fresh metric table for Logistic Regression and a selected CNN checkpoint:

```bash
python classification_metrics.py --checkpoint artifacts/clockwise_cnn.pt
```

The script fits Logistic Regression using `training/` only, loads the specified CNN checkpoint without updating its weights, evaluates both models on all three fixed splits, prints the comparison table, and saves a JSON report beside the checkpoint. Logistic Regression is a baseline for measuring the value of CNN spatial features. The prediction system continues to use only the saved CNN checkpoint.

### 4. Launch the prediction system

For the guided interface, use `python program.py` and select **Test image**. For a direct prediction on a specified image, run:

```bash
python inference.py path/to/figure.png --checkpoint artifacts/clockwise_cnn.pt
```

The inference script restores the model checkpoint and its saved preprocessing values. It reports the predicted class, confidence for that class, and the probability of class `+`.

## Appendix: source-code guide

Do not paste every source file into the report. The complete code is available in the [GitHub repository](https://github.com/khairuonwork/NFU-SE-midterm), while this README provides the user manual required by the Appendix. The table below separates the training, evaluation, and inference components so a reader can locate each part quickly.

| File | Responsibility |
|---|---|
| [labels.py](labels.py) | Defines the `ABC → + → 1` and `ACB → - → 0` training convention. |
| [model.py](model.py) | Defines the `ClockwiseCNN` architecture. |
| [train.py](train.py) | Loads fixed splits, trains models, evaluates them, and saves artifacts. |
| [classification_metrics.py](classification_metrics.py) | Recalculates reproducible split metrics for the baseline and one CNN checkpoint. |
| [inference.py](inference.py) | Loads one CNN checkpoint and predicts one image. |
| [program.py](program.py) | Provides the interactive training and testing menu. |
| [explained_architecture.md](explained_architecture.md) | Explains how image pixels move through the CNN. |

Training records and additional figures are saved in `artifacts/` after training. These include the selected checkpoint, JSON metrics, training curves, and the test confusion matrix.
