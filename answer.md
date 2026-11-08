# Clockwise Order Image Classification Report

## 1 Project Motivation and Problem Statement

This project is a supervised binary image-classification task. Its input is a 400 by 400 PNG containing a coloured closed curve and labels A, B, and C. Its output is one of two classes: `+` when the labels follow clockwise order `A → B → C`, and `-` when they follow `A → C → B`.

The training convention is explicit: `ABC` maps to target `1` and `ACB` maps to target `0`. This convention is stored in `labels.py`. It defines the correct answer for each labelled training image, but it is not a hand-written image-reading rule. The CNN must learn the visual relationship between pixels and those targets.

The practical purpose is to build a prediction system that learns a visual spatial-order rule from labelled examples. The final system accepts a previously unseen image and returns a class plus a confidence score. It does not use a hand-written geometric rule at inference time.

## 2 Dataset Description

The teacher supplied 200 labelled PNG images in `figures/`. The numeric prefix is only an identifier. The filename suffix provides the supervised target: `_+.png` is class `+`, target `1`, and clockwise order `ABC`; `_-.png` is class `-`, target `0`, and clockwise order `ACB`.

| Item | Value |
|---|---:|
| Total images | 200 |
| Image size | 400 × 400 pixels |
| Positive class `+` | 104 |
| Negative class `-` | 96 |
| Missing or unreadable images | 0 |

The class difference is eight samples, so the original dataset is close to balanced. The files are separated by fixed numeric ID before training. No random train-validation-test split is created by the program.

| Split | Images | `+` | `-` |
|---|---:|---:|---:|
| Train, IDs 0–139 | 140 | 67 | 73 |
| Validation, IDs 140–169 | 30 | 18 | 12 |
| Test, IDs 170–199 | 30 | 19 | 11 |

The test split is held out from gradient updates and is used only for final evaluation after selecting the saved CNN checkpoint by validation loss.

## 3 Data Preprocessing Pipeline

The CNN preprocessing pipeline is: load PNG, convert to greyscale, apply a small training-only affine transformation, resize to 128 × 128, convert to a tensor, then normalize. Greyscale is used because the curve colour varies but does not define the target class. The normalization values are calculated from the 140 training images only: mean 0.99210 and standard deviation 0.05817.

Training augmentation uses rotation in the range -8 to +8 degrees, translation up to 4 percent, and scale from 0.95 to 1.05. Horizontal and vertical flips are excluded because either reflection reverses clockwise order and would change the label. Validation, test, and inference images use only resize and the saved normalization values.

The split happens before normalization statistics are calculated. Therefore, no pixel statistics from validation or test images enter training preprocessing. Before training starts, the program calculates a SHA-256 hash for every image and stops if the same image content exists in more than one split folder.

## 4 Model Architecture and Algorithm Choice

Two models are compared.

1. **Baseline: Logistic Regression.** It receives a flattened 64 × 64 grayscale image. This establishes how well a linear classifier can use raw pixels without convolutional spatial features.
2. **Main model: ClockwiseCNN.** It receives one 128 × 128 greyscale channel and contains three convolution blocks:
   - Conv2D 1→16, 3 × 3 kernel, BatchNorm, ReLU, MaxPool 2 × 2
   - Conv2D 16→32, 3 × 3 kernel, BatchNorm, ReLU, MaxPool 2 × 2
   - Conv2D 32→64, 3 × 3 kernel, BatchNorm, ReLU
   - Adaptive average pooling to a 4 × 4 spatial grid
   - Flatten, Linear 1024→128, ReLU, Dropout 0.40, Linear 128→1

The 4 × 4 feature grid is important. A 1 × 1 global average would discard the positional information needed to distinguish the clockwise order of A, B, and C.

The output is one logit, so the loss is `BCEWithLogitsLoss`. It combines sigmoid and binary cross entropy in a numerically stable form. The optimizer is AdamW with learning rate 0.001 and weight decay 0.0001. Batch size is 16, the maximum number of epochs is 100, and validation-loss early stopping is configured with patience 20. Hyperparameters were manually chosen for this small dataset; the checkpoint with the lowest validation loss is saved.

## 5 Experimental Design and Training Process

The experiment ran with Python 3.11.15, PyTorch 2.15.0, torchvision 0.29.2, scikit-learn 1.9.1, and the PyTorch MPS device backend. The random seed 42 controls augmentation and DataLoader order only; it does not change membership of any split. At every epoch, the script records training loss, validation loss, training F1, and validation F1. The selected checkpoint was saved at epoch 93 with validation loss 0.0341. Each training run saves a named checkpoint, metrics JSON file, loss/F1 curve, and test confusion matrix under `artifacts/`.

Metrics are accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix. F1 is included alongside accuracy even though the classes are nearly balanced.

## 6 Experimental Results and Analysis

| Model | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 0.986 | 1.000 | 0.970 | 0.985 | 1.000 |
| Logistic Regression | Validation | 0.567 | 0.692 | 0.500 | 0.581 | 0.630 |
| Logistic Regression | Test | 0.633 | 0.750 | 0.632 | 0.686 | 0.632 |
| ClockwiseCNN | Train | 0.986 | 1.000 | 0.970 | 0.985 | 1.000 |
| ClockwiseCNN | Validation | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ClockwiseCNN | Test | 0.933 | 1.000 | 0.895 | 0.944 | 0.986 |

The CNN is the selected model because it substantially exceeds the linear baseline on validation and test data. The CNN's training and validation curves both improve during training. The small train-validation gap at the selected checkpoint does not indicate severe overfitting, although the dataset is too small to claim broad generalization.

The test confusion matrix is:

| True class | Predicted `-` | Predicted `+` |
|---|---:|---:|
| `-` | 11 | 0 |
| `+` | 2 | 17 |

The two test errors are `177_+.png` and `192_+.png`, both positive images predicted negative with positive probabilities 0.365 and 0.032. In both images, B and C are close together, which makes the visual distinction between the two possible clockwise sequences less clear at the resized resolution. The model makes no false-positive prediction on this test split.

## 7 Prediction System

The program gateway, `program.py`, first asks the user to choose either testing or training. In the testing sequence it lists the `.pt` checkpoints in the fixed `artifacts/` directory, asks the user to choose one model, then displays the saved held-out test metrics: accuracy, precision, recall, F1-score, and ROC-AUC. It next lists the PNG images in the fixed `testing/` directory for a second numbered selection. The user does not type an image path, and the menu hides filename labels before selection because the filename suffix contains the true class. The response shows the prediction, confidence, expected label, and whether that individual prediction is correct. In the training sequence it asks for a new model name and saves the result as `artifacts/<model-name>.pt`; it then exits so the user can run the gateway again and choose that model. `inference.py` performs the actual prediction: it loads the selected checkpoint, reads the saved normalization mean and standard deviation, applies the same grayscale preprocessing, runs the CNN, and prints the predicted label with the probability of `+`.

Example commands:

```bash
python program.py

# Optional direct commands
python train.py --training-dir training --validation-dir validation --testing-dir testing --output-dir artifacts --model-name clockwise_cnn
python inference.py testing/170_+.png --checkpoint artifacts/clockwise_cnn.pt
```

For `170_+.png`, the saved model returned `+ (clockwise ABC)` with confidence 0.668. For `174_-.png`, it returned `- (clockwise ACB)` with confidence 0.860.

The inference system is intended for images that match the training format: a white background, one coloured closed curve, and readable A/B/C labels at a similar scale. Images with extra text, missing labels, multiple curves, extreme rotations, or a substantially different layout may produce unstable confidence scores.

## 8 Discussion Limitations and Future Improvements

The main limitation is dataset size. There are only 200 synthetic-looking images, so one split can produce optimistic or pessimistic estimates. The CNN also depends on the consistent visual style of the labels and the curve. A new font, noisy background, occluded label, or changed image scale could reduce accuracy.

Future work should collect more diverse labelled samples, evaluate repeated stratified splits or cross-validation, tune learning rate and dropout systematically, and test stronger transfer-learning models. A separate held-out dataset with changed colours, fonts, and curve shapes would measure real generalization more convincingly.

## 9 Conclusion

This project built a supervised prediction system for clockwise-order image classification. It uses an explicit, reproducible training pipeline and a separate inference script. On the fixed 30-image test folder, the selected CNN achieved 93.3 percent accuracy and 0.944 F1, compared with 63.3 percent accuracy and 0.686 F1 for the Logistic Regression baseline.

## 10 References

1. PyTorch documentation. `BCEWithLogitsLoss`, `AdamW`, and convolutional neural-network modules.
2. scikit-learn documentation. `LogisticRegression` and classification metrics.
3. torchvision documentation. Image transforms and affine augmentation.

## 11 Appendix

- `labels.py`: the shared `ABC → + → 1` and `ACB → - → 0` target convention.
- `model.py`: the CNN architecture.
- `train.py`: split, preprocessing, baseline, CNN training, evaluation, checkpointing, and figures.
- `program.py`: the interactive gateway for model selection, training, and testing.
- `inference.py`: standalone image prediction used by the gateway.
- `requirements.txt`: required Python packages.
