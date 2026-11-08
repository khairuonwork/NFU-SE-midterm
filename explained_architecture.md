# ClockwiseCNN Architecture Explained

## What the model learns

`ClockwiseCNN` is a binary image classifier. It learns to distinguish two clockwise label orders:

| Clockwise order | Symbol | Binary target |
|---|---:|---:|
| `A → B → C` | `+` | `1` |
| `A → C → B` | `-` | `0` |

The mapping is explicit in [labels.py](labels.py):

```python
CLOCKWISE_ORDER_TO_CLASS_ID = {
    "ABC": 1,
    "ACB": 0,
}
```

This code defines the answer used while training. It does not manually solve an image by tracing the curve. The CNN receives grayscale pixels and learns the visual evidence for each target from the examples in `training/`.

## From PNG file to tensor

Each source PNG is 400 × 400 pixels. Before the image enters the CNN, `train.py` performs these steps:

1. Convert the image to grayscale. The curve colour changes between images, but colour does not define the class.
2. During training only, apply a small affine transformation: rotation up to 8 degrees, translation up to 4 percent, and scaling from 0.95 to 1.05.
3. Resize the image to 128 × 128 pixels.
4. Convert pixel values from 0 through 255 into a tensor from 0 through 1.
5. Normalize the tensor with mean and standard deviation calculated from `training/` only.

Horizontal and vertical flips are intentionally excluded. A reflection reverses clockwise order, so it can change the correct class.

After preprocessing, one batch has this shape:

```text
[batch size, channels, height, width]
[B,          1,        128,    128]
```

`B` is the number of images in the batch. The default batch size is 16.

## What `__init__` does

`__init__` builds the layers and creates their learnable parameters. At this stage, no image has moved through the network. PyTorch registers convolution kernels, BatchNorm parameters, linear weights, and biases inside the `ClockwiseCNN` object.

Image processing starts only when `forward(images)` is called:

```text
images
  │
  ▼
self.features
  │
  ▼
self.classifier
  │
  ▼
one logit per image
```

The model has 154,849 trainable parameters. AdamW updates those parameters through backpropagation during training.

## Feature extractor: `self.features`

`self.features` turns raw pixels into spatial features. Early layers detect small visual details. Later layers combine those details into higher-level positional patterns.

| Stage | Layer | Output shape | What it can represent |
|---|---|---:|---|
| Input | Grayscale image | `[B, 1, 128, 128]` | The curve, label circles, and letter strokes. |
| Block 1 | Conv2d 1→16, BatchNorm, ReLU | `[B, 16, 128, 128]` | Small edges, curved line segments, circle outlines, and letter strokes. |
| Block 1 | MaxPool2d(2) | `[B, 16, 64, 64]` | The strongest feature in each 2 × 2 area. |
| Block 2 | Conv2d 16→32, BatchNorm, ReLU | `[B, 32, 64, 64]` | Local combinations of labels and nearby curve segments. |
| Block 2 | MaxPool2d(2) | `[B, 32, 32, 32]` | A more compact view with a larger receptive field. |
| Block 3 | Conv2d 32→64, BatchNorm, ReLU | `[B, 64, 32, 32]` | Higher-level spatial relationships between areas of the figure. |
| Spatial pooling | AdaptiveAvgPool2d((4, 4)) | `[B, 64, 4, 4]` | A compact grid that retains coarse position. |

### Convolution

Each convolution uses a 3 × 3 kernel with `padding=1`. Padding preserves the height and width of a feature map during convolution. For example, the first convolution changes `[B, 1, 128, 128]` into `[B, 16, 128, 128]`.

The first kernels can learn simple structures such as a black letter stroke or a curved line. Later layers receive several feature maps at once, so they can combine simple structures into more meaningful local relationships.

The channel count increases from 1 to 16, then 32, then 64. This gives the network more distinct feature types as visual information becomes more complex.

### Batch normalization

`BatchNorm2d` normalizes activations for each channel during training. This keeps feature scales more stable and makes it easier for the optimizer to update weights.

### ReLU

`ReLU()` replaces negative values with zero and keeps positive values. Without nonlinear activations, a stack of convolution and linear layers would behave like one large linear transformation. ReLU lets the network learn nonlinear visual patterns.

### Max pooling

`MaxPool2d(2)` keeps the largest value in each 2 × 2 region. It reduces spatial resolution from 128 × 128 to 64 × 64, then to 32 × 32. The next layer can then use information from a wider part of the original image with less computation.

### Why the 4 × 4 grid matters

`AdaptiveAvgPool2d((4, 4))` changes every 32 × 32 feature map into a 4 × 4 grid. This preserves coarse locations such as upper-left, lower-right, and the areas between them.

That location information is important. A global average pool with output size 1 × 1 would tell the classifier that the image contains a curve and three labels, but it would discard much of the evidence about where B and C occur relative to A. The 4 × 4 grid lets the classifier compare evidence from different parts of the figure.

## Classifier: `self.classifier`

The feature extractor produces `[B, 64, 4, 4]`. The classifier converts that representation into one number per image.

| Layer | Output shape | Purpose |
|---|---:|---|
| `Flatten()` | `[B, 1024]` | Converts 64 × 4 × 4 features into one vector. |
| `Linear(1024, 128)` | `[B, 128]` | Combines features from different areas of the 4 × 4 grid. |
| `ReLU()` | `[B, 128]` | Adds a nonlinear decision step. |
| `Dropout(0.4)` | `[B, 128]` during training | Randomly disables about 40 percent of activations to reduce dependence on individual features. It is inactive during validation and inference. |
| `Linear(128, 1)` | `[B, 1]` | Produces one logit for binary classification. |

`Flatten()` produces 1024 values because `64 × 4 × 4 = 1024`. The first linear layer does not see raw pixels. It sees CNN features and their coarse locations.

## Logit, probability, and final class

The final layer produces a logit, not a probability:

```text
logit < 0  supports class -
logit = 0  gives probability(+) = 0.5
logit > 0  supports class +
```

During training, `BCEWithLogitsLoss` receives that logit and compares it with the target `1` for `+` or `0` for `-`. The loss combines sigmoid and binary cross-entropy in a numerically stable operation. For that reason, `model.py` does not contain an explicit `Sigmoid()` layer.

During inference, `inference.py` converts the logit into the probability of class `+`:

```python
positive_probability = torch.sigmoid(logit).item()
```

If the probability is at least 0.5, the prediction is `+ (clockwise ABC)`. Otherwise, the prediction is `- (clockwise ACB)`. Confidence for the negative class is `1 - positive_probability`.

## One-image flow

```text
PNG, 400 × 400
  → grayscale
  → resize to 128 × 128
  → normalize
  → Conv 1→16, BatchNorm, ReLU, MaxPool
  → Conv 16→32, BatchNorm, ReLU, MaxPool
  → Conv 32→64, BatchNorm, ReLU
  → adaptive average pooling to 4 × 4
  → flatten 1024 values
  → Linear 1024→128, ReLU, Dropout
  → Linear 128→1 logit
  → sigmoid
  → probability and predicted class
```

The architecture is compact enough for 140 training images while preserving spatial information needed for clockwise-order classification. The final result should still be interpreted with the held-out test metrics and the confusion matrix, especially when B and C are close together.
