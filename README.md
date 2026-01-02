# CNN-Digit-Classifier (MNIST-as-JPG)

Train a small **Keras CNN** to classify handwritten digits **0–9** from **grayscale JPGs on disk** (MNIST-as-JPG).

### Quick glance (sample inputs)

| | | |
|---|---|---|
| ![Sample digit](images/sample_4.jpg) | ![Sample digit](images/sample_2.jpg) | ![Sample digit](images/sample_8.jpg) |

## Dataset

- **Kaggle**: `https://www.kaggle.com/datasets/scolianni/mnistasjpg`
- **Expected folder layout** (what the notebook assumes conceptually):

```text
DATA_DIR/
  0/*.jpg
  1/*.jpg
  ...
  9/*.jpg
```

## Notebook: step-by-step workflow

All steps below map directly to the code + markdown narrative in `cnn-digit-classifier-mnist-jpg.ipynb`.

### Kaggle notebook (this project, ready to run)

- **Kaggle code**: `https://www.kaggle.com/code/mehrabgheibi/cnn-digit-classifier-accuracy-99`

### 1) Image preview (sanity check)

The notebook starts by displaying a few sample images to confirm they load correctly and look like digits.

### 2) Data pipeline (`tf.data`)

For each image path:
- **Read** JPG from disk
- **Decode** as grayscale (`channels=1`)
- **Resize** to **28×28**
- **Rescale** pixel values to **[0, 1]**
- **One-hot encode** labels for 10 classes (0–9)

### 3) Stratified train/val/test split

The dataset is split **per digit folder** to keep class balance:
- **70%** train
- **15%** validation
- **15%** test

Then each split is shuffled independently, and the code checks that the splits are disjoint.

### 4) Data augmentation (train only)

To improve robustness, training batches are augmented with:
- small **rotation**
- small **translation**
- small **zoom**

Validation/test data are **not augmented**.

## Model

A compact CNN for 28×28×1 inputs:
- `Conv2D(32)` → `MaxPool`
- `Conv2D(64)` → `MaxPool`
- `Flatten` → `Dense(128)` → `Dropout(0.3)`
- `Dense(10, softmax)`

### Model summary

![Model summary](images/model_sequential_1.png)

## Training

Compiled with **Adam** + **categorical cross-entropy**, tracking **accuracy**.

The notebook uses these callbacks:
- **ModelCheckpoint**: save best model by `val_accuracy`
- **EarlyStopping**: stop when `val_accuracy` stops improving
- **ReduceLROnPlateau**: lower LR when `val_loss` plateaus

### Train vs validation accuracy

![Train vs validation accuracy](images/train_val_plot.png)

## Evaluation (held-out test set)

The saved best model is evaluated on the held-out test set (no augmentation). Results are visualized with a confusion matrix.

### Confusion matrix (test set)

![Confusion matrix](images/test_confusion_matrix.png)

## How to run

### Option A: Run on Kaggle (recommended)

1. Create a Kaggle notebook.
2. Add the dataset: `https://www.kaggle.com/datasets/scolianni/mnistasjpg`
3. Upload `cnn-digit-classifier-mnist-jpg.ipynb` (or copy its cells).
4. Ensure `DATA_DIR` points to the dataset folder containing the digit subfolders `0..9` (the notebook uses a Kaggle path by default).

### Option B: Run locally

1. Download the Kaggle dataset and arrange it as:

```text
DATA_DIR/0..9/*.jpg
```

2. Open and run `cnn-digit-classifier-mnist-jpg.ipynb`.
3. Update the notebook’s `DATA_DIR` to your local path.

### Minimal dependencies

This repo is notebook-based. You’ll need:
- Python 3.x
- TensorFlow (includes Keras)
- NumPy
- Matplotlib
