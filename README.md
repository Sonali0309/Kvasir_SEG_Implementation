# Kvasir-SEG Binary Segmentation Experiment

A reproducible controlled comparison between:
1. Baseline U-Net (bottleneck dilation = 1)
2. Dilated U-Net (bottleneck dilation = 2)

Dataset: Kvasir-SEG (1,000 gastrointestinal polyp images with corresponding masks).
Official dataset page: https://datasets.simula.no/kvasir-seg/
Official archive: https://datasets.simula.no/downloads/kvasir-seg.zip

## Run in Google Colab
1. Open `notebooks/Kvasir_SEG_Experiment.ipynb` in Google Colab.
2. Select Runtime > Change runtime type > T4 GPU (if available).
3. Run all cells from top to bottom.
4. The notebook downloads the official archive, creates a reproducible 80/10/10 split (seed 42), trains both models, evaluates them, measures latency, and saves outputs in `outputs/`.
5. Download the generated `results_summary.csv`, plots, predictions and checkpoints for your final evidence.

## Data contract
- Class 0: background.
- Class 1: polyp foreground.
- Source masks are grayscale/JPEG in the official archive; convert to binary using threshold > 127.
- Images: RGB, resized to 256 x 256 with bilinear interpolation.
- Masks: single channel, resized with nearest-neighbour interpolation.
- Split: 80% train, 10% validation, 10% test, deterministic random seed 42.
- Augmentation: random horizontal flip on training data only.
- No pseudo-ground-truth masks are generated.

## Models and training
- U-Net encoder channels: 32, 64, 128, 256; bottleneck: 512.
- Two 3x3 convolutions per block, BatchNorm + ReLU.
- MaxPool downsampling, transposed-convolution upsampling and skip concatenation.
- Baseline bottleneck dilation = 1; controlled variant = 2.
- Adam, learning rate 1e-3, batch size 8, max 20 epochs.
- BCEWithLogitsLoss, early stopping patience 5 based on validation Dice.
- Fixed seed 42.

## Metrics
Foreground IoU, background IoU, mean IoU, Dice, precision, recall, trainable parameters and inference latency (ms/image, batch size 1). Latency is measured on the active runtime device and printed with the device name.

## Important
The Word reports in `reports/` contain an example comparison table to show the required interpretation format. The values marked as illustrative are not experiment measurements. Replace them with the notebook's `outputs/results_summary.csv` before submitting empirical claims.

## Citation
Jha, D. et al. (2020). Kvasir-SEG: A Segmented Polyp Dataset. In: MultiMedia Modeling (MMM 2020), pp. 451–462. https://doi.org/10.1007/978-3-030-37734-2_37

Dataset terms: research and educational use; commercial use requires prior written permission. Cite the original dataset paper in publications/reports.
