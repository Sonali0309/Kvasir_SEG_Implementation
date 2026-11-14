# Kvasir-SEG Data Contract

- Dataset: Kvasir-SEG, 1,000 gastrointestinal polyp images and pixel masks.
- Class IDs: 0 = background; 1 = polyp foreground.
- Masks: grayscale images converted to binary with threshold > 127 (the source masks may be JPEG-compressed).
- Pairing: image and mask are paired by identical filename stem.
- Split: deterministic shuffled 80/10/10 train/validation/test using seed 42.
- Input: RGB 256 x 256, bilinear resize.
- Target: one-channel binary mask 256 x 256, nearest-neighbour resize.
- Normalisation: image pixels divided by 255.
- Augmentation: random horizontal flip on training only.
- No ignore class is defined for this dataset; all binary mask pixels are evaluated.

Official source: https://datasets.simula.no/kvasir-seg/
