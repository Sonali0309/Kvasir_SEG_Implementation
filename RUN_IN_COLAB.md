# Google Colab quick start

1. Upload/open `notebooks/Kvasir_SEG_Experiment.ipynb` in Colab.
2. Choose a GPU runtime if available.
3. Run all cells.
4. The notebook will create `/content/outputs/results_summary.csv` with measured values from the two trained models.

The notebook is self-contained and downloads the official Kvasir-SEG ZIP. If the official archive layout changes, inspect `/content/kvasir_data/raw` and adjust the image/mask folder detection in `src/train.py`.
