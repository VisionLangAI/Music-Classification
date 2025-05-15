
# Instrument Classification using DNN and SVM

This repository contains the full pipeline for a music instrument classification task using a Deep Neural Network (DNN) and Support Vector Machine (SVM). The project uses a 28-class instrument dataset and applies advanced acoustic features such as MFCC, STFT, ZCR, and Mel Spectrogram for classification.

## Project Structure

- `data/`: Contains raw and processed audio files.
- `preprocessing/`: Scripts for trimming, digitization, and noise removal.
- `features/`: Feature extraction scripts (MFCC, STFT, ZCR, RMS, etc.).
- `models/`: DNN and SVM training scripts.
- `evaluation/`: Evaluation metrics and performance reporting.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visualizations.

## Reproducibility

- Python version: 3.9
- Libraries: librosa, numpy, pandas, sklearn, tensorflow, matplotlib
- Hardware: Tested on NVIDIA Tesla T4 GPU, 16 GB RAM
- Training Time: ~30 minutes for 100 epochs on DNN

## How to Run

```bash
git clone https://github.com/yourusername/instrument-classification-dnn-svm.git
cd instrument-classification-dnn-svm
pip install -r requirements.txt
python preprocessing/preprocess_audio.py
python features/extract_features.py
python models/train_dnn.py
python models/train_svm.py
python evaluation/evaluate_models.py
```

## Citation

If you use this code or dataset, please cite the original dataset and our paper.

## License

MIT License
