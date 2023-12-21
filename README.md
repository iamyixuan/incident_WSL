
# Weak Supervised Traffic Incident Detection 

## Overview
This repository supports the research presented in the paper, "A Data-Centric Weak Supervised Learning for Highway Traffic Incident Detection", available at [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0001457522002147). It includes scripts for data preprocessing using weak supervised learning and LSTM model training.

### Background
Traffic data often contain noise due to manual reporting and labeling inaccuracies. This project leverages weak supervised learning, utilizing a set of labeling functions, to enhance data quality. Improved data quality subsequently boosts the performance of supervised learning models in detecting traffic incidents.

### Data
The raw and processed datasets used in this study can be downloaded [here](https://anl.box.com/s/6g347oehx6otdu4g20njnr2gh6oywopq).

## Usage Instructions

### Data Preprocessing
To apply weak supervision for data labeling, run:
```python
python data_prcss/snorkel_labeling.py
```
**Note:** For processing raw data, additional preprocessing steps are required. These steps are detailed in the paper and involve the use of other scripts in the `data_prcss` directory.

### Model Training
- **LSTM Classifier**: To train the LSTM classifier using the enhanced data, execute:
  ```python
  python spvsd_models/lstmClassifier.py
  ```

- **Ensemble Creation**: For uncertainty quantification, create a deep ensemble model with:
  ```python
  python spvsd_models/deepEnsemble.py
  ```

## Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{sun2022data,
  title={A data-centric weak supervised learning for highway traffic incident detection},
  author={Sun, Yixuan and Mallick, Tanwi and Balaprakash, Prasanna and Macfarlane, Jane},
  journal={Accident Analysis & Prevention},
  volume={176},
  pages={106779},
  year={2022},
  publisher={Elsevier}
}
```
