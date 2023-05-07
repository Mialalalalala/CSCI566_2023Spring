# MIMIC-III_ICU_Readmission_Analysis

### Part I: Reconstruction of time series readmission dataset
You need to get all the .csv files from MIMIC-III database. We cannot provide the data here.
Please follow the steps to get the results:

```
1. python3 scripts/extract_subjects.py [PATH TO MIMIC-III CSVs] data/root/
2. python3 scripts/validate_events.py data/root/
3. python3 scripts/create_readmission.py data/root/
4. python3 scripts/extract_episodes_from_subjects.py data/root/
5. python3 scripts/split_train_and_test.py data/root/
6. python3 scripts/create_readmission_data.py data/root/ data/readmission/
```

## Citation
The codes for this part is adapted from this [paper](https://www.biorxiv.org/content/early/2018/08/06/385518/):

```
@article{lin2018analysis,
  title={Analysis and Prediction of Unplanned Intensive Care Unit Readmission using Recurrent Neural Networks with Long Short-Term Memory},
  author={Lin, Yu-Wei and Zhou, Yuqian and Faghri, Faraz and Shaw, Michael J and Campbell, Roy H},
  journal={bioRxiv},
  pages={385518},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}
```

### Part II: Baseline models and LSTM models
```
Baseline: CSCI566_project_baseline.ipynb
LSTM: LSTM_baseline+LDAM.py
```
