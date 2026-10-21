# Predicting rifampicin and isoniazid resistance in *M. tuberculosis* from SNP profiles

Machine learning on pre-called genomic variants from the CRyPTIC data compendium.
Personal learning project (Master's level), 2026.

## Data
- CRyPTIC reuse table and per-isolate VCFs (EMBL-EBI FTP, release_june2022).
- 1,500 isolates sampled at random from the 9,395 that have binary rifampicin and isoniazid
  phenotypes, both with HIGH or MEDIUM phenotype quality.
- Variants were already called by the CRyPTIC pipeline. I did not align reads or call variants myself.
- Resistant fraction in my sample: 41.8% (rifampicin), 51.1% (isoniazid).

## Method
1. `01_download.py`: sample isolates and download VCFs.
2. `02_matrix.py`: build a 0/1 isolate x SNP matrix (PASS SNPs only; variants seen in
   <15 or >n-15 isolates dropped; 6,407 features).
3. `03_train.py`: logistic regression, random forest, XGBoost; stratified 5-fold CV.
4. `04_shap.py`: SHAP feature importance on XGBoost trained on all isolates.
5. `05_group_cv.py`: lineage-held-out CV. Isolates clustered (PCA + k-means, 8 clusters) as a
   lineage proxy, then GroupKFold (4 folds) so test clusters are unseen in training.

## Results
Stratified 5-fold CV (threshold 0.5 for sensitivity/specificity):

| Drug | Model | AUC | Sensitivity | Specificity |
|---|---|---|---|---|
| Rifampicin | Logistic regression | 0.956 | 0.872 | 0.935 |
| Rifampicin | Random forest | 0.932 | 0.793 | 0.921 |
| Rifampicin | XGBoost | 0.971 | 0.882 | 0.953 |
| Isoniazid | Logistic regression | 0.957 | 0.903 | 0.956 |
| Isoniazid | Random forest | 0.939 | 0.824 | 0.915 |
| Isoniazid | XGBoost | 0.961 | 0.920 | 0.967 |

Lineage-held-out CV (XGBoost): rifampicin AUC 0.960 (worst fold 0.923),
isoniazid AUC 0.949 (worst fold 0.897).

SHAP: the top feature for rifampicin is position 761155 (rpoB S450L) and for isoniazid
position 2155168 (katG S315T), i.e. the model recovers the best-known determinants.
Full top-50 lists are in `results/`.

## Limitations
- This largely rediscovers known mechanisms; it is not a discovery study.
- Labels are binary calls from MIC breakpoints (ECOFF), so they carry assay noise.
- SNPs only; indels and structural variants are ignored. Variants seen in fewer than 15
  isolates are dropped, so rare resistance mutations cannot be learned.
- Features are genomic positions, not annotated amino-acid changes. I did not run snpEff.
- Isoniazid and rifampicin resistance often co-occur (MDR), so SHAP importance for one drug
  includes mutations of the other (e.g. rpoB S450L ranks 3rd for isoniazid). This reflects
  correlation, not causation.
- Lineage clusters are a k-means proxy, not true phylogenetic lineages.
- Hyperparameters were not tuned, and results come from a single 1,500-isolate sample.

## Reproduce
```
conda create -n mtb python=3.11 -y && conda activate mtb
pip install pandas numpy scikit-learn xgboost shap matplotlib requests
python scripts/01_download.py   # set N and run
python scripts/02_matrix.py
python scripts/03_train.py
python scripts/04_shap.py
python scripts/05_group_cv.py
```

## Data citation
The CRyPTIC Consortium. A data compendium associating the genomes of 12,289 *Mycobacterium
tuberculosis* isolates with quantitative resistance phenotypes to 13 antibiotics.
PLoS Biology 20(8): e3001721 (2022).