# Delivery Last-Mile Delay Intelligence Pipeline

A synthetic dispatch-time analytics exercise: estimate delivery hours, flag potential promise breaches, and turn those flags into actions using operating segments for the hubs. The data are synthetic; these are exercise results, not estimates of Delhivery's actual performance.

**Before submission:** `generate_data.py` was transcribed from screenshots. Its generated values and all numerical checks match the brief, but byte identity with the instructor's source cannot be established from an image. Replace this file with the original code copied verbatim from the assignment, rerun the commands below, and review the results. This project has not been published to GitHub. Read and understand the implementation, adapt the explanations into your own words, and follow your programme's rules on outside assistance.

## How to run

Use a Python environment compatible with the pinned packages; this build was executed with Python 3.12. Run these commands from the repository root, in order:

```bash
python -m pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
python verify_project.py
```

The scripts use only local inputs and run offline after dependencies are installed. `generate_data.py` prints `24 1224 48`. `verify_project.py` checks the brief's reference values, pass bands and output consistency. It also regenerates the source data and repeats scoring, checking byte equality. It does not prove byte identity of the generator source with the instructor's original.

`python package_model.py` regenerates `models/delay_model.pkl`. The committed pickle is reliable only with the scikit-learn version pinned in `requirements.txt`; regenerate it rather than loading it when installed versions differ. Only load model files from a trusted source. Packaging fits the same training rows used by Part C; it does not retrain on all historical data.

`outputs/monitoring_plan.csv` contains proposed future monitoring rules, not observed production results. It is a documented planning artifact; the training and scoring scripts do not modify it.

The repository contains the generator, its three data exports, the modelling table, shared preprocessing, each analysis script, the packaged model and supporting CSV files. CSV outputs use no pandas index column and use four decimal places for floating values. The correlation matrix retains its row names explicitly in its first column. No plots, screenshots, notebooks, slides or PDF files are needed in the submitted repository.

For GitHub, create a public repository with `main` as its default and only branch, and upload the extracted project contents directly into its root. Submit that repository link. Do not upload the ZIP archive itself as the project.

## Data findings

Part A removes 24 exact duplicate rows from 1224 raw records, leaving 1200 unique shipments. It normalizes service text, records missing payment mode as `Unknown`, parses INR amounts and timestamps, treats weights above the sentinel cutoff as missing, and fills the three specified numeric columns with their historical medians. The fill values are 314.95 km, 1.83 kg, and INR 1,121.50. These values are computed from historical inputs only and reused unchanged for new shipments.

There are 278 delayed shipments, with delay rate 0.2317. After the hub merge there are still 1200 rows and 0 missing model inputs or targets. Delay means `delivery_hours > promised_hours`; arriving exactly at the promise is on time.

| dimension | group | shipments | delayed | delay_rate | mean_hours |
| --- | --- | --- | --- | --- | --- |
| service_type | Economy | 315 | 27 | 0.0857 | 82.0931 |
| service_type | Express | 243 | 119 | 0.4897 | 36.8056 |
| service_type | Standard | 642 | 132 | 0.2056 | 52.2569 |
| hub_tier | 1 | 400 | 16 | 0.0400 | 48.5664 |
| hub_tier | 2 | 400 | 86 | 0.2150 | 58.8194 |
| hub_tier | 3 | 400 | 176 | 0.4400 | 63.4943 |
| distance_band | 0-200 | 274 | 6 | 0.0219 | 46.7559 |
| distance_band | 201-400 | 563 | 77 | 0.1368 | 55.9660 |
| distance_band | 401-600 | 275 | 120 | 0.4364 | 64.2660 |
| distance_band | 601+ | 88 | 75 | 0.8523 | 72.2608 |

Express has the highest delay rate, 0.4897, compared with Standard at 0.2056 and Economy at 0.0857. A shorter promise makes breach more likely; this does not mean Express has the longest absolute delivery time. Delay rates rise from 0.0400 for Tier 1 to 0.2150 for Tier 2 and 0.4400 for Tier 3. Across distance bands, rates rise from 0.0219 in the shortest band to 0.8523 in the longest. These are descriptive associations, not causal effects or case-mix-adjusted hub comparisons.

The three numeric columns most strongly correlated with delivery hours, ranked by absolute Pearson correlation, are `promised_hours` (+0.8372), `distance_km` (+0.4013), `n_stops` (+0.3247). These values come from `outputs/part_a_correlations.csv`; correlation alone does not establish causation.

Declared value has mean INR 1,533.0542 and median INR 1,121.5000, showing a right-skewed distribution. The natural-log feature has mean 7.0277 and median 7.0224, which are much closer; the transformation compresses high-value parcels. These calculations use the median-filled amount column, not only non-missing raw amounts. The delivery-hours IQR rule identifies 1 outlier; the exercise retains it rather than deleting valid outcomes.

## Model results — regression

The split is 840 training, 180 validation and 180 test rows. Only training rows fit the scaler and regressors. `part_b_checks.csv` records a training scaled maximum absolute mean of 0.0000 after CSV rounding, versus 0.2187 on the test rows. The scaler was fitted before transforming test inputs, using training rows alone.

| model | split | MAE | MSE | RMSE | R2 |
| --- | --- | --- | --- | --- | --- |
| mean_baseline | val | 15.7952 | 355.5204 | 18.8552 | -0.0212 |
| linear_regression | val | 3.9222 | 23.9235 | 4.8912 | 0.9313 |
| decision_tree | val | 5.0971 | 39.4338 | 6.2796 | 0.8867 |
| random_forest | val | 4.2882 | 30.3550 | 5.5095 | 0.9128 |
| gradient_boosting | val | 3.9166 | 24.1722 | 4.9165 | 0.9306 |
| linear_regression | test | 3.5671 | 20.4624 | 4.5235 | 0.9493 |

The mean baseline has validation R² -0.0212: predicting the training mean performs slightly worse than predicting the validation mean. Linear regression wins by validation RMSE, 4.8912 hours, narrowly ahead of gradient boosting at 4.9165. The final test RMSE is 4.5235 hours and test R² is 0.9493. Test results are reported after selection; they were not used to choose the model.

| max_depth | train_rmse | val_rmse |
| --- | --- | --- |
| 1 | 11.6334 | 12.1324 |
| 2 | 9.3687 | 10.4114 |
| 3 | 7.5295 | 8.7033 |
| 4 | 6.0999 | 7.3896 |
| 6 | 4.7114 | 6.2796 |
| 8 | 3.3360 | 6.7901 |
| 12 | 1.3613 | 7.3540 |
| unlimited | 0.0000 | 7.8368 |

The depth-1 tree has high bias: both training and validation errors are high. The unlimited tree has high variance: training RMSE is 0.0000 while validation RMSE is 7.8368. Depth 6 gives the smallest validation RMSE, 6.2796. Greater complexity does not guarantee better predictions on unseen parcels.

| learning_rate | val_mse | val_r2 |
| --- | --- | --- |
| 0.0100 | 100.0352 | 0.7127 |
| 0.1000 | 24.1722 | 0.9306 |
| 1.0000 | 54.3179 | 0.8440 |

The best tested learning rate is 0.1. With a fixed number of trees, 0.01 learns too slowly; 1.0 takes aggressive updates and performs worse on validation data, consistent with overshooting or overfitting.

| model | mean_r2 | std_r2 |
| --- | --- | --- |
| linear_regression | 0.9261 | 0.0048 |
| gradient_boosting | 0.9353 | 0.0045 |

Both candidates use the same shuffled KFold object on training-plus-validation rows. Gradient boosting wins the CV comparison, whereas linear regression wins the specified validation comparison. I retain linear regression for this submission because the brief explicitly defines selection by validation RMSE. In a future deployment comparison, repeated or nested evaluation would help resolve this small disagreement without repeatedly consulting the held-out test set.

## Model results — classification

The stratified split has 195, 42 and 41 delayed parcels in training, validation and test, respectively. Feature selection, candidate fitting and searches use training rows; validation chooses the configuration and threshold. Test outcomes do not enter those choices.

| model | TN | FP | FN | TP | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blind_all_on_time | 138 | 0 | 42 | 0 | 0.7667 | 0.0000 | 0.0000 | 0.0000 |
| logistic_regression | 128 | 10 | 7 | 35 | 0.9056 | 0.7778 | 0.8333 | 0.8046 |
| decision_tree_depth_6 | 128 | 10 | 11 | 31 | 0.8833 | 0.7561 | 0.7381 | 0.7470 |
| random_forest | 132 | 6 | 11 | 31 | 0.9056 | 0.8378 | 0.7381 | 0.7848 |

The blind baseline is 0.7667 accurate but catches no delayed parcels: recall and F1 are both 0.0000. Accuracy is therefore insufficient for this imbalanced task. Logistic regression has validation F1 0.8046; the selected decision-tree depth is 6 with F1 0.7470. The full-feature forest has F1 0.7848. The configuration comparison below is a separate comparison using the fixed seven selected features.

The five bookkeeping columns together receive 0.2245 of the full forest's impurity importance despite being generated without signal. Continuous and high-cardinality inputs can receive incidental split importance; that importance does not prove that these fields predict delays. All five lie outside the top seven by ANOVA F-score and have F-scores below 10. The seven retained features are distance, hub tier, hub capacity, route stops, promised hours, rain and sale-event status. See `part_c_feature_scores.csv` for their exact ranks and scores.

| k | val_f1 |
| --- | --- |
| 5 | 0.6988 |
| 7 | 0.8333 |
| 10 | 0.7857 |
| 12 | 0.8049 |
| 17 | 0.7848 |

The k table supports removing weak inputs: k=7 gives validation F1 0.8333, while retaining all 17 gives 0.7848. This is an exercise-level comparison on one validation split, not proof that seven features always generalize best. The later stages use k=7 as required by the brief.

| search | candidates | cv_folds | total_fits | best_cv_accuracy | val_f1 |
| --- | --- | --- | --- | --- | --- |
| grid_search | 27 | 5 | 135 | 0.8952 | 0.7750 |
| randomized_search | 9 | 5 | 45 | 0.8905 | 0.7848 |
| default_forest_same_7_features | 1 | 0 | 1 | not applicable | 0.8333 |

The grid and randomized searches try 27 and 9 candidates, requiring 135 and 45 CV fits. The default forest using the same seven features wins the validation comparison with F1 0.8333. Tuning does not guarantee improvement: the searches optimize CV accuracy and compare a finite parameter set, while the final choice optimizes validation F1. Differences can also reflect the limited validation sample. `best_params` is recorded in `part_c_tuning.csv`; `part_c_final_config.csv` records the actual deployed configuration.

The exercise fits the selector once on the training set before the tuning CV. This follows its explicit instructions, but CV validation folds have influenced the supervised selection. Production CV should include feature selection inside each fold; the untouched outer validation and test splits remain separate here.

| threshold | TN | FP | FN | TP | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1000 | 94 | 44 | 1 | 41 | 0.7500 | 0.4824 | 0.9762 | 0.6457 |
| 0.2000 | 109 | 29 | 3 | 39 | 0.8222 | 0.5735 | 0.9286 | 0.7091 |
| 0.3000 | 121 | 17 | 3 | 39 | 0.8889 | 0.6964 | 0.9286 | 0.7959 |
| 0.4000 | 125 | 13 | 5 | 37 | 0.9000 | 0.7400 | 0.8810 | 0.8043 |
| 0.5000 | 129 | 9 | 7 | 35 | 0.9111 | 0.7955 | 0.8333 | 0.8140 |
| 0.6000 | 134 | 4 | 10 | 32 | 0.9222 | 0.8889 | 0.7619 | 0.8205 |
| 0.7000 | 134 | 4 | 15 | 27 | 0.8944 | 0.8710 | 0.6429 | 0.7397 |
| 0.8000 | 134 | 4 | 23 | 19 | 0.8500 | 0.8261 | 0.4524 | 0.5846 |
| 0.9000 | 136 | 2 | 30 | 12 | 0.8222 | 0.8571 | 0.2857 | 0.4286 |

The chosen threshold is 0.5000, the highest tested threshold with validation recall at least 0.80. It gives validation precision 0.7955 and recall 0.8333. The next threshold fails the recall constraint. Increasing the threshold reduces the number of alerts; recall cannot increase, although precision need not change monotonically.

Random-forest `predict` breaks an exact class-probability tie in favor of the first class, while the required alert rule uses probability **greater than or equal to** the threshold. Thus a probability exactly equal to the threshold can be flagged even when default `predict` returns on time. This explains why the threshold-table F1 at 0.5 differs from the default forest's model-comparison F1; the underlying configuration has not changed.

| threshold | TN | FP | FN | TP | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 125 | 14 | 10 | 31 | 0.8667 | 0.6889 | 0.7561 | 0.7209 |

Test precision is 0.6889, recall 0.7561, and F1 0.7209. Test recall is lower than validation recall 0.8333; the validation target is not a guarantee for new data. The test split contains only 41 delayed parcels, so individual errors noticeably affect recall. The threshold was not retuned after this evaluation.

## Operations recommendation

Profiles use historical shipments for each hub. K-Means uses standardized mean delivery hours, delay rate, mean distance and mean stops, with hubs sorted by ascending ID. Segment names are assigned by increasing mean hub delay rate, rather than arbitrary cluster numbers.

| segment | hubs | mean_delivery_hours | delay_rate | mean_distance_km | mean_n_stops |
| --- | --- | --- | --- | --- | --- |
| Standard | 7 | 47.2396 | 0.0257 | 188.9296 | 55.3686 |
| Watch | 8 | 56.8137 | 0.1550 | 314.1762 | 70.7175 |
| Escalate | 9 | 64.6504 | 0.4600 | 470.4757 | 84.3444 |

Use the Escalate playbook for these hubs and their observed delay rates:

| hub_id | delay_rate |
| --- | --- |
| HUB13 | 0.4600 |
| HUB15 | 0.3600 |
| HUB17 | 0.4600 |
| HUB18 | 0.3800 |
| HUB19 | 0.6200 |
| HUB20 | 0.4800 |
| HUB21 | 0.4000 |
| HUB23 | 0.4400 |
| HUB24 | 0.5400 |

Tier labels and observed operating segments disagree for these hubs:

| hub_id | segment | hub_tier |
| --- | --- | --- |
| HUB07 | Watch | 1 |
| HUB13 | Escalate | 2 |
| HUB15 | Escalate | 2 |
| HUB22 | Watch | 3 |

HUB07 (Kolkata) should use Watch despite its Tier 1 label. HUB13 (Nagpur) and HUB15 (Kochi) should use Escalate despite Tier 2 labels. HUB22 (Dehradun) should use Watch despite Tier 3. These are operating-playbook recommendations based on this synthetic history, rather than changes to administrative tier definitions.

The alert threshold is 0.5000. From test precision 0.6889, about 6.889 out of every ten flagged parcels are delayed. From recall 0.7561, about 7.561 out of every ten delayed parcels are caught. These are observed test estimates, not guaranteed operational yields.

The new batch has 48 parcels; 8 are flagged and 6 need a hub-manager escalation. Flagged parcels in other segments receive a proactive customer call; unflagged parcels receive `No action`. The action list is `outputs/new_shipment_actions.csv`. Exported probabilities are rounded before comparison with the stored threshold so the flags can be recomputed directly from the CSV.

| K | across_over_within | cluster_sizes |
| --- | --- | --- |
| 2 | 2.8940 | 15-9 |
| 3 | 5.3928 | 9-8-7 |
| 4 | 6.3047 | 8-7-5-4 |
| 5 | 6.8866 | 8-7-4-3-2 |

K=3 is fixed by the exercise's operating plan. It provides Standard, Watch and Escalate groups with 7, 8 and 9 hubs. Although the across-to-within distance ratio rises for K=4 and K=5, those solutions introduce groups with fewer than six hubs and are less useful for this three-playbook policy. The ratio alone is not a model-selection criterion; smaller within-cluster distances tend to favor more clusters.

## Production plan

| Characteristic | Concrete implementation |
| --- | --- |
| Accuracy | Held-out regression and classification metrics are exported; threshold selection enforces the validation recall rule. |
| Reliability | Preprocessing rejects unknown services and hubs; joins validate their cardinality and scoring checks missing outputs. |
| Reproducibility | Random seeds, feature order and exact dependency versions are fixed; verification checks regenerated data and repeated scoring. |
| Performance | Batch scoring loads one saved model and scores a DataFrame without repeating training or searches. Runtime and service load still require production benchmarking. |
| Maintainability | Shared preprocessing lives in `src/shipment_prep.py`; each pipeline stage has a separate executable script. |
| Consistency | Training and prediction reuse the same cleaning and feature functions, stored medians and ordered feature list. |
| Monitorability | A weekly job should compare alerts, matured outcomes and input distributions; proposed triggers are recorded below. |

| quantity | baseline | trigger | rule |
| --- | --- | --- | --- |
| weekly_delayed_share | 0.2317 | 0.2817 | review if above trigger in two consecutive comparable weeks |
| weekly_flagged_share | 0.1667 | 0.2667 | review if above trigger in two consecutive comparable weeks |
| mature_outcome_recall | 0.7561 | 0.7000 | review if below trigger after at least 100 delayed outcomes |

These monitoring triggers are proposed starting rules, not validated production thresholds. The flagged-share baseline comes from a small new batch. Recall monitoring must use outcomes that have had time to mature; report sample size and uncertainty and avoid treating pending deliveries as on time. Investigate service mix, route distance, missingness and sale-event composition before attributing an alert to model degradation.

Review monitoring weekly and schedule a monthly retraining review using newly matured labels. Retrain and version preprocessing, model and threshold together, validate on later dispatches, and deploy only after the candidate passes the agreed evaluation. Keep a previous artifact available for rollback.

Treat the September sale window as a distinct operating regime: compare sale-event parcels with prior comparable sale-event parcels and ordinary weeks with ordinary weeks. A planned sale-related increase is not by itself unexpected drift. Drift includes unexplained shifts within those comparable groups in distance, service mix, missingness, probability distributions or observed prediction errors. Preserve the dispatch-time sale indicator during retraining and scoring.

## Limitations

- Synthetic relationships and fixed hub behavior cannot establish real-world performance, causation or treatment effects.
- The random row splits share hubs and nearby calendar periods. They do not measure transfer to unseen hubs or future operating conditions; temporal evaluation is a necessary next step.
- The classifier has a small held-out delayed sample, and one validation split controls both configuration and threshold selection. A larger later-period evaluation is needed.
- ANOVA selection before inner CV is mandated here and makes the tuning CV less strict than a fully nested pipeline. Historical full-table median filling is also required by the exercise; a production pipeline should learn preprocessing inside training folds.
- Distance, tier and capacity are related, and raw delay rates reflect parcel mix. Hub segments describe this historical sample, not risk-adjusted management quality.
- Forest probabilities are not calibrated in this exercise, and no service-wide latency, availability or economic benefit has been measured. Alert costs are qualitative because the brief supplies no explicit INR cost matrix.
- Source byte identity still requires the original generator copied from the assignment. Matching numerical outputs cannot certify source byte identity.

## Learning guide: understand each part before submitting

### Part A: turn raw records into model inputs

Start with `src/shipment_prep.py`. `FEATURES` is the input contract: every stage selects these names in the same order. `clean_shipments(raw, medians=None)` returns a tuple containing a cleaned DataFrame and a dictionary of fill values. Historical calls learn medians; scoring calls pass the saved medians. Copying the input avoids silently modifying the caller's table.

`drop_duplicates` removes exact duplicate rows. String methods repair service spelling. `fillna` supplies missing payment mode and numeric values. Rupee signs and commas must be removed before amount text can become numeric. A weight sentinel is converted to a missing value before computing its median, so it cannot inflate the fill value.

`build_features(clean, hubs)` attaches hub metadata with a left merge. A many-to-one validation prevents duplicate hub records from multiplying shipments. It creates the promise, COD indicator, weekend indicator, dispatch hour and natural-log amount. It never reads delivery hours, which are unavailable at dispatch. `part_a_clean_eda.py` adds the two outcomes only for historical rows, creates the modelling table, and exports numerical evidence.

### Part B: predict a continuous quantity

Read `part_b_regression.py`. Its target is delivery hours. Training rows fit parameters; validation rows compare candidates; test rows provide the final evaluation. The mean baseline establishes whether a learned model improves on a simple constant prediction.

MAE is the average absolute error in hours. MSE averages squared errors, and RMSE takes its square root to return to hours. R² compares squared prediction error with an outcome-mean baseline. A negative R² indicates performance worse than that reference. The depth table demonstrates underfitting and overfitting; the learning-rate table shows how update size affects boosting.

StandardScaler learns means and standard deviations from training inputs. The linear model uses a Pipeline so CV fits its scaler separately inside each fold. Tree models use their raw features because split decisions do not require this standardization. The final selected model remains fitted on the prescribed training rows.

### Part C: identify likely delays and decide when to call

Read `part_c_classification.py`. Its target is the binary delayed flag. Stratification keeps the delayed share similar across splits. A confusion matrix separates true negatives, false positives, false negatives and true positives. Precision answers how often an alert is correct; recall answers how many actual delays are caught; F1 balances precision and recall.

`SelectKBest(f_classif, ...)` ranks features by differences between delayed and on-time classes. It is fitted without looking at the outer validation or test outcomes. `GridSearchCV` tries every specified parameter combination; `RandomizedSearchCV` samples a fixed subset. Their default classifier scoring is accuracy, while the required final comparison uses validation F1.

After choosing a configuration, `predict_proba` returns delay probabilities. The threshold converts those values into alerts. The operations rule chooses the highest tested threshold that still meets the validation recall floor, limiting unnecessary calls while prioritizing missed delays. Test results are reported only after that decision.

### Part D: make the output operational

Read `part_d_segments.py`. Each row now represents a hub rather than a parcel. Standardizing the four profile variables prevents large-scale distance values from dominating Euclidean distances. K-Means groups similar hub profiles; the names Standard, Watch and Escalate come from their mean delay rates.

`package_model.py` rebuilds the final selector-plus-forest Pipeline on the same training split. It saves the model, threshold, feature order and imputation medians. Reloaded test probabilities must match the in-memory model. This is an artifact-integrity check, not another opportunity to choose a model from test performance.

`predict_delay.py` reads the new raw shipments, applies the shared functions using saved medians, scores the exact ordered inputs, joins the hub segment, and assigns the action. Run it again to confirm deterministic output. `verify_project.py` checks the resulting files against the exercise's numerical requirements.

### Suggested reading and execution order

Run one stage at a time using the command sequence above. After Part A, open the checks and grouped-delay CSVs. After Part B, compare validation metrics, depth errors and CV results. After Part C, inspect the confusion matrices, feature scores, search table and threshold table. After Part D, examine mismatched tiers, model reload checks and the new-shipment action list. Explain these choices in your own words before you submit the public repository.
