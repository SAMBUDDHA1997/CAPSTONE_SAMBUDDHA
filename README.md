Delivery Last-Mile Delay Intelligence Pipeline
A synthetic dispatch-time analytics exercise: estimate delivery hours, flag potential promise breaches, and turn those flags into actions using operating segments for the hubs. The data are synthetic; these are exercise results, not estimates of Delhivery's actual performance.
Before submission: `generate_data.py` was transcribed from screenshots. Its generated values and all numerical checks match the brief, but byte identity with the instructor's source cannot be established from an image. Replace this file with the original code copied verbatim from the assignment, rerun the commands below, and review the results. This project has not been published to GitHub. Read and understand the implementation, adapt the explanations into your own words, and follow your programme's rules on outside assistance.
How to run
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
