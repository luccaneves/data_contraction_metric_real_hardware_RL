# data_contraction_metric_real_hardware_RL

This repository contains training scripts and experiment code used for "Safe Reinforcement Learning Force Control for a Hydraulic Actuator with Real-World Training Using a Learned Contraction Metric". It includes tools to learn a contraction metric from recorded physical experiments and to train/validate controllers (feedback linearization and sliding mode) using that metric.

**Quick summary**
- Purpose: Learn a contraction metric from real hardware data and use it to train safe RL controllers.
- Contents: metric training code, RL environment and training scripts, and MATLAB/Simulink models used for experiments.

**Files of interest**
- Data and experiments: [Data](Data)
- Hydraulic model learning: [Data/HydraulicModelLearning](Data/HydraulicModelLearning)
- Metric learning scripts and models: [Data/LearnContractionMetricScripts](Data/LearnContractionMetricScripts)
- Real-world RL environments and checkpoints: [Data/Real_Life_FeedbackLin_Train](Data/Real_Life_FeedbackLin_Train) and [Data/Real_Life_Sliding_Mode_Train](Data/Real_Life_Sliding_Mode_Train)
- Top-level RL environment script: [Data/RL_Train_Environment_Script.py](Data/RL_Train_Environment_Script.py)
- Simulink code used for the dSpace MicroLabBox hardware implementation: [Data/RealLifeTrainModel.slx](Data/RealLifeTrainModel.slx)

Repository structure (high level)
- Data/: recorded measurements, MATLAB/Simulink models (.slx/.slxc) and experiment folders (FL, SMC, RL variants).
- Data/HydraulicModelLearning/: Python code for supervised hydraulic model learning.
- Data/LearnContractionMetricScripts/: training/validation code, model definition and a saved model (`learned_metric.pth`). Key files:
  - [Data/LearnContractionMetricScripts/MetricNet.py](Data/LearnContractionMetricScripts/MetricNet.py)
  - [Data/LearnContractionMetricScripts/train_metric.py](Data/LearnContractionMetricScripts/train_metric.py)
  - [Data/LearnContractionMetricScripts/validate_contraction.py](Data/LearnContractionMetricScripts/validate_contraction.py)
- Data/Real_Life_FeedbackLin_Train/ and Data/Real_Life_Sliding_Mode_Train/: environment definitions and saved checkpoints for controller training.

Hydraulic model learning
- [Data/HydraulicModelLearning/train_model.py](Data/HydraulicModelLearning/train_model.py): trains a multi-step predictor for the hydraulic system with PyTorch.
- [Data/HydraulicModelLearning/dataset.py](Data/HydraulicModelLearning/dataset.py): builds supervised samples from normalized CSV data, groups trajectories by `t == 0.0`, and uses a 100-step prediction horizon.
- [Data/HydraulicModelLearning/mlp_lyap_mean_bb.py](Data/HydraulicModelLearning/mlp_lyap_mean_bb.py): defines the neural network used by the training script.
- [Data/HydraulicModelLearning/model_399.pth](Data/HydraulicModelLearning/model_399.pth): saved model weights.
- [Data/HydraulicModelLearning/validate_hybrid.py](Data/HydraulicModelLearning/validate_hybrid.py): validation and export script for the hydraulic model.

Training details for the hydraulic model
- The dataset uses 9 base inputs: actuator force and derivative, load force and derivative, pressures `Pa`/`Pb`, position `x`, velocity `x_deriv`, and current `i`.
- The model also consumes `last_x` plus a sequence of future current values over the prediction horizon.
- The target is an 8-variable rollout across 100 future steps: actuator force, actuator force derivative, load force, load force derivative, `Pa`, `Pb`, `x`, and `x_deriv`.
- Training uses PyTorch, Adam, MSE loss, gradient clipping, checkpointing, and TensorBoard logging.

Prerequisites
- Python 3.8+ (for the metric training and validation scripts).
- Common Python packages: numpy, scipy, matplotlib, pandas, torch (PyTorch). Install via pip into a virtual environment.
- MATLAB / Simulink (optional) to open and run `RealLifeTrainModel.slx` and related experiment models.
- Optional: tools to read .mf4 recordings (e.g. asammdf) if you want to parse raw measurement files.

Quick start
1. Create a Python virtual environment and install the typical packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pandas torch
# install extras if needed (e.g. asammdf)
```

2. Train or fine-tune the contraction metric (example):

```bash
python Data/LearnContractionMetricScripts/train_metric.py
```

3. Validate the learned metric:

```bash
python Data/LearnContractionMetricScripts/validate_contraction.py
```

4. Use the learned metric within RL training or controller evaluation. See:
- [Data/RL_Train_Environment_Script.py](Data/RL_Train_Environment_Script.py)
- environment files in [Data/Real_Life_FeedbackLin_Train](Data/Real_Life_FeedbackLin_Train) and [Data/Real_Life_Sliding_Mode_Train](Data/Real_Life_Sliding_Mode_Train)

Notes and tips
- The repository mixes MATLAB/Simulink assets and Python code. Use MATLAB to inspect and simulate `.slx` models, and Python for metric learning and RL experiments.
- `learned_metric.pth` in [Data/LearnContractionMetricScripts](Data/LearnContractionMetricScripts) is an example pretrained metric — load it from the training/validation scripts.
- Many experiment recordings are stored as `.mf4` files under `Data/Experiments/FL` and subfolders.
- The hydraulic model learning scripts expect normalized CSV data with future-step columns such as `F_actuator_1`, `Pa_1`, `x_1`, and so on.
- The real-life RL environment folders each contain an `Env.py` and a GUI image.


