# Towards Reliable Time Series Forecasting: Bridging Model Evaluation and Adaptive Prediction Intervals

## Description
Forecasting real-world time series is challenging due to evolving patterns and local variations. Traditional models overlook transient behaviors and rely on static prediction intervals (PIs), failing to adapt to concept drift.

We introduce an adaptive pattern-based framework that:
- Clusters time series subsequences to capture dominant patterns.
- Evaluates local model performance to detect stress regions.
- Dynamically adjusts PIs for better uncertainty quantification.

## Project Structure
The project contains the following key directories and files:

- **test/**: Includes the data needed for the experiment.
- **models_config.py**: Contains the configuration of parameters and architectures of the models.

## Running the Experiment
### **Prerequisites**
Ensure that the required dependencies are installed and directories are correctly adjusted before running the scripts.

### **Execution Steps**
Run the following scripts in order:

1. **`train_base.py`** → Train baseline models.
2. **`train_experts.py`** → Train expert models in the clusters.
3. **`evaluate_base.py`** → Evaluate baseline models.
4. **`evaluate_top_models.py`** → Evaluate "expert," "top-3-expert," and "closest-3clusters" models, which represent the best model of the closest cluster, the best three models of the closest cluster, and the three closest clusters.
5. **`evaluate_clusters.py`** → Evaluate the models inside each cluster with detailed information about the predictions.

## Contact
For further inquiries or contributions, please reach out via GitHub Issues or email the project maintainer.

---
This README provides an overview of the project structure and execution steps. Feel free to modify it according to any updates or additional details in your project.

