from coral_pytorch.dataset import corn_label_from_logits
from sklearn.metrics import accuracy_score, cohen_kappa_score, root_mean_squared_error
import torch
import numpy as np

def enem_accuracy_score(true_values, predicted_values):
    assert len(true_values) == len(predicted_values), "Mismatched length between true and predicted values."

    non_divergent_count = sum([1 for t, p in zip(true_values, predicted_values) if abs(t - p) <= 80])
    
    return non_divergent_count / len(true_values)

def compute_metrics(eval_pred, model):
    logits, all_true_labels = eval_pred
    if model.config.problem_type == "single_label_classification":
        all_predictions = np.argmax(logits, axis=1)
    elif model.config.problem_type == "regression":
        rounded_tensor = np.round(logits)
        # Clamp the values to the range [0, 5]
        clamped_tensor = np.clip(rounded_tensor, a_min=0, a_max=5)
        all_predictions = np.argmax(clamped_tensor, axis=1)
    else:
        raise AttributeError("problem_type from model.config is None!")
    #all_predictions = corn_label_from_logits(torch.from_numpy(logits))
    #revert back
    all_true_labels = list(map(lambda x: x * 40, all_true_labels))
    all_predictions = list(map(lambda x: x * 40, all_predictions))
    # Initialize the metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    qwk = cohen_kappa_score(all_true_labels, all_predictions, weights="quadratic", labels=[0,40,80,120,160,200]) 
    rmse = root_mean_squared_error(all_true_labels, all_predictions)
    horizontal_discrepancy = enem_accuracy_score(all_true_labels, all_predictions)
    results = {
        'accuracy': accuracy,
        'RMSE': rmse,
        'QWK': qwk,
        'HDIV': 1- horizontal_discrepancy
    }

    return results