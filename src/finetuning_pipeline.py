import csv
from functools import partial
import os
from logging import Logger
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    Phi3ForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
import torch
from model.phi3_for_ordinal_regression import Phi3ForOrdinalRegression

from data_processor import DataProcessor
from metrics import compute_metrics

CACHE_DIR = "/media/data/tmp/"
SEED_PARAM = 42
NUM_LABELS = 6
PROBLEM_TYPE = "single_label_classification" #single_label_classification or regression


def save_evaluation_results_to_csv(
    training_id, evaluation_results, timestamp, file_path="evaluation_results.csv"
):
    # Add a timestamp to the evaluation results
    evaluation_results_with_timestamp = evaluation_results.copy()
    evaluation_results_with_timestamp["timestamp"] = timestamp
    evaluation_results_with_timestamp["id"] = training_id

    # Determine if we need to write headers
    write_headers = not os.path.exists(file_path)

    with open(file_path, "a", newline="") as csvfile:
        fieldnames = list(evaluation_results_with_timestamp.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_headers:
            writer.writeheader()
        writer.writerow(evaluation_results_with_timestamp)

def print_model_parameters(model):
    all_param = 0
    trainable_params = 0
    
    for param in model.parameters():
        # params.numel() returns the total number of elements in the parameter tensor
        all_param += param.numel()
        
        if param.requires_grad:
            # Only count parameters that require gradients (trainable)
            trainable_params += param.numel()
    
    print(f"Total parameters: {all_param}")
    print(f"Trainable parameters: {trainable_params}")
    print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )


def finetuning_pipeline(
    dataset_name: str,
    dataset_config: str,
    reference_concept: int,
    max_length: int,
    batch_size: int,
    gradient_acc: int,
    model_name: str,
    tokenizer_name: str,
    logger: Logger,
):
    # Initialize data processor and load dataset
    processor = DataProcessor(
        dataset_name, tokenizer_name, reference_concept, max_length
    )
    datasets = load_dataset(dataset_name, dataset_config, cache_dir=CACHE_DIR)

    # Preprocess datasets
    tokenized_data = processor.preprocess_dataset(datasets)

    lora_config = LoraConfig(
        r=256,
        lora_alpha=32,
        lora_dropout=0.15,
        task_type=TaskType.SEQ_CLS,
        target_modules="all-linear",
    )

    model = Phi3ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2"
    )

    model = get_peft_model(model, lora_config)
    logger.info(model.print_trainable_parameters())
    logger.info(print_model_parameters(model))
    model.config.pad_token_id = model.config.eos_token_id
    model.config.problem_type = PROBLEM_TYPE

    training_args = TrainingArguments(
        seed=SEED_PARAM,
        data_seed=SEED_PARAM,
        output_dir=f"./results/C{reference_concept+1}",  # Output directory for checkpoints and predictions
        overwrite_output_dir=True,  # Overwrite the content of the output directory
        per_device_train_batch_size=batch_size,  # Batch size for training
        per_device_eval_batch_size=batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=gradient_acc,  # number of steps before optimizing
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=10,  # Number of warmup steps
        num_train_epochs=12,  # Number of training epochs
        learning_rate=5e-5,  # Learning rate
        weight_decay=0.01,  # Weight decay
        # For logging and saving
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=1,  # Limit the total number of checkpoints
        eval_strategy="epoch",
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="QWK",
        report_to=["none"],
        bf16=True,
        half_precision_backend="cpu_amp"
    )

    compute_metrics_partial = partial(compute_metrics, model=model)

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        args=training_args,
        compute_metrics=compute_metrics_partial,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=5)]
            if tokenized_data["validation"]
            else []
        ),
    )

    return trainer, tokenized_data
