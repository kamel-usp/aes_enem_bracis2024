# Fine-Tuning Scripts for Phi3 Models

This repository contains all the necessary scripts for fine-tuning Phi3 models. The provided scripts are organized to facilitate the fine-tuning process, including data processing, training, evaluation, and inference.

## Environment Setup

To set up the environment, follow these steps:

1. Create a new conda environment:

    ```bash
    conda create --name phi3_finetuning python=3.12.2
    ```

2. Activate the environment:

    ```bash
    conda activate phi3_finetuning
    ```

3. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

## Folder Structure

The repository is organized as follows:

```
src/
├── adversarial_attacks/
├── model/
├── init.py
├── custom_trainer.py
├── data_processor.py
├── finetuning_pipeline.py
├── inference_test.py
├── main.py
├── metrics.py
.env
.gitignore
evaluation_results.csv
requirements.txt
```

Where:

## Scripts

- **adversarial_attacks/**: Contains scripts for adversarial attacks related to the fine-tuning process.
- **model/**: Contains model architecture and related scripts.
- **__init__.py**: Initialization file for the src module.
- **custom_trainer.py**: Custom training script for fine-tuning the model. (deprecated by now)
- **data_processor.py**: Script for processing and preparing the data.
- **finetuning_pipeline.py**: Pipeline script that integrates the fine-tuning process.
- **inference_test.py**: Script for testing the model inference.
- **main.py**: Main entry point for the fine-tuning process.
- **metrics.py**: Script for calculating evaluation metrics.

## Additional Files

- **.env**: Environment variables file.
- **.gitignore**: Git ignore file to exclude unnecessary files from version control.
- **evaluation_results.csv**: CSV file containing evaluation results.
- **requirements.txt**: File listing all the required Python packages.

## Usage

1. Ensure you have created and activated the conda environment as described above.
2. Run the scripts in the appropriate order for your fine-tuning process.
