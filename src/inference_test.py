
import logging
import transformers
from data_processor import DataProcessor
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from metrics import compute_metrics
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "kamel-usp/aes_enem_dataset"
DATASET_CONFIG = "PROPOR2024"
TOKENIZER_NAME = "microsoft/Phi-3-medium-128k-instruct"
CACHE_DIR = "/media/data/tmp/"
PROBLEM_TYPE = "single_label_classification"
MAX_SEQUENCE = 8192
BATCH_SIZE = 2
REFERENCE_CONCEPT = 0
TRAINED_MODEL = f"best-models/phi3-lora-single_label_classification-C{REFERENCE_CONCEPT+1}"
NUM_LABELS = 6
SEED = 42
transformers.set_seed(SEED)


def main():
    logger.info("Starting the training process.")

    processor = DataProcessor(
        DATASET_NAME, TOKENIZER_NAME, REFERENCE_CONCEPT, MAX_SEQUENCE
    )
    datasets = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=CACHE_DIR)
    tokenized_data = processor.preprocess_dataset(datasets)

    logger.info("Running on Test")
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAINED_MODEL,
        num_labels=NUM_LABELS,
        device_map="cuda", 
        torch_dtype="auto", 
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2"
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.config.problem_type = PROBLEM_TYPE
    model.eval()
    logits_list = []
    true_labels_list = []
    with torch.no_grad():
        tokenized_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        dataloader = DataLoader(tokenized_data["test"], batch_size=BATCH_SIZE)
        for batch in tqdm(dataloader, total=len(dataloader), desc="Applying Inference in Batches"):
            inputs = {k: v.to(model.device) for k, v in batch.items() if k != 'label'}
            outputs = model(**inputs)
            logits = outputs.logits.to(torch.float32).cpu().numpy()
            true_labels = batch['label'].cpu().numpy()
            logits_list.extend(logits)
            true_labels_list.extend(true_labels)
        logits_array = np.array(logits_list)
        true_labels_array = np.array(true_labels_list)
        results = compute_metrics((logits_array, true_labels_array), model)
        print(results)

    logger.info("Inference Finished.")


if __name__ == "__main__":
    main()
