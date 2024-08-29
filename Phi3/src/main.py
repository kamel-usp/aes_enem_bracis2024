from datetime import datetime
import logging
import transformers
from finetuning_pipeline import finetuning_pipeline, save_evaluation_results_to_csv
import os


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

"""
Checklist to train new models:
1. Update Reference Concept
2. Update Training id
3. To train classification:
3.1 Change num labels to 6 [0,1,2,3,4,5]
3.2 Change problem_type to single_label_classification
4. To train regression:
4.1 Update trainer so that `inputs["labels"] = inputs["labels"].float()`
4.2 Change num_labels to 1
4.3 Change problem_type to regression
"""

REFERENCE_CONCEPT = 0
DATASET_NAME = "kamel-usp/aes_enem_dataset"
DATASET_CONFIG = "PROPOR2024"
MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct"
TOKENIZER_NAME = "microsoft/Phi-3-medium-128k-instruct"
PROBLEM_TYPE = "single_label_classification"
TRAINING_ID = f"phi-3-medium-lora-instructions-with-dropout-{PROBLEM_TYPE}-C{REFERENCE_CONCEPT+1}"
MAX_SEQUENCE = 8192
BATCH_SIZE = 2
GRADIENT_ACC = 16
SEED = 42
transformers.set_seed(SEED)


def main():
    logger.info("Starting the training process.")

    trainer, tokenized_data = finetuning_pipeline(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        reference_concept=REFERENCE_CONCEPT,
        max_length=MAX_SEQUENCE,
        batch_size=BATCH_SIZE,
        gradient_acc=GRADIENT_ACC,
        model_name=MODEL_NAME,
        tokenizer_name=TOKENIZER_NAME,
        logger=logger,
    )
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    evaluate_baseline = trainer.evaluate()
    save_evaluation_results_to_csv(TRAINING_ID, evaluate_baseline, current_time)

    trainer.train()

    evaluate_after_training = trainer.evaluate()
    save_evaluation_results_to_csv(TRAINING_ID, evaluate_after_training, current_time)
    logger.info("Training completed successfully.")
    logger.info("Running on Test")
    evaluate_test = trainer.evaluate(tokenized_data["test"])
    save_evaluation_results_to_csv(TRAINING_ID, evaluate_test, current_time)
    trainer.save_model(f"./best-models/phi3-lora-{PROBLEM_TYPE}-C{REFERENCE_CONCEPT+1}")
    logger.info("Fine Tuning Finished.")


if __name__ == "__main__":
    main()
