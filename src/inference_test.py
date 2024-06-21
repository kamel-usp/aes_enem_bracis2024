
from datetime import datetime
import logging
import transformers
from finetuning_pipeline import finetuning_pipeline, save_evaluation_results_to_csv
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("transformers").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "kamel-usp/aes_enem_dataset"
DATASET_CONFIG = "PROPOR2024"
MAX_SEQUENCE = 8192
BATCH_SIZE = 2
GRADIENT_ACC = 16
MODEL_NAME = "microsoft/Phi-3-medium-128k-instruct"
TOKENIZER_NAME = "microsoft/Phi-3-medium-128k-instruct"
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
    logger.info("Running on Test")
    evaluate_test = trainer.evaluate(tokenized_data["test"])
    logger.info("Fine Tuning Finished.")


if __name__ == "__main__":
    main()
