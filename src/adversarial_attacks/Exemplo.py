from GeradorAdversarial import gerar_ataque
from data_processor import DataProcessor
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np

DATASET_NAME = "kamel-usp/aes_enem_dataset"
TOKENIZER_NAME = "microsoft/Phi-3-medium-128k-instruct"
CACHE_DIR = "/media/data/tmp/"
PROBLEM_TYPE = "single_label_classification"
NUM_LABELS = 6
MAX_SEQUENCE = 8192
REFERENCE_CONCEPT = 0
TRAINED_MODEL = (
    f"best-models/phi3-lora-single_label_classification-C{REFERENCE_CONCEPT+1}"
)
processor = DataProcessor(DATASET_NAME, TOKENIZER_NAME, REFERENCE_CONCEPT, MAX_SEQUENCE)
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


def calcular_nota(texto):
    # usa o modelo para calcular quanto vale o texto
    processed_text = processor._prompt_template(texto)
    tokenized_output = processor.tokenizer(
        processed_text,
        return_tensors="pt",
        max_length=processor.max_length,
        padding="longest",
    )
    inputs = {k: v.to(model.device) for k, v in tokenized_output.items()}
    outputs = model(**inputs)
    logits = outputs.logits.to(torch.float32).cpu().numpy()
    all_predictions = np.argmax(logits, axis=1)
    return all_predictions*40

def run_attacks():
    ataque = ["1a", "1b", "1c", "2a", "2b", "2c", "3a", "3b", "3c", "4", "5", "6"]
    for a in ataque:
        frases = gerar_ataque(a)
        for index, texto in enumerate(frases):
            nota = calcular_nota(texto)
            print(f"Ataque {a} frase {index} tirou {nota}")

if __name__ == "__main__":
    print(f"Running attacks for {TRAINED_MODEL}")
    run_attacks()
