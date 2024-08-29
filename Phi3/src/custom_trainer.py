import torch
from transformers import (
    AdamW,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)


class CustomTrainer(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Custom loss function (if None, the default is used)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False):
        # If no custom loss_function is provided, use the model's default.
        if self.loss_function is None:
            return super().compute_loss(model, inputs, return_outputs)

        # Compute custom loss
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = self.loss_function(logits, labels)

        return (loss, outputs) if return_outputs else loss
