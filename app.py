# import HF model, tokenizer, training args, trainer / dataset import / pandas
from transformers import AutoTokenizer,AutoModel,TrainingArguments,Trainer
from datasets import load_dataset
import pandas as pd

# instantiation of all components
dataset = load_dataset('ajaykarthick/imdb-movie-reviews')
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased", num_labels=2)

# takes data and returns tokenized data
def tokenize(example):
    return tokenizer(
        example['review'],
        padding='max_length',
        truncation=True,
        max_length=256
    )

# collect tokenized datasets from 'tokenize'
tokenized_datasets = dataset.map(
    tokenize,
    batched=True
)

training_args = TrainingArguments(
    output_dir="./sentiment_model",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

#TODO make all this work:
# trainer.train()
# testing
# results = trainer.evaluate
# print(results)