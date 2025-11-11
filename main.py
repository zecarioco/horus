import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

splits = {'train': 'multilabel/multilabel_train.csv', 'test': 'multilabel/multilabel_test.csv'}
train_df = pd.read_csv("hf://datasets/Silly-Machine/TuPyE-Dataset/" + splits["train"])
test_df  = pd.read_csv("hf://datasets/Silly-Machine/TuPyE-Dataset/" + splits["test"])

print("Train size:", len(train_df))
print("Test size:", len(test_df))

label_cols = ['aggressive','hate','ageism','aporophobia','body_shame',
              'capacitism','lgbtphobia','political','racism','religious_intolerance',
              'misogyny','xenophobia','other']

train_labels = train_df[label_cols].values
test_labels  = test_df[label_cols].values

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
test_encodings  = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)

class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

train_dataset = MultilabelDataset(train_encodings, train_labels)
test_dataset  = MultilabelDataset(test_encodings, test_labels)

model_dir = "./results/checkpoint-final"

if os.path.exists(model_dir):
    print("Modelo existe")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        "neuralmind/bert-base-portuguese-cased",
        num_labels=len(label_cols),
        problem_type="multi_label_classification"
    ).to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        fp16=True
    )

    def compute_metrics(pred):
        logits = torch.sigmoid(torch.tensor(pred.predictions))
        preds = (logits > 0.5).int()
        labels = torch.tensor(pred.label_ids).int()
        true_positives = (preds & labels).sum(dim=0).float()
        predicted_positives = preds.sum(dim=0).float()
        actual_positives = labels.sum(dim=0).float()
        precision = (true_positives / (predicted_positives + 1e-8)).mean().item()
        recall = (true_positives / (actual_positives + 1e-8)).mean().item()
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(model_dir)

results = Trainer(model=model, args=TrainingArguments(output_dir='./tmp'), eval_dataset=test_dataset).evaluate()
print(results)

sample_texts = ["Essa pessoa é uma vergonha", "Tua mãe é uma vagabunda", "Aquele viadinho sem graça", "bicha, te amo", "Além de ser gorda é burra", "Lindaaa! te vejo na sexta"]
sample_encodings = tokenizer(sample_texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
sample_encodings = {k: v.to(device) for k, v in sample_encodings.items()}

outputs = model(**sample_encodings)
preds = torch.sigmoid(outputs.logits) > 0.5

for i, text in enumerate(sample_texts):
    labels = [label_cols[j] for j, p in enumerate(preds[i]) if p]
    print(text, "=>", labels)