#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm.auto import tqdm


# In[2]:


from accelerate import Accelerator
accelerator = Accelerator()


# In[3]:


torch.cuda.device_count()


# In[4]:


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint) 

def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['sentence1', 'sentence2', 'idx'])
tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
tokenized_datasets.set_format('torch')

data_collator = DataCollatorWithPadding(tokenizer)


# In[5]:


train_dataloader = DataLoader(tokenized_datasets['train'],
                             shuffle=True,
                             batch_size=100,
                             collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets['validation'],                             
                             batch_size=100,
                             collate_fn=data_collator)


# In[6]:


checkpoint = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


# In[7]:


optimizer = AdamW(model.parameters(), lr=5e-5)


# In[8]:


train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer)


# In[9]:


num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
  "linear",
  optimizer=optimizer,
  num_warmup_steps=0,
  num_training_steps=num_training_steps
)


# In[10]:
print('NUm of steps:', len(train_dataloader)/100)

progress_bar = tqdm(range(num_training_steps))

metric = load_metric("glue", "mrpc")


for epoch in range(num_epochs):
    print('Epoch:', epoch)
    
    model.train()
    for batch in train_dataloader:        
        outputs = model(**batch)
        loss = outputs.loss        
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
    model.eval()
    all_predictions = []
    all_labels = []
    
    for batch in eval_dataloader:
        
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)

        all_predictions.append(accelerator.gather(predictions))
        all_labels.append(accelerator.gather(batch["labels"]))

    all_predictions = torch.cat(all_predictions)[:len(tokenized_datasets["validation"])]
    all_labels = torch.cat(all_labels)[:len(tokenized_datasets["validation"])]

    eval_metric = metric.compute(predictions=all_predictions, references=all_labels)

    # Use accelerator.print to print only on the main process.
    accelerator.print(f"epoch {epoch}:", eval_metric)





