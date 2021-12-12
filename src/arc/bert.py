#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[2]:


preprocessor = mz.models.Bert.get_default_preprocessor()


# In[3]:


train_pack_processed = preprocessor.transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[4]:


trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1
    resample=True,
    sort=False,
    batch_size=20,
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed
    batch_size=20,
)


# In[5]:


padding_callback = mz.models.Bert.get_default_padding_callback()
trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    stage='dev',
    callback=padding_callback
)


# In[6]:


model = mz.models.Bert()

model.params['task'] = ranking_task
model.params['mode'] = 'bert-base-uncased'
model.params['dropout_rate'] = 0.2

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[7]:


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

from pytorch_transformers import AdamW, WarmupLinearSchedule

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, betas=(0.9, 0.98), eps=1e-8)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=6, t_total=-1)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=10
)


# In[8]:


trainer.run()

