#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[2]:


ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]


# In[3]:


preprocessor = mz.models.DSSM.get_default_preprocessor(ngram_size=3)
train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[4]:


preprocessor.context


# In[5]:


triletter_callback = mz.dataloader.callbacks.Ngram(
    preprocessor, mode='aggregate')

trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=1,
    num_neg=4,
    callbacks=[triletter_callback]
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    callbacks=[triletter_callback]
)


# In[6]:


padding_callback = mz.models.DSSM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=32,
    stage='train',
    resample=True,
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=32,
    stage='dev',
    callback=padding_callback
)


# In[7]:


model = mz.models.DSSM()

model.params['task'] = ranking_task
model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
model.params['mlp_num_layers'] = 3
model.params['mlp_num_units'] = 300
model.params['mlp_num_fan_out'] = 128
model.params['mlp_activation_func'] = 'relu'

model.build()

print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[8]:


optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=10
)


# In[9]:


trainer.run()

