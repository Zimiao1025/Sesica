#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[2]:


ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=1))
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]


# In[3]:


preprocessor = mz.models.aNMM.get_default_preprocessor()


# In[4]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[5]:


preprocessor.context


# In[6]:


glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


# In[7]:


trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed
)


# In[8]:


padding_callback = mz.models.aNMM.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=20,
    stage='train',
    resample=True,
    sort=False,
    shuffle=True,
    callback=padding_callback,
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=20,
    stage='dev',
    sort=False,
    shuffle=False,
    callback=padding_callback,
)


# In[9]:


model = mz.models.aNMM()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix

model.params['dropout_rate'] = 0.1

model.build()

print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[10]:


optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=15
)

trainer.run()


# In[ ]:




