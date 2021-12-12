#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[2]:


preprocessor = mz.models.MatchSRNN.get_default_preprocessor()


# In[3]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[4]:


preprocessor.context


# In[5]:


glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=100)
term_index = preprocessor.context['vocab_unit'].state['term_index']
embedding_matrix = glove_embedding.build_matrix(term_index)
l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]


# In[6]:


trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed
)


# In[7]:


padding_callback = mz.models.MatchSRNN.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=20,
    stage='train',
    resample=True,
    sort=False,
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=20,
    stage='dev',
    callback=padding_callback
)


# In[8]:


model = mz.models.MatchSRNN()

model.params['task'] = ranking_task
model.params['embedding'] = embedding_matrix
model.params['channels'] = 4
model.params['units'] = 10
model.params['dropout'] = 0.2
model.params['direction'] = 'lt'

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[9]:


optimizer = torch.optim.Adadelta(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=testloader,
    validate_interval=None,
    epochs=10
)


# In[10]:


trainer.run()

