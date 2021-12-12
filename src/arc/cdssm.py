#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[2]:


preprocessor = mz.models.CDSSM.get_default_preprocessor(
    ngram_size = 3
)


# In[3]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
valid_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[4]:


preprocessor.context


# In[5]:


triletter_callback = mz.dataloader.callbacks.Ngram(preprocessor, mode='sum')
trainset = mz.dataloader.Dataset(
    data_pack=train_pack_processed,
    mode='pair',
    num_dup=2,
    num_neg=1,
    callbacks=[triletter_callback]
)
testset = mz.dataloader.Dataset(
    data_pack=test_pack_processed,
    callbacks=[triletter_callback]
)


# In[6]:


padding_callback = mz.models.CDSSM.get_default_padding_callback(
    with_ngram=True,
    fixed_ngram_length=preprocessor.context['ngram_vocab_size']
)

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    batch_size=20,
    stage='train',
    sort=False,
    resample=True,
    callback=padding_callback
)
testloader = mz.dataloader.DataLoader(
    dataset=testset,
    batch_size=20,
    stage='dev',
    sort=False,
    callback=padding_callback
)


# In[7]:


model = mz.models.CDSSM()

model.params['task'] = ranking_task
model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
model.params['filters'] = 64
model.params['kernel_size'] = 3
model.params['conv_activation_func'] = 'tanh'
model.params['mlp_num_layers'] = 1
model.params['mlp_num_units'] = 64
model.params['mlp_num_fan_out'] = 64
model.params['mlp_activation_func'] = 'tanh'
model.params['dropout_rate'] = 0.8

model.build()

print(model)
print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))


# In[8]:


optimizer = torch.optim.Adadelta(model.parameters())

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

