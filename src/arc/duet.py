#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('/data/users/fyx/NTMC-Community/MatchZoo-py/')
import matchzoo as mz


# In[2]:


get_ipython().run_line_magic('run', 'init.ipynb')


# In[3]:


preprocessor = mz.models.DUET.get_default_preprocessor(
    filter_mode='df',
    filter_low_freq=2,
    truncated_mode='post',
    truncated_length_left=10,
    truncated_length_right=40,
    ngram_size=3
)


# In[4]:


train_pack_processed = preprocessor.fit_transform(train_pack_raw)
dev_pack_processed = preprocessor.transform(dev_pack_raw)
test_pack_processed = preprocessor.transform(test_pack_raw)


# In[5]:


preprocessor.context


# In[6]:


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


# In[7]:


padding_callback = mz.models.DUET.get_default_padding_callback(
    fixed_length_left=10,
    fixed_length_right=40,
    pad_word_value=0,
    pad_word_mode='pre',
    with_ngram=True
)

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


model = mz.models.DUET()

model.params['task'] = ranking_task
model.params['left_length'] = 10
model.params['right_length'] = 40
model.params['lm_filters'] = 100
model.params['mlp_num_layers'] = 2
model.params['mlp_num_units'] = 100
model.params['mlp_num_fan_out'] = 100
model.params['mlp_activation_func'] = 'tanh'

model.params['vocab_size'] = preprocessor.context['ngram_vocab_size']
model.params['dm_conv_activation_func'] = 'relu'
model.params['dm_filters'] = 100
model.params['dm_kernel_size'] = 3
model.params['dm_right_pool_size'] = 4
model.params['dropout_rate'] = 0.2


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


# In[ ]:


trainer.run()


# In[ ]:




