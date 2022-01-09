import matchzoo as mz
import torch


def duet_train(train_set, valid_set, model_path):
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]

    padding_callback = mz.models.DUET.get_default_padding_callback(
        fixed_length_left=10,
        fixed_length_right=40,
        pad_word_value=0,
        pad_word_mode='pre',
        with_ngram=True
    )

    train_loader = mz.dataloader.DataLoader(
        dataset=train_set,
        stage='train',
        callback=padding_callback
    )
    valid_loader = mz.dataloader.DataLoader(
        dataset=valid_set,
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

    model.params['vocab_size'] = 3000  # preprocessor.context['ngram_vocab_size']
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
        trainloader=train_loader,
        validloader=valid_loader,
        validate_interval=None,
        epochs=10,
        model_path=model_path
    )

    # In[ ]:

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
