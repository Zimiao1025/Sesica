import torch
import matchzoo as mz


def cdssm_train(train_set, valid_set, model_path):
    # ranking_task = mz.tasks.Ranking(losses=mz.losses.RankHingeLoss())
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]
    # print("`ranking_task` initialized with metrics", ranking_task.metrics)

    padding_callback = mz.models.CDSSM.get_default_padding_callback(
        with_ngram=True,
        fixed_length_left=19,
        fixed_length_right=19,
        fixed_ngram_length=19
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

    # In[7]:

    model = mz.models.CDSSM()

    model.params['task'] = ranking_task
    model.params['vocab_size'] = 19
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
        trainloader=train_loader,
        validloader=valid_loader,
        validate_interval=None,
        epochs=10,
        model_path=model_path
    )

    # In[9]:

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
