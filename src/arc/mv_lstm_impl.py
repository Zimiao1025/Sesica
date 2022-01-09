import numpy as np
import torch
import matchzoo as mz


def mv_lstm_train(train_set, valid_set, model_path):
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]

    padding_callback = mz.models.ESIM.get_default_padding_callback()

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

    # In[9]:

    model = mz.models.MVLSTM()
    model.params['hidden_size'] = 32
    model.params['top_k'] = 50
    model.params['mlp_num_layers'] = 2
    model.params['mlp_num_units'] = 20
    model.params['mlp_num_fan_out'] = 10
    model.params['mlp_activation_func'] = 'relu'
    model.params['dropout_rate'] = 0.0
    model.guess_and_fill_missing_params(verbose=0)

    model.build()

    print(model)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # In[10]:

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

    # In[11]:

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
