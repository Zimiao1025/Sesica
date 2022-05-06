import matchzoo as mz
import torch


def duet_train(train_set, valid_set, test_set, model_path, ind_set=None, params=None):
    # Make use of MatchZoo customized loss functions and evaluation metrics to define a task:
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=params['num_neg']['duet']))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]

    padding_callback = mz.models.DUET.get_default_padding_callback(
        fixed_length_left=params['vocab_size'],
        fixed_length_right=params['vocab_size'],
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
    test_loader = mz.dataloader.DataLoader(
        dataset=test_set,
        stage='dev',
        callback=padding_callback
    )
    if ind_set:
        ind_loader = mz.dataloader.DataLoader(
            dataset=ind_set,
            stage='dev',
            callback=padding_callback
        )
    else:
        ind_loader = None

    model = mz.models.DUET()
    model.params['task'] = ranking_task
    model.params['left_length'] = params['vocab_size']
    model.params['right_length'] = params['vocab_size']
    model.params['mlp_num_layers'] = params['duet_layers']
    model.params['mlp_num_units'] = params['duet_units']
    model.params['vocab_size'] = params['vocab_size']
    model.params['dropout_rate'] = params['duet_dropout']
    # params['mlp_num_fan_out'] = 3
    model.params['lm_filters'] = 3
    model.params['dm_filters'] = 3
    model.build()
    print(model)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adadelta(model.parameters(), lr=params['duet_lr'])

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        validloader=valid_loader,
        validate_interval=None,
        epochs=params['duet_epoch'],
        model_path=model_path
    )

    trainer.run()
    trainer.save_model()

    return trainer, valid_loader, test_loader, ind_loader
