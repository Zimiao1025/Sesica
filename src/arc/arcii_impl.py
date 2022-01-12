import numpy as np
import torch
import matchzoo as mz


def arcii_train(train_set, valid_set, test_set, model_path, ind_set=None, params=None):
    # Make use of MatchZoo customized loss functions and evaluation metrics to define a task:
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=params['num_neg']['arcii']))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]

    padding_callback = mz.models.ArcII.get_default_padding_callback(
        fixed_length_left=10,
        fixed_length_right=100,
        pad_word_value=0,
        pad_word_mode='pre'
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

    model = mz.models.ArcII()

    model.params['task'] = ranking_task
    model.params['embedding'] = np.empty([10000, 100], dtype=float)
    model.params['left_length'] = 10
    model.params['right_length'] = 100
    model.params['kernel_1d_count'] = 32
    model.params['kernel_1d_size'] = 3
    model.params['kernel_2d_count'] = [64, 64]
    model.params['kernel_2d_size'] = [(3, 3), (3, 3)]
    model.params['pool_2d_size'] = [(3, 3), (3, 3)]
    model.params['dropout_rate'] = 0.3
    model.build()
    print(model)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters())

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        validloader=valid_loader,
        validate_interval=None,
        epochs=10,
        model_path=model_path
    )

    trainer.run()
    trainer.save_model()

    return trainer, valid_loader, test_loader, ind_loader
