import numpy as np
import torch
import matchzoo as mz


def anmm_train(train_set, valid_set, test_set, model_path, ind_set=None, params=None):
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=params['num_neg']['anmm']))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]
    padding_callback = mz.models.aNMM.get_default_padding_callback()

    train_loader = mz.dataloader.DataLoader(
        dataset=train_set,
        stage='train',
        callback=padding_callback,
    )
    valid_loader = mz.dataloader.DataLoader(
        dataset=valid_set,
        stage='dev',
        callback=padding_callback,
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

    model = mz.models.aNMM()
    model.params['task'] = ranking_task
    model.params['embedding'] = np.empty([10000, 100], dtype=float)
    model.params['dropout_rate'] = 0.1
    model.build()
    print(model)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

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
