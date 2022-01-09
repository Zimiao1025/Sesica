import numpy as np
import torch
import matchzoo as mz


def knrm_train(train_set, valid_set, model_path):
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]

    padding_callback = mz.models.KNRM.get_default_padding_callback()

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

    model = mz.models.KNRM()

    model.params['task'] = ranking_task
    model.params['embedding'] = np.empty([10000, 100], dtype=float)
    model.params['kernel_num'] = 21
    model.params['sigma'] = 0.1
    model.params['exact_sigma'] = 0.001

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

    # In[10]:

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
