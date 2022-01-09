import numpy as np
import torch
import matchzoo as mz


def diin_train(train_set, valid_set, model_path):
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

    model = mz.models.DIIN()
    model.params['embedding_input_dim'] = 10000
    model.params['embedding_output_dim'] = 300
    model.params['mask_value'] = 0
    model.params['char_embedding_input_dim'] = 100
    model.params['char_embedding_output_dim'] = 8
    model.params['char_conv_filters'] = 100
    model.params['char_conv_kernel_size'] = 5
    model.params['first_scale_down_ratio'] = 0.3
    model.params['nb_dense_blocks'] = 3
    model.params['layers_per_dense_block'] = 8
    model.params['growth_rate'] = 20
    model.params['transition_scale_down_ratio'] = 0.5
    model.params['conv_kernel_size'] = (3, 3)
    model.params['pool_kernel_size'] = (2, 2)
    model.params['dropout_rate'] = 0.2
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
