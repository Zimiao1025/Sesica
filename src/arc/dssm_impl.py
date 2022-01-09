import torch
import matchzoo as mz


def dssm_train(train_set, valid_set, model_path):
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]
    # Define padding callback and generate data loader:
    padding_callback = mz.models.DSSM.get_default_padding_callback()

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

    model = mz.models.DSSM()
    model.params['task'] = ranking_task
    model.params['vocab_size'] = 19  # preprocessor.context['ngram_vocab_size'] --> embedding_input_dim
    model.params['mlp_num_layers'] = 3
    model.params['mlp_num_units'] = 300
    model.params['mlp_num_fan_out'] = 128
    model.params['mlp_activation_func'] = 'relu'

    model.build()

    optimizer = torch.optim.Adam(model.parameters())

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        validloader=valid_loader,
        validate_interval=None,
        epochs=10,
        model_path=model_path,
    )

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
