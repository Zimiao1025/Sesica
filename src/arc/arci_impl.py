import matchzoo as mz
import torch


def arci_train(train_set, valid_set, test_set, model_path, ind_set=None, params=None):
    # Make use of MatchZoo customized loss functions and evaluation metrics to define a task:
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=params['num_neg']['arci']))
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
        mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
        mz.metrics.MeanAveragePrecision()
    ]
    # Define padding callback and generate data loader:
    padding_callback = mz.models.ArcI.get_default_padding_callback()

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

    # Initialize the model, fine-tune the hyper-parameters:
    model = mz.models.ArcI()
    model.params['task'] = ranking_task
    model.params['embedding_input_dim'] = params['arci_emb_in']
    model.params['embedding_output_dim'] = params['arci_emb_out']
    model.params['mlp_num_layers'] = params['arci_layers']
    model.params['mlp_num_units'] = params['arci_units']
    model.params['dropout_rate'] = params['arci_dropout']
    model.guess_and_fill_missing_params()
    model.build()
    print(model)
    print('Trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Trainer is used to control the training flow:
    optimizer = torch.optim.Adam(model.parameters(), lr=params['arci_lr'])

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        validloader=valid_loader,
        epochs=params['arci_epoch'],
        save_dir=model_path
    )

    trainer.run()
    trainer.save_model()

    return trainer, valid_loader, test_loader, ind_loader
