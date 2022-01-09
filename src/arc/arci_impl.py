import torch
import matchzoo as mz


def arci_train(train_set, valid_set, model_path):
    # To train a DSSM, make use of MatchZoo customized loss functions and evaluation metrics to define a task:
    ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
    # print(ranking_task)
    # exit()
    ranking_task.metrics = [
        mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
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

    # Initialize the model, fine-tune the hyper-parameters:
    model = mz.models.ArcI()
    model.params['task'] = ranking_task
    model.params['embedding_output_dim'] = 100
    model.params['embedding_input_dim'] = 30001  # preprocessor.context['embedding_input_dim']
    model.guess_and_fill_missing_params()
    model.build()

    # Trainer is used to control the training flow:
    optimizer = torch.optim.Adam(model.parameters())

    trainer = mz.trainers.Trainer(
        model=model,
        optimizer=optimizer,
        trainloader=train_loader,
        validloader=valid_loader,
        epochs=10,
        save_dir=model_path
    )

    trainer.run()
    valid_prob = trainer.predict(valid_loader)
    print(valid_prob)
    trainer.save_model()
