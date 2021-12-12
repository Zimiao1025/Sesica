import torch
import matchzoo as mz

from pd_test import trans_pd

# Prepare input data:
train_data = trans_pd('train_processed.csv')
# print(train_data['text_left'].values)
# exit()
valid_data = trans_pd('valid_processed.csv')

train_processed = mz.pack(train_data)
valid_processed = mz.pack(valid_data)

# To train a DSSM, make use of MatchZoo customized loss functions and evaluation metrics to define a task:
ranking_task = mz.tasks.Ranking(losses=mz.losses.RankCrossEntropyLoss(num_neg=4))
# print(ranking_task)
# exit()
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.MeanAveragePrecision()
]

# Generate pair-wise training data on-the-fly:
trainset = mz.dataloader.Dataset(
    data_pack=train_processed,
    mode='pair',
    num_dup=1,
    num_neg=4,
    batch_size=32
)
validset = mz.dataloader.Dataset(
    data_pack=valid_processed,
    mode='point',
    batch_size=32
)

# Define padding callback and generate data loader:
padding_callback = mz.models.ArcI.get_default_padding_callback()

trainloader = mz.dataloader.DataLoader(
    dataset=trainset,
    stage='train',
    callback=padding_callback
)
validloader = mz.dataloader.DataLoader(
    dataset=validset,
    stage='dev',
    callback=padding_callback
)

# Initialize the model, fine-tune the hyper-parameters:
model = mz.models.ArcI()
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 100
model.params['embedding_input_dim'] = 30059  # preprocessor.context['embedding_input_dim']
model.guess_and_fill_missing_params()
model.build()

# Trainer is used to control the training flow:
optimizer = torch.optim.Adam(model.parameters())

trainer = mz.trainers.Trainer(
    model=model,
    optimizer=optimizer,
    trainloader=trainloader,
    validloader=validloader,
    epochs=10
)

trainer.run()
