import numpy as np

from utils.util_data import normalize_prob
from utils.util_eval import evaluation


def prob_metric_cal(trainer, model_path, valid_y, valid_loader, valid_g, test_y, test_loader, test_g,
                    ind_loader=None, ind_y=None, ind_g=None, params=None):
    # valid
    valid_prob = trainer.predict(valid_loader)
    # test
    test_prob = trainer.predict(test_loader)
    # ind
    if ind_loader:
        ind_prob = trainer.predict(ind_loader)
    else:
        ind_prob = None
    # save normalized prob and evaluation results.
    # valid
    # valid_prob = normalize_prob(valid_prob)
    valid_prob = valid_prob.flatten()
    np.save(model_path + 'valid_prob.npy', valid_prob)
    # print(params['metrics'])
    # print('normalized valid label: ', valid_y)
    print('Valid prob: ', valid_prob)
    metric_df = evaluation(params['metrics'], valid_y, valid_prob, valid_g)
    metric_df.to_csv(model_path + 'valid_results.csv')
    metric_list = metric_df.mean().tolist()
    print('Evaluation on validation dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
    # test
    # test_prob = normalize_prob(test_prob)
    test_prob = test_prob.flatten()
    np.save(model_path + 'test_prob.npy', test_prob)
    metric_df = evaluation(params['metrics'], test_y, test_prob, test_g)
    metric_list = metric_df.mean().tolist()
    print('Evaluation on testing dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
    # ind
    if ind_prob:
        # ind_prob = normalize_prob(ind_prob)
        ind_prob = ind_prob.flatten()
        np.save(model_path + 'ind_prob.npy', ind_prob)
        metric_df = evaluation(params['metrics'], ind_y, ind_prob, ind_g)
        metric_list = metric_df.mean().tolist()
        metric_df.to_csv(model_path + 'ind_results.csv')
        print('Evaluation on independent dataset: %s = %.4f\n' % (params['metrics'][0], metric_list[0]))
