import time

import numpy as np

from utils.util_eval import evaluation


def weight_mean(method, prob_arr, label, group, int_path, params):
    top_n = params['top_n']
    n_dim = sum(list(top_n.values()))

    def normalize(weight):
        return weight / np.sum(weight)

    def target_func(weight):
        weight_nor = normalize(weight)
        wm_prob = np.dot(prob_arr, weight_nor)
        metric_pd = evaluation(params['metrics'], label, wm_prob, group)
        metric_val = metric_pd.mean().tolist()[0]
        print(' --- The value of %s = %.7f ---' % (params['metrics'][0], metric_val))
        return 1 - metric_val

    if method == 'de':
        start_time = time.time()
        print(' DE processing......\n')
        from sko.DE import DE

        de = DE(func=target_func, n_dim=n_dim, size_pop=params['pop_size'], max_iter=params['max_iter'],
                lb=[0] * n_dim, ub=[1] * n_dim)
        best_x, best_y = de.run()
        time_elapsed = time.time() - start_time
        print('The de code run: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    else:
        start_time = time.time()
        print(' GA processing......\n')
        from sko.DE import GA

        ga = GA(func=target_func, n_dim=n_dim, size_pop=params['pop_size'], max_iter=params['max_iter'],
                prob_mut=0.001, lb=[0] * n_dim, ub=[1] * n_dim, precision=1e-5)
        best_x, best_y = ga.run()
        time_elapsed = time.time() - start_time
        print('The ga code run: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    opt_weight = normalize(best_x)
    print('opt_weight:', opt_weight)
    print('best_y: %.4f\n' % (1 - best_y))
    opt_prob = np.dot(prob_arr, opt_weight)
    # save the best weight
    np.save(int_path + 'opt_weight.npy', opt_weight)
    # save the prob for plot
    np.save(int_path + 'prob.npy', opt_prob)
    # save the evaluation results
    metric_df = evaluation(params['metrics'], label, opt_prob, group)
    metric_df.to_csv(int_path + 'eval_results.csv')
