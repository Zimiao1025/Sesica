import numpy as np
from sko.DE import DE
from sko.GA import GA

from utils.util_eval import evaluation


def weight_mean(method, prob_arr, label, group, int_path, params):
    top_n = params['top_n']
    n_dim = sum(list(top_n.values()))

    def target_func(weight):
        wm_prob = np.dot(prob_arr, weight)
        metric_pd = evaluation(params['metrics'], wm_prob, label, group)
        return metric_pd.mean().tolist()[0]

    def constraint_func(weight):
        return 1 - sum(weight)

    if method == 'de':
        de = DE(func=target_func, n_dim=n_dim, size_pop=params['size_pop'], max_iter=params['max_iter'],
                lb=[0]*n_dim, ub=[1]*n_dim, constraint_eq=constraint_func)
        best_x, best_y = de.run()
        print(' DE processing......\n')
        print('best_x:', best_x, '\n', 'best_y:', best_y)
    else:
        ga = GA(func=target_func, n_dim=n_dim, size_pop=params['size_pop'], max_iter=params['max_iter'],
                prob_mut=0.001, lb=[0]*n_dim, ub=[1]*n_dim, precision=1e-7)
        best_x, best_y = ga.run()
        print(' GA processing......\n')
        print('best_x:', best_x, '\n', 'best_y:', best_y)

    opt_prob = np.dot(prob_arr, best_x)
    # save the best weight
    np.save(int_path + 'opt_weight.npy', best_x)
    # save the prob for plot
    np.save(int_path + 'opt_prob.npy', opt_prob)
    # save the evaluation results
    metric_df = evaluation(params['metrics'], opt_prob, label, group)
    metric_df.to_csv(int_path + 'eval_results.csv')
