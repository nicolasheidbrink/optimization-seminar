import numpy as np
import time
import functions as fn

class Algorithm_Tester:
    def __init__(self):
        np.random.seed(42)

        
        # Define starring points: the parameters are (lb, ub, (nr_tests, dim))
        shubert_2d_starting_points = np.random.uniform(-10, 10, (40, 2))
        shubert_3d_starting_points = np.random.uniform(-10, 10, (10, 3))
        shubert_6d_starting_points = np.random.uniform(-10, 10, (10, 6))
        six_camel_hump_starting_points = np.random.uniform(low=[-3, -2], high=[3, 2], size=(30, 2))
        altered_shubert_starting_points = np.random.uniform(-10, 10, (30, 2))
        function_15_2d_starting_points = np.random.uniform(-10, 10, (30, 2))
        function_15_3d_starting_points = np.random.uniform(-10, 10, (20, 3))
        function_15_4d_starting_points = np.random.uniform(-10, 10, (10, 4))
        function_16_5d_starting_points = np.random.uniform(-10, 10, (10, 5))
        function_16_8d_starting_points = np.random.uniform(-10, 10, (10, 8))
        function_16_10d_starting_points = np.random.uniform(-10, 10, (10, 10))


        self.test_list = []
        self.test_list.append({
            'name': 'Shubert 2D',
            'function': fn.shubert, 
            'grad': fn.shubert_grad, 
            'bounds': [[-10, 10], [-10, 10]], 
            'starting_points': shubert_2d_starting_points, 
            'global_min': -186.730909, 
            'minima_count': 18,
            'nr_trials': shubert_2d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Shubert 3D',
            'function': fn.shubert, 
            'grad': fn.shubert_grad, 
            'bounds': [[-10, 10], [-10, 10], [-10, 10]], 
            'starting_points': shubert_3d_starting_points, 
            'global_min': -2709.093506, 
            'minima_count': 81,
            'nr_trials': shubert_3d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Shubert 6D',
            'function': fn.shubert, 
            'grad': fn.shubert_grad, 
            'bounds': [[-10, 10]]*6, 
            'starting_points': shubert_6d_starting_points, 
            'global_min': -8272701.3783975, 
            'minima_count': 4374,
            'nr_trials': shubert_6d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Six Hump Camel',
            'function': fn.six_hump_camel, 
            'grad': fn.six_hump_camel_grad, 
            'bounds': [[-3, 3], [-2, 2]], 
            'starting_points': six_camel_hump_starting_points, 
            'global_min': -1.031628, 
            'minima_count': 2,
            'nr_trials': six_camel_hump_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Altered Shubert 2D',
            'function': fn.altered_shubert, 
            'grad': fn.altered_shubert_grad, 
            'bounds': [[-10, 10], [-10, 10]], 
            'starting_points': altered_shubert_starting_points, 
            'global_min': -186.730909, 
            'minima_count': 1,
            'nr_trials': altered_shubert_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 15 2D',
            'function': fn.function_15, 
            'grad': fn.function_15_grad,
            'bounds': [[-10, 10], [-10, 10]],
            'starting_points': function_15_2d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_15_2d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 15 3D',
            'function': fn.function_15, 
            'grad': fn.function_15_grad,
            'bounds': [[-10, 10], [-10, 10], [-10, 10]],
            'starting_points': function_15_3d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_15_3d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 15 4D',
            'function': fn.function_15, 
            'grad': fn.function_15_grad,
            'bounds': [[-10, 10], [-10, 10], [-10, 10], [-10, 10]],
            'starting_points': function_15_4d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_15_4d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 16 5D',
            'function': fn.function_16, 
            'grad': fn.function_16_grad,
            'bounds': [[-10, 10]]*5,
            'starting_points': function_16_5d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_16_5d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 16 8D',
            'function': fn.function_16, 
            'grad': fn.function_16_grad,
            'bounds': [[-10, 10]]*8,
            'starting_points': function_16_8d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_16_8d_starting_points.shape[0],
        })
        self.test_list.append({
            'name': 'Function 16 10D',
            'function': fn.function_16, 
            'grad': fn.function_16_grad,
            'bounds': [[-10, 10]]*10,
            'starting_points': function_16_10d_starting_points,
            'global_min': 0.0,
            'minima_count': 1,
            'nr_trials': function_16_10d_starting_points.shape[0],
        })

        self.results_list = []

    
    def run_tests(self, algorithm, **kwargs):
        for test in self.test_list:
            start_time = time.perf_counter()
            avg_nr_minima_found, nr_failures = self.test_algorithm(algorithm, test, **kwargs)
            end_time = time.perf_counter()
            time_per_trial = (end_time - start_time) / test['nr_trials']
            print(f"Test: {test['name']}, average number of minima found when successful: {avg_nr_minima_found:.2f} out of {test['minima_count']}, failure rate: {nr_failures/test['nr_trials']:.2%} over {test['nr_trials']} trials, time per trial: {time_per_trial:.3f} seconds.")

        self.results_list.append({
            'function': test['name'], 
            'avg_pct_minima_found': avg_nr_minima_found/test['nr_trials'], 
            'total_minima': test['minima_count'], 
            'failure_rate': nr_failures/test['nr_trials'],
            'nr_trials': test['nr_trials']
        })

    def test_algorithm(self, algorithm, test, **kwargs):
        np.random.seed(42)
        nr_failures = 0
        found_minima_count = []
        for x0 in test['starting_points']:
            algo = algorithm(f=test['function'], f_grad=test['grad'], bounds=test['bounds'], **kwargs)
            minima, f_min = algo.apply_algorithm(x0)
            if abs(f_min - test['global_min']) < 1e-5:
                found_minima_count.append(len(minima))
            else:
                nr_failures += 1
        mean_found_count = np.mean(found_minima_count) if len(found_minima_count) > 0 else 0
        return mean_found_count, nr_failures
            
