import numpy as np
import itertools
import copy
import re

class GridSearch(object):
    """Class to perform grid searches across algorithm and problem settings

    Special cases:
        If param_grid_algorithm['pop_size'] is any of the following strings, a special case is applied:
            'problem_length': pop_size will be set to param_grid_problem['length']
            '2*problem_length': pop_size will be set to 2*param_grid_problem['length']
    """
    def __init__(self, algorithm = None, param_grid_algorithm = None, problem = None, 
                 param_grid_problem = None, iters = 1):
        """more docs
        """

        self.iters = iters

        self.problem = problem
        self._param_grid_problem = None
        self._param_grid_problem_combinations = None
        self.param_grid_problem = param_grid_problem

        self.algorithm = algorithm
        self._param_grid_algorithm = None
        self._param_grid_algorithm_combinations = None
        self.param_grid_algorithm = param_grid_algorithm

        self.results = []

    def evaluate(self):
        n_settings = len(self._param_grid_problem_combinations) * len(self._param_grid_algorithm_combinations)
        i_settings = 0
        for problem_settings in self._param_grid_problem_combinations:
            for algorithm_settings in self._param_grid_algorithm_combinations:
                try:
                    match = re.match(r'(\d*\*)?problem_length', algorithm_settings['pop_size'])
                    if match:
                        if match.group(1) is None:
                            multiplier = 1
                        else:
                            multiplier = int(match.group(1).strip('*'))
                        # Work on a copy so you don't overwrite the special case for later iterations
                        algorithm_settings = copy.deepcopy(algorithm_settings)
                        algorithm_settings['pop_size'] = multiplier * problem_settings['length']
                except KeyError:
                    pass
                except TypeError:
                    pass
                i_settings += 1
                results = []
                for i in range(self.iters):
                    print(f"Running case {i_settings}/{n_settings}: iteration {i}")
                    thisProblem = self.problem(**problem_settings)
                    solution, fitness, statistics = self.algorithm(thisProblem, **algorithm_settings)
                    results.append(statistics)

                self.record_results(results=results, 
                                    problem_settings=problem_settings,
                                    algorithm_settings=algorithm_settings,
                                    maximize=thisProblem.maximize) 

        # for i in range(self._param_grid_problem_combinations):
        #     pg_a = self._param_grid_algorithm_combinations[i]
        #     pg_p = self._param_grid_problem_combinations[i]

        #     thisProblem = self.problem()

    def record_results(self, results = None, problem_settings = None, 
                       algorithm_settings = None, maximize = 1):
        """Add a record of these results to the ledger

        if maximize=-1 (nomenclature for a minimization problem in mlrose), we find the argmin of the fitness
        to define best evals.  Otherwise use argmax
        """
        resultDict = {}

        # Add settings
        for setting_name, setting_value in problem_settings.items():
            resultDict['param_problem_' + setting_name] = setting_value
        for setting_name, setting_value in algorithm_settings.items():
            resultDict['param_algorithm_' + setting_name] = setting_value

        # Metrics to be recorded from each case
        metrics = ['best_fitness', 'fitness_evals', 'time', 'iters', 'best_state']

        # Add aggregated results
        for key in metrics:
            # Aggregate, unless for the best state vector
            if key != 'best_state':
                collected = np.array([result[key] for result in results])
                resultDict['mean_' + key] = collected.mean()
                resultDict['std_' + key] = collected.std()
                resultDict['max_' + key] = collected.max()
            for i in range(len(results)):
                resultDict[f'it{i}_' + key] = results[i][key]

        # History metrics to be recorded and aggregated for each case
        history_metrics = ['fitness_by_iteration', 'fitness_evals_history']
        for key in history_metrics:
            for i in range(len(results)):
                resultDict[f'it{i}_' + key] = results[i][key]

        # Derived statistics from some history metrics
        # Record the mean of fitness evaluations when we first hit best state
        # If this algorithm has restarts, take the first instance of the best
        resultDict[f'mean_iters_to_best_state'] = 0
        resultDict[f'mean_fitness_evals_to_best_state'] = 0
        for i in range(len(results)):
            # Flatten and aggregate the fitness_by_iteration and
            # fitness_evals_history arrays
            flat_fitness = results[i]['fitness_by_iteration'][0]
            flat_evals = results[i]['fitness_evals_history'][0]

            for j in range(1, len(results[i]['fitness_by_iteration'])):
                flat_fitness = np.concatenate((flat_fitness, 
                    results[i]['fitness_by_iteration'][j]))
                flat_evals = np.concatenate((flat_evals, 
                    results[i]['fitness_evals_history'][j] + flat_evals[-1]))

            iBestState = np.argmax(flat_fitness * maximize)
            resultDict[f'mean_iters_to_best_state'] += iBestState + 1
            resultDict[f'mean_fitness_evals_to_best_state'] += \
                flat_evals[iBestState]
            resultDict[f'it{i}_fitness_by_iteration_flattened'] = flat_fitness
            resultDict[f'it{i}_fitness_evals_history_flattened'] = flat_evals
        resultDict[f'mean_iters_to_best_state'] /= float(len(results))
        resultDict[f'mean_fitness_evals_to_best_state'] /= float(len(results))

        self.results.append(resultDict)

    @property
    def param_grid_problem(self):
        return self._param_grid_problem
    
    @param_grid_problem.setter
    def param_grid_problem(self, param_grid_problem):
        self._param_grid_problem = param_grid_problem
        self._param_grid_problem_combinations = param_grid_combinations(param_grid_problem)

    @property
    def param_grid_algorithm(self):
        return self._param_grid_algorithm
    
    @param_grid_algorithm.setter
    def param_grid_algorithm(self, param_grid_algorithm):
        self._param_grid_algorithm = param_grid_algorithm
        self._param_grid_algorithm_combinations = param_grid_combinations(param_grid_algorithm)


# Helpers
def param_grid_combinations(param_grid):
    """Returns a list of dicts of all param_grid settings combinations"""
    param_val_tuple_lists = []
    for param_name in sorted(param_grid):
        param_val_tuple_lists.append([])
        values = param_grid[param_name]

        # Catch strings which may look like iterables but don't make the cut below...
        if isinstance(values, str):
            values = [values]

        # Does it look like an iterable?  If not, promote
        try:
            values = values[:]
        except TypeError:
            values = [values]

        # Store as a list of (param_name, param_value) tuples
        for value in values:
            param_val_tuple_lists[-1].append((param_name, value))

    settings_combinations = list(itertools.product(*param_val_tuple_lists))
    settings_as_dicts = [dict(settings) for settings in settings_combinations]
    return settings_as_dicts