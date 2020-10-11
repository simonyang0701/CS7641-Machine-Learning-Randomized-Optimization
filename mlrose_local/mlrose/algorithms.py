""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
from .decay import GeomDecay
from time import perf_counter
import itertools

def hill_climb(problem, max_iters=np.inf, restarts=0, init_state=None, return_statistics=False):
    """Use standard hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    return_statistics: bool, default: False
        If True, return includes dictionary of optimization run statistics

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    statistics: dict
        (Optional) Dictionary containing statistics from optimization:
            iters int total number of iterations over all restarts
            time: total time in seconds over all restarts
            fitness_evals: int total number of fitness evaluations over all
                           restarts
            fitness_by_iteration: list of 1D arrays of fitness reached at each
                                  iteration.  Each list item is for a 
                                  corresponding restart  
            fitness_history: 1D array of fitness reached on each restart
            state_history: 1D array of state reached on each restart
            iters_history: 1D array of iters run on each restart
            time_history: 1D array of time taken to run each restart
            fitness_evals_history: list of 1D arrays of number of evaluations
                                   invoked up to this iteration on this restart

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    # Initialize storage for statistics from each restart
    if return_statistics:
        fitness_history = []
        state_history = []
        iters_history = []
        time_history = []
        fitness_evals_history = []
        fitness_by_iteration = []

    for i_restart in range(restarts + 1):
        if return_statistics:
            fitness_evals_start = problem.fitness_evals # Could move this to a self.get_fitness()
            time_start = perf_counter()
            fitness_by_iteration.append([])
            fitness_evals_history.append([])

        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        iters = 0

        while iters < max_iters:
            # Store fitness of previous iteration
            if return_statistics:
                fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
                fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

            iters += 1

            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            next_state = problem.best_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)

            else:
                break

        # Store statistics
        if return_statistics:
            fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)
            fitness_history.append(problem.get_fitness() * problem.get_maximize())
            state_history.append(problem.get_state())
            iters_history.append(iters)
            time_history.append(perf_counter() - time_start)
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness

    if return_statistics:
        statistics = {
            'fitness_history': np.array(fitness_history),
            'state_history': np.array(state_history),
            'iters_history': np.array(iters_history),
            'time_history': np.array(time_history),
            'fitness_evals_history': [np.array(feh) for feh in fitness_evals_history],
            'fitness_by_iteration': [np.array(fbi) for fbi in fitness_by_iteration],
            'iters': int(np.sum(iters_history)),
            'time': np.sum(time_history),
            'fitness_evals': int(np.sum([feh[-1] for feh in fitness_evals_history])),
            'best_state': best_state,
            'best_fitness': best_fitness,
        }

        return best_state, best_fitness, statistics
    else:
        return best_state, best_fitness


def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, return_statistics=False):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    return_statistics: bool, default: False
        If True, return includes dictionary of optimization run statistics

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    statistics: dict
        (Optional) Dictionary containing statistics from optimization:
            iters int total number of iterations over all restarts
            time: total time in seconds over all restarts
            fitness_evals: int total number of fitness evaluations over all
                           restarts
            fitness_by_iteration: list of 1D arrays of fitness reached at each
                                  iteration.  Each list item is for a 
                                  corresponding restart  
            fitness_history: 1D array of fitness reached on each restart
            state_history: 1D array of state reached on each restart
            iters_history: 1D array of iters run on each restart
            time_history: 1D array of time taken to run each restart
            fitness_evals_history: list of 1D arrays of number of evaluations
                                   invoked up to this iteration on this restart

    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming
    Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    # Initialize storage for statistics from each restart
    if return_statistics:
        fitness_history = []
        state_history = []
        iters_history = []
        time_history = []
        fitness_evals_history = []
        fitness_by_iteration = []

    for _ in range(restarts + 1):
        if return_statistics:
            fitness_evals_start = problem.fitness_evals # Could move this to a self.get_fitness()
            time_start = perf_counter()
            fitness_by_iteration.append([])
            fitness_evals_history.append([])

        # Initialize optimization problem and attempts counter
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            if return_statistics:
                fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
                fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

            iters += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1

        # Store statistics
        if return_statistics:
            fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)
            fitness_history.append(problem.get_fitness() * problem.get_maximize())
            state_history.append(problem.get_state())
            iters_history.append(iters)
            time_history.append(perf_counter() - time_start)
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness

    if return_statistics:
        statistics = {
            'fitness_history': np.array(fitness_history),
            'state_history': np.array(state_history),
            'iters_history': np.array(iters_history),
            'time_history': np.array(time_history),
            'fitness_evals_history': [np.array(feh) for feh in fitness_evals_history],
            'fitness_by_iteration': [np.array(fbi) for fbi in fitness_by_iteration],
            'iters': int(np.sum(iters_history)),
            'time': np.sum(time_history),
            'fitness_evals': int(np.sum([feh[-1] for feh in fitness_evals_history])),
            'best_state': best_state,
            'best_fitness': best_fitness,
        }

        return best_state, best_fitness, statistics
    else:
        return best_state, best_fitness


def simulated_annealing(problem, schedule=GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, 
                        return_statistics=False):
    """Use simulated annealing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.
    return_statistics: bool, default: False
        If True, return includes dictionary of optimization run statistics

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    statistics: dict
        (Optional) Dictionary containing statistics from optimization:
            iters int total number of iterations over all restarts
            time: total time in seconds over all restarts
            fitness_evals: int total number of fitness evaluations over all
                           restarts
            fitness_by_iteration: list of 1D arrays of fitness reached at each
                                  iteration.  Each list item is for a 
                                  corresponding restart (for Simulated 
                                  Annealing, this is always a single element
                                  list)
            fitness_evals_history: list of 1D arrays of number of evaluations
                                   invoked up to this iteration on this restart
                                   (for Simulated Annealing, this is always a
                                   single element list)

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0

    # Initialize storage for statistics
    if return_statistics:
        fitness_evals_start = problem.fitness_evals # Could move this to a self.get_fitness()
        time_start = perf_counter()
        fitness_by_iteration = [[]]
        fitness_evals_history = [[]]


    while (attempts < max_attempts) and (iters < max_iters):
        # Store fitness of previous iteration
        if return_statistics:
            fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        temp = schedule.evaluate(iters)
        iters += 1

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            chance = np.random.uniform()
            if (delta_e > 0) or (chance < prob):
                # if delta_e > 0:
                #     print('found improvement')
                # elif chance < prob:
                #     print('found chance < prob')
                # else:
                #     raise Exception("how did I get here?")
                problem.set_state(next_state)

                # Cound lateral steps as attempts.  They're not really 
                # improvements.  Otherwise, max_attempts is rarely reached
                if abs(delta_e) < 0.000001:
                    attempts += 1
                else:
                    attempts = 0

            else:
                attempts += 1

    if attempts >= max_attempts:
        print(f"Search ended with attempts>max_attempts ({attempts}>{max_attempts}).  Iters={iters}")
    elif iters >= max_iters:
        print(f"Search ended with attempts>max_attempts ({iters}>{max_iters})")
    else:
        raise Exception("How did I get here?")

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if return_statistics:
        fitness_by_iteration[-1].append(problem.get_maximize()*problem.get_fitness())
        fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)
        statistics = {
            'iters': iters,
            'time': perf_counter() - time_start,
            'fitness_evals': problem.fitness_evals - fitness_evals_start,
            'best_state': best_state,
            'best_fitness': best_fitness,
            'fitness_by_iteration': [np.array(fbi) for fbi in fitness_by_iteration],
            'fitness_evals_history': [np.array(feh) for feh in fitness_evals_history],
        }

        return best_state, best_fitness, statistics
    else:
        return best_state, best_fitness


def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf, elite=0, return_statistics=False):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    elite: float, default: 0
        Fraction of population's best members to keep from one iteration to next
        This is the maximum number of members transferred between generations. 
        If the elite group has duplicates, they are removed before passing to 
        next generation
    return_statistics: bool, default: False
        If True, return includes dictionary of optimization run statistics

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    statistics: dict
        (Optional) Dictionary containing statistics from optimization:
            iters int total number of iterations over all restarts
            time: total time in seconds over all restarts
            fitness_evals: int total number of fitness evaluations over all
                           restarts
            fitness_by_iteration: list of 1D arrays of fitness reached at each
                      iteration.  Each list item is for a 
                      corresponding restart (for Genetic Algorithm, this is 
                      always a single element list)
            fitness_evals_history: list of 1D arrays of number of evaluations
                                   invoked up to this iteration on this restart
                                   (for Genetic Algorithm, this is always a
                                   single element list)

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    # Initialize storage for statistics
    if return_statistics:
        fitness_evals_start = problem.fitness_evals # Could move this to a self.get_fitness()
        time_start = perf_counter()
        fitness_by_iteration = [[]]
        fitness_evals_history = [[]]

    print(f"running with elite = {elite}")
    while (attempts < max_attempts) and (iters < max_iters):
        # Store fitness of previous iteration
        if return_statistics:
            fitness_by_iteration[-1].append(problem.get_fitness() * problem.get_maximize())
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        iters += 1

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        # Save elite parents
        if elite > 0:
            problem.find_top_pct(elite)
            top_pct = problem.get_keep_sample()
            # top_pct returns the top percent including ties.  Truncate if needed
            top_pct = top_pct[:int(pop_size*elite)]

            # Filter out identical items from top_pct
            top_pct = np.unique(top_pct, axis=0)
            
            # print(f'top_pct ({len(top_pct)}): {top_pct}')
            # for tp in top_pct:
            #     print(problem.eval_fitness(tp))
            next_gen.extend(top_pct)

        for _ in range(pop_size - len(next_gen)):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    if attempts >= max_attempts:
        print(f"Search ended with attempts>max_attempts ({attempts}>{max_attempts}).  Iters={iters}")
    elif iters >= max_iters:
        print(f"Search ended with attempts>max_attempts ({iters}>{max_iters})")
    else:
        raise Exception("How did I get here?")

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    if return_statistics:
        fitness_by_iteration[-1].append(problem.get_maximize()*problem.get_fitness())
        fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        statistics = {
            'iters': iters,
            'time': perf_counter() - time_start,
            'fitness_evals': problem.fitness_evals - fitness_evals_start,
            'best_state': best_state,
            'best_fitness': best_fitness,
            'fitness_by_iteration': [np.array(fbi) for fbi in fitness_by_iteration],
            'fitness_evals_history': [np.array(feh) for feh in fitness_evals_history],
        }
        return best_state, best_fitness, statistics
    else:
        return best_state, best_fitness


def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
          max_iters=np.inf, return_statistics=False):
    """Use MIMIC to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()` or :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in algorithm.
    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm,
        expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    return_statistics: bool, default: False
        If True, return includes dictionary of optimization run statistics

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    statistics: dict
        (Optional) Dictionary containing statistics from optimization:
            iters int total number of iterations over all restarts
            time: total time in seconds over all restarts
            fitness_evals: int total number of fitness evaluations over all
                           restarts
            fitness_by_iteration: list of 1D arrays of fitness reached at each
                      iteration.  Each list item is for a 
                      corresponding restart (for MIMIC, this is always a single
                      element list)
            fitness_evals_history: list of 1D arrays of number of evaluations
                                   invoked up to this iteration on this restart
                                   (for MIMIC, this is always a single element 
                                   list)
    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    MIMIC cannot be used for solving continuous-state optimization problems.
    """
    if problem.get_prob_type() == 'continuous':
        raise Exception("""problem type must be discrete or tsp.""")

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (keep_pct < 0) or (keep_pct > 1):
        raise Exception("""keep_pct must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    # Initialize storage for statistics
    if return_statistics:
        fitness_evals_start = problem.fitness_evals # Could move this to a self.get_fitness()
        time_start = perf_counter()
        fitness_by_iteration = [[]]
        fitness_evals_history = [[]]

    while (attempts < max_attempts) and (iters < max_iters):
        # Store fitness of previous iteration
        if return_statistics:
            # This the appropriate measure of fitness here?
            fitness_by_iteration[-1].append(problem.eval_fitness(problem.best_child()) * problem.get_maximize())
            fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        iters += 1

        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        next_state = problem.best_child()

        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1
    print(f"MIMIC finished after using {attempts}/{max_attempts} attempts, {iters}/{max_iters}, iters")

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)
    
    if return_statistics:
        fitness_by_iteration[-1].append(problem.get_maximize()*problem.get_fitness())
        fitness_evals_history[-1].append(problem.fitness_evals - fitness_evals_start)

        statistics = {
            'iters': iters,
            'time': perf_counter() - time_start,
            'fitness_evals': problem.fitness_evals - fitness_evals_start,
            'best_state': best_state,
            'best_fitness': best_fitness,
            'fitness_by_iteration': [np.array(fbi) for fbi in fitness_by_iteration],
            'fitness_evals_history': [np.array(feh) for feh in fitness_evals_history],
        }
        return best_state, best_fitness, statistics
    else:
        return best_state, best_fitness
