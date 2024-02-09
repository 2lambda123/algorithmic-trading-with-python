from pypm import metrics, signals, data_io, simulation

import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from itertools import product
from timeit import default_timer
from typing import Dict, Tuple, List, Callable, Iterable, Any, NewType, Mapping

import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 

# Performance data and parameter inputs are dictionaries
Parameters = NewType('Parameters', Dict[str, float])
Performance = simulation.PortfolioHistory.PerformancePayload # Dict[str, float]

# Simulation function must take parameters as keyword arguments pointing to 
# iterables and return a performance metric dictionary
SimKwargs = NewType('Kwargs', Mapping[str, Iterable[Any]])
SimFunction = NewType('SimFunction', Callable[[SimKwargs], Performance])

class OptimizationResult(object):
    """Simple container class for optimization data"""

    def __init__(self, parameters: Parameters, performance: Performance):
        """Initializes an object with the given parameters and performance metrics.
        Parameters:
            - parameters (dict): Dictionary of parameters used for the model.
            - performance (dict): Dictionary of performance metrics for the model.
        Returns:
            - None: Does not return anything.
        Processing Logic:
            - Check for collisions between parameter names and performance metric names.
            - Store the parameters and performance metrics in the object.
            - Assert that the length of the intersection between parameter names and performance metric names is 0.
            - Use assert statement to raise an error if there are any collisions.
            - NEVER write more than 4 bullets
        Example:
            >>> parameters = {'learning_rate': 0.01, 'batch_size': 32}
            >>> performance = {'accuracy': 0.85, 'loss': 0.4}
            >>> model = __init__(parameters, performance)
            >>> model.parameters
            {'learning_rate': 0.01, 'batch_size': 32}
            >>> model.performance
            {'accuracy': 0.85, 'loss': 0.4}"""
        

        # Make sure no collisions between performance metrics and params
        assert len(parameters.keys() & performance.keys()) == 0, \
            'parameter name matches performance metric name'

        self.parameters = parameters
        self.performance = performance

    @property
    def as_dict(self) -> Dict[str, float]:
        """Combines the dictionaries after we are sure of no collisions"""
        return {**self.parameters, **self.performance}
    

class GridSearchOptimizer(object):
    """
    A generic grid search optimizer that requires only a simulation function and
    a series of parameter ranges. Provides timing, summary, and plotting 
    utilities with return data.
    """

    def __init__(self, simulation_function: SimFunction):
        """Initializes an instance of the class with a simulation function and sets up attributes for storing results.
        Parameters:
            - simulation_function (SimFunction): A simulation function that takes in parameters and returns a result.
        Returns:
            - None
        Processing Logic:
            - Initialize simulation function.
            - Set up results list and dataframe.
            - Set optimization finished flag to False."""
        

        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self._results_df = pd.DataFrame()

        self._optimization_finished = False

    def add_results(self, parameters: Parameters, performance: Performance):
        """Adds a new optimization result to the list of results.
        Parameters:
            - parameters (Parameters): The parameters used for the optimization.
            - performance (Performance): The performance of the optimization.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Create a new OptimizationResult object.
            - Append the new result to the list.
            - No other processing logic is performed."""
        
        _results = OptimizationResult(parameters, performance)
        self._results_list.append(_results)

    def optimize(self, **optimization_ranges: SimKwargs):
        """Optimizes simulation parameters and runs simulations for each combination of provided ranges.
        Parameters:
            - optimization_ranges (dict): Dictionary of simulation parameter names and their corresponding ranges.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Converts all iterables to lists.
            - Counts total number of simulations.
            - Prints simulation progress and estimated remaining time.
            - Runs simulations for each combination of parameter ranges.
            - Prints total number of simulations and elapsed time.
            - Sets optimization_finished flag to True."""
        

        assert optimization_ranges, 'Must provide non-empty parameters.'

        # Convert all iterables to lists
        param_ranges = {k: list(v) for k, v in optimization_ranges.items()}
        self.param_names = param_names = list(param_ranges.keys())

        # Count total simulation
        n = total_simulations = np.prod([len(r) for r in param_ranges.values()])

        total_time_elapsed = 0

        print('Starting simulation ...')
        print(f'Simulating 1 / {n} ...', end='\r')
        for i, params in enumerate(product(*param_ranges.values())):
            if i > 0:
                _avg = avg_time = total_time_elapsed / i
                _rem = remaining_time = (n - (i + 1)) * avg_time
                s =  f'Simulating {i+1} / {n} ... '
                s += f'{_rem:.0f}s remaining ({_avg:.1f}s avg)'
                s += ' '*8
                print(s, end='\r')

            timer_start = default_timer()

            parameters = {n: param for n, param in zip(param_names, params)}
            results = self.simulate(**parameters)
            self.add_results(parameters, results)

            timer_end = default_timer()
            total_time_elapsed += timer_end - timer_start 

        print(f'Simulated {total_simulations} / {total_simulations} ...')
        print(f'Elapsed time: {total_time_elapsed:.0f}s')
        print('Done.')

        self._optimization_finished = True

    def _assert_finished(self):
        """This function checks if the optimization process has finished before accessing the method.
        Parameters:
            - self (object): The object instance.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Check if optimization is finished.
            - Assert that optimization is finished.
            - Raise an error if optimization is not finished.
            - Run self.optimize before accessing method."""
        
        assert self._optimization_finished, \
            'Run self.optimize before accessing this method.'

    @property
    def results(self) -> pd.DataFrame:
        """Returns a Pandas DataFrame containing the results of the function.
        Parameters:
            - self (object): Instance of the class.
        Returns:
            - results_df (pd.DataFrame): DataFrame containing the results of the function.
        Processing Logic:
            - Asserts that the function has finished running.
            - If the results DataFrame is empty, creates a new DataFrame using the results list.
            - Sets the metric names by subtracting the parameter names from the columns of the results DataFrame."""
        
        self._assert_finished()
        if self._results_df.empty:

            _results_list = self._results_list
            self._results_df = pd.DataFrame([r.as_dict for r in _results_list])

            _columns = set(list(self._results_df.columns.values))
            _params = set(self.param_names)
            self.metric_names = list(_columns - _params)

        return self._results_df

    def print_summary(self):
        """Prints a summary of the statistics of the results dataframe.
        Parameters:
            - self (object): The object containing the results dataframe and metric names.
        Returns:
            - None: This function does not return anything, it only prints the summary.
        Processing Logic:
            - Creates a dataframe and stores it in the variable 'df'.
            - Stores the metric names in the variable 'metric_names'.
            - Prints the summary statistics of the dataframe using the describe() function.
            - Transposes the dataframe using the T attribute."""
        
        df = self.results
        metric_names = self.metric_names

        print('Summary statistics')
        print(df[metric_names].describe().T)

    def get_best(self, metric_name: str) -> pd.DataFrame:
        """
        Sort the results by a specific performance metric
        """
        self._assert_finished()

        results = self.results
        param_names = self.param_names
        metric_names = self.metric_names

        assert metric_name in metric_names, 'Not a performance metric'
        partial_df = self.results[param_names+[metric_name]]

        return partial_df.sort_values(metric_name, ascending=False)

    def plot_1d_hist(self, x, show=True):
        """Plots a 1-dimensional histogram of the given data and displays it if specified.
        Parameters:
            - x (array-like): The data to be plotted.
            - show (bool): Whether or not to display the plot. Default is True.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Plot histogram using self.results.hist().
            - If show is True, display the plot using plt.show()."""
        
        self.results.hist(x)
        if show:
            plt.show()

    def plot_2d_line(self, x, y, show=True, **filter_kwargs):
        """Plots a 2D line graph of the given x and y values from the results dataframe. Can filter the results dataframe by passing in keyword arguments.
        Parameters:
            - x (str): Column name for the x-axis values.
            - y (str): Column name for the y-axis values.
            - show (bool): Optional parameter to determine whether or not to display the plot. Default is True.
            - **filter_kwargs (dict): Optional keyword arguments to filter the results dataframe. The key should be a column name and the value should be the desired value for that column.
        Returns:
            - None
        Processing Logic:
            - Filters the results dataframe based on the keyword arguments provided.
            - Plots the x and y values from the filtered results dataframe.
            - If keyword arguments were provided, adds a legend to the plot indicating the filter criteria used.
            - If show is True, displays the plot."""
        
        _results = self.results
        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        ax = _results.plot(x, y)
        if filter_kwargs:
            k_str = ', '.join([f'{k}={v}' for k,v in filter_kwargs.items()])
            ax.legend([f'{x} ({k_str})'])

        if show:
            plt.show()

    def plot_2d_violin(self, x, y, show=True):
        """
        Group y along x then plot violin charts
        """
        x_values = self.results[x].unique()
        x_values.sort()

        y_by_x = OrderedDict([(v, []) for v in x_values])
        for _, row in self.results.iterrows():
            y_by_x[row[x]].append(row[y])

        fig, ax = plt.subplots()

        ax.violinplot(dataset=list(y_by_x.values()), showmedians=True)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(range(0, len(y_by_x)+1))
        ax.set_xticklabels([''] + list(y_by_x.keys()))
        if show:
            plt.show()

    def plot_3d_mesh(self, x, y, z, show=True, **filter_kwargs):
        """
        Plot interactive 3d mesh. z axis should typically be performance metric
        """
        _results = self.results
        fig = plt.figure()
        ax = Axes3D(fig)

        for k, v in filter_kwargs.items():
            _results = _results[getattr(_results, k) == v]

        X, Y, Z = [getattr(_results, attr) for attr in (x, y, z)]
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0.2)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        if show:
            plt.show()

    def plot(self, *attrs: Tuple[str], show=True, 
        **filter_kwargs: Dict[str, Any]):
        """
        Attempt to intelligently dispatch plotting functions based on the number
        and type of attributes. Last argument should typically be the 
        performance metric.
        """
        self._assert_finished()
        param_names = self.param_names
        metric_names = self.metric_names

        if len(attrs) == 3:
            assert attrs[0] in param_names and attrs[1] in param_names, \
                'First two positional arguments must be parameter names.'

            assert attrs[2] in metric_names, \
                'Last positional argument must be a metric name.'

            assert len(filter_kwargs) + 2 == len(param_names), \
                'Must filter remaining parameters. e.g. p_three=some_number.'

            self.plot_3d_mesh(*attrs, show=show, **filter_kwargs)

        elif len(attrs) == 2:
            if len(param_names) == 1 or filter_kwargs:
                self.plot_2d_line(*attrs, show=show, **filter_kwargs)

            elif len(param_names) > 1:
                self.plot_2d_violin(*attrs, show=show)

        elif len(attrs) == 1:
            self.plot_1d_hist(*attrs, show=show)

        else:
            raise ValueError('Must pass between one and three column names.')




