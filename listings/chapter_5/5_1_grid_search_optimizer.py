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
        """Initializes the class with the given parameters and performance metrics.
        Parameters:
            - parameters (dict): Dictionary of parameters for the model.
            - performance (dict): Dictionary of performance metrics for the model.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Checks for any collisions between parameter names and performance metric names.
            - Sets the class attributes for parameters and performance.
        Example:
            parameters = {'learning_rate': 0.01, 'batch_size': 32}
            performance = {'accuracy': 0.85, 'loss': 0.4}
            model = Model(parameters, performance)
            # model.parameters = {'learning_rate': 0.01, 'batch_size': 32}
            # model.performance = {'accuracy': 0.85, 'loss': 0.4}"""
        

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
        """"Initializes the class with a simulation function and sets up attributes for storing optimization results.
        Parameters:
            - simulation_function (SimFunction): A simulation function that takes in parameters and returns a result.
        Returns:
            - None
        Processing Logic:
            - Store simulation function as attribute.
            - Initialize empty list for storing results.
            - Initialize empty dataframe for storing results.
            - Set optimization_finished flag to False.""""
        

        self.simulate = simulation_function
        self._results_list: List[OptimizationResult] = list()
        self._results_df = pd.DataFrame()

        self._optimization_finished = False

    def add_results(self, parameters: Parameters, performance: Performance):
        """Adds a new OptimizationResult to the list of results.
        Parameters:
            - parameters (Parameters): The parameters used for the optimization.
            - performance (Performance): The performance of the optimization.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Create a new OptimizationResult object.
            - Append the new result to the list.
            - No other processing logic is applied."""
        
        _results = OptimizationResult(parameters, performance)
        self._results_list.append(_results)

    def optimize(self, **optimization_ranges: SimKwargs):
        """Optimizes simulation parameters and returns results.
        Parameters:
            - optimization_ranges (SimKwargs): Dictionary of simulation parameters and their ranges.
        Returns:
            - None: No return value.
        Processing Logic:
            - Convert all iterables to lists.
            - Count total simulations.
            - Print simulation progress and remaining time.
            - Start timer.
            - Simulate with given parameters.
            - Add results to simulation.
            - End timer.
            - Print total simulations and elapsed time.
            - Set optimization_finished flag to True."""
        

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
        """This function checks if the optimization process has finished and raises an error if it hasn't.
        Parameters:
            - self (object): The object containing the optimization process.
        Returns:
            - None: This function does not return any value.
        Processing Logic:
            - Checks if optimization is finished.
            - Raises error if not finished.
            - No additional processing logic."""
        
        assert self._optimization_finished, \
            'Run self.optimize before accessing this method.'

    @property
    def results(self) -> pd.DataFrame:
        """"Returns a Pandas DataFrame containing the results of the function's computation."
        Parameters:
            - self (object): The object containing the function.
        Returns:
            - pd.DataFrame: A DataFrame containing the results of the computation.
        Processing Logic:
            - Asserts that the function has finished running.
            - Creates a list of results from the function.
            - Converts the list into a DataFrame.
            - Sets the columns of the DataFrame to be the metric names.
            - Returns the DataFrame."""
        
        self._assert_finished()
        if self._results_df.empty:

            _results_list = self._results_list
            self._results_df = pd.DataFrame([r.as_dict for r in _results_list])

            _columns = set(list(self._results_df.columns.values))
            _params = set(self.param_names)
            self.metric_names = list(_columns - _params)

        return self._results_df

    def print_summary(self):
        """Prints a summary of the results dataframe.
        Parameters:
            - self (object): The object containing the results dataframe and metric names.
        Returns:
            - None: This function does not return anything.
        Processing Logic:
            - Get the results dataframe and metric names.
            - Print the summary statistics of the dataframe.
            - Transpose the dataframe for easier reading."""
        
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
        """Function to plot a 1-dimensional histogram using the results of a given dataset.
        Parameters:
            - x (array): Array of values to be plotted.
            - show (bool): If True, the histogram will be displayed. Defaults to True.
        Returns:
            - None: This function does not return any values.
        Processing Logic:
            - Plot histogram using self.results.hist().
            - If show is True, display the histogram using plt.show()."""
        
        self.results.hist(x)
        if show:
            plt.show()

    def plot_2d_line(self, x, y, show=True, **filter_kwargs):
        """Plots a 2D line graph using the given x and y values, with the option to filter the results based on keyword arguments.
        Parameters:
            - x (array-like): The x values to plot.
            - y (array-like): The y values to plot.
            - show (bool): Whether or not to display the plot. Defaults to True.
            - **filter_kwargs (kwargs): Keyword arguments used to filter the results before plotting.
        Returns:
            - ax (matplotlib.axes.Axes): The axes object containing the plot.
        Processing Logic:
            - Filters the results based on the given keyword arguments.
            - Creates a legend if filter_kwargs is not empty.
            - Displays the plot if show is True.
            - Uses the x and y values to plot the data."""
        
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
