import numpy as np

def make_timeseries(
        n_timesteps: int=1000,
        ar_coefficients: dict[int, float]={1: 0.9},
        integration_order: int=0,
        noise: float=1.0,
        x0: float=None,
        xmin: float=None,
        xmax: float=None
) -> np.ndarray:
    """
    Generate a univariate time series using an ARIMA process.
    
    Args:
        n_timesteps (int): Number of time steps to generate.
        ar_coefficients (dict): Autoregressive coefficients as a dictionary.
        integration_order (int): Order of differencing.
        noise (float): Standard deviation of the Gaussian noise.
        x0 (float): Initial value of the time series.
        xmin (float): Minimum value of the time series.
        xmax (float): Maximum value of the time series.
        
    Returns:
        np.ndarray: Generated time series of shape (n_timesteps,).
    """
    # if xmin > xmax, raise error
    if xmin is not None and xmax is not None:
        if xmin > xmax:
            raise ValueError("xmin must be less than or equal to xmax.")
        
    # if x0 is outside xmin and xmax, raise error
    if x0 is not None and xmin is not None and xmax is not None:
        if x0 < xmin or x0 > xmax:
            raise ValueError("x0 must be within the range [xmin, xmax].")
        
    # if lags are not integers, send error
    for lag in ar_coefficients:
        if not isinstance(lag, int):
            raise TypeError("AR coefficients keys must be integers representing lags.")
        # if lags < 1, send error
        if lag < 1:
            raise TypeError("AR coefficients keys must be positive integers representing lags.")
    
    # if duplicate lags, raise input error with list of duplicate lags
    from collections import Counter
    lag_counts = Counter(ar_coefficients.keys())
    duplicate_lags = [lag for lag, count in lag_counts.items() if count > 1]
    if duplicate_lags:
        raise ValueError(f"Duplicate lags found in AR coefficients: {duplicate_lags}")
    
    # if lags are not in ascending order, sort them
    ar_coefficients = dict(sorted(ar_coefficients.items()))
    
    # Initialize the time series array
    x = np.zeros(n_timesteps)
    
    if x0 is not None:
        x[0] = x0
    else:
        x[0] = np.random.normal(0, noise)
    
    # Generate the AR process
    for t in range(1, n_timesteps):
        for lag in ar_coefficients:
            coeff = ar_coefficients[lag]
            x[t] += coeff * x[t - lag]
        x[t] += np.random.normal(0, noise)
    
    # Integrate if necessary
    if integration_order > 0:
        raise NotImplementedError("Integration order greater than 0 is not implemented.")
    
    return x

class TimeSeries:
    def __init__(self, n_timesteps: int=100):
        self.n_timesteps = n_timesteps
        self.lags = int(np.random.uniform(1, n_timesteps/10))
        self.x0 = np.random.normal(0, 1) # NB: This is a choice. Almost all interesting time series are non-negative.
        self.noise = self.x0 * np.random.uniform(0.1, 0.9) # TODO: play around with this
        
        # make AR coefficients exponentially decay, with random negatives and zeros
        values = [np.random.uniform(0.01, 0.99)]
        for l in range(1, self.lags):
            values.append(values[-1]*0.75*np.random.choice([-1, 1])*np.random.binomial(n=1, p=.7)) # TODO: adjust sparsity with p, could make it random
        self.ar_coefficients = {i+1: values[i] for i in range(len(values))}
        self.parameters_set = True

    def generate(self):
        self.series = make_timeseries(
            n_timesteps=self.n_timesteps,
            ar_coefficients=self.ar_coefficients,
            x0=self.x0
        )

    # have a nice printed representation
    def __repr__(self):
        ar_coefficients_str = ', '.join([f"{k}: {v:.4f}" for k, v in self.ar_coefficients.items()])
        return f"TimeSeries(n_timesteps={self.n_timesteps}, lags={self.lags}, x0={self.x0}, noise={self.noise}, ar_coefficients={{ {ar_coefficients_str} }})"
    
def make_correlated_timeseries(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Generate correlated time series from a list of uncorrelated time series.
    """
    L = np.linalg.cholesky(R)    
    return X @ L.T
    
class MVTimeSeries:
    """Generate `p` correlated univariate time series."""
    def __init__(self, n_timesteps: int=100, p: int=3):
        """
        Docstring for __init__
        
        :param self: Description
        :param n_timesteps: Number of time steps
        :param p: Number of time series
        """
        self.n_timesteps = n_timesteps
        self.p = p
        self.data = np.zeros((n_timesteps, p))
        
        # symmetric correlation matrix with values between -1 and 1
        A = np.random.randn(p, p)
        cov = np.dot(A, A.T)
        d = np.sqrt(np.diag(cov))
        self.correlation_matrix = cov / np.outer(d, d)

    def generate(self):
        for i in range(self.p):
            x = TimeSeries(n_timesteps=self.n_timesteps)
            x.generate()
            self.data[:, i] = x.series
        
        # induce correlated series
        self.data = make_correlated_timeseries(self.data, self.correlation_matrix)

    def __repr__(self):
        correlation_matrix_str = self.correlation_matrix.round(4)
        return f"MVTimeSeries(n_timesteps={self.n_timesteps}, p={self.p}, rho_X=\n{correlation_matrix_str})"