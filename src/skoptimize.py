import numpy as np
np.warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import scipy.stats as sc
from sko.DE import DE
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.AFSA import AFSA


class RandomSearch:
    def __init__(self, n_iter):
        self.__class__.__name__ = 'RandomSearch'
        self.n_iter = n_iter
        self.weights = None
    
    def fit(self, X, y):
        d = X.shape[1]
        func = lambda dim: np.random.dirichlet(np.ones(dim),size=1)
        weights_candidate = [[func(d)] for i in range(self.n_iter)]
        weights_candidate = np.array(weights_candidate).reshape(d, self.n_iter)
        y_pred_candidate = np.dot(X, weights_candidate)
        mse = lambda v: mean_squared_error(y, v, squared=False)
        errors = [mse(y_pred_candidate[:, i]) for i in range(self.n_iter)]
        min_value = min(errors)
        min_index = errors.index(min_value)
        self.weights = weights_candidate[:, min_index]
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

class LinearSearch:
    def __init__(self, non_negative=True, sum_to_one=True):
        self.__class__.__name__ = 'LinearSearch'
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one
        self.weights = None
        
    def fit(self, X, y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
        if self.non_negative:
            bound = [(0.0, 1.0) for i in range(d)]
        else:
            bound = None
            
        if self.sum_to_one:
            constraint = ({'type': 'eq', 'fun' : lambda v: np.sum(v) - 1.0})
        else:
            constraint = ()
            
        if self.non_negative & self.sum_to_one:
            methods = 'L-BFGS-B'
        else:
            methods = None
            
        w0 = np.full((d,), 1/d, dtype=float)
        result = minimize(loss, w0, method=methods, constraints=constraint, bounds=bound)
        self.weights = result.x
        
    def predict(self, X):
        return np.dot(X, self.weights)

class DifferentialEvolution:
    def __init__(self, size_pop, max_iter, non_negative=False, sum_to_one=False):
        self.__class__.__name__ = 'DifferentialEvolution'
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one
        self.weights = None
        
    def fit(self, X,  y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
        if self.non_negative:
            lb = 0.0
            ub = 1.0
        else:
            lb = -1.0
            ub = 1.0
            
        if self.sum_to_one:
            constraint = [lambda v: np.sum(v) - 1.0]
        else:
            constraint = tuple()
            
        opt = DE(func=loss, n_dim=d, size_pop=self.size_pop, max_iter=self.max_iter, lb=lb, ub=ub, constraint_eq=constraint)
        best_w, _ = opt.run()
        self.weights = np.array(best_w).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)

class GeneticAlgorithm:
    def __init__(self, size_pop, max_iter, non_negative=False, sum_to_one=False):
        self.__class__.__name__ = 'GeneticAlgorithm'
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one
        self.weights = None
        
    def fit(self, X,  y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
        if self.non_negative:
            lb = 0.0
            ub = 1.0
        else:
            lb = -1.0
            ub = 1.0
            
        if self.sum_to_one:
            constraint = [lambda v: np.sum(v) - 1.0]
        else:
            constraint = tuple()
            
        opt = GA(func=loss, n_dim=d, size_pop=self.size_pop, max_iter=self.max_iter, lb=lb, ub=ub, constraint_eq=constraint)
        best_w, _ = opt.run()
        self.weights = np.array(best_w).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)

class ParticleSwarmOptimization:
    def __init__(self, pop, max_iter, w, c1, c2, non_negative=False, sum_to_one=False):
        self.__class__.__name__ = 'ParticleSwarmOptimization'
        self.pop = pop
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.non_negative = non_negative
        self.sum_to_one = sum_to_one
        self.weights = None
        
    def fit(self, X,  y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
        if self.non_negative:
            lb = 0.0
            ub = 1.0
        else:
            lb = -1.0
            ub = 1.0
            
        if self.sum_to_one:
            constraint = [lambda v: np.sum(v) - 1.0]
        else:
            constraint = tuple()
            
        opt = PSO(func=loss, n_dim=d, pop=self.pop, max_iter=self.max_iter, w=self.w, c1=self.c1, c2=self.c2, lb=lb, ub=ub, constraint_eq=constraint)
        best_w, _ = opt.run()
        self.weights = np.array(best_w).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)

class SimulatedAnnealing:
    def __init__(self, T_max, T_min, L, max_stay_counter):
        self.__class__.__name__ = 'SimulatedAnnealing'
        self.T_max = T_max
        self.T_min = T_min
        self.L = L
        self.max_stay_counter = max_stay_counter
        self.weights = None
        
    def fit(self, X,  y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
            
        w0 = list(np.full((d,), 1/d, dtype=float))
        opt = SA(func=loss, x0=w0, T_max=self.T_max, T_min=self.T_min, L=self.L, max_stay_counter=self.max_stay_counter)
        best_w, _ = opt.run()
        self.weights = np.array(best_w).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)

class ArtificialFishSwarm:
    def __init__(self, size_pop, max_iter, max_try_num, step, visual, q, delta):
        self.__class__.__name__ = 'ArtificialFishSwarm'
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.max_try_num = max_try_num
        self.step = step
        self.visual = visual
        self.q = q
        self.delta = delta
        self.weights = None
        
    def fit(self, X,  y):
        loss = lambda w: np.sum(np.square(np.dot(X, w) - y))
        d = X.shape[1]
            
        opt = AFSA(func=loss, n_dim=d, size_pop=self.size_pop, max_iter=self.max_iter, max_try_num=self.max_try_num, step=self.step, visual=self.visual, q=self.q, delta=self.delta)
        best_w, _ = opt.run()
        self.weights = np.array(best_w).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)

class GreedySearch:
    def __init__(self, convergence, epsilon):
        self.__class__.__name__ = 'GreedySearch'
        self.convergence = convergence
        self.epsilon = epsilon
        self.weights = None
        
    def metric_spearman(self, x, Y):
        return sc.stats.spearmanr(x, Y)[0]
    
    def conv_manhattan(self, n, vec, eps):
        return n >= 1/eps
    
    def conv_euclidan(self, n, vec, eps):
        return np.linalg.norm(vec) >= 1/eps
    
    def fit(self, X, y):
        d = X.shape[1]
        weights = np.zeros(d, dtype=int) # weights for the columns of X
        sums_transposed = np.zeros(X.transpose().shape)
        num_weights = 0 # integer makes increment fast and stable
        best = 0
        
        if self.convergence == 'manhattan':
            convergence_func = self.conv_manhattan
        elif self.convergence == 'euclidean':
            convergence_func = self.conv_euclidan
            
        while not convergence_func(num_weights, weights, self.epsilon):
            num_weights += 1
            sums_transposed = sums_transposed[best, :] + X.transpose()
            err = [self.metric_spearman(sums_transposed[i,:] / float(num_weights), y) for i in range(sums_transposed.shape[0])]       
            best = np.argmax(err)
            weights[best] += 1
            
        self.weights = np.array(weights/float(num_weights)).reshape(d,)
        
    def predict(self, X):
        return np.dot(X, self.weights)