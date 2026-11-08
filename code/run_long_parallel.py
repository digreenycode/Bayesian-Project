"""BO vs CMA-ES long-run experiment runner.
Converted from Sections 1-12 of bo_cma_active_subspace_pilot(2).ipynb.
Plotting and analysis are intentionally omitted.
"""

import json
import os
import platform
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cma
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

warnings.filterwarnings('ignore', category=ConvergenceWarning)


@dataclass
class BOConfig:
    n_gp_restarts: int = 0       # Additional GP log-marginal-likelihood restarts
    n_candidates: int = 1500      # Feasible random candidates used before local EI optimization
    n_acq_starts: int = 1        # SLSQP multi-start count
    acq_maxiter: int = 20       # Maximum SLSQP iterations per start
    acq_ftol: float = 1e-6       # SLSQP tolerance
    alpha: float = 1e-8          # Fixed numerical jitter for deterministic objectives
    length_scale_init: float = 0.5
    length_scale_bounds: tuple = (1e-3, 1e3)
    signal_init: float = 1.0
    signal_bounds: tuple = (1e-3, 1e3)

@dataclass
class CMAConfig:
    sigma0: float = 0.5          # Initial global CMA-ES step size
    popsize: int | None = None   # None = PyCMA default


@dataclass
class ExperimentConfig:
    objectives: tuple = ('norm', 'branin')
    dimensions: tuple = (5,) #(2, 5, 10, 20)  # Pilot; use (2, 5, 10, 20) for main experiment
    
    k: int = 2                   # Effective dimension
    n_init_sets: int = 1         # Number of X_init sets
    n_rotations: int = 1         # Number of A: orthogonal matrices
    budget: int = 150            # Total true objective evaluations, including X_init
    
    n_inclusion: int = 1         # Redundant variable, explicit for readability
    n_init_pts: int = 5          # Size of X_init
    seed: int = 42   
    
    domain: str = 'unit_ball'
    data_dir: str = 'data_smoke2'
    bo: BOConfig = field(default_factory=BOConfig)
    cma: CMAConfig = field(default_factory=CMAConfig)

    @property
    def n_eval(self):
        return self.budget - self.n_init_pts

smoke_cfg = ExperimentConfig( # Used as a smoke test
    objectives = ('norm', 'branin'),
    dimensions = (2, 5,), 
    
    n_init_sets = 1,         # Number of X_init sets
    n_rotations = 2,         # Number of A: orthogonal matrices
    budget = 150,            # Total true objective evaluations, including X_init
    
    seed = 42,

    
    data_dir = '../data_smoke',
)    

exp_cfg = ExperimentConfig( # Experimental config, takes 10h to run with parallelization of CPU
    objectives = ('norm', 'branin'),
    dimensions = (2, 5, 10, 20),  # Pilot; use (2, 5, 10, 20) for main experiment
    
    n_init_sets = 20,         # Number of X_init sets
    n_rotations = 20,         # Number of A: orthogonal matrices
    budget = 150,            # Total true objective evaluations, including X_init
    
    seed = 42,

    
    data_dir = '../data_long3',
)

# C:/Users/CZ/Documents/GitHub/Bayesian-Optimisation-CMA-ES-project
# ============================================================
# EDIT EXPERIMENT PARAMETERS HERE

cfg = smoke_cfg

SOLVERS_TO_RUN = ('random', 'bo_isotropic', 'bo_ard', 'bo_oracle', 'cma_es')

# Start conservatively (e.g. 2 or 4) and benchmark on your machine.
N_JOBS = 8

# ============================================================


def sample_unit_ball(n, d, rng):
    W = rng.normal(size=(n, d))
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    while np.any(norms == 0):
        mask = (norms[:, 0] == 0)
        W[mask] = rng.normal(size=(mask.sum(), d))
        norms = np.linalg.norm(W, axis=1, keepdims=True)
    V = W / norms
    R = rng.uniform(size=(n, 1)) ** (1.0 / d)
    return R * V


def project_unit_ball(X):
    X = np.asarray(X, dtype=float)
    one = (X.ndim == 1)
    X2 = X[None, :] if one else X.copy()
    norms = np.linalg.norm(X2, axis=1, keepdims=True)
    scale = np.maximum(1.0, norms)
    out = X2 / scale
    return out[0] if one else out


def norm_squared(z):
    z = np.asarray(z)
    return float(np.dot(z, z))


def branin_native(u, v):
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a * (v - b*u*u + c*u - r)**2 + s * (1.0 - t) * np.cos(u) + s


def branin_normalized(z):
    z = np.asarray(z, dtype=float)
    u = 2.5 + 7.5 * z[0]   # [-1,1] -> [-5,10]
    v = 7.5 + 7.5 * z[1]   # [-1,1] -> [0,15]
    return float(branin_native(u, v))

OBJECTIVES = {
    'norm': norm_squared,
    'branin': branin_normalized,
}

GLOBAL_MIN = {
    'norm': 0.0,
    'branin': 0.39788735772973816,
}

def make_A_inclusion(d, k=2):
    A = np.zeros((d, k))
    A[:k, :] = np.eye(k)
    return A


def make_A_rotated(d, k, rng):
    G = rng.normal(size=(d, k))
    Q, R = np.linalg.qr(G, mode='reduced')
    # Fix QR sign ambiguity for deterministic representation.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q

@dataclass
class Problem:
    problem_id: str
    objective_name: str
    d: int
    k: int
    embedding: str
    A_index: int
    init_index: int
    A: np.ndarray
    X_init: np.ndarray
    y_init: np.ndarray
    problem_seed: int
    domain: str = 'unit_ball'
    branin_scaling: dict | None = None

    def evaluate(self, x):
        z = self.A.T @ np.asarray(x)
        return OBJECTIVES[self.objective_name](z)

    def evaluate_z(self, z):
        return OBJECTIVES[self.objective_name](np.asarray(z))


@dataclass(frozen=True)
class RunSpec:
    problem_id: str
    solver: str
    solver_seed: int

SOLVERS = ('random', 'bo_isotropic', 'bo_ard', 'bo_oracle', 'cma_es')

OBJECTIVE_CODE = {'norm': 1, 'branin': 2}
EMBED_CODE = {'inclusion': 1, 'rotated': 2}
SOLVER_CODE = {name: i + 1 for i, name in enumerate(SOLVERS)}


def derived_seed(*parts):
    ss = np.random.SeedSequence([int(p) for p in parts])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def generate_problems(cfg):
    problems = {}

    for d in cfg.dimensions:
        # X_init depends on d and init_index, and is shared across objectives/embeddings/solvers.
        X_inits = []
        for init_idx in range(cfg.n_init_sets):
            init_seed = derived_seed(cfg.seed, 100, d, init_idx)
            Xi = sample_unit_ball(cfg.n_init_pts, d, np.random.default_rng(init_seed))
            X_inits.append((init_seed, Xi))

        embeddings = [('inclusion', 0, derived_seed(cfg.seed, 200, d, 0), make_A_inclusion(d, cfg.k))]
        for a_idx in range(cfg.n_rotations):
            a_seed = derived_seed(cfg.seed, 201, d, a_idx)
            A = make_A_rotated(d, cfg.k, np.random.default_rng(a_seed))
            embeddings.append(('rotated', a_idx, a_seed, A))

        for objective_name in cfg.objectives:
            for embedding, a_idx, a_seed, A in embeddings:
                assert np.allclose(A.T @ A, np.eye(cfg.k), atol=1e-10)
                for init_idx, (init_seed, Xi) in enumerate(X_inits):
                    problem_seed = derived_seed(
                        cfg.seed, 300, OBJECTIVE_CODE[objective_name], d,
                        EMBED_CODE[embedding], a_idx, init_idx
                    )
                    y_init = np.array([OBJECTIVES[objective_name](A.T @ x) for x in Xi])
                    problem_id = f'{objective_name}_d{d}_{embedding}_A{a_idx}_init{init_idx}'
                    scaling = None
                    if objective_name == 'branin':
                        scaling = {
                            'normalized_domain': '[-1,1]^2',
                            'reachable_domain': 'unit disk in normalized coordinates',
                            'u_from_z1': 'u = 2.5 + 7.5*z1',
                            'v_from_z2': 'v = 7.5 + 7.5*z2',
                            'native_domain': '[-5,10] x [0,15]',
                        }
                    problems[problem_id] = Problem(
                        problem_id=problem_id,
                        objective_name=objective_name,
                        d=d,
                        k=cfg.k,
                        embedding=embedding,
                        A_index=a_idx,
                        init_index=init_idx,
                        A=A.copy(),
                        X_init=Xi.copy(),
                        y_init=y_init,
                        problem_seed=problem_seed,
                        domain=cfg.domain,
                        branin_scaling=scaling,
                    )
    return problems


def generate_run_specs(problems, cfg):
    specs = []
    for problem_id, p in problems.items():
        for solver in SOLVERS:
            seed = derived_seed(
                cfg.seed, 400, OBJECTIVE_CODE[p.objective_name], p.d,
                EMBED_CODE[p.embedding], p.A_index, p.init_index,
                SOLVER_CODE[solver]
            )
            specs.append(RunSpec(problem_id, solver, seed))
    return specs

def make_gp(d_model, ard, cfg_bo, seed):
    if ard:
        ls0 = np.full(d_model, cfg_bo.length_scale_init)
        ls_bounds = [cfg_bo.length_scale_bounds] * d_model
    else:
        ls0 = cfg_bo.length_scale_init
        ls_bounds = cfg_bo.length_scale_bounds

    kernel = C(cfg_bo.signal_init, cfg_bo.signal_bounds) * RBF(
        length_scale=ls0,
        length_scale_bounds=ls_bounds,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=cfg_bo.alpha,
        normalize_y=True,
        n_restarts_optimizer=cfg_bo.n_gp_restarts,
        random_state=int(seed),
    )


def expected_improvement_min(X_model, gp, best_y):
    mu, std = gp.predict(np.atleast_2d(X_model), return_std=True)
    std = np.maximum(std, 1e-15)
    improvement = best_y - mu
    z = improvement / std
    ei = improvement * norm.cdf(z) + std * norm.pdf(z)
    return np.maximum(ei, 0.0)


def gp_diagnostics(gp):
    # Kernel is ConstantKernel * RBF.
    k = gp.kernel_
    return {
        'signal_variance': float(k.k1.constant_value),
        'length_scale': np.atleast_1d(k.k2.length_scale).astype(float).copy(),
        'log_marginal_likelihood': float(gp.log_marginal_likelihood_value_),
    }


def transform_for_gp(problem, X, oracle=False):
    X = np.atleast_2d(X)
    return X @ problem.A if oracle else X

def gp_mean_std_grad(gp, x_model):
    """
    Return GP predictive mean, std, and their gradients with respect
    to the GP input coordinates.

    Assumes fitted kernel = ConstantKernel * RBF.
    """
    x = np.asarray(x_model, dtype=float).reshape(-1)
    X_train = gp.X_train_

    # Fitted kernel parameters
    signal_var = float(gp.kernel_.k1.constant_value)
    length_scale = np.asarray(gp.kernel_.k2.length_scale, dtype=float)

    # Isotropic RBF: turn scalar length scale into vector
    if length_scale.ndim == 0:
        length_scale = np.full(x.size, float(length_scale))

    # X_train - x
    diff = X_train - x

    # k(X_train, x)
    sqdist = np.sum((diff / length_scale) ** 2, axis=1)
    k_x = signal_var * np.exp(-0.5 * sqdist)

    # Gradient of k(X_train, x) wrt x
    dk_dx = k_x[:, None] * diff / (length_scale ** 2)

    # sklearn uses normalized y internally when normalize_y=True.
    # gp.alpha_ corresponds to that normalized target.
    mu_norm = k_x @ gp.alpha_
    grad_mu_norm = dk_dx.T @ gp.alpha_

    # Convert mean and its gradient back to original y scale
    y_scale = float(np.asarray(gp._y_train_std).reshape(-1)[0])
    y_mean = float(np.asarray(gp._y_train_mean).reshape(-1)[0])

    mu = y_mean + y_scale * mu_norm
    grad_mu = y_scale * grad_mu_norm

    # K^{-1} k
    Kinv_k = cho_solve((gp.L_, True), k_x)

    # Predictive latent variance
    var = signal_var - k_x @ Kinv_k
    var = max(float(var), 1e-15)

    sigma_norm = np.sqrt(var)

    # d var / dx
    grad_var = -2.0 * (dk_dx.T @ Kinv_k)

    # d sigma / dx
    grad_sigma_norm = grad_var / (2.0 * sigma_norm)

    # sklearn's return_std is also rescaled by training-y std
    sigma = y_scale * sigma_norm
    grad_sigma = y_scale * grad_sigma_norm

    return float(mu), float(sigma), grad_mu, grad_sigma


def neg_ei_with_grad(x_model, gp, best_y):
    """
    Return (-EI, gradient of -EI) for minimization by scipy.
    """
    mu, sigma, grad_mu, grad_sigma = gp_mean_std_grad(gp, x_model)

    if sigma <= 1e-12:
        return 0.0, np.zeros_like(x_model, dtype=float)

    improvement = best_y - mu
    z = improvement / sigma

    Phi = norm.cdf(z)
    phi = norm.pdf(z)

    ei = improvement * Phi + sigma * phi

    # For minimization EI:
    # grad EI = -Phi(z) grad_mu + phi(z) grad_sigma
    grad_ei = -Phi * grad_mu + phi * grad_sigma

    return -float(ei), -np.asarray(grad_ei)

def optimize_ei_ball(problem, gp, X_obs, y_obs, rng, cfg_bo, oracle=False):
    d = problem.d
    candidates = sample_unit_ball(cfg_bo.n_candidates, d, rng)
    cand_model = transform_for_gp(problem, candidates, oracle=oracle)
    cand_ei = expected_improvement_min(cand_model, gp, np.min(y_obs))

    order = np.argsort(cand_ei)[::-1]
    starts = candidates[order[:cfg_bo.n_acq_starts]]
    best_random = candidates[order[0]].copy()
    best_random_ei = float(cand_ei[order[0]])

    constraint = {
        'type': 'ineq',
        'fun': lambda x: 1.0 - np.dot(x, x),
        'jac': lambda x: -2.0 * x,
    }

    successful = []
    for x0 in starts:
        def objective_with_grad(x):
            x = np.asarray(x, dtype=float)
            if oracle:
                # GP operates on z = A.T x
                z = x @ problem.A
                value, grad_z = neg_ei_with_grad(
                    z,
                    gp,
                    np.min(y_obs)
                )                # Chain rule: z = A.T x  =>  grad_x EI = A grad_z
                grad_x = problem.A @ grad_z

                return value, grad_x

            else:
                # Isotropic / ARD GP operates directly on ambient x
                return neg_ei_with_grad(
                    x,
                    gp,
                    np.min(y_obs)
                )

        res = minimize(
            objective_with_grad,
            x0=x0,
            method='SLSQP',
            jac=True,
            constraints=[constraint],
            options={
                'maxiter': cfg_bo.acq_maxiter,
                'ftol': cfg_bo.acq_ftol,
                'disp': False,
            },
        )


        feasible = np.dot(res.x, res.x) <= 1.0 + 1e-8
        if res.success and feasible and np.isfinite(res.fun):
            successful.append((float(-res.fun), project_unit_ball(res.x), res.message))

    if successful:
        successful.sort(key=lambda t: t[0], reverse=True)
        best_ei, x_next, message = successful[0]
        fallback = False
    else:
        x_next = best_random
        best_ei = best_random_ei
        message = 'all SLSQP starts failed; used best random EI candidate'
        fallback = True

    min_dist = float(np.min(np.linalg.norm(X_obs - x_next, axis=1)))
    return x_next, {
        'acquisition_value': float(best_ei),
        'slsqp_fallback': bool(fallback),
        'min_distance_to_previous': min_dist,
        'near_duplicate': bool(min_dist < 1e-8),
        'slsqp_message': str(message),
    }

def run_random(problem, cfg, seed):
    rng = np.random.default_rng(seed)
    X_new = sample_unit_ball(cfg.n_eval, problem.d, rng)
    y_new = np.array([problem.evaluate(x) for x in X_new])
    return {'X_new': X_new, 'y_new': y_new, 'diagnostics': {}}


def run_bo(problem, cfg, seed, mode):
    assert mode in ('isotropic', 'ard', 'oracle')
    rng = np.random.default_rng(seed)
    oracle = (mode == 'oracle')
    ard = (mode == 'ard')
    d_model = problem.k if oracle else problem.d

    X_obs = problem.X_init.copy()
    y_obs = problem.y_init.copy()
    X_new, y_new = [], []
    diagnostics = []

    for iteration in range(cfg.n_eval):
        gp_seed = derived_seed(seed, 500, iteration)
        gp = make_gp(d_model, ard=ard, cfg_bo=cfg.bo, seed=gp_seed)
        gp.fit(transform_for_gp(problem, X_obs, oracle=oracle), y_obs)
        gp_diag = gp_diagnostics(gp)

        x_next, acq_diag = optimize_ei_ball(
            problem, gp, X_obs, y_obs, rng, cfg.bo, oracle=oracle
        )
        y_next = problem.evaluate(x_next)

        X_new.append(x_next.copy())
        y_new.append(float(y_next))
        diagnostics.append({
            'iteration': iteration + 1,
            **gp_diag,
            **acq_diag,
        })

        X_obs = np.vstack([X_obs, x_next])
        y_obs = np.append(y_obs, y_next)

    return {
        'X_new': np.asarray(X_new),
        'y_new': np.asarray(y_new),
        'diagnostics': diagnostics,
    }


def run_cma(problem, cfg, seed):
    best_idx = int(np.argmin(problem.y_init))
    x0 = problem.X_init[best_idx].copy()

    opts = {
        'seed': int(seed),
        'verbose': -9,
        'verb_disp': 0,
        'verb_log': 0,
    }
    if cfg.cma.popsize is not None:
        opts['popsize'] = int(cfg.cma.popsize)

    es = cma.CMAEvolutionStrategy(x0, cfg.cma.sigma0, opts)

    X_raw_all, X_eval_all, y_all, generation_all = [], [], [], []
    diagnostics = []
    remaining = cfg.n_eval
    generation = 0

    while remaining > 0:
        generation += 1
        X_raw_full = np.asarray(es.ask(), dtype=float)
        n_take = min(remaining, len(X_raw_full))
        X_raw = X_raw_full[:n_take]
        X_eval = project_unit_ball(X_raw)
        y = np.array([problem.evaluate(x) for x in X_eval])
        projected = np.linalg.norm(X_raw - X_eval, axis=1) > 1e-12

        X_raw_all.append(X_raw.copy())
        X_eval_all.append(X_eval.copy())
        y_all.append(y.copy())
        generation_all.extend([generation] * n_take)

        # Complete generations update CMA-ES. The final partial generation is evaluated
        # for exact budget accounting but is not used in an incomplete tell() update.
        complete = (n_take == len(X_raw_full))
        if complete:
            es.tell(X_eval.tolist(), y.tolist())

        diagnostics.append({
            'generation': generation,
            'n_evaluated': n_take,
            'complete_generation': bool(complete),
            'mean': np.asarray(es.mean, dtype=float).copy(),
            'sigma': float(es.sigma),
            'covariance': np.asarray(es.C, dtype=float).copy(),
            'n_projected': int(projected.sum()),
            'projection_fraction': float(projected.mean()),
        })

        remaining -= n_take

    return {
        'X_new': np.vstack(X_eval_all),
        'y_new': np.concatenate(y_all),
        'X_raw': np.vstack(X_raw_all),
        'generation': np.asarray(generation_all, dtype=int),
        'diagnostics': diagnostics,
    }

def run_experiment(problem, spec, cfg):
    assert spec.problem_id == problem.problem_id

    # Absolute timestamps allow overlap between parallel runs to be reconstructed.
    start_time = time.time()
    start_perf = time.perf_counter()

    if spec.solver == 'random':
        result = run_random(problem, cfg, spec.solver_seed)
    elif spec.solver == 'bo_isotropic':
        result = run_bo(problem, cfg, spec.solver_seed, mode='isotropic')
    elif spec.solver == 'bo_ard':
        result = run_bo(problem, cfg, spec.solver_seed, mode='ard')
    elif spec.solver == 'bo_oracle':
        result = run_bo(problem, cfg, spec.solver_seed, mode='oracle')
    elif spec.solver == 'cma_es':
        result = run_cma(problem, cfg, spec.solver_seed)
    else:
        raise ValueError(f'Unknown solver: {spec.solver}')

    end_perf = time.perf_counter()
    end_time = time.time()

    assert len(result['y_new']) == cfg.n_eval
    assert cfg.n_init_pts + len(result['y_new']) == cfg.budget

    result['runtime_seconds'] = float(end_perf - start_perf)
    result['start_time'] = float(start_time)
    result['end_time'] = float(end_time)
    result['worker_pid'] = int(os.getpid())
    return result

def jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def prepare_data_dirs(cfg):
    root = Path(cfg.data_dir)
    (root / 'problems').mkdir(parents=True, exist_ok=True)
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    (root / 'figures').mkdir(parents=True, exist_ok=True)
    return root


def save_global_metadata(cfg):
    root = prepare_data_dirs(cfg)
    with open(root / 'config.json', 'w') as f:
        json.dump(jsonable(asdict(cfg)), f, indent=2)
    versions = {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'sklearn': sklearn.__version__,
        'cma': getattr(cma, '__version__', 'unknown'),
    }
    with open(root / 'versions.json', 'w') as f:
        json.dump(versions, f, indent=2)


def save_problem(problem, cfg):
    root = prepare_data_dirs(cfg) / 'problems'
    np.savez_compressed(
        root / f'{problem.problem_id}.npz',
        A=problem.A,
        X_init=problem.X_init,
        y_init=problem.y_init,
    )
    meta = {
        'problem_id': problem.problem_id,
        'objective_name': problem.objective_name,
        'd': problem.d,
        'k': problem.k,
        'embedding': problem.embedding,
        'A_index': problem.A_index,
        'init_index': problem.init_index,
        'problem_seed': problem.problem_seed,
        'domain': problem.domain,
        'branin_scaling': problem.branin_scaling,
    }
    with open(root / f'{problem.problem_id}.json', 'w') as f:
        json.dump(jsonable(meta), f, indent=2)


def run_id(spec):
    return f'{spec.problem_id}__{spec.solver}'


def save_run(problem, spec, result, cfg):
    root = prepare_data_dirs(cfg) / 'runs'
    rid = run_id(spec)
    arrays = {
        'X_new': result['X_new'],
        'y_new': result['y_new'],
    }
    if 'X_raw' in result:
        arrays['X_raw'] = result['X_raw']
    if 'generation' in result:
        arrays['generation'] = result['generation']
    np.savez_compressed(root / f'{rid}.npz', **arrays)

    meta = {
        'run_id': rid,
        'problem_id': spec.problem_id,
        'solver': spec.solver,
        'solver_seed': spec.solver_seed,
        'runtime_seconds': result['runtime_seconds'],
        'start_time': result['start_time'],
        'end_time': result['end_time'],
        'worker_pid': result['worker_pid'],
        'budget': cfg.budget,
        'n_init_pts': cfg.n_init_pts,
        'n_eval_after_init': cfg.n_eval,
        'diagnostics': result['diagnostics'],
    }
    with open(root / f'{rid}.json', 'w') as f:
        json.dump(jsonable(meta), f, indent=2)



assert cfg.k == 2
assert cfg.n_inclusion == 1
assert 0 < cfg.n_init_pts < cfg.budget
assert cfg.bo.n_candidates >= cfg.bo.n_acq_starts >= 1


def run_file_exists(spec, cfg):
    """
    A run counts as complete only if both files exist and the NPZ contains
    exactly cfg.n_eval new objective evaluations.
    """
    root = Path(cfg.data_dir) / 'runs'
    rid = run_id(spec)
    npz_path = root / f'{rid}.npz'
    json_path = root / f'{rid}.json'

    if not (npz_path.exists() and json_path.exists()):
        return False

    try:
        with np.load(npz_path, allow_pickle=False) as arr:
            if len(arr['y_new']) != cfg.n_eval:
                return False

        with open(json_path, 'r') as f:
            meta = json.load(f)

        return (
            meta.get('run_id') == rid
            and int(meta.get('budget', -1)) == cfg.budget
        )
    except Exception:
        return False

def filter_run_specs(run_specs):
    return [spec for spec in run_specs if spec.solver in SOLVERS_TO_RUN]


def run_and_save(problem, spec, cfg):
    """
    Execute one independent (problem, solver) run inside a worker process,
    save it immediately, and return only a small status dictionary.
    """
    warnings.filterwarnings(
        "ignore",
        category=ConvergenceWarning,
    )
    try:
        result = run_experiment(problem, spec, cfg)
        save_run(problem, spec, result, cfg)

        best = min(problem.y_init.min(), result['y_new'].min())

        return {
            'status': 'saved',
            'run_id': run_id(spec),
            'problem_id': spec.problem_id,
            'solver': spec.solver,
            'runtime_seconds': result['runtime_seconds'],
            'start_time': result['start_time'],
            'end_time': result['end_time'],
            'worker_pid': result['worker_pid'],
            'best': float(best),
        }

    except Exception as exc:
        return {
            'status': 'failed',
            'run_id': run_id(spec),
            'problem_id': spec.problem_id,
            'solver': spec.solver,
            'error': f'{type(exc).__name__}: {exc}',
        }


def main():
    # Generate/freeze all problems in the main process before starting workers.
    problems = generate_problems(cfg)
    run_specs = filter_run_specs(generate_run_specs(problems, cfg))

    save_global_metadata(cfg)
    for problem in problems.values():
        save_problem(problem, cfg)

    print(f'Problems: {len(problems)}')
    print(f'Runs: {len(run_specs)}')
    print(f'Remaining evaluations after initialization: {cfg.n_eval}')
    print(f'Data directory: {cfg.data_dir}')
    print(f'Parallel workers: {N_JOBS}')

    remaining_specs = [
        spec for spec in run_specs
        if not run_file_exists(spec, cfg)
    ]

    n_existing = len(run_specs) - len(remaining_specs)
    print(f'Already saved: {n_existing}')
    print(f'To run: {len(remaining_specs)}')

    if not remaining_specs:
        print('All requested runs are already complete.')
        return

    overall_start = time.perf_counter()

    # Process-based parallelism. Each worker executes one independent run.
    # Limiting inner numerical-library threads avoids CPU oversubscription.
    with parallel_config(
        backend='loky',
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=N_JOBS,
            verbose=10,
            return_as='generator',
        )(
            delayed(run_and_save)(
                problems[spec.problem_id],
                spec,
                cfg,
            )
            for spec in remaining_specs
        )

        n_saved = 0
        failures = []

        # Results arrive as jobs complete. Each worker has already saved its
        # run before yielding this small status dictionary.
        for completed, status in enumerate(results, start=1):
            elapsed_hours = (time.perf_counter() - overall_start) / 3600

            if status['status'] == 'saved':
                n_saved += 1
                print(
                    f'[{completed}/{len(remaining_specs)} completed] '
                    f'{status["problem_id"]} | {status["solver"]} | '
                    f'runtime={status["runtime_seconds"]:.2f}s | '
                    f'best={status["best"]:.6g} | '
                    f'worker={status["worker_pid"]} | '
                    f'elapsed={elapsed_hours:.2f}h',
                    flush=True,
                )
            else:
                failures.append(status)
                print(
                    f'[{completed}/{len(remaining_specs)} completed] '
                    f'{status["problem_id"]} | {status["solver"]} | '
                    f'FAILED: {status["error"]}',
                    flush=True,
                )

    print()
    print('Parallel run finished.')
    print(f'New runs saved: {n_saved}')
    print(f'Failed runs: {len(failures)}')
    print(f'Session wall time: {(time.perf_counter() - overall_start) / 3600:.2f} h')

    if failures:
        failure_path = Path(cfg.data_dir) / 'failures.json'
        with open(failure_path, 'w') as f:
            json.dump(jsonable(failures), f, indent=2)
        print(f'Failure details saved to: {failure_path}')



if __name__ == '__main__':
    main()
