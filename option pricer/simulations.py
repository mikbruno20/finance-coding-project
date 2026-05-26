import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm




## Geometric Brownian Motion single path

def simulate_gbm_single(S0, mu, sigma, T, n_steps):
    dt = T / n_steps
    Z = np.random.normal(0, 1 , n_steps)

    increments = ( mu - 0.5 * sigma **2 )* dt + sigma * np.sqrt(dt) * Z
    path = np.cumsum(increments)

    S_t = S0*np.exp(np.insert(path,0,0))

    S_t = pd.DataFrame(S_t)
    fig = S_t.plot(title = "Stock Price Dynamics using GBM")
    S_t.xlabel("time")
    S_t.ylabel("price")

    return fig

## Geometric Brownian Motion multiple path

def simulate_gbm_multiple(S0, mu, sigma, T, n_paths, n_steps):
    dt = T / n_steps
    Z = np.random.normal(0, 1, (n_paths, n_steps))

    increments = ( mu - 0.5 * sigma ** 2 )* dt + sigma * np.sqrt(dt) * Z

    paths = np.cumsum(increments, axis = 1 )

    S_t = S0*np.exp(np.hstack[np.zeros((n_paths, 1)), n_steps])

    S_t = pd.DataFrame()
    fig = S_t.plot(title = "Stock Price Dynamics of GBM with multiple paths")
    S_t.xlabel("time")
    S_t.ylabel("price")
    S_t.show()

    return fig



## Arithemtic Brownian Motion multiple path

def simulate_abm(S0, mu, sigma, T, n_paths, n_steps):
    dt = T / n_steps
    Z = np.random.normal(0, 1, (n_paths, n_steps))

    increments = mu * dt + sigma * np.sqrt(dt) * Z

    ## cumulative process 


    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 + np.cumsum(increments, axis=1)
    
    return paths


## Stochastic Variance process -- Heston Model

def simulate_heston(
    S0, v0,
    mu,
    kappa, theta, xi, rho,
    T,
    n_paths,
    n_steps
):
    dt = T / n_steps
    
    # Pre-allocate arrays
    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Generate correlated Brownian motions
    Z1 = np.random.normal(size=(n_paths, n_steps))
    Z2 = np.random.normal(size=(n_paths, n_steps))
    W1 = Z1
    W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    for t in range(n_steps):
        
        # Ensure variance stays positive (Full truncation)
        v_t = np.maximum(v[:, t], 0)
        
        # Variance process
        v[:, t+1] = (
            v[:, t]
            + kappa * (theta - v_t) * dt
            + xi * np.sqrt(v_t * dt) * W2[:, t]
        )
        
        v[:, t+1] = np.maximum(v[:, t+1], 0)
        
        # Stock process
        S[:, t+1] = (
            S[:, t]
            * np.exp(
                (mu - 0.5 * v_t) * dt
                + np.sqrt(v_t * dt) * W1[:, t]
            )
        )
    
    return S, v


















