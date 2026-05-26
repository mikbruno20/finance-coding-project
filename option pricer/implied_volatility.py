import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# Black-Scholes
def black_scholes(S, K, T, r, sigma, option_type="call"):
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


# Implied Volatility
def implied_vol(price, S, K, T, r, option_type="call"):
    
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - price
    
    return brentq(objective, 1e-6, 5.0)


# Parametri
S = 100
r = 0.01

strikes = np.array([80, 90, 100, 110, 120])
maturities = np.array([0.25, 0.5, 1.0])

market_prices = {
    0.25: [22, 13, 6, 2, 0.5],
    0.5:  [24, 15, 8, 4, 1.5],
    1.0:  [28, 18, 11, 7, 3]
}

# Costruzione surface
vol_surface = np.zeros((len(maturities), len(strikes)))

for i, T in enumerate(maturities):
    for j, K in enumerate(strikes):
        price = market_prices[T][j]
        vol_surface[i, j] = implied_vol(price, S, K, T, r)

# Plot 3D
from mpl_toolkits.mplot3d import Axes3D

K_grid, T_grid = np.meshgrid(strikes, maturities)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(K_grid, T_grid, vol_surface)

ax.set_xlabel('Strike')
ax.set_ylabel('Maturity')
ax.set_zlabel('Implied Volatility')

plt.show()