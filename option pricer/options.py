import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# european options - Black Scholes & Monte Carlo

def european_option_bs(S, K,  sigma, T, r, type = "call"):

    d1 = (np.log(S/K) + (r + 0.5 * sigma **2 ) * T ) / np.sqrt(T) * sigma
    d2 = d1 - np.sqrt(T) * sigma

    try:
        if type == "call":
            price = S * np.cdf(d1) - K * np.exp(-r * T) * np.cdf(d2)

            return price
        elif type == "put":
            price = K * np.exp(-r * T) * np.cdf(-d2) - S * np.cdf(-d1)

            return price
        else:
            return None
        
    except Exception as e:
        print(f"an error occured: {e}")

        return None


def european_option_mc(S0, K, T, r, sigma, n_sim, type="call"):
    
    # Generazione variabili normali
    Z = np.random.normal(size=n_sim)
    
    # Simulazione prezzo finale
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    # Payoff
    try: 
        if type == "call":
            payoff = np.maximum(ST - K, 0)
            price = np.exp(-r*T)* np.mean(payoff)
            return price
        
        else:
            payoff = np.maximum(K - ST, 0)
            price = np.exp(-r*T)* np.mean(payoff)
            return price

    except Exception as e:
        print(f"an error occured : {e}")
        return None
  


# american options  -- study !!!!

def american_option_binomial(S0, K, T, r, sigma, N, type="put"):
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # prezzi del sottostante alla maturità
    S = np.array([S0 * (u**j) * (d**(N-j)) for j in range(N+1)])

    if type == "call":
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # backward induction
    for i in range(N-1, -1, -1):

        S = S[:i+1] / d

        continuation = discount * (p * V[1:i+2] + (1-p) * V[0:i+1])

        if type == "call":
            exercise = np.maximum(S - K, 0)
        else:
            exercise = np.maximum(K - S, 0)

        V = np.maximum(continuation, exercise)

    return V[0]


# put-call parity

def put_call_parity(S, K, T, r, price, type="call"):

    try:
        if type == "call":
            put_price = price + K * np.exp(-r * T) - S
            return put_price
        elif type == "put":
            call_price = price + K * np.exp(-r * T) - S
            return call_price
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


#  Greeks

def Greeks(S, K, T, r, sigma, type="call"):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:

        if type == "call":
            delta = norm.cdf(d1)
            delta = float(delta)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

        elif type == "put":
            delta = norm.cdf(d1) - 1
            delta = float(delta)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return {
        "Delta": delta, 
        "Gamma": gamma, 
        "Vega": vega, 
        "Theta": theta, 
        "Rho": rho
        }  


# asian options with Monte Carlo

def asian_option_mc(S0, K, mu, r, T, sigma, n_paths, n_steps, type = "call"):

    dt = T / n_steps
    dW = np.random.normal(0 , 1, size = (n_paths, n_steps))

    increment = ( mu - 0.5 * sigma ** 2 ) * dt + sigma * np.sqrt(dt) * dW
    increments = np.cumsum(increment, axis = 1)

    S = S0*np.exp(increments)
    S_mean = np.mean(S, axis = 1)
    discount = np.exp(-r*T)

    try:
        if type == "call":
            payoff = np.maximum(S_mean - K, 0)
            call_price = discount * payoff
            return call_price 
        
        elif type == "put":
            payoff = np.maximum(K - S_mean, 0)
            put_price = discount * payoff
            return put_price
        
        else:
            return None
        
    except Exception as e:
        print(f"an error occured as: {e}")
        return None


# barrier options

def barrer_option_mc(S0, B, K, mu, sigma, r, T, n_paths, n_steps, type = "in", option_type = "call"):


    dt = T / n_steps
    dW = np.random.normal(0, 1, size = (n_paths, n_steps))

    increment = ( mu - 0.5 * sigma **2 ) * dt + sigma * np.sqrt(dt) * dW
    increments = np.cumsum(increment, axis = 1)

    price_t = S0*np.exp(increments)
    discount = np.exp(-r*T)

    try:
        if type == "in":   #  l'opzione si attiva
            if option_type == "call":
                if float(price_t.max()) >= B:
                    option_price = 0
                    return option_price
                
                else:
                    price_t = pd.DataFrame(price_t)
                    last_prices = price_t.iloc[n_steps - 1 , :]
                    last_prices = list(last_prices)
                    close = np.random.choice(last_prices)
                    option_price = float(discount * np.maximum(close - K, 0))

                    return option_price

            
            elif option_type == "put":
                if float(price_t.min())<= B : 
                    option_price = 0

                    return option_price
                
                else:
                    price_t = pd.DataFrame(price_t)
                    last_prices = price_t.iloc[n_steps - 1 , :]
                    last_prices = list(last_prices)
                    close = np.random.choice(last_prices)
                    option_price = float(discount * np.maximum(K - close, 0))

                    return option_price
            
            else:
                return None
        
        elif type == "out":      #  l'opzione si annulla
            if option_type == "call":
                if float(price_t.max()) >= B:
                    price_t = pd.DataFrame(price_t)
                    last_prices = price_t.iloc[n_steps - 1 , :]
                    last_prices = list(last_prices)
                    close = np.random.choice(last_prices)
                    option_price = float(discount * np.maximum(close - K, 0))
                    
                    return option_price

                else:
                    option_price = 0
                    return option_price

            elif option_type == "put":
                if float(price_t.min()) >= B:
                    price_t = pd.DataFrame(price_t)
                    last_prices = price_t.iloc[n_steps - 1 , :]
                    last_prices = list(last_prices)
                    close = np.random.choice(last_prices)
                    option_price = float(discount * np.maximum(K - close, 0))

                else:
                    option_price = 0
                    return option_price

            else:
                return None
    except Exception as e:
        print(f"an error occured as: {e}")
        return None



