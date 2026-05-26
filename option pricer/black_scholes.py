import pandas as pd
import numpy as np
from scipy.stats import norm


##Black-Scholes formula for option pricing

def Black_Scholes(S, K, T, r, sigma, type="Call"):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:

        if type == "Call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif type == "Put":
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return price


##Calculating the Greeks

def Greeks(S, K, T, r, sigma, type="Call"):
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:

        if type == "Call":
            delta = norm.cdf(d1)
            delta = float(delta)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

        elif type == "Put":
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

def put_call_parity(S, K, T, r, price, type="Call"):

    try:
        if type == "Call":
            
            put_price = price + K * np.exp(-r * T) - S

            return put_price
        elif type == "Put":
            call_price = price + K * np.exp(-r * T) - S
            return call_price
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None








