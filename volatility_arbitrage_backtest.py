#!/usr/bin/env python3
"""
Volatility Arbitrage Backtesting System

This script implements a comprehensive volatility arbitrage strategy using SABR model
for implied volatility surface fitting and prediction. It compares predicted vs 
actual implied volatilities and executes trades when differences exceed thresholds,
with daily delta hedging.

Features:
- SABR model calibration per maturity
- IV prediction and comparison with market
- Options trading simulation
- Daily delta hedging
- P&L tracking and performance analysis

Usage:
1. Calibration mode (run SABR calibration and save results):
   python volatility_arbitrage_backtest.py calibrate --data-file <path> --start-date 2019-04-11 --end-date 2019-04-30

2. Backtest mode (run full backtest, optionally using pre-computed calibration):
   python volatility_arbitrage_backtest.py backtest --data-file <path> --start-date 2019-04-11 --end-date 2019-04-30
   python volatility_arbitrage_backtest.py backtest --calibration-file <calibration.pkl> --start-date 2019-04-11 --end-date 2019-04-30

3. Backtest with live plotting (real-time visualization):
   python volatility_arbitrage_backtest.py backtest --live-plotting --start-date 2019-04-11 --end-date 2019-04-30

Examples:
   # First run calibration
   python volatility_arbitrage_backtest.py calibrate
   
   # Then run backtest using pre-computed calibration
   python volatility_arbitrage_backtest.py backtest --calibration-file sabr_calibration_20190411_20190430.pkl
   
   # Or run backtest with on-the-fly calibration and live plotting
   python volatility_arbitrage_backtest.py backtest --live-plotting
   
   # Run backtest with pre-computed calibration and live plotting
   python volatility_arbitrage_backtest.py backtest --calibration-file sabr_calibration_20190411_20190430.pkl --live-plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import argparse
import pickle
import os
import glob
import re
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FuncFormatter
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable matplotlib font manager logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')


class LivePlotter:
    """Live plotting class for real-time backtest visualization"""
    
    def __init__(self, title="Volatility Arbitrage Backtest Results"):
        """Initialize the live plotter"""
        self.title = title
        self.dates = []
        self.portfolio_values = []
        self.pnl_values = []
        self.num_positions = []
        self.cash_values = []
        self.stock_values = []
        self.underlying_prices = []
        self.stock_positions = []
        self.buy_calls = []
        self.sell_calls = []
        self.buy_puts = []
        self.sell_puts = []
        
        # Set up the figure and axes
        plt.ion()  # Turn on interactive mode
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle(self.title, fontsize=16)
        
        # Initialize empty plots
        self.portfolio_line, = self.axes[0, 0].plot([], [], 'b-', linewidth=2, label='Portfolio Value')
        self.pnl_line, = self.axes[0, 1].plot([], [], 'g-', linewidth=2)
        
        # Create secondary y-axis for portfolio plot (underlying price)
        self.portfolio_ax2 = self.axes[0, 0].twinx()
        self.portfolio_ax2.set_ylabel('Underlying Price ($)', color='orange')
        self.portfolio_ax2.tick_params(axis='y', labelcolor='orange')
        self.underlying_line, = self.portfolio_ax2.plot([], [], 'orange', linewidth=2, label='Underlying Price', alpha=0.7)
        
        # Positions plot with multiple lines and secondary y-axis
        self.total_positions_line, = self.axes[1, 0].plot([], [], 'k-', linewidth=2, label='Total')
        self.buy_calls_line, = self.axes[1, 0].plot([], [], 'b-', linewidth=1.5, label='Buy Calls')
        self.sell_calls_line, = self.axes[1, 0].plot([], [], 'b--', linewidth=1.5, label='Sell Calls')
        self.buy_puts_line, = self.axes[1, 0].plot([], [], 'r-', linewidth=1.5, label='Buy Puts')
        self.sell_puts_line, = self.axes[1, 0].plot([], [], 'r--', linewidth=1.5, label='Sell Puts')
        
        # Create secondary y-axis for positions
        self.positions_ax2 = self.axes[1, 0].twinx()
        self.positions_ax2.set_ylabel('Individual Position Types', color='gray')
        self.positions_ax2.tick_params(axis='y', labelcolor='gray')
        
        # Move individual position lines to secondary axis
        self.buy_calls_line_right, = self.positions_ax2.plot([], [], 'b-', linewidth=1.5, label='Buy Calls', alpha=0.7)
        self.sell_calls_line_right, = self.positions_ax2.plot([], [], 'b--', linewidth=1.5, label='Sell Calls', alpha=0.7)
        self.buy_puts_line_right, = self.positions_ax2.plot([], [], 'r-', linewidth=1.5, label='Buy Puts', alpha=0.7)
        self.sell_puts_line_right, = self.positions_ax2.plot([], [], 'r--', linewidth=1.5, label='Sell Puts', alpha=0.7)
        
        # Keep only total on left axis
        self.buy_calls_line.remove()
        self.sell_calls_line.remove() 
        self.buy_puts_line.remove()
        self.sell_puts_line.remove()
        
        # Cash and stock plot with secondary y-axis
        self.cash_line, = self.axes[1, 1].plot([], [], 'm-', linewidth=2, label='Cash')
        self.stock_line, = self.axes[1, 1].plot([], [], 'orange', linewidth=2, label='Stock Value')
        
        # Create secondary y-axis for cash & stock plot (stock position count)
        self.cash_ax2 = self.axes[1, 1].twinx()
        self.cash_ax2.set_ylabel('Stock Position (Shares)', color='brown')
        self.cash_ax2.tick_params(axis='y', labelcolor='brown')
        self.stock_position_line, = self.cash_ax2.plot([], [], 'brown', linewidth=2, label='Stock Position', alpha=0.7)
        
        # Set up axis properties
        self.axes[0, 0].set_title('Portfolio Value Over Time')
        self.axes[0, 0].set_ylabel('Portfolio Value ($)', color='blue')
        self.axes[0, 0].grid(True, alpha=0.3)
        self.axes[0, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Setup portfolio plot legend with both axes
        lines1, labels1 = self.axes[0, 0].get_legend_handles_labels()
        lines2, labels2 = self.portfolio_ax2.get_legend_handles_labels()
        self.axes[0, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        self.axes[0, 1].set_title('Cumulative P&L')
        self.axes[0, 1].set_ylabel('P&L ($)')
        self.axes[0, 1].grid(True, alpha=0.3)
        self.axes[0, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
        
        self.axes[1, 0].set_title('Options Positions by Type')
        self.axes[1, 0].set_ylabel('Total Positions', color='black')
        self.axes[1, 0].grid(True, alpha=0.3)
        self.axes[1, 0].legend(loc='upper left', fontsize=8)
        
        # Setup secondary axis legend
        lines1, labels1 = self.axes[1, 0].get_legend_handles_labels()
        lines2, labels2 = self.positions_ax2.get_legend_handles_labels()
        self.axes[1, 0].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        self.axes[1, 1].set_title('Cash & Stock Positions')
        self.axes[1, 1].set_ylabel('Value ($)', color='purple')
        self.axes[1, 1].grid(True, alpha=0.3)
        self.axes[1, 1].yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
        
        # Setup cash & stock plot legend with both axes
        lines3, labels3 = self.axes[1, 1].get_legend_handles_labels()
        lines4, labels4 = self.cash_ax2.get_legend_handles_labels()
        self.axes[1, 1].legend(lines3 + lines4, labels3 + labels4, loc='upper left', fontsize=8)
        
        # Format x-axis for dates
        for ax in self.axes.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.dates)//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show(block=False)
        
    def update_data(self, date, portfolio_value, pnl, num_positions, cash, stock_value, position_breakdown, underlying_price, stock_position):
        """Update the data points
        
        Parameters:
        position_breakdown: dict with keys 'buy_calls', 'sell_calls', 'buy_puts', 'sell_puts'
        underlying_price: current underlying asset price
        stock_position: number of shares held
        """
        self.dates.append(date)
        self.portfolio_values.append(portfolio_value)
        self.pnl_values.append(pnl)
        self.num_positions.append(num_positions)
        self.cash_values.append(cash)
        self.stock_values.append(stock_value)
        self.underlying_prices.append(underlying_price)
        self.stock_positions.append(stock_position)
        self.buy_calls.append(position_breakdown.get('buy_calls', 0))
        self.sell_calls.append(position_breakdown.get('sell_calls', 0))
        self.buy_puts.append(position_breakdown.get('buy_puts', 0))
        self.sell_puts.append(position_breakdown.get('sell_puts', 0))
        
    def update_plots(self):
        """Update all plots with new data"""
        if not self.dates:
            return
            
        # Update data for each line
        dates_num = mdates.date2num(self.dates)
        
        # Portfolio plot - primary and secondary axes
        self.portfolio_line.set_data(dates_num, self.portfolio_values)
        self.underlying_line.set_data(dates_num, self.underlying_prices)
        
        self.pnl_line.set_data(dates_num, self.pnl_values)
        
        # Update positions lines
        self.total_positions_line.set_data(dates_num, self.num_positions)
        
        # Update individual position lines on secondary axis
        self.buy_calls_line_right.set_data(dates_num, self.buy_calls)
        self.sell_calls_line_right.set_data(dates_num, self.sell_calls)
        self.buy_puts_line_right.set_data(dates_num, self.buy_puts)
        self.sell_puts_line_right.set_data(dates_num, self.sell_puts)
        
        # Update cash and stock lines
        self.cash_line.set_data(dates_num, self.cash_values)
        self.stock_line.set_data(dates_num, self.stock_values)
        self.stock_position_line.set_data(dates_num, self.stock_positions)
        
        # Rescale primary axes
        for ax in self.axes.flat:
            ax.relim()
            ax.autoscale_view()
            
            # Format x-axis dates
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(self.dates)//10)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Rescale secondary axes
        self.portfolio_ax2.relim()
        self.portfolio_ax2.autoscale_view()
        
        self.positions_ax2.relim()
        self.positions_ax2.autoscale_view()
        
        self.cash_ax2.relim()
        self.cash_ax2.autoscale_view()
        
        # Add latest values as text annotations
        if len(self.dates) > 0:
            latest_date = self.dates[-1].strftime('%Y-%m-%d')
            latest_portfolio = self.portfolio_values[-1]
            latest_pnl = self.pnl_values[-1]
            latest_positions = self.num_positions[-1]
            latest_cash = self.cash_values[-1]
            latest_stock = self.stock_values[-1]
            latest_buy_calls = self.buy_calls[-1]
            latest_sell_calls = self.sell_calls[-1]
            latest_buy_puts = self.buy_puts[-1]
            latest_sell_puts = self.sell_puts[-1]
            
            # Clear previous annotations and add new ones
            for ax in self.axes.flat:
                for txt in ax.texts:
                    if hasattr(txt, '_live_annotation'):
                        txt.remove()
            
            # Portfolio value annotation
            txt = self.axes[0, 0].text(0.02, 0.98, f'Latest: ${latest_portfolio:,.0f}', 
                                      transform=self.axes[0, 0].transAxes, 
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            txt._live_annotation = True
            
            # P&L annotation
            pnl_color = 'green' if latest_pnl >= 0 else 'red'
            txt = self.axes[0, 1].text(0.02, 0.98, f'Latest: ${latest_pnl:,.0f}', 
                                      transform=self.axes[0, 1].transAxes, 
                                      verticalalignment='top', color=pnl_color,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            txt._live_annotation = True
            
            # Positions annotation
            positions_text = f'Total: {latest_positions}\nBuy Calls: {latest_buy_calls}\nSell Calls: {latest_sell_calls}\nBuy Puts: {latest_buy_puts}\nSell Puts: {latest_sell_puts}'
            txt = self.axes[1, 0].text(0.02, 0.98, positions_text, 
                                      transform=self.axes[1, 0].transAxes, 
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            txt._live_annotation = True
            
            # Cash and stock annotation
            cash_stock_text = f'Cash: ${latest_cash:,.0f}\nStock: ${latest_stock:,.0f}\nTotal: ${latest_cash + latest_stock:,.0f}'
            txt = self.axes[1, 1].text(0.02, 0.98, cash_stock_text, 
                                      transform=self.axes[1, 1].transAxes, 
                                      verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            txt._live_annotation = True
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def save_final_plot(self, filename='live_volatility_arbitrage_results.png'):
        """Save the final plot"""
        plt.ioff()  # Turn off interactive mode
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Final live plot saved to {filename}")
        
    def close(self):
        """Close the live plot"""
        plt.ioff()
        plt.close(self.fig)


@dataclass
class OptionPosition:
    """Class to track an option position"""
    # option_id: strs
    symbol: str
    expiration_date: datetime
    strike: float
    option_type: str  # 'C' or 'P'
    quantity: int  # positive for long, negative for short
    entry_price: float
    entry_date: datetime
    entry_iv: float
    predicted_iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    underlying_price: float
    cur_price: float
    prev_price: float
    cur_date: datetime

@dataclass
class SABRParameters:
    """Class to store SABR model parameters"""
    alpha: float
    beta: float
    rho: float
    nu: float
    forward_rate: float
    time_to_maturity: float

class BlackScholes:
    """Black-Scholes option pricing and Greeks calculations"""
    
    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1 parameter"""
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2 parameter"""
        return BlackScholes.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate European call option price"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate European put option price"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='C'):
        """Calculate option delta"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == 'C':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate option gamma"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def theta(S, K, T, r, sigma, option_type='C'):
        """Calculate option theta"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        if option_type == 'C':
            return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r*T) * norm.cdf(d2)) / 365
        else:
            return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r*T) * norm.cdf(-d2)) / 365
    
    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate option vega"""
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100
    
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='C'):
        """Calculate implied volatility using Brent's method"""
        def objective(sigma):
            if option_type == 'C':
                theoretical_price = BlackScholes.call_price(S, K, T, r, sigma)
            else:
                theoretical_price = BlackScholes.put_price(S, K, T, r, sigma)
            return (theoretical_price - market_price) ** 2
        
        try:
            result = minimize(objective, x0=0.2, bounds=[(0.001, 5.0)], method='L-BFGS-B')
            return result.x[0] if result.success else np.nan
        except:
            return np.nan

class SABRModel:
    """SABR (Stochastic Alpha Beta Rho) volatility model implementation"""
    
    def __init__(self, beta=1.0):
        """Initialize SABR model with fixed beta"""
        self.beta = beta
    
    def sabr_volatility(self, F, K, T, alpha, beta, rho, nu):
        """
        Calculate SABR implied volatility using Hagan's formula
        
        Parameters:
        F: Forward rate
        K: Strike price
        T: Time to maturity
        alpha: Initial volatility
        beta: CEV parameter (usually 0 for normal model, 1 for lognormal)
        rho: Correlation between asset and volatility
        nu: Volatility of volatility
        """
        if T <= 0:
            return 0
        
        if beta==1:
            ### this same as in the thesis
            z = (nu*np.log(F/K))/alpha
            x = np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
            sigma = alpha*(z/x)*(1+0.125*T*(2*nu*rho*alpha+nu**2*((2/3)-rho**2)))
            return sigma
        
        if abs(F - K) < 1e-6:  # At-the-money case
            sigma = alpha * (F**(beta-1)) * (1 + ((beta-1)**2/24) * (np.log(F/K))**2 + 
                                           (beta-1)**4/1920 * (np.log(F/K))**4) * \
                    (1 + (((beta-1)**2/24) * alpha**2/(F**(2-2*beta)) + 
                          (rho*beta*nu*alpha)/(4*F**(1-beta)) + 
                          ((2-3*rho**2)/24)*nu**2) * T)
        else:
            # General case
            z = (nu/alpha) * (F*K)**((1-beta)/2) * np.log(F/K)
            if abs(z) < 1e-6:
                x_z = 1
            else:
                x_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)/(1 - rho))
            
            sigma = (nu * (F*K)**((1-beta)/2) * x_z * 
                    (1 + ((beta-1)**2/24) * (np.log(F/K))**2 + 
                     (beta-1)**4/1920 * (np.log(F/K))**4) * 
                    (1 + (((beta-1)**2/24) * alpha**2/(F*K)**(1-beta) + 
                          (rho*beta*nu*alpha)/(4*(F*K)**((1-beta)/2)) + 
                          ((2-3*rho**2)/24)*nu**2) * T))
        
        return max(sigma, 1e-6)  # Ensure positive volatility
    
    def calibrate(self, strikes, market_ivs, forward, time_to_maturity, initial_guess=None):
        """
        Calibrate SABR parameters to market implied volatilities
        
        Parameters:
        strikes: Array of strike prices
        market_ivs: Array of market implied volatilities
        forward: Forward price
        time_to_maturity: Time to maturity in years
        initial_guess: Initial parameter guess [alpha, rho, nu]
        
        Returns:
        SABRParameters object with calibrated parameters
        """
        if initial_guess is None:
            initial_guess = [0.2, -0.5, 0.3]  # [alpha, rho, nu]
        
        def objective(params):
            alpha, rho, nu = params
            predicted_ivs = []
            for K in strikes:
                try:
                    iv = self.sabr_volatility(forward, K, time_to_maturity, alpha, self.beta, rho, nu)
                    predicted_ivs.append(iv)
                except:
                    predicted_ivs.append(np.nan)
            
            predicted_ivs = np.array(predicted_ivs)
            valid_mask = ~np.isnan(predicted_ivs) & ~np.isnan(market_ivs)
            
            if np.sum(valid_mask) < 2:
                return 1e6
            
            error = np.sum((predicted_ivs[valid_mask] - np.array(market_ivs)[valid_mask])**2)
            return error
        
        # Bounds: alpha > 0, -1 < rho < 1, nu > 0
        bounds = [(0.0001, 2.0), (-0.999, 0.999), (0., 10.0)]
        
        try:
            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
            
            if result.success:
                alpha, rho, nu = result.x
                return SABRParameters(
                    alpha=alpha,
                    beta=self.beta,
                    rho=rho,
                    nu=nu,
                    forward_rate=forward,
                    time_to_maturity=time_to_maturity
                )
            else:
                logger.warning(f"SABR calibration failed: {result.message}")
                return None
        except Exception as e:
            logger.error(f"SABR calibration error: {str(e)}")
            return None
    
    def predict_iv(self, strike, sabr_params):
        """Predict implied volatility for a given strike using calibrated SABR parameters"""
        if sabr_params is None:
            return np.nan
        
        return self.sabr_volatility(
            sabr_params.forward_rate, 
            strike, 
            sabr_params.time_to_maturity,
            sabr_params.alpha, 
            sabr_params.beta, 
            sabr_params.rho, 
            sabr_params.nu
        )

class VolatilityArbitrageStrategy:
    """Main volatility arbitrage strategy class"""
    
    def __init__(self, 
                 iv_threshold=0.02,  # 2% IV difference threshold
                 risk_free_rate=0.02,
                 transaction_cost=0.005,  # 0.5% transaction cost
                 max_position_size=100,
                 delta_hedge_threshold=0.1,
                 start_date=None,
                 end_date=None):
        """
        Initialize the volatility arbitrage strategy
        
        Parameters:
        iv_threshold: Minimum IV difference to trigger trades
        risk_free_rate: Risk-free interest rate
        transaction_cost: Transaction cost as percentage of notional
        max_position_size: Maximum position size per trade
        delta_hedge_threshold: Delta threshold for rehedging
        start_date: Start date for backtest (for CSV naming)
        end_date: End date for backtest (for CSV naming)
        """
        self.iv_threshold = iv_threshold
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.delta_hedge_threshold = delta_hedge_threshold
        self.start_date = start_date
        self.end_date = end_date
        
        self.sabr_model = SABRModel()
        self.positions = {}
        self.cash = 0
        self.stock_position = 0
        self.pnl_history = []
        self.trade_log = []
        self.least_dte_to_open = 40
        self.max_moneyness_to_open = 0.2
        
        # Initialize data attributes
        self.options_data = None
        self.available_files = []
        self.current_file_index = 0
        
        # Live plotting
        self.live_plotter = None
        
    def discover_data_files(self, data_file_pattern):
        """
        Discover all data files matching the pattern and sort them by date range
        
        Parameters:
        data_file_pattern: Base file path pattern (e.g., ~/data/quant/volarb/SPX_options_2019-04-11_2019-04-30.parquet)
        
        Returns:
        List of tuples: (file_path, start_date, end_date)
        """
        # Extract directory and pattern from the file path
        data_file_expanded = os.path.expanduser(data_file_pattern)
        data_dir = os.path.dirname(data_file_expanded)
        filename_base = os.path.basename(data_file_expanded)
        
        # Create a pattern to match files with date ranges
        # Convert SPX_options_2019-04-11_2019-04-30.parquet to SPX_options_*_*.parquet
        pattern_parts = filename_base.split('_')
        if len(pattern_parts) >= 4:
            # Assuming pattern: PREFIX_YYYY-MM-DD_YYYY-MM-DD.parquet
            file_pattern = f"{data_dir}/{pattern_parts[0]}_{pattern_parts[1]}_*_*.parquet"
        else:
            # Fallback: just look for all parquet files in directory
            file_pattern = f"{data_dir}/*.parquet"
        
        logger.info(f"Searching for files with pattern: {file_pattern}")
        
        files_with_dates = []
        for file_path in glob.glob(file_pattern):
            filename = os.path.basename(file_path)
            
            # Extract dates from filename using regex
            # Pattern: SPX_options_YYYY-MM-DD_YYYY-MM-DD.parquet
            date_pattern = r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})'
            match = re.search(date_pattern, filename)
            
            if match:
                start_date_str, end_date_str = match.groups()
                try:
                    start_date = pd.Timestamp(start_date_str)
                    end_date = pd.Timestamp(end_date_str)
                    files_with_dates.append((file_path, start_date, end_date))
                except:
                    logger.warning(f"Could not parse dates from filename: {filename}")
                    continue
        
        # Sort files by start date
        files_with_dates.sort(key=lambda x: x[1])
        
        logger.info(f"Found {len(files_with_dates)} data files:")
        for file_path, start_date, end_date in files_with_dates:
            logger.info(f"  {os.path.basename(file_path)}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return files_with_dates

    def load_single_file(self, data_file):
        """Load a single parquet file and process it"""
        try:
            logger.info(f"Loading file: {os.path.basename(data_file)}")
            data = pd.read_parquet(data_file)
            
            # Map actual column names to expected names
            column_mapping = {
                'c_date': 'trade_date',
                'call_put': 'cp_flag',
                'price': 'price_close',
                'Bid': 'price_bid',
                'Ask': 'price_ask'
            }
            
            # Rename columns
            data = data.rename(columns=column_mapping)
            
            # Convert date columns
            date_columns = ['trade_date', 'expiration_date']
            for col in date_columns:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col])
            
            # Calculate time to maturity
            data['tte'] = (
                data['expiration_date'] - data['trade_date']
            ).dt.days / 365.25
            
            # Filter out options with very short or long maturities
            data = data[
                (data['tte'] > 0.01) & 
                (data['tte'] < 2.0)
            ]
            
            # Calculate bid-ask spread for filtering
            if 'price_bid' in data.columns and 'price_ask' in data.columns:
                data['spread'] = data['price_ask'] - data['price_bid']
                data['mid_price'] = (data['price_bid'] + data['price_ask']) / 2
            else:
                # If no bid/ask data, use close price as mid
                data['spread'] = 0
                data['mid_price'] = data['price_close']
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading file {data_file}: {str(e)}")
            return None

    def load_data(self, data_file_pattern):
        """Load options data from parquet files (handles multiple sequential files)"""
        logger.info("Discovering and loading options data...")
        
        # Discover all available data files
        self.available_files = self.discover_data_files(data_file_pattern)
        
        if not self.available_files:
            logger.error("No data files found matching the pattern")
            return False
        
        # Load the first file initially
        first_file_path, _, _ = self.available_files[0]
        self.options_data = self.load_single_file(first_file_path)
        self.current_file_index = 0
        
        if self.options_data is None:
            logger.error("Failed to load initial data file")
            return False
        
        logger.info(f"Loaded {len(self.options_data)} option records from initial file")
        return True

    def get_all_trading_dates(self, start_date=None, end_date=None):
        """
        Get all trading dates from all available files
        
        Parameters:
        start_date: Optional start date filter
        end_date: Optional end date filter
        
        Returns:
        Sorted list of trading dates
        """
        if not hasattr(self, 'available_files') or not self.available_files:
            # Fallback to current data if available
            if self.options_data is not None:
                trading_dates = sorted(self.options_data['trade_date'].unique())
            else:
                return []
        else:
            # Collect dates from all files
            all_dates = set()
            for file_path, file_start, file_end in self.available_files:
                try:
                    temp_data = self.load_single_file(file_path)
                    if temp_data is not None:
                        file_dates = temp_data['trade_date'].unique()
                        all_dates.update(file_dates)
                except Exception as e:
                    logger.warning(f"Could not load dates from {file_path}: {str(e)}")
                    continue
            
            trading_dates = sorted(list(all_dates))
        
        # Apply date filters
        if start_date:
            trading_dates = [d for d in trading_dates if d >= start_date]
        if end_date:
            trading_dates = [d for d in trading_dates if d <= end_date]
        
        return trading_dates

    def check_and_load_next_file(self, current_date):
        """
        Check if we need to load the next file based on current date
        
        Parameters:
        current_date: Current trading date
        
        Returns:
        True if new file was loaded, False otherwise
        """
        if not hasattr(self, 'available_files') or not self.available_files:
            return False
        
        # Check if current date exceeds the range of current file
        current_file_path, current_start, current_end = self.available_files[self.current_file_index]
        
        if current_date > current_end:
            # Need to load next file
            next_file_index = self.current_file_index + 1
            
            if next_file_index < len(self.available_files):
                next_file_path, next_start, next_end = self.available_files[next_file_index]
                
                logger.info(f"Current date {current_date.strftime('%Y-%m-%d')} exceeds current file range "
                           f"({current_end.strftime('%Y-%m-%d')}). Loading next file...")
                
                # Load next file
                new_data = self.load_single_file(next_file_path)
                
                if new_data is not None:
                    # Replace current data with new data
                    self.options_data = new_data
                    self.current_file_index = next_file_index
                    
                    logger.info(f"Successfully loaded next file: {os.path.basename(next_file_path)}")
                    logger.info(f"New data range: {next_start.strftime('%Y-%m-%d')} to {next_end.strftime('%Y-%m-%d')}")
                    logger.info(f"Loaded {len(self.options_data)} option records")
                    return True
                else:
                    logger.error(f"Failed to load next file: {next_file_path}")
                    return False
            else:
                logger.warning(f"Current date {current_date.strftime('%Y-%m-%d')} exceeds all available data files")
                return False
        
        return False
    
    def get_risk_free_rate(self, date):
        """Get risk-free rate for a given date (simplified - uses constant rate)"""
        return self.risk_free_rate
    
    def get_forward_price(self, spot_price, risk_free_rate, time_to_maturity):
        """Calculate forward price"""
        return spot_price * np.exp(risk_free_rate * time_to_maturity)
    
    def filter_options_for_calibration(self, exp_group, forward_price, method='default'):
        """
        Filter options for SABR calibration:
        - default: Use puts for K < F and calls for K > F, around ATM take the leg with tighter spread
        - put_only: Use only put options without further filtration
        - call_only: Use only call options without further filtration
        
        Parameters:
        exp_group: DataFrame with options for a specific expiration
        forward_price: Forward price for comparison
        method: Filtering method ('default', 'put_only', 'call_only')
        
        Returns:
        Filtered DataFrame with selected options for calibration
        """
        if method == 'put_only':
            # Return all put options
            puts = exp_group[exp_group['cp_flag'] == 'P']
            puts = puts[puts['volume'] != 0]  # Ensure positive volumes
            return puts
        
        elif method == 'call_only':
            # Return all call options
            calls = exp_group[exp_group['cp_flag'] == 'C']
            calls = calls[calls['volume'] != 0]  # Ensure positive volumes
            return calls
        
        else:  # method == 'default'
            atm_threshold = 0.05  # 5% around ATM
            
            filtered_options = []
            
            # Group by strike
            for strike, strike_group in exp_group.groupby('price_strike'):
                moneyness = strike / forward_price
                
                # Get call and put data for this strike
                calls = strike_group[strike_group['cp_flag'] == 'C']
                puts = strike_group[strike_group['cp_flag'] == 'P']
                
                if len(calls) == 0 and len(puts) == 0:
                    continue
                
                # Determine which option to use
                if moneyness < (1 - atm_threshold):  # OTM puts (K < F)
                    if len(puts) > 0:
                        filtered_options.append(puts.iloc[0])
                elif moneyness > (1 + atm_threshold):  # OTM calls (K > F)
                    if len(calls) > 0:
                        filtered_options.append(calls.iloc[0])
                else:  # ATM region - choose based on spread
                    if len(calls) > 0 and len(puts) > 0:
                        call_spread = calls.iloc[0]['spread']
                        put_spread = puts.iloc[0]['spread']
                        
                        if call_spread <= put_spread:
                            filtered_options.append(calls.iloc[0])
                        else:
                            filtered_options.append(puts.iloc[0])
                    elif len(calls) > 0:
                        filtered_options.append(calls.iloc[0])
                    elif len(puts) > 0:
                        filtered_options.append(puts.iloc[0])
            
            return pd.DataFrame(filtered_options) if filtered_options else pd.DataFrame()

    def ensure_data_loaded_for_date(self, target_date):
        """
        Ensure the correct data file is loaded for the target date
        
        Parameters:
        target_date: The date for which we need data
        
        Returns:
        True if data is available for the date, False otherwise
        """
        if not hasattr(self, 'available_files') or not self.available_files:
            return self.options_data is not None
        
        # Check if current file covers the target date
        current_file_path, current_start, current_end = self.available_files[self.current_file_index]
        
        if current_start <= target_date <= current_end:
            # Current file is correct
            return self.options_data is not None
        
        # Find the correct file for this date
        for i, (file_path, start_date, end_date) in enumerate(self.available_files):
            if start_date <= target_date <= end_date:
                # Need to load this file
                if i != self.current_file_index:
                    logger.info(f"Loading file for date {target_date.strftime('%Y-%m-%d')}: {os.path.basename(file_path)}")
                    new_data = self.load_single_file(file_path)
                    if new_data is not None:
                        self.options_data = new_data
                        self.current_file_index = i
                        return True
                    else:
                        logger.error(f"Failed to load file: {file_path}")
                        return False
                else:
                    return self.options_data is not None
        
        logger.warning(f"No data file found for date: {target_date.strftime('%Y-%m-%d')}")
        return False

    def calibrate_sabr_by_maturity(self, trade_date, save_plots=False, calculate_iv= False, plot_valids_only= False):
        """
        Calibrate SABR model for each maturity on a given trade date
        
        Parameters:
        trade_date: Trading date for calibration
        save_plots: Whether to save calibration plots
        plot_valids_only: Whether to plot only valid strikes

        Returns:
        Dictionary mapping expiration dates to SABRParameters
        """
        # Ensure correct data is loaded for this date
        if not self.ensure_data_loaded_for_date(trade_date):
            logger.error(f"Could not load data for date: {trade_date}")
            return {}
        
        filter_method = 'default'
        day_data = self.options_data[self.options_data['trade_date'] == trade_date]
        
        if len(day_data) == 0:
            return {}
        
        sabr_params_by_maturity = {}
        calibration_results_calls = []  # Store results for plotting
        calibration_results_puts = []  # Store results for plotting
        calibration_results = []

        # Group by expiration date
        underlying_price_tot_median = day_data['underlying_price'].median()
        for exp_date, exp_group in day_data.groupby('expiration_date'):
            if len(exp_group) < 5:  # Need minimum number of strikes
                continue
            
            # Get underlying price (use median to avoid outliers)
            underlying_price = exp_group['underlying_price'].median()
            time_to_maturity = exp_group['tte'].iloc[0]
            
            if time_to_maturity <= 0:
                continue
            
            # Calculate forward price
            risk_free_rate = self.get_risk_free_rate(trade_date)
            forward_price = self.get_forward_price(underlying_price, risk_free_rate, time_to_maturity)
            
            # Filter options for calibration using the new filtering logic
            filtered_group = self.filter_options_for_calibration(exp_group, forward_price, method= filter_method)
            
            if len(filtered_group) < 5:  # Need minimum number of filtered options
                continue
            
            # Prepare data for calibration using filtered options
            strikes = filtered_group['price_strike'].values
            market_prices = filtered_group['mid_price'].values  # Use mid price for calibration
            
            if calculate_iv:
                # Calculate implied volatilities
                implied_vols = []
                valid_strikes = []
                
                for i, (strike, market_price) in enumerate(zip(strikes, market_prices)):
                    option_type = filtered_group.iloc[i]['cp_flag']
                    
                    try:
                        iv = BlackScholes.implied_volatility(
                            market_price, underlying_price, strike, 
                            time_to_maturity, risk_free_rate, option_type
                        )
                        
                        if not np.isnan(iv) and 0.01 < iv < 2.0:  # Reasonable IV range
                            implied_vols.append(iv)
                            valid_strikes.append(strike)
                            
                    except Exception as e:
                        print(f"IV calculation error for strike {strike}: {str(e)}")
                        continue
                
                if len(implied_vols) < 5:
                    continue
            else:
                implied_vols = filtered_group['iv'].values
                valid_strikes = strikes
            
            # Calibrate SABR model
            sabr_params = self.sabr_model.calibrate(
                valid_strikes, 
                implied_vols, 
                forward_price, 
                time_to_maturity
            )
            
            if sabr_params is not None:
                sabr_params_by_maturity[exp_date] = sabr_params
                
                # Generate predicted IVs for plotting
                if save_plots:
                    if plot_valids_only:
                        predicted_ivs = []

                        for strike in valid_strikes:
                            pred_iv = self.sabr_model.predict_iv(strike, sabr_params)
                            predicted_ivs.append(pred_iv)
                        calibration_results_calls.append({
                            'expiration_date': exp_date,
                            'strikes': valid_strikes,
                            'market_ivs': implied_vols,
                            'predicted_ivs': predicted_ivs,
                            'time_to_maturity': time_to_maturity,
                            'sabr_params': sabr_params
                        })
                    else:
                        call_predicted_ivs = []
                        put_predicted_ivs = []
                        call_market_ivs = []
                        put_market_ivs = []
                        call_strikes = []
                        put_strikes = []
                        for row in exp_group.itertuples():
                            strike = row.price_strike
                            option_type = row.cp_flag
                            pred_iv = self.sabr_model.predict_iv(strike, sabr_params)
                            if option_type == 'C':
                                call_predicted_ivs.append(pred_iv)
                                call_strikes.append(strike)
                                call_market_ivs.append(row.iv)
                            else:
                                put_predicted_ivs.append(pred_iv)
                                put_strikes.append(strike)
                                put_market_ivs.append(row.iv)
                        
                        calibration_results_calls.append({
                            'expiration_date': exp_date,
                            'strikes': call_strikes,
                            'market_ivs': call_market_ivs,
                            'predicted_ivs': call_predicted_ivs,
                            'time_to_maturity': time_to_maturity,
                            'sabr_params': sabr_params
                        })
                        calibration_results_puts.append({
                            'expiration_date': exp_date,
                            'strikes': put_strikes,
                            'market_ivs': put_market_ivs,
                            'predicted_ivs': put_predicted_ivs,
                            'time_to_maturity': time_to_maturity,
                            'sabr_params': sabr_params
                        })
                
                logger.debug(f"Calibrated SABR for {exp_date}: α={sabr_params.alpha:.4f}, "
                           f"ρ={sabr_params.rho:.4f}, ν={sabr_params.nu:.4f}")
        
        # Save plots if requested
        if save_plots:
            if plot_valids_only:
                self.save_calibration_plots(filter_method, underlying_price_tot_median, trade_date, calibration_results_calls)
            else:
                self.save_calibration_plots(filter_method, underlying_price_tot_median, trade_date, calibration_results_calls, calibration_results_puts)

        return sabr_params_by_maturity

    def save_calibration_plots(self, label, underlying_price_tot_median, trade_date, calibration_results_calls, calibration_results_puts= None):
        """
        Save calibration plots showing market vs SABR fitted IVs for each maturity
        Plots calls and puts with different colors on the same plot
        
        Parameters:
        trade_date: Trading date
        calibration_results_calls: List of calibration results for calls
        calibration_results_puts: List of calibration results for puts
        """
        if not calibration_results_calls and not calibration_results_puts:
            return
        
        # Create plots directory if it doesn't exist
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # Get unique expiration dates from both calls and puts
        exp_dates = set()
        if calibration_results_calls:
            exp_dates.update([result['expiration_date'] for result in calibration_results_calls])
        if calibration_results_puts:
            exp_dates.update([result['expiration_date'] for result in calibration_results_puts])
        
        exp_dates = sorted(list(exp_dates))
        n_maturities = len(exp_dates)
        
        if n_maturities == 0:
            return
        
        # Calculate subplot grid
        cols = min(4, n_maturities)
        rows = (n_maturities + cols - 1) // cols
        
        plt.figure(figsize=(5*cols, 4*rows))
        
        # Create dictionaries for quick lookup
        calls_dict = {result['expiration_date']: result for result in calibration_results_calls} if calibration_results_calls else {}
        puts_dict = {result['expiration_date']: result for result in calibration_results_puts} if calibration_results_puts else {}
    
        for i, exp_date in enumerate(exp_dates):
            ax1 = plt.subplot(rows, cols, i + 1)
            
            # Get results for this expiration date
            call_result = calls_dict.get(exp_date)
            put_result = puts_dict.get(exp_date)
            
            # Plot calls if available
            if call_result:
                call_strikes = call_result['strikes']
                call_market_ivs = call_result['market_ivs']
                call_predicted_ivs = call_result['predicted_ivs']
                
                # Plot call market IVs
                ax1.plot(call_strikes, call_market_ivs, 'o', color='blue', label='Call Market IV', 
                        markersize=4, alpha=0.7)
                
                # Plot call SABR fitted IVs
                ax1.plot(call_strikes, call_predicted_ivs, '-', color='blue', label='Call SABR IV', 
                        linewidth=2)
            
            # Plot puts if available
            if put_result:
                put_strikes = put_result['strikes']
                put_market_ivs = put_result['market_ivs']
                put_predicted_ivs = put_result['predicted_ivs']
                
                # Plot put market IVs
                ax1.plot(put_strikes, put_market_ivs, 's', color='red', label='Put Market IV', 
                        markersize=4, alpha=0.7)
                
                # Plot put SABR fitted IVs
                ax1.plot(put_strikes, put_predicted_ivs, '-', color='red', label='Put SABR IV', 
                        linewidth=2)
            
            # Create secondary y-axis for difference plot
            ax2 = ax1.twinx()
            
            # Plot predicted_iv - market_iv difference
            if call_result:
                call_strikes = call_result['strikes']
                call_diff = np.array(call_result['predicted_ivs']) - np.array(call_result['market_ivs'])
                ax2.plot(call_strikes, call_diff, '--', color='darkblue', label='Call (Pred-Mkt)', 
                        linewidth=1.5, alpha=0.8)            
            if put_result:
                put_strikes = put_result['strikes']
                put_diff = np.array(put_result['predicted_ivs']) - np.array(put_result['market_ivs'])
                ax2.plot(put_strikes, put_diff, '--', color='darkred', label='Put (Pred-Mkt)', 
                        linewidth=1.5, alpha=0.8)
            ax2.set_ylim(-0.08, 0.08)
            
            # Get parameters from either calls or puts (should be the same)
            result = call_result if call_result else put_result
            if result:
                tte = result['time_to_maturity']
                sabr_params = result['sabr_params']
                
                ax1.set_xlabel('Strike (K)')
                ax1.set_ylabel('Implied Volatility', color='black')
                ax2.set_ylabel('Predicted - Market IV', color='gray')
                ax1.set_title(f'{exp_date.strftime("%Y-%m-%d")} (T={tte:.3f})\n'
                             f'α={sabr_params.alpha:.3f}, ρ={sabr_params.rho:.3f}, ν={sabr_params.nu:.3f}')
                
                # Combine legends from both axes
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                
                ax1.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='green', linestyle=':', alpha=0.9)  # Add zero line for difference
                ax2.axhline(y=self.iv_threshold , color='grey', linestyle=':', alpha=0.7)  # Add zero line for difference
                ax2.axhline(y=-self.iv_threshold, color='grey', linestyle=':', alpha=0.7)  # Add zero line for difference
                ax2.axvline(x=underlying_price_tot_median, color='green', linestyle=':', alpha=0.7)
                ax2.axvline(x=underlying_price_tot_median*(1+self.max_moneyness_to_open), color='grey', linestyle=':', alpha=0.7)
                ax2.axvline(x=underlying_price_tot_median*(1-self.max_moneyness_to_open), color='grey', linestyle=':', alpha=0.7)
                # Calculate and display R-squared for combined data
                all_market_ivs = []
                all_predicted_ivs = []
                
                if call_result:
                    all_market_ivs.extend(call_result['market_ivs'])
                    all_predicted_ivs.extend(call_result['predicted_ivs'])
                
                if put_result:
                    all_market_ivs.extend(put_result['market_ivs'])
                    all_predicted_ivs.extend(put_result['predicted_ivs'])
                
                if len(all_market_ivs) > 1:
                    market_ivs_array = np.array(all_market_ivs)
                    predicted_ivs_array = np.array(all_predicted_ivs)
                    
                    ss_res = np.sum((market_ivs_array - predicted_ivs_array) ** 2)
                    ss_tot = np.sum((market_ivs_array - np.mean(market_ivs_array)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    ax1.text(0.05, 0.95, f'R²={r_squared:.3f}', 
                            transform=ax1.transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        if calibration_results_puts:
            plot_filename = f"{plots_dir}/sabr_calib_{label}_{trade_date.strftime('%Y%m%d')}.png"
        else:
            plot_filename = f"{plots_dir}/sabr_calib_{label}_callsOrValids_{trade_date.strftime('%Y%m%d')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved calibration plot: {plot_filename}")
    
    def save_calibration_results(self, calibration_data, filename):
        """Save calibration results to file"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(calibration_data, f)
            logger.info(f"Saved calibration results to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving calibration results: {str(e)}")
            return False
    
    def load_calibration_results(self, filename):
        """Load calibration results from file"""
        try:
            with open(filename, 'rb') as f:
                calibration_data = pickle.load(f)
            logger.info(f"Loaded calibration results from {filename}")
            return calibration_data
        except Exception as e:
            logger.error(f"Error loading calibration results: {str(e)}")
            return None
    
    def run_calibration_only(self, start_date=None, end_date=None, output_file=None):
        """
        Run SABR calibration for all trading dates and save results
        
        Parameters:
        start_date: Start date for calibration (datetime)
        end_date: End date for calibration (datetime)
        output_file: Output filename for calibration results
        """
        logger.info("Starting SABR calibration process...")
        
        # Get unique trading dates from all files
        trading_dates = self.get_all_trading_dates(start_date, end_date)
        
        logger.info(f"Calibrating SABR models from {trading_dates[0]} to {trading_dates[-1]}")
        logger.info(f"Total trading dates: {len(trading_dates)}")
        
        all_calibration_results = {}
        
        for i, trade_date in enumerate(trading_dates):
            logger.info(f"Calibrating for {trade_date.strftime('%Y-%m-%d')} ({i+1}/{len(trading_dates)})")
            
            # Calibrate SABR models for this date with plotting enabled
            sabr_params_by_maturity = self.calibrate_sabr_by_maturity(trade_date, save_plots=True)
            
            if sabr_params_by_maturity:
                all_calibration_results[trade_date] = sabr_params_by_maturity
                logger.info(f"Successfully calibrated {len(sabr_params_by_maturity)} maturities for {trade_date.strftime('%Y-%m-%d')}")
            else:
                logger.warning(f"No successful calibrations for {trade_date.strftime('%Y-%m-%d')}")
        
        # Save results
        if output_file is None:
            output_file = f"sabr_calibration_{trading_dates[0].strftime('%Y%m%d')}_{trading_dates[-1].strftime('%Y%m%d')}.pkl"
        
        if self.save_calibration_results(all_calibration_results, output_file):
            logger.info(f"Calibration completed. Results saved to {output_file}")
            logger.info(f"Total dates with successful calibrations: {len(all_calibration_results)}")
            
            # Print summary statistics
            total_maturities = sum(len(params) for params in all_calibration_results.values())
            logger.info(f"Total calibrated maturities across all dates: {total_maturities}")
            
            return output_file
        else:
            logger.error("Failed to save calibration results")
            return None
    
    def identify_arbitrage_opportunities(self, trade_date, sabr_params_by_maturity, calibrate_greeks= False):
        """
        Identify arbitrage opportunities by comparing predicted vs market IVs
        
        Returns:
        List of trading opportunities
        """
        # Ensure correct data is loaded for this date
        if not self.ensure_data_loaded_for_date(trade_date):
            logger.error(f"Could not load data for date: {trade_date}")
            return []
        
        opportunities = []
        day_data = self.options_data[self.options_data['trade_date'] == trade_date]
        
        for exp_date, sabr_params in sabr_params_by_maturity.items():
            exp_data = day_data[day_data['expiration_date'] == exp_date]
            
            for _, option in exp_data.iterrows():
                strike = option['price_strike']
                market_price = option['price_close']
                underlying_price = option['underlying_price']
                option_type = option['cp_flag']
                time_to_maturity = option['tte']
                option_id = option['option_symbol']
                
                if option_id not in self.positions or self.positions[option_id].quantity ==0:
                    if time_to_maturity < self.least_dte_to_open / 365.25:
                        continue
                    moneyness = abs(strike / underlying_price - 1)
                    if moneyness > self.max_moneyness_to_open:
                        continue

                # Calculate market IV
                risk_free_rate = self.get_risk_free_rate(trade_date)
                if not calibrate_greeks:
                    market_iv = option['iv']
                else:
                    market_iv = BlackScholes.implied_volatility(
                        market_price, underlying_price, strike, 
                        time_to_maturity, risk_free_rate, option_type
                    )
                
                if np.isnan(market_iv):
                    continue
                
                # Predict IV using SABR
                predicted_iv = self.sabr_model.predict_iv(strike, sabr_params)
                
                if np.isnan(predicted_iv):
                    continue
                
                # Calculate IV difference
                iv_diff = predicted_iv - market_iv
                
                # Check if difference exceeds threshold
                if abs(iv_diff) > self.iv_threshold:
                    # Calculate Greeks
                    if not calibrate_greeks:
                        delta = option['delta']
                        gamma = option['gamma']
                        theta = option['theta']
                        vega = option['vega']
                    else:
                        delta = BlackScholes.delta(
                            underlying_price, strike, time_to_maturity, 
                            risk_free_rate, market_iv, option_type
                        )
                        gamma = BlackScholes.gamma(
                            underlying_price, strike, time_to_maturity, 
                            risk_free_rate, market_iv
                        )
                        theta = BlackScholes.theta(
                            underlying_price, strike, time_to_maturity, 
                            risk_free_rate, market_iv, option_type
                        )
                        vega = BlackScholes.vega(
                            underlying_price, strike, time_to_maturity, 
                            risk_free_rate, market_iv
                        )
                    
                    opportunity = {
                        'option_id': option_id,
                        'trade_date': trade_date,
                        'expiration_date': exp_date,
                        'strike': strike,
                        'option_type': option_type,
                        'market_price': market_price,
                        'market_iv': market_iv,
                        'predicted_iv': predicted_iv,
                        'iv_diff': iv_diff,
                        'underlying_price': underlying_price,
                        'delta': delta,
                        'gamma': gamma,
                        'theta': theta,
                        'vega': vega,
                        'action': 'BUY' if iv_diff > 0 else 'SELL'  # Buy if underpriced, sell if overpriced
                    }
                    # print(f"Arbitrage opportunity: {opportunity}")
                    opportunities.append(opportunity)
        
        return opportunities
    
    def execute_trade(self, opportunity):
        """Execute a volatility arbitrage trade"""
        action = opportunity['action']
        position_size = min(self.max_position_size, 10)  # Conservative sizing
        
        if action == 'BUY':
            quantity = position_size
        else:
            quantity = -position_size
        
        # Calculate transaction cost
        notional = opportunity['market_price'] * abs(quantity) * 100  # Options are per 100 shares
        transaction_cost = notional * self.transaction_cost
        
        # Create position
        # option_id = f"{opportunity['expiration_date'].strftime('%Y%m%d')}_{opportunity['strike']}_{opportunity['option_type']}"
        
        option_id=opportunity['option_id']
        position = OptionPosition(
            symbol='SPX',
            expiration_date=opportunity['expiration_date'],
            strike=opportunity['strike'],
            option_type=opportunity['option_type'],
            quantity=quantity,
            entry_price=opportunity['market_price'],
            entry_date=opportunity['trade_date'],
            entry_iv=opportunity['market_iv'],
            predicted_iv=opportunity['predicted_iv'],
            delta=opportunity['delta'],
            gamma=opportunity['gamma'],
            theta=opportunity['theta'],
            vega=opportunity['vega'],
            underlying_price=opportunity['underlying_price'],
            cur_price=opportunity['market_price'],  # Initialize current price
            prev_price=opportunity['market_price'],  # Initialize previous price
            cur_date=opportunity['trade_date']  # Initialize current date
        )
        
        self.positions[option_id] = position
        
        # Update cash position (no individual delta hedging - will be handled by daily_delta_hedge)
        cash_flow = -quantity * opportunity['market_price'] * 100 - transaction_cost
        self.cash += cash_flow
        
        # Log trade (no individual hedging recorded)
        trade_record = {
            'date': opportunity['trade_date'],
            'action': action,
            'option_id': option_id,
            'quantity': quantity,
            'price': opportunity['market_price'],
            'iv_diff': opportunity['iv_diff'],
            'delta_hedge': 0,  # No individual hedging
            'cash_flow': cash_flow,
            'transaction_cost': transaction_cost
        }
        
        self.trade_log.append(trade_record)
        
        logger.info(f"Executed {action} {abs(quantity)} {option_id} @ {opportunity['market_price']:.2f}, "
                   f"IV diff: {opportunity['iv_diff']:.1%}, (delta: {opportunity['delta']:.3f}), underlyingP:{opportunity['underlying_price']}, ")
    
    def _save_position_hedge_details(self, position_details):
        """Save individual position hedge details to CSV"""
        if not position_details:
            return
            
        import csv
        import os
        
        # Create filename with date range
        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d") 
            filename = f'position_hedge_details_{start_str}_{end_str}.csv'
        else:
            filename = 'position_hedge_details.csv'
            
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [
                'date', 'option_id', 'quantity', 'strike', 'option_type', 
                'expiration_date', 'time_to_maturity', 'current_delta', 
                'previous_delta', 'delta_exposure', 'current_price', 
                'current_iv', 'underlying_price'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write all position details
            for detail in position_details:
                writer.writerow(detail)
    
    def _save_hedge_summary(self, summary_data):
        """Save hedge summary data to CSV"""
        import csv
        import os
        
        # Create filename with date range
        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d")
            filename = f'hedge_summary_{start_str}_{end_str}.csv'
        else:
            filename = 'hedge_summary.csv'
            
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [
                'date', 'action_type', 'total_delta_exposure', 'current_stock_position',
                'total_delta_adjustment', 'threshold_shares', 'hedge_executed',
                'underlying_price', 'num_positions', 'old_stock_position',
                'new_stock_position', 'stock_adjustment', 'cash_flow',
                'option_id', 'payoff', 'expiration_date', 'strike', 'option_type',
                'quantity', 'final_underlying'  # Additional fields for expired positions
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(summary_data)
    
    def _cleanup_existing_csv_files(self):
        """Remove existing CSV files to ensure fresh logs for new backtest"""
        import os
        
        # Generate filenames for cleanup
        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d")
            position_hedge_file = f'position_hedge_details_{start_str}_{end_str}.csv'
            hedge_summary_file = f'hedge_summary_{start_str}_{end_str}.csv'
            unified_position_file = f'unified_position_details_{start_str}_{end_str}.csv'
            # Legacy files to clean up
            position_details_file = f'position_details_{start_str}_{end_str}.csv'
        else:
            position_hedge_file = 'position_hedge_details.csv'
            hedge_summary_file = 'hedge_summary.csv'
            unified_position_file = 'unified_position_details.csv'
            # Legacy files to clean up
            position_details_file = 'position_details.csv'
        
        # Remove files if they exist
        files_to_remove = [position_hedge_file, hedge_summary_file, unified_position_file, position_details_file]
        
        for filename in files_to_remove:
            if os.path.exists(filename):
                try:
                    os.remove(filename)
                    logger.info(f"Removed existing CSV file: {filename}")
                except Exception as e:
                    logger.warning(f"Could not remove file {filename}: {str(e)}")
    
    def _save_position_details(self, position_details):
        """Save unified daily position details to CSV (combines position and hedging info)"""
        if not position_details:
            return
        
        import csv
        import os
        
        # Create filename with date range
        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d")
            filename = f'unified_position_details_{start_str}_{end_str}.csv'
        else:
            filename = 'unified_position_details.csv'
            
        file_exists = os.path.exists(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = [
                # Core position identification
                'date', 'option_id', 'symbol', 'expiration_date', 'strike', 'option_type', 'quantity',
                # Entry information
                'entry_price', 'entry_date', 'entry_iv', 'predicted_iv', 'entry_value',
                # Current pricing and P&L
                'prev_price', 'cur_price', 'price_change', 'current_value', 'position_pnl',
                # Time tracking
                'days_held', 'days_to_expiry', 'time_to_maturity',
                # Greeks and hedging
                'current_delta', 'previous_delta', 'delta_change', 'delta_exposure',
                'current_gamma', 'current_theta', 'current_vega', 'current_iv',
                # Market context
                'underlying_price', 'underlying_change'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write all position details
            for detail in position_details:
                writer.writerow(detail)
    
    def daily_delta_hedge(self, current_date, calculate_greeks=False):
        """Perform daily delta hedging for all positions"""
        # Ensure correct data is loaded for this date
        if not self.ensure_data_loaded_for_date(current_date):
            logger.warning(f"Could not load data for delta hedging on date: {current_date}")
            return
        
        # Get current market data
        current_data = self.options_data[self.options_data['trade_date'] == current_date]
        
        if len(current_data) == 0:
            return
        
        # Calculate total current delta exposure across all positions
        total_current_delta_exposure = 0
        positions_updated = []
        position_details = []  # For CSV logging
        
        for position in self.positions.values():
            if position.quantity == 0:
                continue
                
            # Find current option data
            option_data = current_data[
                (current_data['expiration_date'] == position.expiration_date) &
                (current_data['price_strike'] == position.strike) &
                (current_data['cp_flag'] == position.option_type)
            ]
            
            if len(option_data) == 0:
                continue
            
            current_option = option_data.iloc[0]
            current_underlying = current_option['underlying_price']
            current_time_to_maturity = current_option['tte']
            
            if current_time_to_maturity <= 0:
                continue
            
            # Calculate current delta
            current_market_price = current_option['price_close']
            risk_free_rate = self.get_risk_free_rate(current_date)
            if not calculate_greeks:
                current_iv = current_option['iv']
            else:
                current_iv = BlackScholes.implied_volatility(
                    current_market_price, current_underlying, position.strike,
                    current_time_to_maturity, risk_free_rate, position.option_type
                )
            
            if np.isnan(current_iv):
                continue
            
            if not calculate_greeks:
                current_delta = current_option['delta']
            else:
                current_delta = BlackScholes.delta(
                    current_underlying, position.strike, current_time_to_maturity,
                    risk_free_rate, current_iv, position.option_type
                )
            
            # Calculate delta exposure for this position
            delta_exposure = position.quantity * current_delta * 100
            total_current_delta_exposure += delta_exposure
            
            # Store updated delta for this position
            positions_updated.append((position, current_delta))
            
            # Log position details for CSV
            option_id = f"{position.expiration_date.strftime('%Y%m%d')}_{position.strike}_{position.option_type}"
            position_details.append({
                'date': current_date,
                'option_id': option_id,
                'quantity': position.quantity,
                'strike': position.strike,
                'option_type': position.option_type,
                'expiration_date': position.expiration_date,
                'time_to_maturity': current_time_to_maturity,
                'current_delta': current_delta,
                'previous_delta': position.delta,
                'delta_exposure': delta_exposure,
                'current_price': current_market_price,
                'current_iv': current_iv,
                'underlying_price': current_underlying
            })
        
        # Save position details to CSV
        self._save_position_hedge_details(position_details)
        
        # Calculate total delta adjustment needed
        # Current stock position should neutralize the delta exposure
        total_delta_adjustment = total_current_delta_exposure + self.stock_position
        
        # Prepare summary data for CSV logging
        hedge_executed = abs(total_delta_adjustment) > self.delta_hedge_threshold * 100
        current_underlying_price = current_data['underlying_price'].median()
        
        # Log summary data
        summary_data = {
            'date': current_date,
            'total_delta_exposure': total_current_delta_exposure,
            'current_stock_position': self.stock_position,
            'total_delta_adjustment': total_delta_adjustment,
            'threshold_shares': self.delta_hedge_threshold * 100,
            'hedge_executed': hedge_executed,
            'underlying_price': current_underlying_price,
            'num_positions': len(position_details),
            'action_type': 'delta_hedge'
        }
        
        # Execute delta hedge if adjustment exceeds threshold
        if hedge_executed:
            # Adjust stock position to neutralize delta
            old_stock_position = self.stock_position
            self.stock_position -= total_delta_adjustment
            cash_flow = total_delta_adjustment * current_underlying_price
            self.cash += cash_flow
            
            # Update stored deltas for all positions
            for position, current_delta in positions_updated:
                position.delta = current_delta
            
            # Update summary data with execution details
            summary_data.update({
                'old_stock_position': old_stock_position,
                'new_stock_position': self.stock_position,
                'stock_adjustment': -total_delta_adjustment,
                'cash_flow': cash_flow
            })
            
            logger.debug(f"Delta hedge: {total_delta_adjustment:.0f} shares @ {current_underlying_price:.2f}, "
                        f"Total delta exposure: {total_current_delta_exposure:.0f}, "
                        f"New stock position: {self.stock_position:.0f}")
        else:
            # No hedge executed
            summary_data.update({
                'old_stock_position': self.stock_position,
                'new_stock_position': self.stock_position,
                'stock_adjustment': 0,
                'cash_flow': 0
            })
        
        # Save summary to CSV
        self._save_hedge_summary(summary_data)
    
    def close_expired_positions(self, current_date):
        """Close positions that have expired"""
        expired_option_ids = []
        
        # Find expired positions
        for option_id, position in self.positions.items():
            if position.expiration_date <= current_date:
                expired_option_ids.append(option_id)
        
        # Remove expired positions
        for option_id in expired_option_ids:
            position = self.positions.pop(option_id)
            
            # Calculate final payoff
            payoff = 0
            cash_flow = 0
            final_underlying = 0
            
            # Ensure data is loaded for expiration calculation
            if self.ensure_data_loaded_for_date(current_date):
                current_data = self.options_data[
                    (self.options_data['trade_date'] <= current_date) &
                    (self.options_data['expiration_date'] == position.expiration_date)
                ]
                
                if len(current_data) > 0:
                    final_underlying = current_data['underlying_price'].iloc[-1]
                    
                    if position.option_type == 'C':
                        payoff = max(0, final_underlying - position.strike)
                    else:
                        payoff = max(0, position.strike - final_underlying)
                    
                    cash_flow = position.quantity * payoff * 100
                    self.cash += cash_flow
                    
                    logger.info(f"Expired: {option_id}, Payoff: {payoff:.2f}, "
                               f"Cash flow: {cash_flow:.2f}")
            
            # Log expired position to CSV
            expiry_data = {
                'date': current_date,
                'action_type': 'position_expiry',
                'option_id': option_id,
                'expiration_date': position.expiration_date,
                'payoff': payoff,
                'cash_flow': cash_flow,
                'quantity': position.quantity,
                'strike': position.strike,
                'option_type': position.option_type,
                'final_underlying': final_underlying,
                'total_delta_exposure': 0,  # Not applicable for expiry
                'current_stock_position': self.stock_position,
                'total_delta_adjustment': 0,  # Not applicable for expiry
                'threshold_shares': 0,  # Not applicable for expiry
                'hedge_executed': False,  # Not applicable for expiry
                'underlying_price': final_underlying,
                'num_positions': 1,
                'old_stock_position': self.stock_position,
                'new_stock_position': self.stock_position,
                'stock_adjustment': 0
            }
            
            self._save_hedge_summary(expiry_data)
    
    def calculate_portfolio_value(self, current_date):
        """Calculate current portfolio value and update position prices"""
        if not self.positions:
            return self.cash
        summary=""
        
        # Ensure correct data is loaded for this date
        if not self.ensure_data_loaded_for_date(current_date):
            logger.warning(f"Could not load data for portfolio valuation on date: {current_date}")
            return self.cash
        
        # Get current market data
        current_data = self.options_data[self.options_data['trade_date'] == current_date]
        
        if len(current_data) == 0:
            return self.cash
        
        portfolio_value = self.cash
        current_underlying_price = current_data['underlying_price'].median()
        summary+=f"Cash:{self.cash}\n"
        summary+=f"Stock:{self.stock_position}*{current_underlying_price}={self.stock_position * current_underlying_price}\n"
        
        # Add stock position value
        portfolio_value += self.stock_position * current_underlying_price
        
        # Collect position details for CSV logging
        position_details = []
        
        # Add option positions value and update position prices
        sorted_positions = sorted([(oid, pos) for oid, pos in self.positions.items() if pos.quantity != 0], key=lambda x: x[0])
        for option_id, position in sorted_positions:
            option_data = current_data[
                (current_data['expiration_date'] == position.expiration_date) &
                (current_data['price_strike'] == position.strike) &
                (current_data['cp_flag'] == position.option_type)
            ]
            
            if len(option_data) > 0:
                current_price = option_data.iloc[0]['price_close']
                position_value = position.quantity * current_price * 100
                summary+=f"{option_id}:{position.quantity}*{current_price}*100={position_value}\n"
                portfolio_value += position_value
                
                # Update position prices and date
                position.prev_price = position.cur_price  # Save previous price
                position.cur_price = current_price        # Update current price
                position.cur_date = current_date          # Update current date
                
                # Calculate P&L for this position
                entry_value = position.quantity * position.entry_price * 100
                current_value = position_value
                position_pnl = current_value - entry_value
                
                # Get current Greeks and IV from market data
                current_option = option_data.iloc[0]
                current_iv = current_option.get('iv', 0)
                current_delta = current_option.get('delta', 0)
                current_gamma = current_option.get('gamma', 0)
                current_theta = current_option.get('theta', 0)
                current_vega = current_option.get('vega', 0)
                current_time_to_maturity = current_option.get('tte', 0)
                
                # Calculate delta exposure and changes
                previous_delta = position.delta
                delta_change = current_delta - previous_delta
                delta_exposure = position.quantity * current_delta * 100
                
                # Calculate underlying price change (if we have previous data)
                # For now, we'll use 0 as placeholder - this could be enhanced with historical tracking
                underlying_change = 0
                
                # Collect unified position details for CSV
                position_details.append({
                    # Core position identification
                    'date': current_date,
                    'option_id': option_id,
                    'symbol': position.symbol,
                    'expiration_date': position.expiration_date,
                    'strike': position.strike,
                    'option_type': position.option_type,
                    'quantity': position.quantity,
                    # Entry information
                    'entry_price': position.entry_price,
                    'entry_date': position.entry_date,
                    'entry_iv': position.entry_iv,
                    'predicted_iv': position.predicted_iv,
                    'entry_value': entry_value,
                    # Current pricing and P&L
                    'prev_price': position.prev_price,
                    'cur_price': position.cur_price,
                    'price_change': position.cur_price - position.prev_price,
                    'current_value': current_value,
                    'position_pnl': position_pnl,
                    # Time tracking
                    'days_held': (current_date - position.entry_date).days,
                    'days_to_expiry': (position.expiration_date - current_date).days,
                    'time_to_maturity': current_time_to_maturity,
                    # Greeks and hedging
                    'current_delta': current_delta,
                    'previous_delta': previous_delta,
                    'delta_change': delta_change,
                    'delta_exposure': delta_exposure,
                    'current_gamma': current_gamma,
                    'current_theta': current_theta,
                    'current_vega': current_vega,
                    'current_iv': current_iv,
                    # Market context
                    'underlying_price': current_underlying_price,
                    'underlying_change': underlying_change
                })
                
                # Update position's stored delta for next day's calculation
                position.delta = current_delta
        
        # Save position details to CSV
        self._save_position_details(position_details)
        
        print("Portfolio Summary:\n"+summary)
        return portfolio_value
    
    def get_position_breakdown(self):
        """Get breakdown of positions by type"""
        breakdown = {
            'buy_calls': 0,
            'sell_calls': 0,
            'buy_puts': 0,
            'sell_puts': 0
        }
        
        for position in self.positions.values():
            if position.quantity == 0:
                continue
                
            if position.option_type == 'C':  # Calls
                if position.quantity > 0:
                    breakdown['buy_calls'] += abs(position.quantity)
                else:
                    breakdown['sell_calls'] += abs(position.quantity)
            else:  # Puts
                if position.quantity > 0:
                    breakdown['buy_puts'] += abs(position.quantity)
                else:
                    breakdown['sell_puts'] += abs(position.quantity)
        
        return breakdown
    
    def run_backtest(self, start_date=None, end_date=None, calibration_file=None, save_plots=False, live_plotting=False):
        """
        Run the volatility arbitrage backtest
        
        Parameters:
        start_date: Start date for backtest (datetime)
        end_date: End date for backtest (datetime)
        calibration_file: Pre-computed calibration results file (optional)
        save_plots: Whether to save calibration plots during backtest
        live_plotting: Whether to show live plotting during backtest
        """
        logger.info("Starting volatility arbitrage backtest...")
        
        # Initialize live plotter if requested
        if live_plotting:
            self.live_plotter = LivePlotter("Volatility Arbitrage Backtest - Live Results")
            logger.info("Live plotting enabled - plots will update in real-time")
        
        # Load pre-computed calibration results if available
        precomputed_calibration = None
        if calibration_file and os.path.exists(calibration_file):
            precomputed_calibration = self.load_calibration_results(calibration_file)
            if precomputed_calibration:
                logger.info(f"Using pre-computed calibration results from {calibration_file}")
        
        # Get unique trading dates from all files
        trading_dates = self.get_all_trading_dates(start_date, end_date)
        
        logger.info(f"Backtesting from {trading_dates[0]} to {trading_dates[-1]}")
        
        # Clean up existing CSV files to ensure fresh logs
        self._cleanup_existing_csv_files()
        
        initial_capital = 1000000  # $1M initial capital
        self.cash = initial_capital
        
        for i, trade_date in enumerate(trading_dates):
            logger.info(f"Processing {trade_date.strftime('%Y-%m-%d')} ({i+1}/{len(trading_dates)})")
            
            # Close expired positions
            self.close_expired_positions(trade_date)
            
            # Get SABR calibration results
            if precomputed_calibration and trade_date in precomputed_calibration:
                # Use pre-computed calibration
                sabr_params_by_maturity = precomputed_calibration[trade_date]
                logger.debug(f"Using pre-computed calibration for {trade_date.strftime('%Y-%m-%d')}")
            else:
                # Calibrate SABR models on-the-fly
                sabr_params_by_maturity = self.calibrate_sabr_by_maturity(trade_date, save_plots=save_plots)
                if not precomputed_calibration:
                    logger.debug(f"Computed SABR calibration on-the-fly for {trade_date.strftime('%Y-%m-%d')}")
                else:
                    logger.warning(f"No pre-computed calibration found for {trade_date.strftime('%Y-%m-%d')}, computing on-the-fly")
                
                # Store calibration results for later saving
                if sabr_params_by_maturity:
                    if not hasattr(self, 'backtest_calibration_results'):
                        self.backtest_calibration_results = {}
                    self.backtest_calibration_results[trade_date] = sabr_params_by_maturity
            
            current_underlying_price = self.options_data[self.options_data['trade_date'] == trade_date]['underlying_price'].median()
            if not sabr_params_by_maturity:
                logger.debug(f"No SABR calibration successful for {trade_date}")
                # Still update live plot with current portfolio value
                if live_plotting and self.live_plotter:
                    portfolio_value = self.calculate_portfolio_value(trade_date)
                    pnl = portfolio_value - initial_capital
                    num_positions = len([pos for pos in self.positions.values() if pos.quantity != 0])
                    stock_value = self.stock_position * current_underlying_price
                    position_breakdown = self.get_position_breakdown()
                    
                    self.live_plotter.update_data(trade_date, portfolio_value, pnl, num_positions, self.cash, stock_value, position_breakdown, current_underlying_price, self.stock_position)
                    self.live_plotter.update_plots()
                    
                    # Add small delay to allow plot updates
                    time.sleep(0.1)
                continue
            
            # Identify arbitrage opportunities
            opportunities = self.identify_arbitrage_opportunities(trade_date, sabr_params_by_maturity)
            
            # Execute top opportunities (limit to avoid over-trading)
            opportunities.sort(key=lambda x: abs(x['iv_diff']), reverse=True)
            for opportunity in opportunities[:5]:  # Top 5 opportunities per day
                self.execute_trade(opportunity)
            
            # Daily delta hedging AFTER executing trades
            # Always run delta hedging to ensure proper portfolio balance
            # (even when positions net to zero, stock position should be adjusted)
            self.daily_delta_hedge(trade_date)
            
            # Calculate and record portfolio value
            portfolio_value = self.calculate_portfolio_value(trade_date)
            pnl = portfolio_value - initial_capital
            num_positions = len([pos for pos in self.positions.values() if pos.quantity != 0])
            
            self.pnl_history.append({
                'date': trade_date,
                'portfolio_value': portfolio_value,
                'pnl': pnl,
                'cash': self.cash,
                'stock_position': self.stock_position,
                'num_positions': num_positions,
                'underlying_price': current_underlying_price,
            })
            
            # Update live plot if enabled
            if live_plotting and self.live_plotter:
                stock_value = self.stock_position * current_underlying_price
                position_breakdown = self.get_position_breakdown()
                print(f"============= {trade_date.strftime('%Y-%m-%d')}, current_underlying_price: {current_underlying_price} =============")
                
                self.live_plotter.update_data(trade_date, portfolio_value, pnl, num_positions, self.cash, stock_value, position_breakdown, current_underlying_price, self.stock_position)
                self.live_plotter.update_plots()
                
                # Add small delay to allow plot updates
                time.sleep(0.1)
            
            if i % 1 == 0:  # Log progress every day
                logger.info(f"Portfolio value: ${portfolio_value:,.0f}, "
                           f"P&L: ${pnl:,.0f}, Positions: {num_positions}")
        
        logger.info("Backtest completed!")
        
        # Save final live plot if enabled
        if live_plotting and self.live_plotter:
            self.live_plotter.save_final_plot('live_volatility_arbitrage_backtest.png')
            logger.info("Live plot saved. Close the plot window to continue...")
            # Keep the plot open for user to see final results
            input("Press Enter after viewing the live plot to continue...")
            self.live_plotter.close()
        
        # Save calibration results if any were computed during backtest
        if hasattr(self, 'backtest_calibration_results') and self.backtest_calibration_results:
            # Generate filename for backtest calibration results
            backtest_calibration_file = f"backtest_calibration_{trading_dates[0].strftime('%Y%m%d')}_{trading_dates[-1].strftime('%Y%m%d')}.pkl"
            if self.save_calibration_results(self.backtest_calibration_results, backtest_calibration_file):
                logger.info(f"Saved backtest calibration results to {backtest_calibration_file}")
                logger.info(f"Total dates with calibrations during backtest: {len(self.backtest_calibration_results)}")
            else:
                logger.error("Failed to save backtest calibration results")
    
    def generate_performance_report(self):
        """Generate performance analysis report"""
        if not self.pnl_history:
            logger.error("No P&L history available")
            return
        
        pnl_df = pd.DataFrame(self.pnl_history)
        pnl_df['returns'] = pnl_df['pnl'].pct_change()
        
        # Performance metrics
        total_return = pnl_df['pnl'].iloc[-1] / 1000000  # Assuming $1M initial
        annualized_return = (1 + total_return) ** (365.25 / len(pnl_df)) - 1
        volatility = pnl_df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        max_drawdown = (pnl_df['pnl'] / pnl_df['pnl'].cummax() - 1).min()
        
        print("\n" + "="*50)
        print("VOLATILITY ARBITRAGE BACKTEST RESULTS")
        print("="*50)
        print(f"Initial Capital:      ${1000000:,.0f}")
        print(f"Final Portfolio Value: ${pnl_df['portfolio_value'].iloc[-1]:,.0f}")
        print(f"Total P&L:            ${pnl_df['pnl'].iloc[-1]:,.0f}")
        print(f"Total Return:         {total_return:.2%}")
        print(f"Annualized Return:    {annualized_return:.2%}")
        print(f"Volatility:           {volatility:.2%}")
        print(f"Sharpe Ratio:         {sharpe_ratio:.2f}")
        print(f"Max Drawdown:         {max_drawdown:.2%}")
        print(f"Total Trades:         {len(self.trade_log)}")
        print("="*50)
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(pnl_df['date'], pnl_df['portfolio_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # P&L over time
        axes[0, 1].plot(pnl_df['date'], pnl_df['pnl'])
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_ylabel('P&L ($)')
        axes[0, 1].grid(True)
        
        # Number of positions over time
        axes[1, 0].plot(pnl_df['date'], pnl_df['num_positions'])
        axes[1, 0].set_title('Number of Open Positions')
        axes[1, 0].set_ylabel('Positions')
        axes[1, 0].grid(True)
        
        # Returns distribution
        axes[1, 1].hist(pnl_df['returns'].dropna(), bins=50, alpha=0.7)
        axes[1, 1].set_title('Daily Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('volatility_arbitrage_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return pnl_df

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Volatility Arbitrage Backtesting System')
    parser.add_argument('mode', choices=['calibrate', 'backtest'], 
                       help='Operation mode: calibrate (run SABR calibration only) or backtest (run full backtest)')
    parser.add_argument('--data-file', default='~/data/quant/volarb/SPX_options_2019-04-11_2019-04-30.parquet',
                       help='Path to options data parquet file')
    parser.add_argument('--start-date', default='2019-04-11',
                       help='Start date for analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2019-04-30',
                       help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--calibration-file', 
                       help='Path to calibration results file (for backtest mode)')
    parser.add_argument('--output-file',
                       help='Output filename for calibration results (calibrate mode only)')
    parser.add_argument('--iv-threshold', type=float, default=0.02,
                       help='IV difference threshold for trades (default: 0.02)')
    parser.add_argument('--risk-free-rate', type=float, default=0.025,
                       help='Risk-free rate (default: 0.025)')
    parser.add_argument('--transaction-cost', type=float, default=0.005,
                       help='Transaction cost as percentage (default: 0.005)')
    parser.add_argument('--max-position-size', type=int, default=50,
                       help='Maximum position size per trade (default: 50)')
    parser.add_argument('--delta-hedge-threshold', type=float, default=0.1,
                       help='Delta threshold for rehedging (default: 0.1)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO',
                       help='Logging level (default: INFO)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save calibration plots (default: False, always True in calibrate mode)')
    parser.add_argument('--live-plotting', action='store_true',
                       help='Enable live plotting during backtest (default: False)')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse dates
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    
    # Expand file paths
    data_file = os.path.expanduser(args.data_file)
    
    # Configuration
    config = {
        'data_file': data_file,
        'iv_threshold': args.iv_threshold,
        'risk_free_rate': args.risk_free_rate,
        'transaction_cost': args.transaction_cost,
        'max_position_size': args.max_position_size,
        'delta_hedge_threshold': args.delta_hedge_threshold
    }
    
    # Initialize strategy
    strategy = VolatilityArbitrageStrategy(
        iv_threshold=config['iv_threshold'],
        risk_free_rate=config['risk_free_rate'],
        transaction_cost=config['transaction_cost'],
        max_position_size=config['max_position_size'],
        delta_hedge_threshold=config['delta_hedge_threshold'],
        start_date=start_date,
        end_date=end_date
    )
    
    # Load data
    if not strategy.load_data(config['data_file']):
        logger.error("Failed to load data")
        return
    
    if args.mode == 'calibrate':
        # Run calibration only
        logger.info("Running SABR calibration mode...")
        output_file = strategy.run_calibration_only(
            start_date=start_date, 
            end_date=end_date,
            output_file=args.output_file
        )
        if output_file:
            print(f"\nCalibration completed successfully!")
            print(f"Results saved to: {output_file}")
            print(f"Use this file with --calibration-file for backtesting")
        else:
            print("Calibration failed!")
            
    elif args.mode == 'backtest':
        # Run backtest
        logger.info("Running backtest mode...")
        
        # Check if calibration file exists
        calibration_file = None
        if args.calibration_file:
            calibration_file = os.path.expanduser(args.calibration_file)
            if not os.path.exists(calibration_file):
                logger.warning(f"Calibration file {calibration_file} not found. Will compute calibration on-the-fly.")
                calibration_file = None
        
        # Run backtest
        strategy.run_backtest(
            start_date=start_date, 
            end_date=end_date,
            calibration_file=calibration_file,
            save_plots=args.save_plots,
            live_plotting=args.live_plotting
        )
        
        # Generate performance report
        results_df = strategy.generate_performance_report()
        
        # Save calibration results if any were computed during backtest
        if hasattr(strategy, 'backtest_calibration_results') and strategy.backtest_calibration_results:
            print(f"Calibration results from backtest saved automatically.")
            print(f"Total dates with calibrations: {len(strategy.backtest_calibration_results)}")
        
        # Save detailed results
        if results_df is not None:
            results_file = f'volatility_arbitrage_pnl_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
            results_df.to_csv(results_file, index=False)
            print(f"\nDetailed P&L results saved to: {results_file}")
            
        # Save trade log
        if strategy.trade_log:
            trade_df = pd.DataFrame(strategy.trade_log)
            trades_file = f'volatility_arbitrage_trades_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
            trade_df.to_csv(trades_file, index=False)
            print(f"Trade log saved to: {trades_file}")
            print(f"Total trades executed: {len(strategy.trade_log)}")

if __name__ == "__main__":
    main()
