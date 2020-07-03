VaR/ES With Hedge Process
=============

Why VaR
----------------
Risk measurement is the basis of risk management. The VaR approach is attractive because it is easy to understand (VaR is measured in monetary units) and it provides an estimate of the amount of capital that is needed to support a certain level of risk. Another advantage of this measure is the ability to incorporate the effects of portfolio diversification. 

Introduction
-----------------------
In this project, we estimate **Value-at-risk (VaR) and Expected Shortfall (ES)** of a portfolio between normal period and coronavirus period that contains several assets which include US stock, US index, with the US stocks hedged with stock put option. 

The assets we used are *Amazon stock price (AMZN), Boeing stock price (BA) and SP500*. Our data starts at 1997-05-15, which is also the date when amazon's stock price starts. We compare the results in the portfolio with fully-hedged, partially hedged or without the hedging. This topic is inspired by the current issue of COVID-19 and by our discussions in class about options and partial risk hedges. 

Conclusion
------------------------

|| VaR (Normal)| ES(Normal)| VaR (Extreme)|ES(Extreme)|
|---|---|---|---|---|
| No-hedged | 19880.08 |28112.97| 29403.74| 38335.94| 
|Amazon |4913.82 |6743.60| 9724.11 |12829.04 |
|S&P500| 22310.73| 29720.14| 25902.09 |33242.23|
|Boeing| 19440.30| 26003.40 |26093.76 |34635.90 |
|Amazon + S&P500| 4372.97| 6821.92| 6471.46 |8750.80|
|Amazon + Boeing| 0.00| 0.00 |3296.04| 4432.60|
|Fully| 0.00| 0.00| 0.00| 0.00| 

VaR/ES estimation of all portfolios without option prices in both periods 



In this project, we explored the effect of a varieties of option hedging strategies in normal period and coronavirus period separately with 50 days moving average bootstrapping simulation. Though counterintuitive, the VaR and ES of fully hedged and partially hedged portfolio tend to be higher than no hedged scenario, which could be explained by the high initial cost of in-the-money put option. 

Besides, since we are hedging with European option that cannot be executed before maturity, the time value of option fades gradually with the proximity of maturity while the inner value of option may rise first due to coronavirus shock but then falls because the US government and Federal Reserve have launched a series of financial stimulus packages to save the economy and stock market from slipping into recession.  
 
