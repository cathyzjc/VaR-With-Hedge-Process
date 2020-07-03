# VaR/ES With Hedge Process


### 1. VaR
Risk measurement is the basis of risk management. The VaR approach is attractive because it is easy to understand (VaR is measured in monetary units) and it provides an estimate of the amount of capital that is needed to support a certain level of risk. Another advantage of this measure is the ability to incorporate the effects of portfolio diversification. 

### 2. Introduction
In this project, we estimate Value-at-risk (VaR) and Expected Shortfall (ES) of a portfolio between normal period and coronavirus period that contains several assets which include US stock, US index, with the US stocks hedged with stock put option. The assets we used are Amazon stock price (AMZN), Boeing stock price (BA) and SP500. Our data starts at 1997-05-15, which is also the date when amazon's stock price starts. We compare the results in the portfolio with fully-hedged, partially hedged or without the hedging. This topic is inspired by the current issue of COVID-19 and by our discussions in class about options and partial risk hedges. 
 
Since coronavirus has just occurred, we set our normal time period as 2020-01-02 to 2020-02-21 and extreme time period as 2020-02-22 to 2020-4-13. We used a rolling window with a 50 days window size, and estimated the VaR and ES (p=1%) for no option hedged, single stock option hedged, double stock option hedged, and fully stocks option hedged in normal period and extreme period. 

The rest of the report is structured as follows. In Section 2, we conduct a literature review on risk management using options and current VaR approaches. Section 3 introduces the methodology of our project. Section 4 includes an overview of the data and presents the results of the full data sample. Section 5 is the reflection of the entire Hedge process. Section 6 explains our final results. Finally, we made some conclusions and suggestion in Section 7. 
