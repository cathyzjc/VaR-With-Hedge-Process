#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 17:49:55 2020

@author: Group 7
"""
# load packages
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optm

# In[]
# load data and create data frame
sp = pd.read_csv("SP500.csv")
sp = sp[['Date','Close']]
sp.index = pd.to_datetime(sp.Date,format="%m/%d/%Y")
del sp['Date']

amzn = pd.read_csv("AMZN.csv")
amzn = amzn[['Date','Close']]
amzn.index = pd.to_datetime(amzn.Date,format="%m/%d/%Y")
del amzn['Date']

##########################################
################### Bond here is Boeing stock price
bond = pd.read_csv('BOND.csv')
bond = bond[['Date','Close']]
bond.index = pd.to_datetime(bond.Date,format="%m/%d/%Y")
del bond['Date']

# In[]
#aa = pd.read_csv('AAL.csv')
#aa = aa[['Date','Close']]
#aa.index = pd.to_datetime(aa.Date,format="%m/%d/%Y")
#del aa['Date']

# check for amazon's volatility
amznnew = amzn.copy()
amznnew['lagclose'] = amznnew.Close.shift(1)
amznnew['ret']=np.log(amznnew['Close'])-np.log(amznnew['lagclose'])
amznnew = amznnew[1:]

amznnewretVec = amznnew['ret'].values
amznnewmean = np.mean(amznnewretVec)
amznnewstd = np.std(amznnewretVec)

amznnew['retv']=(amznnew['ret']-amznnewmean)**2.
amznnewrollwindow = amznnew.rolling(window=50,win_type="boxcar")
amznnewrollmean = amznnewrollwindow.mean()
amznnewrollmean['retsd'] = np.sqrt(amznnewrollmean['retv'])

plt.plot(amznnewrollmean['retsd'])
plt.grid()
plt.xlabel("Year")
plt.ylabel('Volatility')

# select data range from 2010-01-01 to 2019-12-22
spa = sp.copy()
spa = spa[(spa.index >= '1997-05-15') & (spa.index <= '2019-12-31')]

amzna = amzn.copy()
amzna = amzna[(amzna.index >= '1997-05-15') & (amzna.index <= '2019-12-31')]

bonda = bond.copy()
bonda = bonda[(bonda.index >= '1997-05-15') & (bonda.index <= '2019-12-31')]

#aaa = aa.copy()
#aaa = aaa[(aaa.index >= '2005-01-01') & (aaa.index <= '2019-12-31')]

data = pd.DataFrame({"sp":spa.Close,
                     "amzn":amzna.Close,
                     "bond":bonda.Close})
data = data.dropna()
data.head()

ret = np.log(data.pct_change()+1)
ret = ret.dropna()


# In[]
# test normality of returns
def normality_test(a):
    print('Norm test p-value %14.3f' % stats.normaltest(a)[1])

for i in ["sp","amzn","bond"]:
    print('\nResults for {}'.format(i))
    print('-'*32)
    log_data = np.array(ret[i])
    normality_test(log_data)

# In[]
# calculate weights
pret = []
pvar = []
prs = []
prt = []
for p in range(1000):
    weights = np.random.random(3)
    weights /=np.sum(weights)
    return1=np.sum(ret.mean()*250*weights)
    std1=np.sqrt(np.dot(weights.T, np.dot(ret.cov()*250, weights)))
    Rs=stats.norm.ppf(0.01,loc=return1,scale=std1)
    Rt = -std1*stats.norm.pdf((Rs-return1)/std1)/0.01+return1
    pret.append(return1)
    pvar.append(std1)
    prs.append(Rs)
    prt.append(Rt)
    
pret = np.array(pret)
pvar = np.array(pvar)
prs = np.array(prs)
prt = np.array(prt)
# plot random 1000 portfolio set: return against volatitliy
rf = 0.03
plt.figure(figsize=(8, 6))
plt.scatter(pvar, pret, c=(pret-rf)/pvar, marker = 'o')
plt.grid(True)
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.colorbar(label = 'Sharpe Ratio')


# In[]
# Weights calculation
def statsRecord(weights):
    weights = np.array(weights)
    pret = np.sum(ret.mean()*weights)*250
    pvar = np.sqrt(np.dot(weights.T, np.dot(ret.cov()*250,weights)))
    sharpe = (pret-rf)/pvar
    return  sharpe

# minimum sharpe
def statsRecord1(weights):
    return -statsRecord(weights)

# Initialize
x0 = 3*[1./3] 

#权重（某股票持仓⽐例）限制在0和1之间。 
bnds1 = tuple((0,1) for x in range(3))
cons = ({'type':'eq', 'fun':lambda x: np.sum(x)-1})

optv = optm.minimize(max_sharpe,
                     x0,
                     method = 'SLSQP', 
                     bounds = bnds1, 
                     constraints = cons)
w = optv['x']
# In[]
# now start for Normal period: 2020-01-02 to 2020-02-21
# get rolling return

spb = sp.copy()
spb = spb[(spb.index >= '1997-05-15') & (spb.index <= '2020-02-21')]

amznb = amzn.copy()
amznb = amznb[(amznb.index >= '1997-05-15') & (amznb.index <= '2020-02-21')]

#googb = goog.copy()
#googb = googb[(googb.index >= '2005-01-01') & (googb.index <= '2020-02-21')]

bondb = bond.copy()
bondb = bondb[(bondb.index >= '1997-05-15') & (bondb.index <= '2020-02-21')]

#aab = aa.copy()
#aab = aab[(aab.index >= '2010-01-01') & (aab.index <= '2020-02-21')]

datanormal = pd.DataFrame({"sp":spb.Close,"amzn":amznb.Close,"bond":bondb.Close})
#datanormal = pd.DataFrame({"sp":spb.Close,"amzn":amznb.Close,"bond":bondb.Close})
datanormal = datanormal.dropna()
datanormal.head()
    
datanormal['splag'] = datanormal.sp.shift(1)
datanormal['spret']=np.log(datanormal['sp'])-np.log(datanormal['splag'])
datanormal['amznlag'] = datanormal.amzn.shift(1)
datanormal['amznret']=np.log(datanormal['amzn'])-np.log(datanormal['amznlag'])
datanormal['bondlag'] = datanormal.bond.shift(1)
datanormal['bondret']=np.log(datanormal['bond'])-np.log(datanormal['bondlag'])
#datanormal['aalag'] = datanormal.aa.shift(1)
#datanormal['aaret']=np.log(datanormal['aa'])-np.log(datanormal['aalag'])
datanormal = datanormal[1:]

rollwindow = datanormal.rolling(window=35,win_type="boxcar").mean()
rollwindow = rollwindow[35:]

T0 = len(rollwindow)

sprollingret = rollwindow['spret'].values
amznrollingret = rollwindow['amznret'].values
bondrollingret = rollwindow['bondret'].values
#aarollingret = rollwindow['aaret'].values


# In[]
# now start for normal period: 2020-02-23 to 2020-04-13
# get rolling return


spc = sp.copy()
spc = spc[(spc.index >= '1997-05-15') & (spc.index <= '2020-04-13')]

amznc = amzn.copy()
amznc = amznc[(amznc.index >= '1997-05-15') & (amznc.index <= '2020-04-13')]

#googc = goog.copy()
#googc = googc[(googc.index >= '2005-01-01') & (googc.index <= '2020-02-21')]

bondc = bond.copy()
bondc = bondc[(bondc.index >= '1997-05-15') & (bondc.index <= '2020-04-13')]

#aab = aa.copy()
#aab = aab[(aab.index >= '2010-01-01') & (aab.index <= '2020-02-21')]

dataExtreme = pd.DataFrame({"sp":spc.Close,"amzn":amznc.Close,"bond":bondc.Close})
#datanormal = pd.DataFrame({"sp":spb.Close,"amzn":amznb.Close,"bond":bondb.Close})
dataExtreme = dataExtreme.dropna()
dataExtreme.head()
    
dataExtreme['splag'] = dataExtreme.sp.shift(1)
dataExtreme['spret']=np.log(dataExtreme['sp'])-np.log(dataExtreme['splag'])
dataExtreme['amznlag'] = dataExtreme.amzn.shift(1)
dataExtreme['amznret']=np.log(dataExtreme['amzn'])-np.log(dataExtreme['amznlag'])
dataExtreme['bondlag'] = dataExtreme.bond.shift(1)
dataExtreme['bondret']=np.log(dataExtreme['bond'])-np.log(dataExtreme['bondlag'])
#dataExtreme['aalag'] = dataExtreme.aa.shift(1)
#dataExtreme['aaret']=np.log(dataExtreme['aa'])-np.log(dataExtreme['aalag'])
dataExtreme = dataExtreme[1:]

rollwindow2 = dataExtreme.rolling(window=35,win_type="boxcar").mean()
rollwindow2 = rollwindow2[35:]
T02 = len(rollwindow2)

sprollingret2 = rollwindow2['spret'].values
amznrollingret2 = rollwindow2['amznret'].values
bondrollingret2 = rollwindow2['bondret'].values
#aarollingret = rollwindow['aaret'].values

# In[]
# estimate option price

# assets mean and std
spreturn = ret['sp'].values
amznreturn = ret['amzn'].values
bondreturn = ret['bond'].values
# mean
spmean = np.mean(spreturn)
spstd = np.std(spreturn)
amznmean = np.mean(amznreturn)
# std
amznstd = np.std(amznreturn)
bondmean = np.mean(bondreturn)
bondstd = np.std(bondreturn)

# year volatility
spyear = np.sqrt(250.)*spstd
amznyear = np.sqrt(250.)*amznstd
bondyear = np.sqrt(250.)*bondstd

# BSM Model
def callput(price,strike,vol,rf,tmat):
    d1 = (np.log(price/strike)+(rf+vol*vol/2.)*tmat)/(vol*np.sqrt(tmat))
    d2 = d1-vol*np.sqrt(tmat)
    nd1 = stats.norm.cdf(d1)
    nnd1 = stats.norm.cdf(-d1) #n(-d1)
    nd2 = stats.norm.cdf(d2)
    nnd2 = stats.norm.cdf(-d2) #n(-d2)
    
    cval = price*nd1 - strike*np.exp(-rf*tmat)*nd2
    pval = strike*np.exp(-rf*tmat)*nnd2-price*nnd1
    delta = nd1-1         #put option should be nd1-1
    return cval, pval, delta

# theoretical option value at time t
# Normal time
# amazon
amznprice = amzna.copy()
amznstartprice = amznprice.iloc[-1]
amznstrike = amznstartprice
h = 35.
T = h/250. 
amzncval, amznpval, amzndelta = callput(amznstartprice,amznstrike,amznyear,rf,T)

# sp
spprice = spa.copy()
spstartprice = spprice.iloc[-1]
spstrike = spstartprice
h = 35.
T = h/250. 
spcval, sppval, spdelta = callput(spstartprice,spstrike,spyear,rf,T)

# bond
bondprice = bonda.copy()
bondstartprice = bondprice.iloc[-1]
bondstrike = bondstartprice
h = 35.
T = h/250. 
bondcval, bondpval, bonddelta = callput(bondstartprice,bondstrike,bondyear,rf,T)

# Extreme time
# amazon
amznprice2 = amznb.copy()
amznstartprice2 = amznprice2.iloc[-1]
amznstrike2 = amznstartprice2
h = 35.
T = h/250. 
amzncval2, amznpval2, amzndelta2 = callput(amznstartprice2,amznstrike2,amznyear,rf,T)

# sp
spprice2 = spb.copy()
spstartprice2 = spprice2.iloc[-1]
spstrike2 = spstartprice2
h = 35.
T = h/250. 
spcval2, sppval2, spdelta2 = callput(spstartprice2,spstrike2,spyear,rf,T)

# bond
bondprice2 = bondb.copy()
bondstartprice2 = bondprice2.iloc[-1]
bondstrike2 = bondstartprice2
h = 35.
T = h/250. 
bondcval2, bondpval2, bonddelta2 = callput(bondstartprice2,bondstrike2,bondyear,rf,T)

# In[]
# NO HEDGE
# estimate VaR

# set up portfolio for no hedge
# normal

P0 = 1000000.
sharessp = P0*w[0]/spstartprice  
sharesamzn = P0*w[1]/amznstartprice 
sharesbond = P0*w[2]/bondstartprice   

# extreme
sharessp2 = P0*w[0]/spstartprice2  
sharesamzn2 = P0*w[1]/amznstartprice2 
sharesbond2 = P0*w[2]/bondstartprice2   

initportv = P0

# In[]
####################################
# Normal period
indexset = range(T0)
nboot = 10000
portval = np.zeros(nboot)
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice[0]
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice[0]
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice[0]
    portval[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal

VaR_noHedge_n = -(np.percentile(portval,100.*p)-initportv)
esP_noHedge_n = initportv-np.mean(portval[portval<=np.percentile(portval,100.*p)])


####################################
# Extreme period
indexset2 = range(T02)
portval2 = np.zeros(nboot)

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2[0]
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2[0]
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2[0]
    portval2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal

VaR_noHedge_e = -(np.percentile(portval2,100.*p)-initportv)
esP_noHedge_e = initportv-np.mean(portval2[portval2<=np.percentile(portval2,100.*p)])

# In[]
# FULLY HEDGED for three stocks
# estimate VaR

# calculate VaR
initportvF = P0+amznpval*sharesamzn+sppval*sharessp+bondpval*sharesbond
initportvF2 = P0+amznpval2*sharesamzn2+sppval2*sharessp2+bondpval2*sharesbond2

####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalF = np.zeros(nboot)

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    amznoptval = np.maximum(amznstrike-amznfinal,0.)
    spoptval = np.maximum(spstrike-spfinal,0.)
    bondoptval = np.maximum(bondstrike-bondfinal,0.)
    portvalF[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesamzn*amznoptval+sharessp*spoptval+sharesbond*bondoptval

VaR_Fully_n = -(np.percentile(portvalF,100.*p)-initportvF)
esP_fully_n = initportvF-np.mean(portvalF[portvalF<=np.percentile(portvalF,100.*p)])


####################################
# Extreme period
indexset2 = range(T02)
portvalF2 = np.zeros(nboot)

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    spoptval = np.maximum(spstrike2-spfinal,0.)
    bondoptval = np.maximum(bondstrike2-bondfinal,0.)
    portvalF2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesamzn2*amznoptval++sharessp2*spoptval++sharesbond2*bondoptval
 
VaR_Fully_e = -(np.percentile(portvalF2,100.*p)-initportvF2)
esP_fully_e = initportvF2-np.mean(portvalF2[portvalF2<=np.percentile(portvalF2,100.*p)])

# In[]
# PARTIAL HEDGED - only sp500
# estimate VaR

# calculate VaR
initportvSP = P0+sppval*sharessp
initportvSP2 = P0+sppval2*sharessp2


####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalSP = np.zeros(nboot)
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    spoptval = np.maximum(spstrike-spfinal,0.)
    portvalSP[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharessp*spoptval

VaR_sp_n = -(np.percentile(portvalSP,100.*p)-initportvSP)
esP_sp_n = initportvSP-np.mean(portvalSP[portvalSP<=np.percentile(portvalSP,100.*p)])

###################################
# Extreme period
indexset2 = range(T02)
portvalSP2 = np.zeros(nboot)

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    spoptval = np.maximum(spstrike2-spfinal,0.)
    portvalSP2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharessp2*spoptval

VaR_sp_e = -(np.percentile(portvalSP2,100.*p)-initportvSP2)
esP_sp_e = initportvSP2-np.mean(portvalSP2[portvalSP2<=np.percentile(portvalSP2,100.*p)])

# In[]
# PARTIAL HEDGED - only amzn
# estimate VaR

# calculate VaR
initportvAMZN = P0+amznpval*sharesamzn
initportvAMZN2 = P0+amznpval2*sharesamzn2

####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalAMZN = np.zeros(nboot) 

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    amznoptval = np.maximum(amznstrike-amznfinal,0.) 
    portvalSPAMZN[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesamzn*amznoptval

VaR_amzn_n = -(np.percentile(portvalAMZN,100.*p)-initportvAMZN)
esP_amzn_n  = initportvAMZN-np.mean(portvalAMZN[portvalAMZN<=np.percentile(portvalAMZN,100.*p)]) 

####################################
# Extreme period
indexset2 = range(T02)
portvalAMZN2 = np.zeros(nboot)
for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    spoptval = np.maximum(spstrike2-spfinal,0.)
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    portvalAMZN2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesamzn2*amznoptval

VaR_spamzn_e = -(np.percentile(portvalAMZN2,100.*p)-initportvAMZN2)
esP_spamzn_e =  initportvAMZN2 - np.mean(portvalAMZN2[portvalAMZN2<=np.percentile(portvalAMZN2,100.*p)])

# In[]
# PARTIAL HEDGED - only bond
# estimate VaR

# calculate VaR
initportvBOND = P0+bondpval*sharesbond
initportvBOND2 = P0+bondpval2*sharesbond2

####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalBOND = np.zeros(nboot) 
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    bondoptval = np.maximum(bondstrike-bondfinal,0.) 
    portvalBOND[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesbond*bondoptval

VaR_bond_n = -(np.percentile(portvalBOND,100.*p)-initportvBOND) 
esP_bond_n = initportvBOND - np.mean(portvalBOND[portvalBOND<=np.percentile(portvalBOND,100.*p)]) 

####################################
# Extreme period
indexset2 = range(T02)
portvalBOND2 = np.zeros(nboot) 

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    bondoptval = np.maximum(bondstrike2-bondfinal,0.)
    portvalBOND2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesbond2*bondoptval

VaR_bond_e = -(np.percentile(portvalBOND2,100.*p)-initportvBOND2) 
esP_bond_e = initportvBOND2 - np.mean(portvalBOND2[portvalBOND2<=np.percentile(portvalBOND2,100.*p)]) 

# In[]
# PARTIAL HEDGED - sp500+amzn
# estimate VaR

# calculate VaR
initportvSPAMZN = P0+sppval*sharessp+amznpval*sharesamzn
initportvSPAMZN2 = P0+sppval2*sharessp2+amznpval2*sharesamzn2  

####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalBOND = np.zeros(nboot) 
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    spoptval = np.maximum(spstrike-spfinal,0.)
    amznoptval = np.maximum(amznstrike-amznfinal,0.) 
    portvalSPAMZN[i]  = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharessp*spoptval + sharesamzn*amznoptval
VaR_spamzn_n = -(np.percentile(portvalSPAMZN,100.*p)-initportvSPAMZN) 
esP_spamzn_n = initportvSPAMZN - np.mean(portvalSPAMZN[portvalSPAMZN<=np.percentile(portvalSPAMZN,100.*p)])

####################################
# Extreme period
indexset2 = range(T02)
portvalSPAMZN2 = np.zeros(nboot)
for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    spoptval = np.maximum(spstrike2-spfinal,0.) 
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    portvalSPAMZN2[i]  = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal + sharessp2*spoptval + sharesamzn2*amznoptval
VaR_spamzn_e = -(np.percentile(portvalSPAMZN2,100.*p)-initportvSPAMZN2) 
esP_spamzn_e = initportvSPAMZN2 - np.mean(portvalSPAMZN2[portvalSPAMZN2<=np.percentile(portvalSPAMZN2,100.*p)])



# In[]
# PARTIAL HEDGED - amzn+bond
# estimate VaR

# calculate VaR
initportvAMZNBOND = P0+amznpval*sharesamzn+bondpval*sharesbond 
initportvAMZNBOND2 = P0+bondpval2*sharesbond2+amznpval2*sharesamzn2  

####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalAMZNBOND = np.zeros(nboot)
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    bondoptval = np.maximum(bondstrike-bondfinal,0.) 
    amznoptval = np.maximum(amznstrike-amznfinal,0.) 
    portvalAMZNBOND[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesbond*bondoptval + sharesamzn*amznoptval
VaR_amznbond_n = -(np.percentile(portvalAMZNBOND,100.*p)-initportvAMZNBOND
esP_amznbond_n = initportvAMZNBOND - np.mean(portvalAMZNBOND[portvalAMZNBOND<=np.percentile(portvalAMZNBOND,100.*p)])

####################################
# Extreme period
indexset2 = range(T02)
portvalSPAMZN2 = np.zeros(nboot)
for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    bondoptval = np.maximum(bondstrike2-bondfinal,0.) 
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    portvalAMZNBOND2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal +  sharesbond2*bondoptval + sharesamzn2*amznoptval
VaR_amznbond_e = -(np.percentile(portvalAMZNBOND2,100.*p)-initportvAMZNBOND2) 
esP_amznbond_e = initportvAMZNBOND2 - np.mean(portvalAMZNBOND2[portvalAMZNBOND2<=np.percentile(portvalAMZNBOND2,100.*p)])  



# In[]
# exclude option cost 
# FULLY HEDGED for three stocks
# estimate VaR

# calculate VaR
initportvF = P0
initportvF2 = P0
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalF = np.zeros(nboot)

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    amznoptval = np.maximum(amznstrike-amznfinal,0.)
    spoptval = np.maximum(spstrike-spfinal,0.)
    bondoptval = np.maximum(bondstrike-bondfinal,0.)
    portvalF[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesamzn*amznoptval+sharessp*spoptval+sharesbond*bondoptval

VaR_Fully_n = -(np.percentile(portvalF,100.*p)-initportvF)
esP_fully_n = initportvF-np.mean(portvalF[portvalF<=np.percentile(portvalF,100.*p)])


####################################
####################################
# Extreme period
indexset2 = range(T02)
portvalF2 = np.zeros(nboot)

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    spoptval = np.maximum(spstrike2-spfinal,0.)
    bondoptval = np.maximum(bondstrike2-bondfinal,0.)
    portvalF2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesamzn2*amznoptval++sharessp2*spoptval++sharesbond2*bondoptval
    
VaR_Fully_e = -(np.percentile(portvalF2,100.*p)-initportvF2)
esP_fully_e = initportvF2-np.mean(portvalF2[portvalF2<=np.percentile(portvalF2,100.*p)])


# In[]
# PARTIAL HEDGED - only sp500
# estimate VaR

# calculate VaR
initportvSP = P0
initportvSP2 = P0
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalSP = np.zeros(nboot)

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    spoptval = np.maximum(spstrike-spfinal,0.)
    portvalSP[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharessp*spoptval

VaR_sp_n = -(np.percentile(portvalSP,100.*p)-initportvSP)
esP_sp_n = initportvSP-np.mean(portvalSP[portvalSP<=np.percentile(portvalSP,100.*p)])

####################################
# Extreme period
indexset2 = range(T02)
portvalSP2 = np.zeros(nboot)

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    spoptval = np.maximum(spstrike2-spfinal,0.)
    portvalSP2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharessp2*spoptval

VaR_sp_e = -(np.percentile(portvalSP2,100.*p)-initportvSP2)
esP_sp_e = initportvSP2-np.mean(portvalSP2[portvalSP2<=np.percentile(portvalSP2,100.*p)])

# In[]
# PARTIAL HEDGED - only amzn
# estimate VaR

# calculate VaR
initportvAMZN = P0 
initportvAMZN2 = P0 
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalAMZN = np.zeros(nboot)  

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    amznoptval = np.maximum(amznstrike-amznfinal,0.) 
    portvalAMZN[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesamzn*amznoptval

VaR_amzn_n = -(np.percentile(portvalAMZN,100.*p)-initportvAMZN)
esP_amzn_n = initportvAMZN - np.mean(portvalAMZN[portvalAMZN<=np.percentile(portvalAMZN,100.*p)])  

####################################
# Extreme period
indexset2 = range(T02)
portvalAMZN2 = np.zeros(nboot) 

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    amznoptval = np.maximum(amznstrike2-amznfinal,0.) 
    portvalAMZN2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesamzn2*amznoptval

VaR_amzn_e = -(np.percentile(portvalAMZN2,100.*p)-initportvAMZN2
esP_amzn_e = initportvAMZN2 - np.mean(portvalAMZN2[portvalAMZN2<=np.percentile(portvalAMZN2,100.*p)]) 

# In[]
# PARTIAL HEDGED - only bond
# estimate VaR

# calculate VaR
initportvBOND = P0
initportvBOND2 = P0 
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalBOND = np.zeros(nboot)

p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    bondoptval = np.maximum(bondstrike-bondfinal,0.) 
    portvalBOND[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesbond*bondoptval

VaR_bond_n = -(np.percentile(portvalBOND,100.*p)-initportvBOND)  
esP_bond_n = initportvBOND - np.mean(portvalBOND[portvalBOND<=np.percentile(portvalBOND,100.*p)])  

####################################
# Extreme period
indexset2 = range(T02)
portvalBOND2 = np.zeros(nboot) 

for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    bondoptval = np.maximum(bondstrike2-bondfinal,0.)  
    portvalBOND2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesbond2*bondoptval

VaR_bond_e = -(np.percentile(portvalBOND2,100.*p)-initportvBOND2) 
esP_bond_e = initportvBOND2 - np.mean(portvalBOND2[portvalBOND2<=np.percentile(portvalBOND2,100.*p)]) 

# In[]
# PARTIAL HEDGED - sp500+amzn
# estimate VaR

# calculate VaR
initportvSPAMZN = P0
initportvSPAMZN2 = P0
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalSPAMZN = np.zeros(nboot)
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    spoptval = np.maximum(spstrike-spfinal,0.)
    amznoptval = np.maximum(amznstrike-amznfinal,0.)
    portvalSPAMZN[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharessp*spoptval+sharesamzn*amznoptval
VaR_spamzn_n = -(np.percentile(portvalSPAMZN,100.*p)-initportvSPAMZN)
esP_spamzn_n = initportvSPAMZN-np.mean(portvalSPAMZN[portvalSPAMZN<=np.percentile(portvalSPAMZN,100.*p)])

####################################
# Extreme period
indexset2 = range(T02)
portvalSPAMZN2 = np.zeros(nboot)
for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    spoptval = np.maximum(spstrike2-spfinal,0.)
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    portvalSPAMZN2[i] = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharessp2*spoptval+sharesamzn2*amznoptval
VaR_spamzn_e = -(np.percentile(portvalSPAMZN2,100.*p)-initportvSPAMZN2)
esP_spamzn_e = initportvSPAMZN2-np.mean(portvalSPAMZN2[portvalSPAMZN2<=np.percentile(portvalSPAMZN2,100.*p)])

# In[]
# PARTIAL HEDGED - amzn+bond
# estimate VaR

# calculate VaR
initportvAMZNBOND = P0 
initportvAMZNBOND2 = P0
####################################
# normal period
indexset = range(T0)
nboot = 10000
portvalSPAMZN = np.zeros(nboot)
p = 0.01

for i in range(nboot):
    bindex = np.random.choice(indexset,size=35)
    spfinal = np.prod(np.exp(sprollingret[bindex]))*spstartprice
    amznfinal = np.prod(np.exp(amznrollingret[bindex]))*amznstartprice
    bondfinal = np.prod(np.exp(bondrollingret[bindex]))*bondstartprice
    bondoptval = np.maximum(bondstrike-bondfinal,0.) 
    amznoptval = np.maximum(amznstrike-amznfinal,0.)
    portvalAMZNBOND[i] = sharessp*spfinal + sharesamzn*amznfinal + sharesbond*bondfinal+sharesbond*bondoptval+sharesamzn*amznoptval
VaR_amznbond_n = -(np.percentile(portvalAMZNBOND,100.*p)-initportvAMZNBOND) 
esP_amznbond_n = initportvAMZNBOND - np.mean(portvalAMZNBOND[portvalAMZNBOND<=np.percentile(portvalAMZNBOND,100.*p)]) 

####################################
# Extreme period
indexset2 = range(T02)
portvalAMZNBOND2 = np.zeros(nboot) 
for i in range(nboot):
    bindex = np.random.choice(indexset2,size=35)
    spfinal = np.prod(np.exp(sprollingret2[bindex]))*spstartprice2
    amznfinal = np.prod(np.exp(amznrollingret2[bindex]))*amznstartprice2
    bondfinal = np.prod(np.exp(bondrollingret2[bindex]))*bondstartprice2
    bondoptval = np.maximum(bondstrike2-bondfinal,0.) 
    amznoptval = np.maximum(amznstrike2-amznfinal,0.)
    portvalAMZNBOND2[i]  = sharessp2*spfinal + sharesamzn2*amznfinal + sharesbond2*bondfinal+sharesbond2*bondoptval+sharesamzn2*amznoptval
VaR_amznbond_e = -(np.percentile(portvalAMZNBOND2,100.*p)-initportvAMZNBOND2)  
esP_amznbond_e = initportvAMZNBOND2 - np.mean(portvalAMZNBOND2[portvalAMZNBOND2<=np.percentile(portvalAMZNBOND2,100.*p)])




