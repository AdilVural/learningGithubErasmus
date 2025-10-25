# Statistics for Data Science
# D. Fok
# Functions to create diagnostic plots based on
# statsmodels.stats.outliers_influence.OLSInfuence()

import statsmodels.nonparametric.smoothers_lowess as smooth
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Some helper functions
def addlabels(ax, x,y,labels, number=4, abs_thres=False):
    if abs_thres==True:
        addlabels_crit(ax, x,y,np.abs(y),labels,number)   
    else:
        addlabels_crit(ax, x,y,np.abs(y),labels,number)   

def addlabels_crit(ax, x,y, crit, labels, number=4):
    threshold = np.sort(crit)[-number-1]
        
    for i, r in enumerate(zip(x, y, labels, crit)):
        if r[3] > threshold:
            ax.text(r[0],r[1],r[2])
                
def addnonparfit(ax, x,y):
    rng = np.linspace(x.min(), x.max())
    nonparfit = smooth.lowess(y,x,xvals=rng)            
    ax.plot(rng,nonparfit)
                
def getstdres(influence):
    return influence.resid/np.sqrt(np.var(influence.resid))

# The actual diagnostic plots, can provide axis to add to an existing plot
# influence is an OLSInfluence() object
def residfitted(influence, ax=None):    
    if ax==None:
        fig, ax = plt.subplots(1,1)
    fit = influence.results.predict()
    ax.scatter(fit, influence.resid)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs fitted")
    ax.plot([fit.min(), fit.max()], [0,0],linestyle="dotted")

    addlabels(ax, fit, influence.resid, influence.results.model.data.row_labels, abs_thres=True)
    addnonparfit(ax, fit, influence.resid)
    return ax

def qqresid(influence, ax=None):    
    if ax==None:
        fig, ax = plt.subplots(1,1)
        
    df = influence.nobs-influence.k_vars-1
    stats.probplot(influence.resid_studentized, dist=stats.distributions.t, sparams=df, fit=False, plot=ax)
    bnds = [influence.resid_studentized.min(), influence.resid_studentized.max()]
    ax.plot(bnds, bnds)

    stdres = pd.DataFrame(data=influence.resid_studentized, columns=["stdres"])
    stdres.index = influence.results.model.data.row_labels
    stdres = stdres.sort_values(by="stdres")

    # sm.qqplot(stdres.stdres, line='r', ax=ax)
    qq = np.arange(1, influence.nobs+1)/(influence.nobs+2)
    qq = stats.t(df=df).ppf(qq)

    addlabels(ax, qq, stdres.stdres, stdres.index, abs_thres=True)                  
    ax.set_title("QQ plot of studentized residuals");

def scalelocation(influence, ax=None):    
    if ax==None:
        fig, ax = plt.subplots(1,1)
    fit = influence.results.predict()
    stdres = getstdres(influence)
    
    target = np.sqrt(np.abs(stdres))
    ax.scatter(fit, target)
    ax.set_xlabel("Fitted values")
    # ax.set_ylabel("Sqrt(abs. standardized residuals)")
    ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
    ax.set_title("Scale vs location")  

    addnonparfit(ax, fit,target)
    addlabels(ax, fit, target, influence.results.model.data.row_labels)      
   
def residleverage(influence, ax=None):
    if ax==None:
        fig, ax = plt.subplots(1,1)
    stdres = getstdres(influence)
    ax.scatter(influence.hat_matrix_diag, stdres)
    
    lims = ax.axes.get_ylim()
    hrng = np.arange(0.01,influence.hat_matrix_diag.max(),0.01)
    hfact = hrng/((1-hrng)**2)
    Dfact = hfact/influence.results.params.size
    for x in [0, 0.5, 1]:
        ax.plot(hrng, np.sqrt(x/Dfact), linestyle='dotted', color="grey")
        ax.plot(hrng, -np.sqrt(x/Dfact), linestyle='dotted', color="grey")
    ax.axes.set_ylim(lims)
    ax.set_xlabel("Leverage")
    ax.set_ylabel("Standardized residuals")
    h = influence.hat_matrix_diag
            
    addlabels_crit(ax, h, stdres, np.abs(influence.cooks_distance[0]), influence.results.model.data.row_labels)        
    addnonparfit(ax, h,stdres)

    ax.set_title("Residuals vs leverage")

def diagnosticplots(influence):
    fig, axs = plt.subplots(2, 2, height_ratios=[0.2,0.2])
    residleverage(influence, axs[1,1])
    residfitted(influence, axs[0,0])
    qqresid(influence, axs[0,1])
    scalelocation(influence, axs[1,0])
    fig.tight_layout()
