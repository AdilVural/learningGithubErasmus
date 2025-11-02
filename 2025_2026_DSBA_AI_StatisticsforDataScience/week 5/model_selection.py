# Statistics for Data Science
# D. Fok
# Functions to perform model selection based on statsmodels

import pandas as pd
import statsmodels.api as sm
import numpy as np

# Backward elimination process based on p-values
# Input m is a statsmodels OLS object (before fit()!)
def backward_elimination_pvalue(model, significance=0.05):    
    y = model.data.endog    
    X = pd.DataFrame(model.data.exog, columns = model.exog_names)
    m = sm.OLS(y, X).fit()
    
    while max(m.pvalues[1:] > significance):
        features = m.model.exog_names
        
        to_remove = m.pvalues[1:].idxmax()
        print(f"- {to_remove} with p-value {max(m.pvalues[1:])}")
        sel = [f != to_remove for f in features]
        X = pd.DataFrame(m.model.data.exog[:,sel], columns = [f for f in features if f != to_remove])
        features.remove(to_remove)       
        
        m = sm.OLS(y, X).fit()
                
    return m

# Backward elmination process based on AIC
# Input m is a statsmodels OLS object (before fit()!)
def backward_elimination_aic(m):    
    y = m.data.endog
    features = m.exog_names
    X = pd.DataFrame(m.data.exog, columns=features)
    used = np.ones(len(features))
    used[0] = 1
    
    curF = [f for (i,f) in enumerate(features) if used[i]]
    stop = False

    curAIC = sm.OLS(y, X[curF]).fit().aic
    print("Initial AIC =", curAIC, "\n")
    while stop==False:
        # try removing each
        best = 0
        for (i,f) in enumerate(features):
            if (i != 0) and (used[i] == 1):
                tmpUsed = np.copy(used)
                tmpUsed[i] = 0
                
                curF = [f for (i,f) in enumerate(features) if tmpUsed[i]]
                aic = sm.OLS(y, X[curF]).fit().aic
                print("-", f, aic)
                if aic < curAIC:
                    curAIC = aic
                    best = i
        
        if best > 0:
            used[best] = 0
            print("Removing", features[best], " AIC = ", curAIC, "\n")
        else:
            print("Removing nothing\n")
            stop = True
        
    curF = [f for (i,f) in enumerate(features) if used[i]]
    return sm.OLS(y, X[curF]).fit()

# Forward selection process based on AIC
# Input m is a statsmodels OLS object (before fit()!)
def forward_selection_aic(m):    
    y = m.data.endog
    features = m.exog_names
    X = pd.DataFrame(m.data.exog, columns=features)
    used = np.zeros(len(features))
    used[0] = 1
    
    curF = [f for (i,f) in enumerate(features) if used[i]]
    stop = False

    curAIC = sm.OLS(y, X[curF]).fit().aic
    print("Initial AIC =", curAIC, "\n")
    while stop==False:
        # try adding each
        best = 0
        for (i,f) in enumerate(features):
            if used[i] == 0:                
                tmpUsed = np.copy(used)
                tmpUsed[i] = 1
                
                curF = [f for (i,f) in enumerate(features) if tmpUsed[i]]
                aic = sm.OLS(y, X[curF]).fit().aic
                print("+", f, aic)
                if aic < curAIC:
                    curAIC = aic
                    best = i
        
        if best > 0:
            used[best] = 1
            print("Adding", features[best], " AIC = ", curAIC, "\n")
        else:
            print("Adding nothing\n")
            stop = True
        
    curF = [f for (i,f) in enumerate(features) if used[i]]
    return sm.OLS(y, X[curF]).fit()

# Forward selection based on p-values
# Input m is a statsmodels OLS object (before fit()!)
def forward_selection_pvalue(m, significance=0.05):    
    y = m.data.endog
    features = m.exog_names
    X = pd.DataFrame(m.data.exog, columns=features)
    used = np.zeros(len(features))
    used[0] = 1
    
    stop = False
    
    while stop==False:
        # try adding each
        best = 0
        curPval = 1
        for (i,f) in enumerate(features):
            if used[i] == 0:                
                tmpUsed = np.copy(used)
                tmpUsed[i] = 1
                
                curF = [f for (i,f) in enumerate(features) if tmpUsed[i]]
                pval = sm.OLS(y, X[curF]).fit().pvalues[f]
                print("+", f, pval)
                if pval < curPval:
                    if pval < significance:
                        curPval = pval
                        best = i
        
        if best > 0:
            used[best] = 1
            print("Adding", features[best], " pvalue = ", curPval, "\n")
        else:
            print("Adding nothing\n")
            stop = True
        
    curF = [f for (i,f) in enumerate(features) if used[i]]
    return sm.OLS(y, X[curF]).fit()


# Helper function to generate all subsets of a set
def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

# Perform all subset regression and return the best `best` models based on AIC
# Input m is a statsmodels OLS object (before fit()!)
def allsubset(m, best=10):       
    y = m.data.endog
    features = m.exog_names
    X = pd.DataFrame(m.data.exog, columns=features)    
    bestModels = []    
    aicList = [] 
    i = 0
    for used in powerset(features[1:]):                
        used.append('Intercept')
        m = sm.OLS(y, X[used]).fit()
        # print(used, m.aic)
        i = i +1
        if i<=best:
            bestModels.append(m)
            aicList.append(m.aic)                
        else:
            maxAIC = max(aicList)
            if m.aic < maxAIC:
                ind = [l for (l, aic) in enumerate(aicList) if aic==maxAIC]
                # print("Change ", ind, "for", m)
                bestModels[ind[0]] = m
                aicList[ind[0]] = m.aic
                
    bestModels = [bestModels[i] for i in np.argsort(aicList)]
    
    print("All subset regression:\nBest", len(bestModels), "models by AIC")
    for m in bestModels:
        print("AIC =", m.aic, end=' Intercept + ')
        for v in m.params.index[:-1]:
            print(v, end=' + ')
        print('\b\b')
    
    return bestModels[0]