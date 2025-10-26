# Statistics for Data Science
# D. Fok
# Functions to plot "ceteris paribus" plot
# This is especially useful for non-linear models
# Please report problems or errors to dfok@ese.eur.nl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using effects package to get ceteris paribus plots
def ploteffect(model, focalvar):
    df = model.model.data.frame
    # Get means for non-categorical variables    
    testdf = pd.DataFrame([df.select_dtypes(exclude=['category']).mean()])
    
    # Remove focal var if needed
    if focalvar in df.select_dtypes(exclude=['category']):
        testdf = testdf.drop(columns=[focalvar])   
    
    # Expand data each categorical variable (excluding focal)
    for v in df.select_dtypes(include=['category']):        
        if v != focalvar:
            cp = testdf.copy()    
            
            # Append column with furst unique value
            testdf[v] = np.full(testdf.shape[0], np.unique(df[v])[0])    
            
            # Add rows for every other unique value
            for value in np.unique(df[v])[1:]:
                cp[v] = value
                testdf = pd.concat([testdf, cp])

    # Add focal variable (grid for numerical, all unique for categorical)          
    if focalvar in df.select_dtypes(exclude=['category']):
        focal = np.linspace(df[focalvar].min(), df[focalvar].max())
    else:
        focal = np.unique(df[focalvar])
    cp = testdf.copy()
    # Add column for first value
    testdf[focalvar] = np.full(testdf.shape[0], focal[0])    
    # Add rows for every other unique value
    for value in focal[1:]:        
        cp[focalvar] = value        
        testdf = pd.concat([testdf, cp]) 

    # Calculate the prediction
    testdf["__pred"] = model.predict(exog=testdf)
    
    # Average over all non-focal variables
    testdf = testdf.groupby(focalvar).agg(__average_pred=("__pred", "mean"),)
    
    # Create plot
    if focalvar in df.select_dtypes(exclude=['category']):
        plt.plot(testdf.index, testdf["__average_pred"], marker='o', markersize=8)
    else:
        plt.scatter(testdf.index, testdf["__average_pred"])
    plt.xlabel(focalvar)
    plt.ylabel("Prediction")    
    # return testdf
