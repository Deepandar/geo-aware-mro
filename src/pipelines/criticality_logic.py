import pandas as pd
import numpy as np

def calculate_criticality(df):
    # Weights and maps from Week 3 NB 
    weights = {'ABC':0.28, 'VED':0.25, 'FNS':0.17, 'LOC':0.15, 'LTR':0.15}
    abc_map = {'A':1, 'B':0.6, 'C':0.3}
    ved_map = {'V':1, 'E':0.6, 'D':0.3}
    fns_map = {'F':1, 'N':0.6, 'S':0.3}
    
    df['Ci'] = (df['ABC'].map(abc_map) * weights['ABC'] +
                df['VED'].map(ved_map) * weights['VED'] +
                df['FNS'].map(fns_map) * weights['FNS'] +
                (df['location_score']/3) * weights['LOC'] +
                (df['LTR'] / df['LTR'].max()) * weights['LTR'])
    return df
