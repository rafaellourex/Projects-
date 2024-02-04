
#Feature Extraction
import numpy as np
import pandas as pd 

def oneHotEncoder (data):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
    non_metric_features =data.select_dtypes(exclude=np.number).columns
    ohc = OneHotEncoder(sparse=False)
    ohc_feat = ohc.fit_transform(data[non_metric_features])
    names = ohc.get_feature_names_out()
    
    ohc_cat = pd.DataFrame(data =ohc_feat ,columns = names, index = data.index)
    return(ohc_cat)
 

def Feature_Discretization (data,strategy = 'quantile',n_bins = 10):
    from sklearn.preprocessing import KBinsDiscretizer
    df_eng = data.copy()
    num_cols = df_eng.select_dtypes(include = np.number).columns
    rest_cols = df_eng.columns[df_eng.columns.isin(num_cols)==False]
    
    kbins = KBinsDiscretizer(n_bins=n_bins ,
                             encode = 'ordinal',
                             strategy=strategy)
    data = kbins.fit_transform(df_eng[num_cols])
    k_bins_df = pd.DataFrame(index = df_eng.index,
                             columns=num_cols,
                             data=data )

    df_bin = pd.concat([k_bins_df,
                        df_eng[rest_cols]],
                       axis=1)
    
    return(df_bin,data)


def quantile_transform_df(df, n_quantiles=10, random_state=0):
    from sklearn.preprocessing import quantile_transform
    # Perform quantile transformation on numerical columns
    transformed_df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        transformed_df[col] = quantile_transform(df[[col]],
                                                 n_quantiles=n_quantiles,
                                                 random_state=random_state)
        
    # Return transformed DataFrame with the same index as the original DataFrame
    transformed_df.index = df.index
    return transformed_df


def encodeCategories (x_train, y_train):
    
    import category_encoders as ce
    catBoostEnc = ce.CatBoostEncoder(random_state=0 )
    
    catBoostEnc.fit(x_train,y_train)
    x_train_enc = catBoostEnc.transform(x_train).iloc[:,:]
    
    return x_train_enc, catBoostEnc