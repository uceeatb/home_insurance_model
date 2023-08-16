import pandas as pd
from collections import Counter

def get_feature_types(df: pd.DataFrame) -> dict:
    
    """
    Returns a dictionary with features bucketed into "numeric" and
    "categorical" 

    Args:
        df (pd.DataFrame): dataframe containing only features

    Returns:
        dict: features types dictionary
    """

    feature_types = {
        "numeric": [],
        "categorical": []
    }
    dtype_dict = df.dtypes.to_dict().items()
    
    for feature, type_ in dtype_dict:
        if type_== "int64" or type_ == "float64":
            feature_types["numeric"].append(feature)
        else:
            feature_types["categorical"].append(feature)
 
    return feature_types


def select_dummy_values(train_df: pd.DataFrame,
                        categorical_features: list,
                        LIMIT_DUMMIES: int=50) -> dict:
        
    """
    Builds a dictionary of dummy values for categorical features,
    ordered by frequency and capped by a certain threshold,
    to facilitate one-hot encoding.

    Args:
        train_df (pd.DataFrame): training data
        categorical_features (list): list of categorical features
        LIMIT_DUMMIES (pd.DataFrame): threshold for cardinality

    Returns:
        dict: features types dictionary
    """

    dummy_values = {}
    for feature in categorical_features:
        values = [value for (value, _) 
                  in Counter(train_df[feature]).most_common(LIMIT_DUMMIES)
        ]
        dummy_values[feature] = values
    return dummy_values


def dummy_encode_dataframe(df: pd.DataFrame,
                           DUMMY_VALUES: dict):
    
    """
    One-hot encodes dummy categorical features using dummy_values of selected
    cardinality.

    Args:
        df (pd.DataFrame): training, test or scoring data
        DUMMY_VALUES (dict): dictionary of dummy values fo categorical features

    Returns:
        None
    """
    for (feature, dummy_values) in DUMMY_VALUES.items():
        for dummy_value in dummy_values:
            dummy_name = '%s_value_%s' % (feature, dummy_value)
            df[dummy_name] = (df[feature] == dummy_value).astype(float)
        del df[feature]