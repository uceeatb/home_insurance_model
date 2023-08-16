import xgboost as xgb

xgb_model_paras_feature_sel = xgb.XGBClassifier(
    verbose = 3,
    n_jobs=16,
    base_score=0.5,
    random_state=1337,
    missing=None,
    scale_pos_weight=2.3,
    cv = 4,
    objective='binary:logistic',
    eval_metric='auc',
    subsample=1.0
)