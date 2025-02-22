import pandas as pd

def create_submission(X_test, y_pred, empty_submission_path="data/zero_submission.csv", target_col='total_fire_size'):
    empty_pred = pd.read_csv(empty_submission_path)
    y_pred = pd.Series(y_pred, name=target_col)
    X_test = X_test.rename(columns={'State': 'STATE', 'year_month': 'month'})
    X_test[target_col] = y_pred
    submission = empty_pred.drop(columns=[target_col])
    submission = submission.merge(X_test[['STATE', 'month', target_col]], on=['STATE', 'month'], how='left')
    submission.to_csv('submission.csv', index=False)

