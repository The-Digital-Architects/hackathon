import pandas as pd
import os

def create_submission(predictions_df, output_path='submissions/submission.csv'):
    """
    Create submission file from predictions DataFrame
    
    Args:
        predictions_df: DataFrame with columns ['STATE', 'month', 'total_fire_size']
        output_path: Path to save submission file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load zero submission template
    submission = pd.read_csv('data/zero_submission.csv')
    
    # Merge predictions with template
    submission = submission[['ID', 'STATE', 'month']].merge(
        predictions_df,
        on=['STATE', 'month'],
        how='left'
    )
    
    submission = submission[['ID', 'STATE', 'month', 'total_fire_size']]

    # Save submission file
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")