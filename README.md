# Hackathon Practice

## Project Structure

```
.
├── config/
│   └── config.yaml         # Main config file
├── data/
│   ├── raw/               # Raw input data
│   └── processed/         # Processed data
├── models/                # Saved model files after training
├── notebooks/            
│   └── template.ipynb     # Jupyter notebook template for experimentation
├── src/
│   ├── data.py           # Data handling utilities
│   ├── models.py         # Model implementations
│   ├── scoring/          # Scoring metrics
│   └── utils/            # Utility functions
└── train.py              # Main training script
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place your dataset in `data/raw/`
   - Update the data path in `config/config.yaml`

3. Train a model:
```bash
python train.py --config config/config.yaml
```

4. For experimentation, use the Jupyter notebook template:
```bash
jupyter notebook notebooks/template.ipynb
```

## Available Models

- RandomForest
- AdaBoost
- Neural Network (MLP)

## Configuration

The `config/config.yaml` file controls:
- Data paths and preprocessing parameters
- Model selection and hyperparameters
- Training settings
- Logging configuration
- Output directories

## Development

To add a new model:

1. Extend the BaseModel class in `src/models.py`
2. Add model configuration in `config/config.yaml`
3. Update the model registry in `train.py`
