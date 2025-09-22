# How to run

# Data Processing Decisions

Drop education-num because it is a numerical representation of education.

# Run training

You can run the two model or a debug version with one of the next command lines:

python -m src.train --config configs/config_xgboost.yaml
python -m src.train --config configs/config_lightgbm.yaml
python -m src.train --config configs/config_debug.yaml

# Screenshots of MLFlow