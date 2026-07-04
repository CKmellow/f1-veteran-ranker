# Model Artifacts Directory

This directory stores trained model binaries produced by the training pipeline.

- `f1_xgb_ranker.pkl`
- `f1_lgb_ranker.pkl`

Training source code is intentionally versioned under:

- `src/models/train_ranker.py`
- `src/models/train_model.py`

This separation keeps executable source code and generated artifacts clearly isolated.
