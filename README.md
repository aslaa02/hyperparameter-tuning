# Hyper Parameter Tuning 

Credit Risk Assessment with Ensemble Learning

Goal: Predict credit default risk using multiple ML models intelligently combined.
- Machine Learning(core): Ensemble of 3 models: Logistic Regression, SVM, and Neural Network trained on German Credit Dataset
- Swarm Intelligence: Ant Colony Optimization(ACO) to select the optimal voting weights for the ensemble (eg., NN gets 0.4 weights, SVM gets 0.3, LR gets 0.3).
- Soft computing: Neural-Fuzzy system (ANFIS) that takes the raw features as input and provide an interpretable risk score("medium-high risk") alongside the ML prediction.
- Mathematical Optimization: Quadratic Programming to solve for the optimal ensemble weights(benchmark against ACO's solution).

How it integrates: Three ML models make prediction -> ACO finds the best combinaion weights -> ANFIS provides interpretable output -> QP validates optimal weights.
