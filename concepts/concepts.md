**Machine Learning Paradigms & Foundational Theory**
* Statistical machine learning
* Mathematical optimization
* Swarm intelligence
* Soft computing
* Bias-variance tradeoff
* VC dimension 
* Regularization necessity for complex models

**Base Estimators & Core Mathematics**
* **Logistic Regression:** Maximum Likelihood Estimation, Cross-Entropy Loss, Gradient Descent.
* **Support Vector Machines (SVM):** Lagrangian multipliers, dual formulation, the kernel trick (Mercer's theorem), optimal margin classifiers, soft-margin regularization ($C$ parameter).
* **Multilayer Perceptron (MLP):** Universal Approximation Theorem, backpropagation calculus (chain rule over computational graphs), activation functions.

**Ensemble Theory & Validation**
* Stacking and Voting mechanisms
* Ensemble weight optimization ($\mathbf{w} = [w_1, w_2, w_3]^T$ subject to $\sum w_i = 1$)
* $K$-fold cross-validation
* Out-of-fold (OOF) predictions
* Meta-learner overfitting prevention

**Swarm Intelligence & Optimization in Continuous Domains**
* Standard Ant Colony Optimization (ACO) and discrete combinatorial problems (e.g., Traveling Salesperson Problem).
* Continuous ACO ($ACO_{\mathbb{R}}$).
* Probability density functions (Gaussian kernels) vs. discrete pheromone tables.
* Pheromone representation via Gaussian mixtures.
* Routing and evaporation mechanisms in $\mathbb{R}^n$.
* Artificial discretization of continuous search spaces.
* Objective function design (negative log-loss or continuous Mean Squared Error).
* Algorithm convergence tracking.

**Soft Computing & Neuro-Fuzzy Systems**
* Adaptive Neuro-Fuzzy Inference System (ANFIS).
* Curse of Dimensionality (ANFIS rule explosion).
* Takagi-Sugeno fuzzy inference.
* Fuzzy membership functions (e.g., High/Low) and fuzzification.
* T-norm operators.
* Mapping continuous probabilities to linguistic terms.
* Hybrid learning algorithms: Least squares (for consequent parameters) + Gradient descent (for premise parameters).

**Mathematical Optimization (Quadratic Programming)**
* Quadratic Programming (QP) formulation.
* Mean Squared Error minimization: $\min_{\mathbf{w}} ||P\mathbf{w} - y||_2^2$ 
* Standard QP form: $\min \frac{1}{2}\mathbf{w}^T Q \mathbf{w} + c^T \mathbf{w}$ (where $Q = 2P^TP$ and $c = -2P^Ty$).
* Convex optimization.
* Affine constraints ($\mathbf{1}^T\mathbf{w} = 1$ and $\mathbf{w} \ge 0$).
* Karush-Kuhn-Tucker (KKT) conditions.
* Constrained least squares.
* Sequential Least SQuares Programming (SLSQP).

**Data Engineering & Pipeline Mechanics**
* Categorical variable handling: One-Hot Encoding.
* Continuous variable scaling: StandardScaler.
* Feature selection techniques: Random Forest, Information Gain.
* Strict data splitting schemas: Train, Validation, and Test sets.
* Prevention of state contamination across pipeline blocks.
* Prediction matrix construction ($P \in \mathbb{R}^{N \times 3}$).
* Evaluation metrics: Precision, Recall, F1-score, ROC-AUC.
