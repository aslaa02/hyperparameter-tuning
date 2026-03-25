# Import Documentation: Ensemble Credit Risk Assessment Pipeline
### A Rigorous, Self-Contained Reference for Every Library and Algorithm

> **How to read this document:** Every technical term that a beginner might not recognise is explained inline inside parentheses immediately after it appears. You should be able to read this document top-to-bottom without needing to consult any external source. Mathematical notation is introduced before it is used.

---

## Preliminary: What an Import Statement Actually Does

Before discussing any specific library, it is worth being precise about what `import X` means computationally.

Python is an interpreted language (meaning a program called the interpreter reads your source code line by line and executes it, rather than compiling it to machine code in advance). When you write `import numpy as np`, Python:

1. Locates the `numpy` package on your filesystem (inside the site-packages directory of your Python installation)
2. Executes the package's initialisation code, which loads compiled C extensions into memory
3. Binds the name `np` in your current namespace (the dictionary of names your script can see) to the loaded module object

The `as np` part is an alias — it means that everywhere you would write `numpy.something`, you can write `np.something` instead. This is purely a brevity convention, not a mathematical operation.

---

## 1. `import numpy as np`

### What NumPy Is

NumPy is the foundational numerical computing library for Python. Its central object is the `ndarray` (n-dimensional array — where n can be 1 for a list of numbers, 2 for a table of numbers, 3 for a cube of numbers, and so on).

The critical property of an `ndarray` compared to a Python `list` is **homogeneous typing** (all elements have the same data type, e.g., all 64-bit floating point numbers) and **contiguous memory layout** (all values are stored next to each other in RAM, rather than scattered as Python objects with individual memory headers). This means NumPy can delegate arithmetic to pre-compiled C and Fortran routines (specifically, libraries called BLAS — Basic Linear Algebra Subprograms — and LAPACK — Linear Algebra Package — that are optimised at the hardware level for matrix operations), executing operations on large arrays orders of magnitude faster than equivalent Python loops.

### The Mathematical Object NumPy Represents

Every two-dimensional `ndarray` of shape `(m, n)` is a matrix in $\mathbb{R}^{m \times n}$ (the set of all matrices with $m$ rows and $n$ columns, where every entry is a real number). NumPy's operations map directly to standard matrix algebra:

- `A @ B` computes the **matrix product** $AB$ where $(AB)_{ij} = \sum_k A_{ik} B_{kj}$
- `A.T` computes the **transpose** $A^\top$ where $(A^\top)_{ij} = A_{ji}$ (rows become columns)
- `np.linalg.norm(v)` computes the **Euclidean norm** (the length of a vector) $\|v\|_2 = \sqrt{\sum_i v_i^2}$
- `np.exp(x)` applies the **exponential function** $e^x$ element-wise (where $e \approx 2.718$ is Euler's number, the base of the natural logarithm)

### Role in This Pipeline

NumPy is the numerical backbone of every quantitative operation outside the ANFIS (Adaptive Neuro-Fuzzy Inference System — a type of neural network described in Section 7):

- Constructing the prediction matrix **P** of shape $(N, 3)$ (a matrix with $N$ rows, one per training sample, and 3 columns, one per base model's predicted probability)
- Computing the QP (Quadratic Programming — an optimisation problem where the objective function is a quadratic polynomial) matrices **Q = 2PᵀP** and **c = −2Pᵀy**
- The softmax function (a mathematical map that takes any vector of real numbers and converts it into a probability vector where all entries are positive and sum to 1): `exp(w) / sum(exp(w))`
- MSE (Mean Squared Error — the average of the squared differences between predictions and true values) computation: `mean((P @ w − y)²)`

### Mathematical Prerequisites

- **Real numbers and arithmetic**: addition, multiplication, division over $\mathbb{R}$
- **Vectors**: ordered lists of numbers, treated as points or directions in space
- **Matrices**: rectangular grids of numbers, treated as linear transformations between vector spaces
- **Matrix multiplication**: specifically, that multiplying an $(m \times k)$ matrix by a $(k \times n)$ matrix produces an $(m \times n)$ matrix
- **The transpose**: swapping rows and columns of a matrix

---

## 2. `import pandas as pd`

### What Pandas Is

Pandas provides the `DataFrame` — a two-dimensional, labelled, heterogeneously typed (meaning different columns can contain different data types, e.g., one column of integers, another of strings) tabular data structure. You can think of a `DataFrame` as a spreadsheet that lives in memory: rows are observations (individual records), and columns are variables (features or measurements).

Each column in a `DataFrame` is a `Series` (a one-dimensional labelled array — like a single column of a spreadsheet with both a name and an index). Internally, each `Series` is backed by a NumPy `ndarray` for numeric data.

### The Critical Operation: `.select_dtypes()`

The call `X.select_dtypes(include=['int64', 'float64'])` returns a new `DataFrame` containing only the columns whose data type is a 64-bit integer (`int64` — a whole number stored using 64 binary digits, capable of representing values from approximately $-9.2 \times 10^{18}$ to $9.2 \times 10^{18}$) or a 64-bit floating point number (`float64` — a number with a decimal component, stored using the IEEE 754 double-precision standard, giving approximately 15-16 significant decimal digits of precision).

The call `X.select_dtypes(include=['category', 'object'])` returns the columns whose dtype is `category` (a Pandas-specific type for variables that take a finite, fixed set of string values, like "car loan" or "education") or `object` (Python's generic string container type).

This partition is mathematically meaningful because:

- **Numerical features** (also called continuous features — variables that can take any value on a continuous number line, like "credit amount in DM" ranging from 250 to 18,424, or "age in years" ranging from 19 to 75) can be directly used in arithmetic. You can compute their mean, standard deviation (a measure of spread: the average distance of values from their mean), and other statistical moments.

- **Categorical features** (also called nominal features — variables that name membership in a discrete, unordered group, like "purpose of credit" which can be "car", "furniture", "education", etc.) cannot be used in arithmetic directly. The string "car" has no numerical value that can be multiplied. They must first be converted to numbers through a process called encoding (described in Section 5).

The partition into these two disjoint sets (sets with no elements in common) is required because `ColumnTransformer` (Section 5) needs to apply different mathematical transformations to each set.

### Role in This Pipeline

Pandas serves two purposes:

1. **Data ingestion and type detection**: `fetch_openml` (a function that downloads datasets from the OpenML repository — an online platform hosting thousands of standardised machine learning datasets) returns a `DataFrame`. The `.select_dtypes()` calls partition the 20 German Credit features into the numerical and categorical sets that feed separate preprocessing transformers (programs that mathematically transform raw data into a form suitable for learning algorithms).

2. **Output formatting**: The final `results_df` in Block 7 assembles per-sample pipeline outputs into a readable table for inspection.

### Mathematical Prerequisites

- **Sets and partitions**: a partition of a set $S$ is a collection of non-empty, disjoint subsets whose union is $S$
- **Data types**: understanding that "car" and 3 are fundamentally different objects requiring different mathematical treatment

---

## 3. `import matplotlib.pyplot as plt`

### What Matplotlib Is

Matplotlib is a 2D plotting library. `pyplot` is its stateful interface (meaning it maintains an internal record of the "current figure" — the canvas you are drawing on — so you do not need to pass the figure object to every function call). It renders NumPy arrays as visualisations by mapping numerical values to pixel positions on a coordinate system.

### Role in This Pipeline

One specific use: plotting the ACO (Ant Colony Optimisation — a bio-inspired algorithm described in Section 8) convergence curve. The `convergence_curve` array of shape `(100,)` records the best MSE in the archive (the collection of candidate solutions the algorithm maintains) at each of the 100 iterations. 

Plotting this array against iteration number produces a **monotonically non-increasing curve** (monotonically means it only moves in one direction — here, it either stays flat or decreases, never increases, because the archive replacement rule only retains solutions that are at least as good as the worst currently stored). Visually, a steep initial drop that flattens toward a horizontal asymptote (a horizontal line that the curve approaches but never crosses, representing the best solution the algorithm can find) indicates healthy convergence. A curve that never flattens suggests the algorithm has not converged and needs more iterations.

### Mathematical Prerequisites

- **Functions of one variable**: understanding a curve as a visual representation of how one quantity changes as another changes
- **Monotone sequences**: a sequence $a_1, a_2, \ldots$ is monotonically non-increasing if $a_{i+1} \leq a_i$ for all $i$
- **Asymptotic behaviour**: the concept that a sequence can approach a limit without reaching it

---

## 4. `import torch` / `import torch.nn as nn` / `import torch.optim as optim`

### What PyTorch Is

PyTorch is a numerical computing framework built around **automatic differentiation** (also called autograd — a system that can automatically compute the derivative of any function that is expressed as a sequence of basic operations, by applying the chain rule mechanically through a computational graph).

The fundamental object is the `Tensor` — structurally identical to a NumPy array, but augmented with a **computational graph** (a directed acyclic graph — a graph where edges have a direction and there are no cycles, meaning you cannot start at a node and return to it by following edges — that records every operation performed on the tensor so that derivatives can be computed by traversing it in reverse).

When you call `.backward()` on a scalar loss (a single number measuring how wrong the model's predictions are), PyTorch traverses this graph in reverse — from output back to inputs — computing exact derivatives of the loss with respect to every trainable parameter using the **chain rule**:

$$\frac{dL}{dw} = \frac{dL}{dz_n} \cdot \frac{dz_n}{dz_{n-1}} \cdots \frac{dz_2}{dz_1} \cdot \frac{dz_1}{dw}$$

where $z_1, z_2, \ldots, z_n$ are intermediate computed values in the forward pass (the computation of the model output from the input), $L$ is the loss, and $w$ is a parameter deep in the network. Each $\frac{dz_{k+1}}{dz_k}$ is a local derivative (the derivative of one step of the computation with respect to its immediate input), computed and stored during the forward pass. The chain rule says the total derivative of $L$ with respect to $w$ is the product of all these local derivatives along the path from $w$ to $L$.

### `torch.nn` — Neural Network Building Blocks

`torch.nn` provides the components for building differentiable computational graphs:

- **`nn.Module`**: the base class (a template that other classes inherit from, receiving all its methods and properties) for any differentiable computation. It defines `forward()` (the function that computes the model output given an input — this is where you write the mathematical operations of your model), registers `nn.Parameter` objects as trainable leaves (leaf nodes are nodes in the computational graph that have no parents — they are the inputs and parameters, the starting points of the computation), and implements `.parameters()` (an iterator — an object that yields items one by one — that recursively collects all trainable parameters in the module and its submodules).

- **`nn.Parameter`**: a `Tensor` wrapper that signals to the autograd engine that this variable is a trainable quantity. Gradients (derivatives of the loss with respect to this parameter) are accumulated in its `.grad` attribute (a property of the object that stores the computed gradient tensor) after `.backward()` is called, and the optimiser uses these gradients to update the parameter values.

- **`nn.MSELoss`**: computes the Mean Squared Error: $\mathcal{L}(\hat{y}, y) = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$, where $\hat{y}_i$ is the model's prediction for sample $i$ and $y_i$ is the true label. This is used as the training objective for the ANFIS.

### `torch.optim` — Parameter Update Rules

`torch.optim` provides **first-order optimisation algorithms** (algorithms that use only first derivatives — gradients — and not second derivatives — Hessians — to determine how to update parameters) that update `nn.Parameter` tensors.

**Adam** (Adaptive Moment Estimation, Kingma & Ba, 2014) is used in this pipeline. It maintains, for each parameter $\theta$, estimates of the first moment (the mean — the expected value, i.e., the average, of the gradient) and second moment (the uncentered variance — the expected value of the squared gradient, which measures how large the gradients typically are) of the gradient:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(exponential moving average of gradients)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(exponential moving average of squared gradients)}$$

where $g_t = \frac{\partial \mathcal{L}}{\partial \theta}\big|_t$ is the gradient at step $t$, and $\beta_1, \beta_2 \in (0, 1)$ are decay rates (numbers strictly between 0 and 1 that control how quickly old information is forgotten — $\beta_1 = 0.9$ means 90% of the previous average is retained and 10% comes from the new gradient). Because $m_t$ and $v_t$ are initialised to zero, they are biased toward zero early in training; the bias-corrected estimates are:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

(dividing by $1 - \beta^t$ inflates the early estimates to correct for the initialisation bias — when $t=1$, $1-\beta_1^1 = 0.1$, so $\hat{m}_1 = m_1 / 0.1 = 10 m_1$, which is a large correction; as $t$ grows, $\beta^t \to 0$ and the correction vanishes). The parameter update is:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

where $\eta = 0.01$ (the learning rate — a positive scalar controlling the step size; too large and parameters oscillate past the minimum; too small and training is prohibitively slow) and $\epsilon = 10^{-8}$ (a small constant added to prevent division by zero when $\hat{v}_t \approx 0$).

The key advantage of Adam over basic gradient descent: $\sqrt{\hat{v}_t}$ in the denominator scales the effective step size by the historical magnitude of the gradient. Parameters with consistently large gradients get smaller steps (preventing overshooting); parameters with consistently small gradients get larger steps (preventing stagnation). This per-parameter adaptive learning rate handles the heterogeneity (different parameters responding to loss on very different scales) of the ANFIS, where the Gaussian centres (which live in $[0,1]$) and the Sugeno consequent slopes (which can take any real value) have very different gradient scales.

### Role in This Pipeline

Exclusively for the ANFIS (Block 6): defining the `GaussianMF` and `ANFIS1D` classes, training them via backpropagation with Adam, and performing inference (generating predictions on new data) at test time.

### Mathematical Prerequisites

- **Differential calculus**: the derivative $\frac{df}{dx}$ measures the instantaneous rate of change of $f$ with respect to $x$; the partial derivative $\frac{\partial f}{\partial x_i}$ measures the rate of change with respect to one variable while holding all others fixed
- **The chain rule**: $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ — the derivative of a composition is the product of derivatives
- **Gradient vectors**: $\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_d}\right]^\top$ — the vector of all partial derivatives, pointing in the direction of steepest ascent
- **Directed acyclic graphs**: nodes connected by directed edges with no cycles
- **Exponential moving averages**: a weighted average where recent values receive exponentially more weight than older values

---

## 5. `from sklearn.preprocessing import StandardScaler, OneHotEncoder` and `from sklearn.compose import ColumnTransformer`

### `StandardScaler` — Normalising the Numerical Feature Space

`StandardScaler` applies the **z-score transformation** (also called standardisation) column-wise. Given a column vector $\mathbf{x} = [x_1, x_2, \ldots, x_N]^\top$ of $N$ values for one feature:

1. Compute the **sample mean** $\hat{\mu} = \frac{1}{N}\sum_{i=1}^N x_i$ (the arithmetic average of all values — the centre of mass of the distribution)
2. Compute the **sample standard deviation** $\hat{\sigma} = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_i - \hat{\mu})^2}$ (the square root of the average squared deviation from the mean — a measure of how spread out the values are; large $\hat{\sigma}$ means values are far from the mean, small $\hat{\sigma}$ means they cluster tightly)
3. Transform each value: $z_i = \frac{x_i - \hat{\mu}}{\hat{\sigma}}$

After this transformation, the feature has **empirical mean 0** (the average of the transformed values is exactly 0, because we subtracted the mean from every value) and **empirical standard deviation 1** (the spread is scaled to exactly 1, because we divided by the standard deviation).

**Why this is required for gradient-based algorithms:** The **loss surface** (the surface you get by plotting the loss function value — how wrong the model is — against all possible parameter values) has a geometry determined by the **Hessian matrix** (the matrix of all second partial derivatives $H_{ij} = \frac{\partial^2 \mathcal{L}}{\partial w_i \partial w_j}$ — a symmetric matrix that characterises the curvature of the loss surface in every direction). When features have very different scales, the Hessian is **ill-conditioned** (its eigenvalues — the values $\lambda$ such that $Hv = \lambda v$ for some non-zero vector $v$; each eigenvalue measures the curvature in the direction of its corresponding eigenvector — span many orders of magnitude). Gradient descent on an ill-conditioned loss surface oscillates: the gradient points diagonally across steep valleys rather than directly downhill, requiring a very small learning rate and many iterations to converge. Standardisation makes the loss surface more **isotropic** (having the same properties in all directions — from the Greek "isos" meaning equal and "tropos" meaning turn), reducing the condition number (the ratio of the largest eigenvalue to the smallest — a condition number close to 1 means the surface is nearly spherical; a very large condition number means it is highly elongated) and allowing faster, more stable convergence.

**Why this is required for the RBF kernel SVM:** The RBF kernel computes $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$ where $\|x_i - x_j\|^2 = \sum_k (x_{ik} - x_{jk})^2$ is the **squared Euclidean distance** (the sum of squared coordinate differences — the standard notion of "how far apart" two points are in space, generalising Pythagoras' theorem to $d$ dimensions). If one feature (credit amount, range 250–18,424) contributes thousands of units to this sum while another (number of dependents, range 0–6) contributes at most 36, the kernel value is dominated entirely by the credit amount. Standardisation ensures each feature contributes equally to the distance calculation.

### `OneHotEncoder` — Mapping Categories to Orthogonal Vectors

`OneHotEncoder` converts a categorical variable (a variable that names membership in a discrete, unordered group) into a set of binary (0 or 1) indicator variables.

Suppose a feature "credit purpose" has $k = 4$ levels: \{car, furniture, education, repairs\}. We cannot assign integers 1, 2, 3, 4 because that would impose a **false ordinal relationship** (an ordering that does not exist — it would imply "repairs" is arithmetically 4 times "car", which is meaningless). Instead, one-hot encoding creates $k = 4$ new binary columns, one per category:

| car | furniture | education | repairs |
|-----|-----------|-----------|---------|
|  1  |     0     |     0     |    0    |  ← a car loan
|  0  |     1     |     0     |    0    |  ← a furniture loan
|  0  |     0     |     1     |    0    |  ← an education loan

With `drop='first'`, the first column (car) is dropped, representing all 4 categories using only 3 binary columns. This is necessary to avoid the **dummy variable trap**:

When you include all $k$ binary columns, their values always sum to 1 (exactly one category applies to any given sample). This means the columns are **linearly dependent** (one column can be expressed as a linear combination — a weighted sum — of the others: $x_\text{car} = 1 - x_\text{furniture} - x_\text{education} - x_\text{repairs}$). In the context of Logistic Regression, the model's **design matrix** $X$ (the matrix whose rows are the feature vectors of training samples) becomes **rank-deficient** (has a rank — the number of linearly independent rows or columns — less than its number of columns), making $X^\top X$ **singular** (non-invertible — having a zero determinant, meaning the matrix equation $X^\top X \beta = X^\top y$ has no unique solution). Dropping one column removes this dependency and restores full rank.

### `ColumnTransformer` — Applying Different Transforms to Different Feature Subsets

`ColumnTransformer` applies different transformers to different column subsets in **parallel** (simultaneously, not sequentially), then **concatenates** (joins side by side) the results:

$$X_\text{out} = [T_\text{num}(X_\text{numerical}) \; \big| \; T_\text{cat}(X_\text{categorical})]$$

where $|$ denotes horizontal concatenation (placing two matrices next to each other to form one wider matrix), $T_\text{num}$ is `StandardScaler`, and $T_\text{cat}$ is `OneHotEncoder`.

### Mathematical Prerequisites

- **Descriptive statistics**: mean ($\mu$), variance ($\sigma^2 = E[(X-\mu)^2]$), standard deviation ($\sigma$)
- **Linear algebra**: linear independence (a set of vectors is linearly independent if no vector in the set can be written as a linear combination of the others), matrix rank, matrix invertibility, eigenvalues and eigenvectors
- **Euclidean geometry**: the distance formula $d(x, y) = \sqrt{\sum_k (x_k - y_k)^2}$, Pythagoras' theorem

---

## 6. `from sklearn.model_selection import train_test_split, StratifiedKFold` and `from sklearn.pipeline import Pipeline`

### `train_test_split` — Partitioning Data to Prevent Evaluation Bias

`train_test_split` partitions two arrays (features and labels) into random train and test subsets. The `stratify=y` argument enables **stratified sampling** (sampling that preserves the class proportions of the full dataset in each subset).

**Why stratification is necessary:** The German Credit dataset has a **class imbalance** (the two classes — "good" and "bad" — do not appear with equal frequency; "good" accounts for 70% and "bad" for 30%). If we split randomly without stratification, by chance the test set might contain 40% "bad" samples (over-representation — appearing more than in the population) or 20% "bad" samples (under-representation), simply due to the randomness of sampling. This would make test set evaluation statistics unreliable — the model appears better or worse depending on the random split, not on its actual predictive capacity.

Stratified sampling constructs each split by sampling **class-conditionally** (separately within each class): it takes 20% of the 700 "good" samples and 20% of the 300 "bad" samples, guaranteeing the test set contains exactly 140 "good" and 60 "bad" samples — preserving the 70/30 ratio. This ensures the empirical class distribution (the proportion of each class actually observed in the data) in each split is an unbiased estimate (an estimate whose expected value equals the true value being estimated) of the population class distribution (the true underlying proportion of each class in the real world).

### `StratifiedKFold` — Generating Unbiased Out-of-Fold Predictions

`StratifiedKFold` implements **stratified K-fold cross-validation** (a technique for using training data efficiently by training and evaluating a model $K$ times, each time on a different partition of the data).

Given a dataset of $N$ training samples with index set $\mathcal{I} = \{1, 2, \ldots, N\}$, it produces a **partition** (a division into non-overlapping subsets that together cover the whole set) into $K$ folds (subsets) $F_1, F_2, \ldots, F_K$, where each fold maintains the class ratio. For fold $k$, the model trains on $\mathcal{I} \setminus F_k = \bigcup_{j \neq k} F_j$ (all indices except those in fold $k$ — the set-theoretic difference) and generates predictions on $F_k$.

**Out-of-Fold (OOF) predictions:** After all $K$ folds, every training sample $i$ has exactly one prediction $\hat{p}_i$ — generated by a model that was trained on $\mathcal{I} \setminus F_{k(i)}$, where $k(i)$ is the fold containing sample $i$. Crucially, sample $i$ was **not seen during training** of the model that predicted it. Collecting all $N$ predictions produces an **unbiased estimate** (an estimate that does not systematically over- or underestimate the true value) of each model's generalisation performance (how well the model performs on data it has never seen).

These OOF predictions form the matrix $P \in \mathbb{R}^{N \times 3}$ (a matrix with $N$ rows and 3 columns, where entry $P_{ij}$ is the OOF probability predicted by model $j$ for training sample $i$). This matrix is passed to the QP and ACO weight optimisers. Using OOF predictions here is essential: if we used predictions on the training data itself (in-sample predictions), each model would have memorised those samples to some degree, producing overconfidently high probabilities (probabilities closer to 0 or 1 than the true uncertainty warrants). The weight optimiser would then tune weights to exploit this overconfidence rather than genuine discriminative ability, resulting in poor generalisation.

### `Pipeline` — Enforcing the Correct Order of Operations to Prevent Leakage

A `Pipeline` chains a sequence of **transformers** (objects that have `.fit()` and `.transform()` methods — fit learns parameters from data, transform applies the learned transformation) and a **final estimator** (an object with `.fit()` and `.predict()` or `.predict_proba()` methods) into a single object:

$$x \xrightarrow{T_1} T_1(x) \xrightarrow{T_2} T_2(T_1(x)) \xrightarrow{f} \text{prediction}$$

When `pipeline.fit(X_train, y)` is called, each transformer $T_i$ is fitted strictly on the output of $T_{i-1}$ applied to `X_train`. When `pipeline.predict_proba(X_val)` is called, each $T_i$ applies its **already-fitted** parameters without any refitting.

**The leakage problem Pipeline solves:** Without a Pipeline, a naive implementation would:
1. Fit `StandardScaler` on the entire training set (all 800 samples)
2. Transform the entire training set using those statistics
3. Then split into folds for cross-validation

In step 1, the scaler computes $\hat{\mu}$ and $\hat{\sigma}$ using all 800 samples — including the samples that will later serve as the validation fold. When the model trains on 640 samples (K-1 folds) and is evaluated on 160 (1 fold), the validation fold's data has already influenced $\hat{\mu}$ and $\hat{\sigma}$. This is **data leakage** (information from the validation set contaminating the training process, causing the model to appear more accurate than it truly is on unseen data, because it has seen a statistical summary of the validation data during preprocessing). The model's evaluation statistics are then **optimistically biased** (systematically higher than the true generalisation performance).

With `Pipeline`, when `model_pipeline.fit(X_fold_train, y)` is called inside the K-fold loop, the scaler computes $\hat{\mu}$ and $\hat{\sigma}$ **only** on the 640 K-1 fold samples, completely ignorant of the 160 validation samples. This is the mathematically correct procedure: the scaler has never seen the validation fold's distribution.

### Mathematical Prerequisites

- **Probability theory**: random variables, expected value ($E[X] = \sum_x x \cdot P(X=x)$ for discrete variables), unbiasedness ($E[\hat{\theta}] = \theta$, where $\hat{\theta}$ is an estimator and $\theta$ is the true parameter)
- **Set theory**: set difference ($A \setminus B$ = elements in $A$ but not in $B$), partition, union, intersection
- **Statistics**: sampling distributions, bias in estimation

---

## 7. `from sklearn.linear_model import LogisticRegression`

### What Logistic Regression Is

Logistic Regression is a **discriminative probabilistic classifier** (a model that directly learns the conditional probability $P(Y = 1 \mid X = x)$ — the probability of the positive class given the observed features — rather than modelling the class-conditional distributions separately and applying Bayes' theorem). It is classified as a **Generalised Linear Model** (a family of statistical models that relate a linear predictor to the response variable through a **link function** — a monotone, differentiable function that transforms the linear output into the appropriate range for the response), specifically using the **logit link** function.

**The sigmoid (logistic) function:** The model maps a linear combination of features to a probability using:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $e \approx 2.718$ is Euler's number. The sigmoid function maps any real number $z \in (-\infty, +\infty)$ to a value in $(0, 1)$ (the open interval between 0 and 1, excluding the endpoints — the model can approach but never reach exactly 0 or exactly 1 for any finite input). Its shape is an S-curve (hence "sigmoid" from the Greek letter sigma): it is approximately 0 for very negative $z$, rises through 0.5 at $z = 0$, and approaches 1 for very positive $z$.

The model is:

$$P(Y = 1 \mid x; w, b) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}$$

where $w \in \mathbb{R}^d$ is the **weight vector** (the vector of $d$ learnable parameters, one per feature, that determine the relative contribution of each feature to the prediction), $b \in \mathbb{R}$ is the **bias** (also called intercept — a learnable scalar offset that shifts the decision boundary away from the origin), and $x \in \mathbb{R}^d$ is the feature vector of one sample. $w^\top x = \sum_{j=1}^d w_j x_j$ is the **inner product** (also called dot product — the sum of element-wise products of two vectors, producing a scalar).

**The decision boundary** (the set of points in feature space where the model is exactly 50% certain, i.e., $P(Y=1 \mid x) = 0.5$) is the **hyperplane** (a flat affine subspace of dimension $d-1$ in a $d$-dimensional space — in 2D it is a line, in 3D it is a plane) $\{x : w^\top x + b = 0\}$, since $\sigma(0) = 0.5$.

### Maximum Likelihood Estimation

The weight vector $w$ and bias $b$ are learned by **Maximum Likelihood Estimation (MLE)** — the principle of choosing parameter values that maximise the probability (likelihood) of observing the training data, assuming the model is correct.

The **likelihood** of the training data $\{(x_i, y_i)\}_{i=1}^N$ given parameters $(w, b)$ is:

$$\mathcal{L}(w, b) = \prod_{i=1}^N P(Y = y_i \mid x_i; w, b) = \prod_{i=1}^N \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

where $\hat{p}_i = \sigma(w^\top x_i + b)$ and the product structure assumes **i.i.d. samples** (independent and identically distributed — each sample is drawn independently from the same distribution, a standard statistical assumption). Taking the **logarithm** (a strictly increasing function, so maximising the log-likelihood is equivalent to maximising the likelihood — this is valid because the logarithm is monotone increasing) converts the product into a sum (using $\log \prod_i a_i = \sum_i \log a_i$), yielding the **log-likelihood**:

$$\ell(w, b) = \sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1-y_i)\log(1 - \hat{p}_i) \right]$$

Maximising this is equivalent to minimising the **binary cross-entropy loss** (also called log-loss):

$$\mathcal{L}(w, b) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1-y_i)\log(1 - \hat{p}_i) \right]$$

The negation converts maximisation to minimisation (a standard convention in optimisation). The loss has an intuitive interpretation: for a sample with $y_i = 1$ (true positive), the loss is $-\log \hat{p}_i$, which is 0 when $\hat{p}_i = 1$ (perfect prediction) and approaches $+\infty$ as $\hat{p}_i \to 0$ (catastrophically wrong prediction). The logarithm provides an asymmetric penalty: being very wrong is punished severely.

**`class_weight='balanced'`:** This reweights each sample's contribution to the loss by $\frac{N}{2 N_{y_i}}$, where $N_{y_i}$ is the count of class $y_i$. For the 70/30 split: "good" samples receive weight $\frac{1000}{2 \times 700} \approx 0.714$ and "bad" samples receive $\frac{1000}{2 \times 300} \approx 1.667$. This up-weights (assigns greater importance to) the minority class ("bad" — defaulters), preventing the model from ignoring them in favour of the majority class.

### Mathematical Prerequisites

- **Probability theory**: random variables, probability distributions, the Bernoulli distribution (a distribution over $\{0,1\}$ with parameter $p$ — $P(X=1)=p$, $P(X=0)=1-p$)
- **Calculus**: logarithms and their properties ($\log(ab) = \log a + \log b$, $\log(a^b) = b \log a$), derivatives
- **Linear algebra**: inner products, hyperplanes
- **Optimisation**: gradient descent, convex functions (a function where the line segment between any two points on the graph lies above or on the graph — the loss surface of Logistic Regression is convex, guaranteeing a unique global minimum)

---

## 8. `from sklearn.svm import SVC`

### What a Support Vector Classifier Is

An SVC (Support Vector Classifier) finds the **maximum-margin hyperplane** separating two classes. The core idea: among all hyperplanes that correctly separate the training data, choose the one that is furthest from the nearest training points of either class — this maximises the **margin** (the perpendicular distance from the hyperplane to the closest point of each class on either side).

**Why maximise the margin?** Intuitively, a decision boundary that passes very close to some training points is sensitive to small perturbations — slightly shifting those points could change the boundary entirely. A maximum-margin boundary is more robust to such noise. Formally, the **VC theory** (Vapnik-Chervonenkis theory — a mathematical framework for bounding the generalisation error of classifiers in terms of their complexity) provides a bound showing that the generalisation error decreases as the margin increases.

**The primal problem** (direct formulation in terms of the hyperplane parameters):

$$\min_{w, b} \frac{1}{2}\|w\|_2^2 \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1 \quad \forall i \in \{1, \ldots, N\}$$

where $\|w\|_2^2 = \sum_j w_j^2$ is the **squared L2 norm** (the sum of squared components of $w$ — also equal to $w^\top w$) and the constraint $y_i(w^\top x_i + b) \geq 1$ (using labels $y_i \in \{-1, +1\}$ in the SVM convention) enforces correct classification with a margin of at least $\frac{2}{\|w\|}$ (the margin is $\frac{2}{\|w\|}$ because the constraint forces the hyperplane to be at distance at least $\frac{1}{\|w\|}$ from each class). Minimising $\|w\|^2$ is equivalent to maximising the margin.

**Soft-margin SVM:** Real data is not linearly separable (there is no hyperplane that perfectly separates all training points). Soft-margin introduces **slack variables** $\xi_i \geq 0$ (non-negative scalars that measure how much sample $i$ violates the margin constraint — $\xi_i = 0$ if the sample is correctly classified and outside the margin, $0 < \xi_i \leq 1$ if inside the margin, $\xi_i > 1$ if misclassified) and a regularisation parameter $C > 0$ (the penalty for margin violations; large $C$ allows few violations but risks overfitting; small $C$ allows more violations but produces a larger, more robust margin):

$$\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C \sum_{i=1}^N \xi_i \quad \text{subject to} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**The Lagrangian Dual:** Via **Lagrangian duality** (a technique that converts a constrained optimisation problem into an unconstrained one by adding penalty terms for constraint violations, multiplied by **Lagrange multipliers** — non-negative scalars that represent the "shadow price" of each constraint), the problem becomes:

$$\max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle$$

subject to $0 \leq \alpha_i \leq C$ and $\sum_i \alpha_i y_i = 0$, where $\langle x_i, x_j \rangle = x_i^\top x_j$ is the inner product. Notice the data appears **only through inner products** — this is the key observation enabling the kernel trick.

**The Kernel Trick:** Replacing $\langle x_i, x_j \rangle$ with a **kernel function** $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_\mathcal{H}$ (the inner product in a high-dimensional feature space $\mathcal{H}$, computed via $\phi$ — a non-linear mapping from the original space into $\mathcal{H}$) allows the SVM to find non-linear boundaries **without explicitly computing $\phi$**. By **Mercer's theorem** (a result from functional analysis — the branch of mathematics studying infinite-dimensional vector spaces and operators on them — stating that any symmetric, positive semi-definite function can be expressed as an inner product in some Hilbert space — a complete inner product space generalising Euclidean geometry to infinite dimensions), any function $K$ satisfying certain conditions is a valid kernel.

The **RBF kernel** used here: $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$. This implicitly maps data into an **infinite-dimensional** feature space (a vector space with infinitely many dimensions — the kernel corresponds to an inner product in the space of all square-integrable functions, which is infinite-dimensional). In this space, any finite dataset is linearly separable.

**`probability=True`** enables **Platt scaling** (a post-hoc calibration technique: after training the SVM, a logistic regression is fitted on the SVM's **decision function** outputs — the signed distances from the decision boundary, $f(x) = w^\top \phi(x) + b$ — to map them to calibrated probabilities $P(Y=1 \mid x)$). This is necessary because the SVM's decision function is not a probability — it is a raw distance that can take any real value.

### Mathematical Prerequisites

- **Vector geometry**: inner products, orthogonality, hyperplanes, signed distances
- **Convex optimisation**: Lagrangian functions, duality theory, KKT conditions
- **Functional analysis**: Hilbert spaces, Mercer's theorem, reproducing kernel Hilbert spaces

---

## 9. `from sklearn.neural_network import MLPClassifier`

### What a Multilayer Perceptron Is

An MLP (Multilayer Perceptron) is a **feedforward artificial neural network** (a network where information flows in one direction — from input to output — with no feedback loops; "artificial" because it is a computational model inspired by biological neural networks, not a biological system; "neural" because it is composed of interconnected processing units called neurons, loosely analogous to biological neurons).

**Architecture:** An MLP consists of:
- An **input layer** (not a processing layer — it simply holds the $d$ input features $x_1, \ldots, x_d$)
- One or more **hidden layers** (intermediate processing layers whose outputs are not directly observed; "hidden" because they are neither the input nor the output)
- An **output layer** (produces the final prediction)

For this pipeline's architecture — one hidden layer with $H = 64$ neurons:

**Hidden layer computation:**
$$h = \text{ReLU}(W^{(1)} x + b^{(1)})$$

where $W^{(1)} \in \mathbb{R}^{64 \times d}$ (a matrix with 64 rows and $d$ columns, $d$ being the number of features after preprocessing) and $b^{(1)} \in \mathbb{R}^{64}$ are learnable parameters.

**ReLU (Rectified Linear Unit):** $\text{ReLU}(z) = \max(0, z)$, applied element-wise. ReLU is preferred over older activations (sigmoid, tanh) because:
- It does not **saturate** (reach a region where the derivative is near zero — causing the **vanishing gradient problem**: gradients near zero propagate essentially zero signal back through the network, stalling learning) for positive inputs
- It is computationally cheap (a simple threshold operation)
- It produces **sparse activations** (most neurons output exactly 0 for any given input, since half the real line maps to 0 — sparsity is computationally efficient and acts as a form of regularisation)

**Output layer computation (binary classification):**
$$\hat{p} = \sigma(W^{(2)} h + b^{(2)})$$

where $W^{(2)} \in \mathbb{R}^{1 \times 64}$ and $b^{(2)} \in \mathbb{R}$ are learnable parameters, and $\sigma$ is the sigmoid function.

**Universal Approximation Theorem** (Cybenko 1989, Hornik et al. 1991): Any continuous function $f: \mathbb{R}^d \to \mathbb{R}$ on a compact (closed and bounded) subset of $\mathbb{R}^d$ can be approximated to arbitrary precision by a feedforward network with a single hidden layer containing a sufficiently large (but finite) number of neurons using a non-polynomial activation. This provides theoretical justification that the MLP has sufficient expressive capacity for this task, regardless of how complex the true decision boundary is.

**Training:** The MLP is trained to minimise the binary cross-entropy loss using the **Adam optimiser** (described in Section 4) and **backpropagation** (described in Section 4). The total number of learnable parameters is $64d + 64 + 64 + 1 = 64(d+1) + 65$ (64 rows of $W^{(1)}$ each with $d$ weights plus a bias, then one output neuron with 64 weights plus a bias).

### Mathematical Prerequisites

- **Linear algebra**: matrix-vector products, composition of linear maps
- **Calculus**: chain rule, partial derivatives
- **Optimisation**: gradient descent, Adam
- **Functional analysis**: function approximation, compactness

---

## 10. `from sklearn.base import clone`

### What `clone` Does

`clone(estimator)` creates a **new estimator object** (a fresh instance of the same class) with identical **hyperparameters** (configuration parameters set before training, such as `max_iter=1000` or `kernel='rbf'` — distinct from **parameters** which are learned from data, such as the weight vector $w$) but with **no fitted state** (no learned parameters, no stored statistics from previous calls to `.fit()`).

Internally, `clone` calls `estimator.get_params(deep=True)` (which returns a dictionary — a key-value mapping — of all hyperparameter names and values), then instantiates a fresh object using those values as constructor arguments.

**Why this is structurally required:** In Python, when you write:

```python
base_models = {
    'LR':  Pipeline([('prep', preprocessor), ('clf', LogisticRegression())]),
    'SVM': Pipeline([('prep', preprocessor), ('clf', SVC())]),
}
```

both `'LR'` and `'SVM'` pipelines contain a **reference** (a pointer to the same memory address — in Python, variable names are labels attached to objects in memory, not containers for the objects themselves; two labels can point to the same object) to the same `preprocessor` object. When `LR_pipeline.fit(X_fold_1)` is called, it calls `preprocessor.fit(X_fold_1)`, which writes the computed $\hat{\mu}$ and $\hat{\sigma}$ values into the `preprocessor` object's internal state. When `SVM_pipeline.fit(X_fold_2)` is subsequently called, it calls the same `preprocessor.fit(X_fold_2)`, **overwriting** the previously computed values. The LR pipeline now points to a scaler fitted on `X_fold_2`, not on `X_fold_1`.

`clone(preprocessor)` creates three independent objects at three distinct memory addresses, so each pipeline's state mutations are isolated.

### Mathematical Prerequisites

- **Computer science**: memory model, references vs. values, object mutability (whether an object can be modified after creation)

---

## 11. `from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve`

### The Confusion Matrix — Foundation of All Classification Metrics

For binary classification with predicted labels $\hat{y}_i \in \{0, 1\}$ and true labels $y_i \in \{0, 1\}$:

| | Predicted 0 (Good) | Predicted 1 (Bad) |
|---|---|---|
| **True 0 (Good)** | TN (True Negative) | FP (False Positive) |
| **True 1 (Bad)** | FN (False Negative) | TP (True Positive) |

- **TP (True Positive):** Model predicted "bad risk" and was correct — a defaulter correctly identified
- **TN (True Negative):** Model predicted "good risk" and was correct — a creditworthy customer correctly approved
- **FP (False Positive):** Model predicted "bad risk" but was wrong — a creditworthy customer incorrectly rejected (Type I error)
- **FN (False Negative):** Model predicted "good risk" but was wrong — a defaulter incorrectly approved (Type II error); in the German Credit cost matrix, this costs 5× a FP

### `precision_score` — Exactness

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all samples predicted as "bad risk", what fraction truly were? High precision means the model rarely raises false alarms. In practice, improving precision (raising the threshold $\tau$) typically reduces recall.

### `recall_score` — Completeness

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all true "bad risk" samples, what fraction did the model catch? High recall means few defaulters slip through. This is the critical metric in credit risk given the 5:1 cost asymmetry — missing a defaulter (FN) is far more expensive than incorrectly rejecting a good customer (FP).

### `f1_score` — Balanced Metric

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The **harmonic mean** (for two positive numbers $a$ and $b$, the harmonic mean is $\frac{2ab}{a+b}$, always $\leq$ the **arithmetic mean** $\frac{a+b}{2}$, with equality only when $a = b$) of precision and recall. The harmonic mean is used rather than the arithmetic mean because it is **dominated by the smaller of the two values** — if either precision or recall is near zero, $F_1$ is near zero, even if the other is near 1. This prevents a degenerate classifier (one that predicts all samples as positive, achieving recall=1 but precision≈0.3) from appearing good.

### `roc_auc_score` — Threshold-Free Discrimination

The ROC (Receiver Operating Characteristic — terminology inherited from radar signal detection theory, where operators had to decide whether a received signal represented a real target) curve plots:

$$\text{TPR}(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)} \quad \text{vs.} \quad \text{FPR}(\tau) = \frac{FP(\tau)}{FP(\tau) + TN(\tau)}$$

as $\tau$ sweeps from 1 (predict all negative) to 0 (predict all positive). TPR (True Positive Rate — also called sensitivity or recall) is plotted on the y-axis; FPR (False Positive Rate — the fraction of true negatives incorrectly predicted as positive, also called the fall-out) on the x-axis.

The AUC (Area Under the Curve — computed as $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt$, the definite integral from 0 to 1) has a **probabilistic interpretation**: AUC = $P(\hat{p}(x^+) > \hat{p}(x^-))$ where $x^+$ is a randomly drawn positive sample and $x^-$ a randomly drawn negative sample. A random classifier has AUC = 0.5 (because it has no discriminative power, so $P(\hat{p}(x^+) > \hat{p}(x^-)) = 0.5$); a perfect classifier has AUC = 1.

AUC is **threshold-invariant** (it summarises performance across all possible thresholds) and **scale-invariant** (it measures the ranking quality of predicted probabilities, not their absolute values). This makes it the primary metric for comparing probability-outputting classifiers before threshold selection.

### `precision_recall_curve` — Calibrated Threshold Selection

Returns `(precisions, recalls, thresholds)` by sweeping $\tau$ from high to low. Used in `calibrate_threshold` to find:

$$\tau^* = \arg\max_\tau F_1(\tau)$$

the threshold that maximises the $F_1$ score on the calibration set.

**Critical indexing note:** sklearn returns arrays where `len(precisions) == len(recalls) == len(thresholds) + 1`. The final element represents the **degenerate case** $\tau > \max_i \hat{p}_i$ (the threshold exceeds every predicted probability, so the classifier predicts no positives — precision is defined as 1 by convention when there are no predictions, but there is no corresponding threshold value). The `[:-1]` slice (removing the last element) aligns the arrays before computing $F_1$ at each threshold.

### Mathematical Prerequisites

- **Set theory**: understanding TP/TN/FP/FN as counts of elements in set intersections
- **Probability theory**: conditional probabilities, expected values
- **Calculus**: definite integral (for AUC interpretation)
- **Means**: arithmetic mean $\frac{a+b}{2}$, harmonic mean $\frac{2ab}{a+b}$, and their comparison

---

## 12. `from scipy.optimize import minimize`

### What SLSQP Is

`minimize` with `method='SLSQP'` solves **constrained nonlinear programming problems**. In this pipeline it solves the **Quadratic Programming (QP)** problem (an optimisation problem where the objective function is quadratic — a polynomial of degree 2 — and the constraints are linear):

$$\min_{w \in \mathbb{R}^3} \frac{1}{2} w^\top Q w + c^\top w \quad \text{subject to} \quad \mathbf{1}^\top w = 1, \quad w \geq 0$$

where $\mathbf{1} = [1, 1, 1]^\top$ is the vector of ones, $\mathbf{1}^\top w = w_1 + w_2 + w_3 = 1$ is the **simplex constraint** (forcing the weights to form a probability distribution — sum to 1 and be non-negative), and $Q = 2P^\top P \in \mathbb{R}^{3 \times 3}$, $c = -2P^\top y \in \mathbb{R}^3$.

This is **strictly convex** (the Hessian of the objective, $Q$, is **positive semi-definite** — a symmetric matrix $M$ is positive semi-definite if $v^\top M v \geq 0$ for all vectors $v \in \mathbb{R}^n$ — in fact positive definite if $P$ has full column rank, meaning no column is a linear combination of the others), guaranteeing a unique global minimum.

**SLSQP operates by:** At each iteration, constructing a **quadratic approximation** (a second-order Taylor expansion — $f(x + d) \approx f(x) + \nabla f(x)^\top d + \frac{1}{2} d^\top H d$ where $H$ is the Hessian — of the objective) and a **linear approximation** (first-order Taylor expansion) of the constraints, solving the resulting quadratic subproblem to find a step direction $d$, then performing a **line search** (an algorithm that selects a step length along $d$ satisfying the Armijo sufficient decrease condition: $f(x + \alpha d) \leq f(x) + c_1 \alpha \nabla f(x)^\top d$ for some $c_1 \in (0,1)$) to ensure progress.

**KKT conditions** (Karush-Kuhn-Tucker — first-order necessary conditions for a local minimum of a constrained optimisation problem, named after Harold Karush who derived them in 1939 and William Karush, Harold Kuhn, and Albert Tucker who independently published them in 1951): at the optimum $w^*$, the following must hold:

1. **Stationarity:** $\nabla_w \mathcal{L} = Q w^* + c + \lambda \mathbf{1} - \mu = 0$ (the gradient of the Lagrangian — the objective plus constraint penalties — is zero)
2. **Primal feasibility:** $\mathbf{1}^\top w^* = 1$ and $w^* \geq 0$
3. **Dual feasibility:** $\mu \geq 0$ (the multipliers for inequality constraints are non-negative)
4. **Complementary slackness:** $\mu_i w_i^* = 0$ for each $i$ (if weight $i$ is non-zero, its non-negativity constraint is not active and its multiplier is zero; if the constraint is active meaning $w_i^* = 0$, the multiplier can be positive)

`assert result.success` verifies that SLSQP satisfied these conditions to within `ftol=1e-9` (a tolerance of $10^{-9}$, meaning the KKT residual is less than one billionth).

### Mathematical Prerequisites

- **Multivariable calculus**: gradients, Hessian matrices, Taylor expansions
- **Linear algebra**: positive (semi-)definite matrices, quadratic forms
- **Optimisation theory**: convexity, Lagrangian duality, KKT conditions, line search

---

## Appendix: Complete Conceptual Dependency Map

```
MATHEMATICAL PREREQUISITES
══════════════════════════

Real Analysis & Calculus          Linear Algebra              Probability Theory
─────────────────────────         ──────────────              ──────────────────
Derivatives, chain rule      →    Matrices, inner products → Distributions, MLE
Gradients, Hessians          →    Eigenvalues, norms       → Likelihood, Bayes
Convexity, optimality        →    Rank, invertibility      → Cross-entropy loss
                                  Positive definiteness    → Imbalance, priors

                    ↓                      ↓                       ↓

COMPUTATIONAL INFRASTRUCTURE
═════════════════════════════

numpy         →  All matrix algebra (P, Q, c, w, MSE)
pandas        →  Data loading, feature type detection
scipy         →  QP solver (SLSQP)
torch         →  Automatic differentiation for ANFIS
matplotlib    →  Convergence curve visualisation
sklearn.base  →  clone() for memory isolation

                    ↓

DATA PREPARATION LAYER
═══════════════════════

StandardScaler    →  Isotropic loss surface for gradient methods
OneHotEncoder     →  Numerical representation of categorical variables
ColumnTransformer →  Parallel application to feature subsets
Pipeline          →  Encapsulates transforms to prevent leakage
clone             →  Isolated preprocessor state per model
train_test_split  →  Stratified 80/20 partition
StratifiedKFold   →  Unbiased OOF prediction generation

                    ↓

BASE MODEL LAYER
═════════════════

LogisticRegression  →  Linear discriminative classifier (convex MLE)
SVC (RBF kernel)    →  Maximum-margin classifier (kernel Hilbert space)
MLPClassifier       →  Non-linear approximator (backpropagation)

                    ↓

WEIGHT OPTIMISATION LAYER
══════════════════════════

scipy.minimize  →  QP exact solution: w_QP (convex benchmark)
numpy (ACO_R)   →  Swarm heuristic: w_ACO (approximation)

                    ↓

SOFT COMPUTING LAYER
═════════════════════

torch / nn / optim  →  ANFIS: probability → linguistic label

                    ↓

EVALUATION LAYER
═════════════════

precision_score, recall_score, f1_score  →  Threshold-dependent metrics
roc_auc_score                            →  Threshold-independent ranking
precision_recall_curve                   →  Optimal threshold calibration
confusion_matrix                         →  Domain cost computation
```

