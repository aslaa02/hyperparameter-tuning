
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Literal, List
import uvicorn

# --------------------- ANFIS CLASS DEFINITIONS (Must match training) ---------------------
def inv_sigmoid(y, min_val, max_val):
    """Inverse sigmoid mapping from bounded to unbounded space."""
    p = (y - min_val) / (max_val - min_val)
    return np.log(p / (1 - p))

class GaussianMF(nn.Module):
    """Gaussian membership function with sigmoid reparameterization."""
    def __init__(self, centers, sigmas):
        super().__init__()
        self.min_c, self.max_c = 0.0, 1.0
        self.min_s, self.max_s = 0.05, 0.3

        raw_centers = [inv_sigmoid(c, self.min_c, self.max_c) for c in centers]
        self._raw_centers = nn.Parameter(torch.tensor(raw_centers, dtype=torch.float32))

        raw_sigmas = [inv_sigmoid(s, self.min_s, self.max_s) for s in sigmas]
        self._raw_sigmas = nn.Parameter(torch.tensor(raw_sigmas, dtype=torch.float32))

    @property
    def centers(self):
        return self.min_c + (self.max_c - self.min_c) * torch.sigmoid(self._raw_centers)

    @property
    def sigmas(self):
        return self.min_s + (self.max_s - self.min_s) * torch.sigmoid(self._raw_sigmas)

    def forward(self, x):
        return torch.exp(-0.5 * ((x - self.centers) / self.sigmas) ** 2)

class ANFISModel(nn.Module):
    """PyTorch module for the ANFIS network."""
    def __init__(self, n_rules=3):
        super().__init__()
        self.fuzzify = GaussianMF([0.1, 0.5, 0.9], [0.2, 0.2, 0.2])
        self.a = nn.Parameter(torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(3, dtype=torch.float32))

    def forward(self, x):
        w = self.fuzzify(x)
        w_norm = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        f = x * self.a + self.b
        return (w_norm * f).sum(dim=1)

class ANFIS(BaseEstimator, TransformerMixin):
    """Adaptive Neuro‑Fuzzy Inference System as a scikit‑learn estimator."""
    def __init__(self, n_rules=3, lr=0.01, epochs=500, random_state=42):
        self.n_rules = n_rules
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.model_ = ANFISModel(n_rules=self.n_rules)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32).view(-1, 1)
        y_t = torch.tensor(y, dtype=torch.float32)

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = nn.MSELoss()(self.model_(X_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def transform(self, X):
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).view(-1, 1)
            return self.model_(X_t).numpy()

    def predict(self, X):
        return self.transform(X)

# --------------------- Load model artifacts ---------------------
artifacts = joblib.load('credit_risk_model.pkl')
trained_pipelines = artifacts['trained_pipelines']
w_aco = artifacts['w_aco']
t_aco = artifacts['t_aco']
anfis_params = artifacts['anfis_params']
feature_names = artifacts['feature_names']
numerical_cols = artifacts['numerical_cols']
categorical_cols = artifacts['categorical_cols']
cat_indices = artifacts['cat_indices']

# Load ANFIS state dict and reconstruct the model
anfis_state_dict = torch.load('anfis_state_dict.pth', map_location='cpu')
anfis = ANFIS(**anfis_params)                 # create a new instance
anfis.model_ = ANFISModel(n_rules=anfis_params['n_rules'])  # create the internal model
anfis.model_.load_state_dict(anfis_state_dict)               # load weights
anfis.model_.eval()                           # set to evaluation mode

# --------------------- Pydantic model for input validation ---------------------
class CreditApplication(BaseModel):
    checking_status: Literal['<0', '0<=X<200', '>=200', 'no checking']
    duration: int = Field(ge=1, le=72)
    credit_history: Literal['no credits/all paid', 'all paid', 'existing paid', 'delayed previously', 'critical/other existing credit']
    purpose: Literal['new car', 'used car', 'furniture/equipment', 'radio/tv', 'appliances', 'repair', 'education', 'vacation', 'retraining', 'business', 'other']
    credit_amount: float = Field(gt=0)
    savings_status: Literal['<100', '100<=X<500', '500<=X<1000', '>=1000', 'no savings account']
    employment: Literal['unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']
    installment_commitment: int = Field(ge=1, le=4)
    personal_status: Literal['male div/sep', 'female div/sep/mar', 'male single', 'male mar/wid', 'female single']
    other_parties: Literal['none', 'co-applicant', 'guarantor']
    residence_since: int = Field(ge=1, le=4)
    property_magnitude: Literal['real estate', 'life insurance', 'car', 'no property']
    age: int = Field(ge=18, le=100)
    other_payment_plans: Literal['bank', 'stores', 'none']
    housing: Literal['rent', 'own', 'for free']
    existing_credits: int = Field(ge=1, le=4)
    job: Literal['unemp/unskilled non-res', 'unskilled resident', 'skilled', 'high qualif/self-emp/mgmt']
    num_dependents: int = Field(ge=1, le=2)
    own_telephone: Literal['none', 'yes']
    foreign_worker: Literal['yes', 'no']

    @validator('duration', 'installment_commitment', 'residence_since', 'age', 'existing_credits', 'num_dependents')
    def check_positive(cls, v):
        if v < 0:
            raise ValueError('must be non-negative')
        return v

# --------------------- FastAPI app ---------------------
app = FastAPI(title="Credit Risk Assessment API")

@app.post("/predict", response_model=dict)
async def predict(application: CreditApplication):
    # Convert input to DataFrame
    input_dict = application.dict()
    df = pd.DataFrame([input_dict])

    # Ensure correct dtypes
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col])

    # Generate predictions from base models
    P = np.zeros((1, len(trained_pipelines)))
    for i, (name, pipeline) in enumerate(trained_pipelines.items()):
        P[0, i] = pipeline.predict_proba(df)[0, 1]

    # Aggregate with ACO weights
    final_prob = float(P @ w_aco)
    decision = "REJECT" if final_prob >= t_aco else "APPROVE"

    # ANFIS linguistic output
    linguistic = float(anfis.transform([[final_prob]])[0])

    return {
        "risk_probability": final_prob,
        "decision": decision,
        "linguistic_score": linguistic
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
