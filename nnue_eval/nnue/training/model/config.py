from dataclasses import dataclass


# 3 layer fully connected network
@dataclass
class ModelConfig:
    L1: int = 3072
    L2: int = 31  # 2x capacity for 10B dataset (L2+1=32, divisible by 16)
    L3: int = 32  # Standard hidden layer size


# parameters needed for the definition of the loss
@dataclass
class LossParams:
    in_offset: float = 270
    out_offset: float = 270
    in_scaling: float = 340
    out_scaling: float = 380
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
