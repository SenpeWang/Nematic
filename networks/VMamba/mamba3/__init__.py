# Mamba3 implementations
# SISO (triton backend)
from .triton.mamba3_siso_combined import mamba3_siso_combined

# MIMO (tilelang backend) - import on demand
def get_mamba3_mimo():
    from .tilelang.mamba3_mimo import mamba3_mimo
    return mamba3_mimo
