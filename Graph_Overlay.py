import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = {
    "Scmr": "S_cmr",
    "Stdt": "S_tdt",
    "Syfp": "S_yfp",
    "Snp": "S_np",
    "Ctdt": "C_tdt",
    "Cyfp": "C_yfp",
    "Cnp": "C_np",
    "Rcmr": "R_cmr",
    "Rtdt": "R_tdt",
    "Ryfp": "R_yfp",
    "Rnp": "R_np",
}

Y_NAMES = np.array(["S_tdt_1", "S_cmr_1", "C_tdt_1", "R_cmr_1", "R_tdt_1", "C_cmr_1", "S_tdtR_yfp_1", "R_tdtS_yfp_1",
            "S_tdt_2", "S_cmr_2", "C_tdt_2", "R_cmr_2", "R_tdt_2", "C_cmr_2", "S_tdtR_yfp_2", "R_tdtS_yfp_2",
            "S_tdt_3", "S_cmr_3", "C_tdt_3", "R_cmr_3", "R_tdt_3", "C_cmr_3", "S_tdtC_yfp", "C_tdtS_yfp",
            "S_yfp_1", "S_np_1", "C_yfp_1", "C_np_1", "R_yfp_1", "R_np_1", "R_tdtC_yfp", "C_tdtR_yfp",
            "S_yfp_2", "S_np_2", "C_yfp_2", "C_np_2", "R_yfp_2", "R_np_2", "S_tdtS_yfp", "C_tdtC_yfp",
            "S_yfp_3", "S_np_3", "C_yfp_3", "C_np_3", "R_yfp_3", "R_np_3", "R_tdtR_yfp"
            ])


x = df.iloc[1,1:].to_numpy(dtype=float)
y = (df.iloc[3:50,1:] - df.iloc[50,1:]).to_numpy(dtype=float)
