from bidict import bidict

model_name_mapping = bidict({
    'neohookean': 'Neo-Hookean',
    'mooney_rivlin': 'Mooney Rivlin',
    'generalized_mooney_rivlin': 'Generalized Mooney Rivlin',
    'beda': 'Beda',
    'yeoh': 'Yeoh',
    'gent': 'Gent',
    # 'gent_gent': 'Gent Gent',
    # 'mod_gent_gent': 'Mod Gent Gent',
    'carroll': 'Carroll'
})

error_functions_name = bidict({
    "R_abs_err_P": "Absolute error in P",
    "R_abs_err_sigma": "Absolute error in σ",  # σ is sigma symbol
    "R_rel_err": "Relative error"
})

if __name__ == "__main__":
    print(*list(model_name_mapping.values()))
