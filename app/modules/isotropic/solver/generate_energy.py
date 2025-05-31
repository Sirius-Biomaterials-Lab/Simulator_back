def build_energy_content(model_name, optimization_params):
    """
    Генерирует строку с описанием .energy-файла,
    в зависимости от выбранной модели и массива оптимизированных параметров.
    Здесь можно задавать свои формулы, объявления Var/Let и т.д.
    """

    # Ниже – примеры шаблонов. Подстройте под ваши реальные обозначения и уравнения.
    # Имена I[1], I[2] предположим, что соответствуют I1 и I2 в софте, куда подключаем .energy.
    # Пример, который вы приводили, скорее напоминает Mooney-Rivlin, но можно свободно менять:
    #
    # Var c0 : "f:c0" = 5.95e6, #[kPa]
    #     c1 = 1.48e-3, c2;     #[1]
    # Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];
    # Potential = c0 * (I3d_1 - 3) + c1 * (I3d_2 - 3);
    #
    # Далее – примеры для разных моделей.

    if model_name == "Neohookean":
        # Здесь 1 параметр: mu
        mu = optimization_params[0]
        content = f"""# Neohookean .energy

Var mu : "f:mu" = {mu:.6g};  # [MPa or kPa, зав. от единиц]

Let I1 = I[1];
Potential = mu/2 * (I1 - 3);

"""
        return content

    elif model_name == "Mooney-Rivlin":
        # Здесь 2 параметра: c1, c2
        c1, c2 = optimization_params
        content = f"""# Mooney-Rivlin .energy

Var c1 : "f:c1" = {c1:.6g}, c2 : "f:c2" = {c2:.6g};  # [MPa?]
Let I1 = I[1], I2 = I[2];

Potential = c1*(I1 - 3) + c2*(I2 - 3);

"""
        return content

    elif model_name == "Generalized Mooney-Rivlin":
        # Параметров 5: C10, C01, C11, C20, C02
        C10, C01, C11, C20, C02 = optimization_params
        content = f"""# Generalized Mooney-Rivlin .energy

Var C10 : "f:C10" = {C10:.6g},
    C01 : "f:C01" = {C01:.6g},
    C11 : "f:C11" = {C11:.6g},
    C20 : "f:C20" = {C20:.6g},
    C02 : "f:C02" = {C02:.6g};

Let I1 = I[1], I2 = I[2];

Potential = C10*(I1 - 3)
          + C01*(I2 - 3)
          + C11*(I1 - 3)*(I2 - 3)
          + C20*(I1 - 3)^2
          + C02*(I2 - 3)^2;

"""
        return content

    elif model_name == "Beda":
        # 7 параметров
        c1, c2, c3, K1, alpha, ksi, beta = optimization_params
        content = f"""# Beda model .energy

Var c1 : "f:c1" = {c1:.6g},
    c2 : "f:c2" = {c2:.6g},
    c3 : "f:c3" = {c3:.6g},
    K1 : "f:K1" = {K1:.6g},
    alpha : "f:alpha" = {alpha:.6g},
    ksi : "f:ksi" = {ksi:.6g},
    beta : "f:beta" = {beta:.6g};

Let I1 = I[1], I2 = I[2];

# (Примерно) Potential = (c1/alpha)*(I1 - 3)^alpha + ...

Potential = (c1/alpha)*(I1 - 3)^alpha
          + c2*(I2 - 3)
          + (c3/ksi)*(I1 - 3)^ksi
          + (K1/beta)*(I2 - 3)^beta;

"""
        return content

    elif model_name == "Yeoh":
        # 3 параметра
        c1, c2, c3 = optimization_params
        content = f"""# Yeoh .energy

Var c1 : "f:c1" = {c1:.6g},
    c2 : "f:c2" = {c2:.6g},
    c3 : "f:c3" = {c3:.6g};

Let I1 = I[1];

Potential = c1*(I1 - 3) + c2*(I1 - 3)^2 + c3*(I1 - 3)^3;

"""
        return content

    elif model_name == "Gent":
        mu, Jm = optimization_params
        content = f"""# Gent .energy

Var mu : "f:mu" = {mu:.6g},
    Jm : "f:Jm" = {Jm:.6g};

Let I1 = I[1];

Potential = -(mu * Jm / 2)*ln(1 - (I1 - 3)/Jm);

"""
        return content

    elif model_name == "Carroll":
        A, B, C = optimization_params
        content = f"""# Carroll .energy

Var A : "f:A" = {A:.6g},
    B : "f:B" = {B:.6g},
    C : "f:C" = {C:.6g};

Let I1 = I[1], I2 = I[2];

Potential = A*I1 + B*(I1^4) + C*sqrt(I2);

"""
        return content

    else:
        # На случай, если вдруг мы забыли где-то модель
        return f"# Unknown model: {model_name}\nPotential = 0;\n"
