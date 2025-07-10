import os
import tempfile

import numpy as np
from fastapi import BackgroundTasks

from app.modules.isotropic.solver import IsotropicModelType


class EnergyInfo:

    @staticmethod
    async def download_energy(energy_text: str, background_tasks: BackgroundTasks) -> str:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".energy", encoding="utf-8") as temp_file:
            temp_file.write(energy_text)
            temp_file_path = temp_file.name

        background_tasks.add_task(os.remove, temp_file_path)

        return temp_file_path

    @staticmethod
    def energy_text(name: str, params: np.ndarray) -> str:
        # TODO Переписать на фабрику моделей
        """
        Возвращает текст .energy в стиле FEBio-Calc.
        3-D инварианты (J=1):
            I3d_1 = I[1] + 1/I[2]
            I3d_2 = I[2] + I[1]/I[2]
        """
        model_name_internal = IsotropicModelType(name)
        if model_name_internal is None:
            raise ValueError(f"Unknown model name alias: {name}")
        hdr = f"# Auto-generated .energy for {name}\n\n"

        if model_name_internal == "NeoHookean":
            (mu,) = params
            return (
                    hdr +
                    f'Var mu : "f:mu" = {mu:.8g};  # [MPa]\n\n'
                    "Let I3d_1 = I[1] + 1/I[2];\n"
                    "Potential = mu/2 * (I3d_1 - 3);\n"
            )

        if model_name_internal == "MooneyRivlin":
            c1, c2 = params
            return (
                    hdr +
                    f'Var c1 : "f:c1" = {c1:.8g}, '
                    f'c2 : "f:c2" = {c2:.8g};  # [MPa]\n\n'
                    "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
                    "Potential = c1 * (I3d_1 - 3) + c2 * (I3d_2 - 3);\n"
            )

        if model_name_internal == "GeneralizedMooneyRivlin":
            C10, C01, C11, C20, C02 = params
            return (
                    hdr +
                    "Var C10=\"f:C10\"={:.8g}, C01=\"f:C01\"={:.8g}, "
                    "C11={:.8g}, C20={:.8g}, C02={:.8g};  # [MPa]\n\n"
                    "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
                    "Potential = C10*(I3d_1-3) + C01*(I3d_2-3) + "
                    "C11*(I3d_1-3)*(I3d_2-3) + C20*(I3d_1-3)^2 + "
                    "C02*(I3d_2-3)^2;\n".format(C10, C01, C11, C20, C02)
            )

        if model_name_internal == "Yeoh":
            c1, c2, c3 = params
            return (
                    hdr +
                    f'Var c1="f:c1"={c1:.8g}, c2={c2:.8g}, c3={c3:.8g};  # [MPa]\n\n'
                    "Let I3d_1 = I[1] + 1/I[2];\n"
                    "Potential = c1*(I3d_1-3) + c2*(I3d_1-3)^2 + "
                    "c3*(I3d_1-3)^3;\n"
            )

        if model_name_internal == "Beda":
            c1, c2, c3, K1, a, k, b = params
            return (
                    hdr +
                    "Var c1={:.8g}, c2={:.8g}, c3={:.8g}, "
                    "K1={:.8g}, a={:.8g}, k={:.8g}, b={:.8g};\n\n"
                    "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
                    "Potential = c1/a*(I3d_1-3)^a + c2*(I3d_2-3) + "
                    "c3/k*(I3d_1-3)^k + K1/b*(I3d_2-3)^b;\n".format(
                        c1, c2, c3, K1, a, k, b
                    )
            )

        if model_name_internal == "Gent":
            mu, Jm = params
            return (
                    hdr +
                    f'Var mu="f:mu"={mu:.8g}, Jm={Jm:.8g};  # [MPa], [-]\n\n'
                    "Let I3d_1 = I[1] + 1/I[2];\n"
                    "Potential = -mu*Jm/2*log(1 - (I3d_1-3)/Jm);\n"
            )

        if model_name_internal == "Carroll":
            A, B, C = params
            return (
                    hdr +
                    f'Var A={A:.8g}, B={B:.8g}, C={C:.8g};  # [MPa]\n\n'
                    "Let I3d_1 = I[1] + 1/I[2], I3d_2 = I[2] + I[1]/I[2];\n"
                    "Potential = A*I3d_1 + B*I3d_1^4 + C*sqrt(I3d_2);\n"
            )

        return hdr + "# Unknown model\nPotential = 0;\n"
