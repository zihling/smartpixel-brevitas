# TODO : Modification
import subprocess
from typing import Any, Literal, Optional

import numpy as np
import torch
from arbolta import DesignConfig, HardwareDesign, PortConfig
from numpy import ndarray
from torch import Tensor

__all__ = ["SystolicArray"]


def require(name: str, val: Any) -> Any:
    if val is None:
        raise ValueError(f"Must set `{name}`")
    return val


def width_to_int(w: int, signed: bool) -> np.dtype:
    if signed:
        dtype = (
            np.int8
            if w <= 8
            else np.int16
            if w <= 16
            else np.int32
            if w <= 32
            else np.int64
        )
    else:
        dtype = (
            np.uint8
            if w <= 8
            else np.uint16
            if w <= 16
            else np.uint32
            if w <= 32
            else np.uint64
        )
    return dtype


class SystolicArray:
    def __init__(
        self,
        mac_type: Literal["integer", "minifloat"],
        rows: int,
        columns: int,
        accumulator_bit_width: int,
        # Integer parameters
        x_bit_width: Optional[int] = None,
        k_bit_width: Optional[int] = None,
        # Minifloat parameters
        x_exponent_bit_width: Optional[int] = None,
        x_mantissa_bit_width: Optional[int] = None,
        k_exponent_bit_width: Optional[int] = None,
        k_mantissa_bit_width: Optional[int] = None,
    ):
        params: dict[str, int]
        x_dtype: np.dtype
        match mac_type:
            case "integer":
                params = {
                    "MacType": 0,
                    "WidthX": require("x_bit_width", x_bit_width),
                    "WidthK": require("k_bit_width", k_bit_width),
                }
                x_dtype = width_to_int(x_bit_width, signed=True)
                k_dtype = width_to_int(k_bit_width, signed=True)

            case "minifloat":
                params = {
                    "MacType": 1,
                    "ExpWidthX": require("x_exponent_bit_width", x_exponent_bit_width),
                    "ManWidthX": require("x_mantissa_bit_width", x_mantissa_bit_width),
                    "ExpWidthK": require("k_exponent_bit_width", k_exponent_bit_width),
                    "ManWidthK": require("k_mantissa_bit_width", k_mantissa_bit_width),
                }

                # For calculating datatype
                # TODO: Conversion w/ ml_dtypes
                x_bit_width = 1 + x_exponent_bit_width + x_mantissa_bit_width
                k_bit_width = 1 + k_exponent_bit_width + k_mantissa_bit_width
                x_dtype = width_to_int(x_bit_width, signed=False)
                k_dtype = width_to_int(k_bit_width, signed=False)

            case _:
                raise ValueError(f"Unsupported MAC type `{mac_type}`")

        params.update({"Rows": rows, "Cols": columns, "WidthY": accumulator_bit_width})

        self.params = params
        self.acc_dtype = width_to_int(accumulator_bit_width, signed=True)
        self.x_dtype = x_dtype
        self.k_dtype = k_dtype
        self.design: HardwareDesign = None

    def load(self, netlist_path: str = "./synth_output/0_proc/synth.json") -> None:
        """
        Load systolic array from Yosys JSON netlist.
        """
        rows, columns = self.params["Rows"], self.params["Cols"]
        config = DesignConfig(
            clk_i=PortConfig(clock=True, polarity=1),
            rst_ni=PortConfig(reset=True, polarity=0),
            s_valid_i=PortConfig(shape=(1, 1)),
            s_last_i=PortConfig(shape=(1, 1)),
            m_ready_i=PortConfig(shape=(1, 1)),
            s_ready_o=PortConfig(shape=(1, 1)),
            m_valid_o=PortConfig(shape=(1, 1)),
            m_last_o=PortConfig(shape=(1, 1)),
            sx_data_i=PortConfig(shape=(1, rows), dtype=self.x_dtype),
            sk_data_i=PortConfig(shape=(1, columns), dtype=self.k_dtype),
            m_data_o=PortConfig(shape=(1, rows), dtype=self.acc_dtype),
        )
        self.design = HardwareDesign("axis_sa", netlist_path, config)

    def synthesize(
        self,
        script_path: str = "../arbolta/experiments/systolic_array/axis-systolic-array/tcl/synth.tcl", # !: Need to be change to corresponding path if used in other repo
        output_dir: str = "./synth_output",
        load: bool = True,
    ) -> None:
        """
        Synthesizes systolic array with Yosys.
        Saves netlists at `./synth_output` and loads.
        """
        synth_params = [f"{p_name}={p_val}" for (p_name, p_val) in self.params.items()]
        synth_params.append(f"output_dir={output_dir}")
        command = ["yosys", "-c", script_path, "--", *synth_params]
        p = subprocess.Popen(command, stdout=subprocess.PIPE)
        # Ignore output for now
        _, err = p.communicate()

        assert err is None, err

        if load:
            # Default to process netlist
            self.load(f"{output_dir}/0_proc/synth.json")

    def run_matmul(self, x: ndarray | Tensor, k: ndarray | Tensor) -> ndarray | Tensor:
        """
        Run inputs through systolic array.
        Expects x: (K,R), k: (K,C) -> y: (C,R)
        """
        if self.design is None:
            raise AttributeError("Must load or synthesize design")

        K, R, C = x.shape[0], x.shape[1], k.shape[1]
        actual = np.zeros((C, R), dtype=self.acc_dtype)

        # Start simulation
        self.design.eval_reset_clocked()
        for i in range(K):
            while self.design.ports.s_ready_o == 0:
                self.design.eval_clocked()

            self.design.ports.s_valid_i = 1
            self.design.ports.sx_data_i = x[i]
            self.design.ports.sk_data_i = k[i]

            if i == K - 1:
                self.design.ports.s_last_i = 1

            self.design.eval_clocked()

        self.design.ports.m_ready_i = 1
        self.design.ports.s_valid_i = 0
        self.design.ports.s_last_i = 0

        idx = 0
        while True:
            if self.design.ports.m_valid_o.item() == 1:
                actual[idx] = self.design.ports.m_data_o
                idx += 1

            if self.design.ports.m_last_o.item() == 1:
                break

            self.design.eval_clocked()

        # TODO: CALCULATE FRACTION BITS

        if isinstance(x, Tensor) or isinstance(k, Tensor):
            return torch.from_numpy(actual)
        else:
            return actual