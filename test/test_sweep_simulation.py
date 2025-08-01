import os
import unittest

import numpy as np

from tricys.manager.config_manager import config_manager
from tricys.simulation.sweep_simulation import run_parameter_sweep


class TestSweepSimulation(unittest.TestCase):
    def test_sweep_simulation_main(self):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        package_path = os.path.join(
            PROJECT_ROOT, config_manager.get("paths.package_path")
        )
        log_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.log_dir"))

        # 定义TEST_ROOT防止多次测试结束删除同一个文件夹
        TEST_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "./test_sweep_simulation")
        )
        temp_dir = os.path.join(TEST_ROOT, config_manager.get("paths.temp_dir"))
        results_dir = os.path.join(TEST_ROOT, config_manager.get("paths.results_dir"))

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        variableFilter = config_manager.get(
            "simulation.variableFilter", "time|sds.I[1]"
        )
        model_name = config_manager.get("simulation.model_name", "example.Cycle")
        param_A_name = config_manager.get("sweep_parameter.parameter_A.name")
        param_B_name = config_manager.get("sweep_parameter.parameter_B.name")
        min_val = config_manager.get("sweep_parameter.parameter_B.min_value")
        max_val = config_manager.get("sweep_parameter.parameter_B.max_value")
        steps = config_manager.get("sweep_parameter.parameter_B.num_steps")

        param_A_vals = config_manager.get("sweep_parameter.parameter_A.values")
        param_A_values = {param_A_name: param_A_vals}
        param_B_vals = np.linspace(min_val, max_val, steps)
        param_B_sweep = {param_B_name: param_B_vals}

        stop_time = config_manager.get("simulation.stop_time", 5000.0)
        step_size = config_manager.get("simulation.step_size", 1)

        try:
            result_path = run_parameter_sweep(
                package_path,
                model_name,
                param_A_values,
                param_B_sweep,
                stop_time,
                step_size,
                temp_dir,
                results_dir,
                variableFilter,
            )
            self.assertTrue(os.path.exists(result_path))
            os.system(f"rm -rf {TEST_ROOT}")
            os.system(f"rm -rf {log_dir}")
        except Exception as e:
            self.fail(f"Sweep simulation failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
