import os
import unittest

from tricys.manager.config_manager import config_manager
from tricys.simulation.single_simulation import simulation


class TestSingleSimulation(unittest.TestCase):
    def test_simulation_main(self):
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        package_path = os.path.join(
            PROJECT_ROOT, config_manager.get("paths.package_path")
        )
        log_dir = os.path.join(PROJECT_ROOT, config_manager.get("paths.log_dir"))
        db_path = os.path.join(PROJECT_ROOT, config_manager.get("paths.db_path"))

        # 定义TEST_ROOT防止多次测试结束删除同一个文件夹
        TEST_ROOT = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "./test_single_simulation")
        )
        temp_dir = os.path.join(TEST_ROOT, config_manager.get("paths.temp_dir"))
        results_dir = os.path.join(TEST_ROOT, config_manager.get("paths.results_dir"))

        model_name = config_manager.get("simulation.model_name", "example.Cycle")
        stop_time = config_manager.get("simulation.stop_time", 5000.0)
        step_size = config_manager.get("simulation.step_size", 1)
        param_overrides = config_manager.get("overrides_parameter")
        variableFilter = config_manager.get(
            "simulation.variableFilter", "time|sds.I[1]"
        )

        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        try:
            result_path = simulation(
                package_path=package_path,
                model_name=model_name,
                stop_time=stop_time,
                step_size=step_size,
                results_dir=results_dir,
                param_values=param_overrides,
                variableFilter=variableFilter,
            )
            self.assertTrue(os.path.exists(result_path))
            os.system(f"rm -rf {TEST_ROOT}")
            os.system(f"rm -rf {log_dir}")
            os.system(f"rm -rf {db_path}")
        except Exception as e:
            self.fail(f"Simulation failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
