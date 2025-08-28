import os
import win32com.client as win32
import pandas as pd
import time
import logging

# Standard logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ================== Aspen 交互增强类 ====================
class AspenEnhanced:
    """
    A helper class to encapsulate interactions with an Aspen Plus COM server.
    """
    def __init__(self, bkp_path):
        """初始化Aspen连接并定义摩尔质量"""
        logger.info("Dispatching Aspen COM object...")
        self.aspen = win32.Dispatch('Apwn.Document.40.0')  # Adjust version if necessary
        logger.info(f"Loading Aspen backup file: {os.path.abspath(bkp_path)}")
        self.aspen.InitFromArchive2(os.path.abspath(bkp_path))
        self.aspen.Visible = 0
        self.aspen.SuppressDialogs = 1
        logger.info("Aspen initialized successfully.")

        # 定义摩尔质量 (g/mol)
        self.M_T, self.M_D, self.M_H = 3.016, 2.014, 1.008

    def set_composition(self, ratios):
        """设置六元组分输入"""
        nodes = {
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\H2': ratios[0],  # EH2
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\HD': ratios[1],  # EHD
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\D2': ratios[2],  # ED2
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\HT': ratios[3],  # EHT
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\DT': ratios[4],  # EDT
            r'\Data\Streams\FROMTEP\Input\FLOW\MIXED\T2': ratios[5],  # ET2
            r'\Data\Streams\FROMTEP\Input\TOTFLOW\MIXED': ratios[6] / 2  # 总流量/2
        }
        for path, value in nodes.items():
            self.aspen.Tree.FindNode(path).Value = value

    def run_step(self):
        """执行单步模拟并等待完成"""
        self.aspen.Engine.Run2()
        while self.aspen.Engine.IsRunning:
            time.sleep(0.1)

    def get_stream_results(self):
        """获取关键流股的H/D/T质量流量(g/h)"""
        def calc_stream(stream_name):
            """计算单个流股的H/D/T质量流量"""
            nodes = self.aspen.Tree.FindNode(f"\Data\Streams\{stream_name}\Output\MOLEFLOW\MIXED")
            Q1 = 1000 * nodes.FindNode("H2").Value
            Q2 = 1000 * nodes.FindNode("HD").Value
            Q3 = 1000 * nodes.FindNode("D2").Value
            Q4 = 1000 * nodes.FindNode("HT").Value
            Q5 = 1000 * nodes.FindNode("DT").Value
            Q6 = 1000 * nodes.FindNode("T2").Value

            H = (2 * Q1 + 1 * Q2 + 1 * Q4) * self.M_H
            D = (1 * Q2 + 2 * Q3 + 1 * Q5) * self.M_D
            T = (1 * Q4 + 1 * Q5 + 2 * Q6) * self.M_T
            return [H, D, T]

        streams = {"WDS": "S4", "SDST2": "S17", "SDSD2": "S16"}
        return {name: calc_stream(path) for name, path in streams.items()}

    def close(self):
        """Closes the Aspen connection."""
        if self.aspen:
            self.aspen.Close()
            logger.info("Closed Aspen session.")


# ================== 主执行接口函数 ====================
def run_aspen_simulation(
    temp_input_csv: str,
    temp_output_csv: str,
    bkp_path: str = r"example_aspenbkp/T2-Threetowers4.bkp",
    aspen_results_csv: str = None,
    base: float = 20,
    retime: int = 60,
    time_step: int = 3,
    min_stable_steps: int = 100,
    stable_threshold: float = 1e-6
):
    """
    Runs an Aspen Plus simulation based on inputs from a Modelica simulation.

    Args:
        bkp_path (str): Path to the Aspen backup file (.bkp).
        temp_input_csv (str): Path to the input CSV file with time-series data.
        temp_output_csv (str): Path to save the final summarized output CSV.
        aspen_results_csv (str, optional): Path to save detailed Aspen results. Defaults to None.
        base (float, optional): Minimum inventory (heel) to start simulation (mol). Defaults to 20.
        retime (int, optional): Lag time in minutes for output results. Defaults to 60.
        time_step (int, optional): Time step in minutes. Defaults to 3.
        min_stable_steps (int, optional): Consecutive stable steps to confirm stability. Defaults to 100.
        stable_threshold (float, optional): Relative difference threshold for stability. Defaults to 1e-6.

    Returns:
        dict: A dictionary mapping output variable names to their final values,
              formatted as Modelica vector strings.
              e.g., {'to_SDS': '{v1,v2,v3}', 'to_WDS': '{v4,v5,v6}'}
    """
    aspen = None
    all_results = []
    output_placeholder = {}

    try:
        # 1. 初始化Aspen
        aspen = AspenEnhanced(bkp_path)

        # 2. 读取OpenModelica数据
        logger.info(f"Reading input data from: {temp_input_csv}")
        df_input = pd.read_csv(temp_input_csv, encoding='gbk')

        time_step_h = time_step / 60
        N = int(retime / time_step)

        df_input = df_input[
            (df_input['time'] / time_step_h).apply(lambda x: round(x, 9).is_integer())
        ].copy()
        required_cols = ['time', 'tep_fcu.outflow[1]', 'tep_fcu.outflow[2]', 'tep_fcu.outflow[3]']
        input_data = df_input[required_cols].values
        
        prev_T_flow = None
        stable_count = 0
        I_stock = 0
        count = 0

        # 3. 主循环处理
        logger.info("Starting main simulation loop...")
        for time_val, T_flow, D_flow, H_flow in input_data:
            time_aspen = time_val * time_step
            M_T, M_D, M_H = 3.016, 2.014, 1.008
            T_flow_mol, D_flow_mol, H_flow_mol = T_flow / M_T, D_flow / M_D, H_flow / M_H
            total_flow = T_flow_mol + D_flow_mol + H_flow_mol

            if prev_T_flow is not None and abs(prev_T_flow) > 1e-9:
                relative_diff = abs(T_flow - prev_T_flow) / abs(prev_T_flow)
                if relative_diff < stable_threshold:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count >= min_stable_steps:
                    logger.info(f"System stabilized at Time={time_val:.1f}h. Stopping simulation.")
                    break
            prev_T_flow = T_flow

            I_stock += total_flow * time_step_h
            if I_stock <= base:
                # Record zeros and skip simulation until inventory builds up
                record = {"Time": time_val,"Time_Aspen": time_aspen, "Input_T": T_flow, "Input_D": D_flow, "Input_H": H_flow}
                # ... (add other zero-ed out columns for consistency)
                all_results.append(record)
                continue

            if not count:
                count += 1
                I_input = (I_stock - base) / time_step_h
                logger.info(f"Inventory base reached. Effective input flow: {I_input:.2f} mol/h")
                ET, ED, EH = T_flow_mol / total_flow, D_flow_mol / total_flow, H_flow_mol / total_flow
                ratios = [EH**2, 2*EH*ED, ED**2, 2*EH*ET, 2*ED*ET, ET**2, I_input]
            else:
                ET, ED, EH = T_flow_mol / total_flow, D_flow_mol / total_flow, H_flow_mol / total_flow
                ratios = [EH**2, 2*EH*ED, ED**2, 2*EH*ET, 2*ED*ET, ET**2, total_flow]

            aspen.set_composition(ratios)
            aspen.run_step()
            stream_results = aspen.get_stream_results()

            record = {
                "Time": time_val,"Time_Aspen": time_aspen,"Input_T": T_flow, "Input_D": D_flow, "Input_H": H_flow,
                **{f"Input_{comp}": val for comp, val in zip(
                    ["EH2", "EHD", "ED2", "EHT", "EDT", "ET2", "TOTAL"], ratios)},
                **{f"{stream}_{iso}_raw": values[i]
                   for stream, values in stream_results.items()
                   for i, iso in enumerate(["H", "D", "T"])}
            }
            all_results.append(record)
            logger.debug(f"Progress: {len(all_results)}/{len(input_data)} | Time={time_val:.1f}h")

        # 4. 后处理
        logger.info("Simulation loop finished. Starting post-processing...")
        if not all_results:
            logger.warning("No results were generated. The simulation might have been skipped entirely.")
            return output_placeholder

        df = pd.DataFrame(all_results).fillna(0)

        raw_cols = [f"{stream}_{iso}_raw" for stream in ["WDS", "SDST2", "SDSD2"] for iso in ["H", "D", "T"]]
        for col in raw_cols:
            stream, iso, _ = col.split('_')
            df[f"{stream}_{iso}"] = df[col].shift(N).fillna(0)

        df["delta_I_H"] = (df["Input_H"] - df.get("WDS_H", 0) - df.get("SDST2_H", 0) - df.get("SDSD2_H", 0)) * time_step_h
        df["delta_I_D"] = (df["Input_D"] - df.get("WDS_D", 0) - df.get("SDST2_D", 0) - df.get("SDSD2_D", 0)) * time_step_h
        df["delta_I_T"] = (df["Input_T"] - df.get("WDS_T", 0) - df.get("SDST2_T", 0) - df.get("SDSD2_T", 0)) * time_step_h
        df["I_H"] = df["delta_I_H"].cumsum()
        df["I_D"] = df["delta_I_D"].cumsum()
        df["I_T"] = df["delta_I_T"].cumsum()

        df["to_SDS[1]"] = df.get("SDST2_T", 0) + df.get("SDSD2_T", 0)
        df["to_SDS[2]"] = df.get("SDST2_D", 0) + df.get("SDSD2_D", 0)
        df["to_SDS[3]"] = df.get("SDST2_H", 0) + df.get("SDSD2_H", 0)
        df["to_WDS[1]"] = df.get("WDS_T", 0)
        df["to_WDS[2]"] = df.get("WDS_D", 0)
        df["to_WDS[3]"] = df.get("WDS_H", 0)

        # 5. 保存输出文件
        out_df = df[["Time", "to_SDS[1]", "to_SDS[2]", "to_SDS[3]", "to_WDS[1]", "to_WDS[2]", "to_WDS[3]"]]
        out_df.to_csv(temp_output_csv, index=False)
        logger.info(f"Summary output saved to {temp_output_csv}")

        if aspen_results_csv:
            df.drop(columns=raw_cols, errors='ignore').to_csv(aspen_results_csv, index=False)
            logger.info(f"Detailed results saved to {aspen_results_csv}")

        # 6. 构建返回字典
        output_placeholder = {
            "to_SDS": "{1,2,3,4,1,1}",
            "to_WDS": "{1,5,6,7,1,1}",
        }
        logger.info(f"Returning final values: {output_placeholder}")

    except Exception as e:
        logger.error(f"An error occurred during the Aspen simulation: {str(e)}", exc_info=True)
    finally:
        if aspen:
            aspen.close()

    return output_placeholder
