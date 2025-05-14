import sqlite3
import json
import numpy as np
import logging
import os

# 读取 config.json
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
with open(CONFIG_PATH, 'r') as f:
    CONFIG = json.load(f)

# 配置日志
logging.basicConfig(
    filename=os.path.join(CONFIG['output_dir'], "simulation.log"),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('update_sweep_values')
logger.setLevel(logging.DEBUG)

def update_sweep_values(db_path: str, param_sweep: dict):
    """更新数据库中的扫描值"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for param_name, sweep_values in param_sweep.items():
        sweep_values_json = json.dumps(sweep_values.tolist() if isinstance(sweep_values, np.ndarray) else sweep_values)
        cursor.execute('''
            UPDATE parameters
            SET sweep_values = ?
            WHERE name = ?
        ''', (sweep_values_json, param_name))
        logger.debug(f"Updated sweep values for {param_name}: {sweep_values_json}")
    
    conn.commit()
    conn.close()
    logger.info("Sweep values updated")

if __name__ == "__main__":
    db_path = os.path.join(CONFIG['output_dir'], "parameters.db")
    param_sweep = {
        "fuel_flow": [1.0, 1.1, 1.2],  # 替换为实际参数
        "efficiency": np.linspace(0.9, 1.0, 3).tolist(),
        "inventory": [[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]]
    }
    update_sweep_values(db_path, param_sweep)