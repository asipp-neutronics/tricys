import sqlite3
import json
import os
from OMPython import OMCSessionZMQ
import logging

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
logger = logging.getLogger('get_parameters')
logger.setLevel(logging.DEBUG)

def create_parameters_table(db_path: str):
    """创建参数表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS parameters (
            name TEXT PRIMARY KEY,
            type TEXT,  -- e.g., Real, Real[5]
            default_value TEXT,  -- JSON encoded
            sweep_values TEXT,  -- JSON encoded, null if not swept
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Parameters table created")

def get_model_parameters(package_path: str, model_name: str, db_path: str):
    """从 OpenModelica 获取模型参数并存储到数据库"""
    omc = OMCSessionZMQ()
    try:
        omc.sendExpression(f'loadFile("{package_path}")')
        logger.info(f"Loaded package: {package_path}")

        params = omc.sendExpression(f'getParameters("{model_name}")')
        logger.debug(f"Raw parameters: {params}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for param in params:
            name = param['name']
            type_info = param.get('type', 'Real')
            default_value = param.get('defaultValue', None)

            if isinstance(default_value, list):
                default_value = json.dumps(default_value)
            elif isinstance(default_value, (int, float)):
                default_value = json.dumps(float(default_value))
            else:
                default_value = json.dumps(None)

            cursor.execute('''
                INSERT OR REPLACE INTO parameters (name, type, default_value, sweep_values, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, type_info, default_value, None, f"Parameter {name} from {model_name}"))
            logger.debug(f"Stored parameter: {name}, type: {type_info}, default: {default_value}")

        conn.commit()
        logger.info("Parameters stored in database")
    except Exception as e:
        logger.error(f"Failed to get parameters: {str(e)}")
        raise
    finally:
        conn.close()
        del omc

if __name__ == "__main__":
    package_path = CONFIG['package_path']
    model_name = "FFCAS.Cycle"
    db_path = os.path.join(CONFIG['output_dir'], "parameters.db")
    
    create_parameters_table(db_path)
    get_model_parameters(package_path, model_name, db_path)