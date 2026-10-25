# Co-Simulation Model Pipeline 

本目录包含了一套专为 TRICYS 协同仿真框架设计的自动化编译与代码生成流水线（Pipeline）。通过这套脚本，您可以从原始的 `.mo` 模型文件中一键生成供协同仿真使用的 `schema.json`、编译所有子组件的 FMI 2.0 Co-Simulation `.fmu` 文件、自动生成 Python 的强类型代理基类以及 SSP 网络拓扑文件。

## 核心脚本概览

流水线被拆分为 3 个核心步骤脚本，以及 1 个一键式总控脚本：

1. **`1_extract_schema.py`**：提取模型接口（Schema）
2. **`2_build_fmus.py`**：批量编译 FMU
3. **`3_generate_processors.py`**：生成强类型 Python 代理基类
4. **`4_export_ssp.py`**：导出原生 SSP 网络拓扑文件
5. **`build_pipeline.py`**：流水线一键总控脚本

---

## 一键式使用方法 (推荐) 

最简单、最高效的方式是直接运行 `build_pipeline.py`，它会自动按顺序执行前 3 个脚本。

**运行示例：**

*适用于 Linux / macOS (Bash) 的多行格式：*
```bash
python script/cosim/build_pipeline.py \
    --mo_file tricys/example/example_data/example_model_single/example_model.mo \
    --package example_model \
    --out_fmu_dir tricys/example/example_data/example_model_ssp/fmus \
    --out_proc_dir tricys/example/example_data/example_model_ssp/typed_base \
    --out_schema tricys/example/example_data/example_model_ssp/schema.json \
    --model_name example_model.Cycle \
    --out_ssp_dir tricys/example/example_data/example_model_ssp/ssp
```

*适用于 Windows (CMD / PowerShell) 的单行格式（推荐直接复制粘贴）：*
```powershell
python script/cosim/build_pipeline.py --mo_file tricys/example/example_data/example_model_single/example_model.mo --package example_model --out_fmu_dir tricys/example/example_data/example_model_ssp/fmus --out_proc_dir tricys/example/example_data/example_model_ssp/typed_base --out_schema tricys/example/example_data/example_model_ssp/schema.json --model_name example_model.Cycle --out_ssp_dir tricys/example/example_data/example_model_ssp/ssp
```

### 参数说明：
- `--mo_file`: (必填) Modelica 模型文件 `.mo` 的相对或绝对路径。
- `--package`: (必填) 模型中定义的顶层包名，例如 `CFEDR` 或是 `example_model`。
- `--out_fmu_dir`: (必填) 编译生成的 `.fmu` 存放的输出目录。
- `--out_proc_dir`: (必填) 生成的 `***_processor_base.py` 存放的 Python 目录（通常为 `tricys/online_cosim/processors/typed_base`）。
- `--out_schema`: (可选) 存放中间产物 `schema.json` 的路径，默认为当前执行目录下的 `schema.json`。
- `--model_name`: (可选) 顶层模型名称，例如 `CFEDR.Cycle`。如果提供此参数，流水线将在最后一步导出 `.ssp` 文件。
- `--out_ssp_dir`: (可选) 导出的 `.ssp` 文件存放目录。默认存放在 `--mo_file` 同级目录下。

---

## 分步使用说明 (高级调试)

如果某个环节出现问题，或者您只想单独更新某一部分（比如仅更新 Python 基类而不想耗费时间重编 FMU），您可以单独执行以下脚本：

### 第一步：提取 Schema
负责启动 OpenModelica，加载 `.mo` 并扫描所有指定的 `package` 下的模型，提取所有的 Connector、Parameter 和 Variable，生成 JSON 文件。

```bash
python script/cosim/1_extract_schema.py \
    --mo_file path/to/model.mo \
    --package MyPackage \
    --out_json path/to/output_schema.json
```

*适用于 Windows (CMD / PowerShell) 的单行格式：*
```powershell
python script/cosim/1_extract_schema.py --mo_file path/to/model.mo --package MyPackage --out_json path/to/output_schema.json
```

### 第二步：编译 FMU
读取 `schema.json` 中的组件清单，启动 OpenModelica 批量将其编译为 `.fmu`（Co-Simulation 模式，FMI 2.0 标准）。

```bash
python script/cosim/2_build_fmus.py \
    --mo_file path/to/model.mo \
    --package MyPackage \
    --schema path/to/input_schema.json \
    --out_dir path/to/fmu_output_dir
```

*适用于 Windows (CMD / PowerShell) 的单行格式：*
```powershell
python script/cosim/2_build_fmus.py --mo_file path/to/model.mo --package MyPackage --schema path/to/input_schema.json --out_dir path/to/fmu_output_dir
```

### 第三步：生成 Python 处理器基类
读取 `schema.json`，利用 Jinja2 模板引擎批量生成 `***_processor_base.py`，为用户提供强类型的代码补全体验。

```bash
python script/cosim/3_generate_processors.py \
    --schema path/to/input_schema.json \
    --out_dir path/to/python_output_dir
```

*适用于 Windows (CMD / PowerShell) 的单行格式：*
```powershell
python script/cosim/3_generate_processors.py --schema path/to/input_schema.json --out_dir path/to/python_output_dir
```

### 第四步：导出 SSP 网络拓扑 (可选)
如果需要原生系统描述，可以利用编译好的 FMU 和模型本身提取系统拓扑，直接导出标准的 `.ssp` 文件，供 FMI 标准工具链（如 OMEdit、Dymola）独立查看与仿真。

```bash
python script/cosim/4_export_ssp.py \
    --mo_file path/to/model.mo \
    --model_name MyPackage.MySystem \
    --fmu_dir path/to/fmu_output_dir \
    --out_ssp path/to/output.ssp
```

*适用于 Windows (CMD / PowerShell) 的单行格式：*
```powershell
python script/cosim/4_export_ssp.py --mo_file path/to/model.mo --model_name MyPackage.MySystem --fmu_dir path/to/fmu_output_dir --out_ssp path/to/output.ssp
```

## 注意事项
1. 运行该脚本前，请确保系统中已经成功安装并配置了 **OpenModelica**，并且 Python 环境中已安装 `OMPython`。
2. FMU 的编译过程可能会比较耗时（每个组件 5-10 秒不等），请耐心等待。
3. 若报错 `UnicodeEncodeError: 'gbk' codec can't encode character...`，请确保您的 Windows 命令行处于 UTF-8 模式，或使用终端（如 PowerShell、Git Bash）运行。脚本内部已去除了不支持的 Emoji 字符以最大化兼容。
