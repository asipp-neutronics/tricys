# P0: DWSIM 等效性验证 — 替代 Aspen Plus 可行性基准测试

## 背景与目标

- **问题/需求描述**：tricys 当前依赖 Aspen Plus（Windows COM 接口）驱动 I_ISS 低温精馏子系统（三塔六组分氢同位素分离）。Aspen 是闭源商业软件，无法在 Linux/HPC 上运行，阻碍批量参数扫描和开源化部署。需要验证开源模拟器 DWSIM 能否在数值精度可接受的范围内复现 Aspen 的三塔稳态计算结果。
- **根因分析**：Aspen 的技术护城河在于 (a) 内置 SRK/NRTL 二元交互参数（BIP）包含氢同位素体系，(b) rigorous 精馏塔求解器的收敛性。P0 的核心是验证这两项能力是否可通过 DWSIM 开源替代。
- **目标**：
  1. 在 Linux 上部署 DWSIM headless 环境
  2. 从 Aspen .bkp 文件中提取三塔拓扑与物性参数
  3. 在 DWSIM 中构建等效三塔流程
  4. 对 ≥5 组进料工况进行 Aspen vs DWSIM 对照测试
  5. 得出 Go/No-Go 结论
- **非目标（不做什么）**：
  - 不修改 tricys 生产代码（`simulation.py`、`i_iss_handler.py` 等）— P0 仅验证
  - 不实现 DWSIM handler 与 OM 联合仿真的完整集成 — 那是 P1 的工作
  - 不涉及 IDAES-PSE 或 OpenModelica 自建塔 — 那是 P2 的工作
  - 不优化 DWSIM 性能或收敛速度
- **已有代码/流程复用分析**：
  - `run_dummy_simulation()` 的 CSV mock 模式：**复用**其接口签名设计作为 DWSIM handler 的参考
  - `AspenEnhanced` 类的 `set_composition/run_step/get_stream_results` 三方法接口：**复用**作为 DWSIM wrapper 的 API 契约
  - `co_simulation_module.json` 的 handler 配置格式：**复用**，P0 的 DWSIM 测试驱动器遵循相同接口
  - Aspen .bkp 文件中的物性参数：**提取后迁移**到 DWSIM

## 技术方案

- **方案概述**：
  1. 在 Windows 端编写 Aspen COM 脚本，从 `T2-Threetowers4.bkp` 中提取全部 BIP、塔规格、流股拓扑到 JSON
  2. 在 Linux 端安装 DWSIM（.NET 8 + 预编译 Release），通过 Python.NET 或独立 C# 脚本 headless 驱动
  3. 用提取的参数在 DWSIM 中注册 6 个自定义化合物并配置 SRK 物性包
  4. 构建三塔精馏流程，保存为 `.dwxmz` 文件
  5. 编写 Python 测试驱动器，对多组进料工况分别调用 Aspen 和 DWSIM，比较 9 个输出（3 流股 × 3 同位素质量流量）

- **关键设计决策**：
  - **DWSIM 自动化方式**：优先采用 Python.NET (`pythonnet`) 直接加载 DWSIM.Automation 程序集；若 Python.NET 在 Linux .NET 8 环境下不稳定，降级为独立 C# 控制台程序（接受 JSON stdin → 输出 JSON stdout），Python 侧通过 `subprocess` 调用
  - **物性参数来源**：优先从 .bkp COM 树提取（最准确）；若 COM 导出不全，用公开文献值（Souers 1986 *Hydrogen Properties for Fusion Energy*，Kinoshita 1981）补充
  - **验收阈值**：9 个输出值中，每个的相对偏差 < 5%；对于绝对值 < 0.01 g/h 的小流量，改用绝对偏差 < 0.01 g/h

- **影响范围**：
  - 新增文件（全部在 `script/dwsim/` 和 `test/dwsim/` 下，不触碰现有代码）：
    - `script/dwsim/extract_aspen_params.py`
    - `script/dwsim/build_dwsim_flowsheet.py` 或 `script/dwsim/BuildFlowsheet.cs`
    - `script/dwsim/run_dwsim_point.py` 或 `script/dwsim/RunPoint.cs`
    - `test/dwsim/test_aspen_dwsim_parity.py`
    - `test/dwsim/fixtures/` — 测试数据
    - `example/example_dwsim/` — DWSIM 流程文件 + 参数 JSON

## Error & Rescue Map（关键失败路径映射）

| 代码路径/操作                | 可能的失败                                    | 错误类型         | 已处理？ | 处理方式                                                     | 用户可见行为        |
| ---------------------------- | --------------------------------------------- | ---------------- | -------- | ------------------------------------------------------------ | ------------------- |
| DWSIM .NET 8 加载            | Linux 缺少 `libicu`/`libssl` 等共享库         | RuntimeError     | Y        | Task 1.1 检查依赖并安装                                      | 安装脚本报错提示    |
| Python.NET 加载 DWSIM 程序集 | `pythonnet` 无法找到 coreclr 或程序集路径错误 | ImportError      | Y        | 降级为 C# subprocess 方案（Task 3.3 备选）                   | 日志警告 + 自动切换 |
| Aspen COM 参数提取           | BIP 节点路径因 Aspen 版本差异不存在           | COM Error        | Y        | 手动从 .bkp 文本解析 + 文献补充                              | 提取脚本报告缺失项  |
| DWSIM 自定义化合物注册       | 缺少临界属性导致 flash 计算失败               | ConvergenceError | Y        | Task 3.1 预检临界属性完整性                                  | 报错并列出缺失属性  |
| DWSIM 精馏塔不收敛           | 氢同位素体系在低温下初值敏感                  | ConvergenceError | Y        | 调整初始温度/压力估计；缩小塔规模先跑 2 组分                 | 测试标记 XFAIL      |
| Aspen 基准数据获取           | 无 Windows 机器可用                           | 环境缺失         | N        | **CRITICAL GAP** — 必须有 Windows + Aspen 许可证生成基准数据 | 阻塞全部对照测试    |

> **CRITICAL GAP**: Aspen 基准数据的生成依赖 Windows + Aspen 许可证环境。若该环境不可用，P0 无法完成。**降级策略**：使用 `run_dummy_simulation()` 的已有 CSV 数据（`v1_examples_legacy/handlers/i_iss_handler.csv`）作为近似基准，但需标注"未经独立验证"。

## 执行计划

### Phase 1: DWSIM Linux 环境部署

#### ✅ Task 1.1: 安装 .NET 8 Runtime 与 DWSIM 依赖
- **目标**：在 Linux 开发机上准备 DWSIM 运行环境
- **依赖**：无
- **修改内容**：
  - 新建 `script/dwsim/install_dwsim_linux.sh` — 安装脚本（.NET 8 Runtime + 系统依赖 + DWSIM 下载解压）
- **修改边界**：不修改系统级包管理器配置文件（仅用户空间安装），不修改项目现有文件
- **测试要求**：
  - 运行 `dotnet --info` 确认 .NET 8 可用
  - 预期输出：版本号包含 `8.0`
- **验收标准**：
  - ✅ `dotnet --info` 输出包含 `.NET Runtime 8.0.x`
  - ✅ DWSIM 目录存在且包含 `DWSIM.Automation.dll`
- **潜在风险**：部分 Linux 发行版（如 CentOS 7）的 glibc 版本过低不支持 .NET 8；需要 Ubuntu 20.04+ 或等效

#### ✅ Task 1.2: DWSIM headless 冒烟测试
- **目标**：验证 DWSIM 可在 Linux 上无 GUI 运行，能创建流程、添加化合物、求解
- **依赖**：T1.1
- **修改内容**：
  - 新建 `script/dwsim/smoke_test_dwsim.py` — Python.NET 冒烟测试（创建流程 → 添加 Water + Ethanol → 创建物料流 → 设温压 → 求解 → 验证结果非零）
  - 新建 `script/dwsim/smoke_test_dwsim.cs`（备选）— C# 控制台冒烟测试
- **修改边界**：不修改 DWSIM 源码，不修改项目现有文件
- **测试要求**：
  - 运行 `python script/dwsim/smoke_test_dwsim.py`
  - 预期输出：打印 "DWSIM smoke test PASSED" 且退出码 0
- **验收标准**：
  - ✅ Python.NET 成功加载 `DWSIM.Automation.dll` **或** C# 备选脚本编译运行成功
  - ✅ 创建的流程可求解，物料流温度/压力结果与设定值一致
- **潜在风险**：Python.NET 的 `coreclr` runtime 加载可能需要设置 `DOTNET_ROOT` 和 `PYTHONNET_RUNTIME` 环境变量

### Phase 2: Aspen 参数提取（Windows 端执行）

#### ✅ Task 2.1: 编写 Aspen COM 参数提取脚本
- **目标**：从 `T2-Threetowers4.bkp` 中提取全部构建 DWSIM 等效模型所需的参数
- **依赖**：无（可与 Phase 1 并行）
- **执行者**：需在 Windows + Aspen Plus 环境执行
- **修改内容**：
  - 新建 `script/dwsim/extract_aspen_params.py` — Aspen COM 提取脚本，输出 JSON
  - 输出文件 `example/example_dwsim/aspen_params.json` 包含：
    - `compounds`: 6 个化合物的临界属性（Tc, Pc, omega, Tb, MW, Vc）
    - `bips`: SRK 二元交互参数矩阵（15 对：kij, lij）
    - `columns`: 3 个塔的配置（塔板数、进料板位置、冷凝器/再沸器类型、操作压力、规格类型与值）
    - `streams`: 流股拓扑（连接关系、初始温压）
- **修改边界**：不修改 `.bkp` 文件本身，不修改 `i_iss_handler.py`
- **测试要求**：
  - 在 Windows 上运行 `python script/dwsim/extract_aspen_params.py --bkp example/example_aspenbkp/T2-Threetowers4.bkp --output example/example_dwsim/aspen_params.json`
  - 预期输出：JSON 文件包含 `compounds`（6 项）、`bips`（≥15 对）、`columns`（3 项）
- **验收标准**：
  - ✅ JSON 中 6 个化合物均有 Tc/Pc/omega/MW 四个临界属性
  - ✅ BIP 矩阵包含 15 对 kij 值（6 组分的组合数 C(6,2)=15）
  - ✅ 3 个塔均有 stage_count、feed_stage、condenser_type、reboiler_type、pressure
- **潜在风险**：Aspen COM 树节点路径因版本（V11 vs V14）不同而变化；脚本需处理多个候选路径

#### ✅ Task 2.2: 生成 Aspen 基准数据集
- **目标**：用 Aspen 跑 5 组有代表性的进料工况，记录精确的输出结果作为对照基准
- **依赖**：T2.1（需要 Aspen 环境可用）
- **执行者**：需在 Windows + Aspen Plus 环境执行
- **修改内容**：
  - 新建 `script/dwsim/generate_aspen_baseline.py` — 调用 `AspenEnhanced` 对 5 组进料逐一运行
  - 输出文件 `test/dwsim/fixtures/aspen_baseline.csv`
- **修改边界**：不修改 `i_iss_handler.py`；仅导入使用 `AspenEnhanced` 类
- **测试要求**：
  - 运行脚本生成 CSV，包含 5 行 × 16 列（7 输入 + 9 输出）
  - 预期：所有 T 流量值 > 0，质量守恒相对误差 < 1%
- **验收标准**：
  - ✅ CSV 包含 5 组工况的完整输入和输出
  - ✅ 每组工况的 H+D+T 质量守恒（入 ≈ 出），相对误差 < 1%
- **潜在风险**：Aspen 许可证不可用 → 降级使用 `i_iss_handler.csv` 已有数据
- **5 组测试工况设计**：

| 工况 | T_flow (g/h) | D_flow (g/h) | H_flow (g/h) | 设计意图              |
| ---- | ------------ | ------------ | ------------ | --------------------- |
| TC1  | 100.0        | 50.0         | 10.0         | 标称工况（高 T 富集） |
| TC2  | 50.0         | 50.0         | 50.0         | 等摩尔进料            |
| TC3  | 150.0        | 10.0         | 5.0          | 近纯 T 进料           |
| TC4  | 10.0         | 100.0        | 50.0         | 低 T 高 D 进料        |
| TC5  | 80.0         | 30.0         | 20.0         | 中间工况              |

### Phase 3: DWSIM 三塔模型构建

#### ✅ Task 3.1: 注册氢同位素自定义化合物
- **目标**：在 DWSIM 中注册 H2/HD/D2/HT/DT/T2 六个化合物，确保临界属性和蒸汽压曲线正确
- **依赖**：T1.2, T2.1
- **修改内容**：
  - 新建 `script/dwsim/register_compounds.py` — 读取 `aspen_params.json`，通过 DWSIM Compound Creator API 注册 6 个化合物
  - 若 DWSIM 内置数据库已有 H2，检查其属性是否与 Aspen 一致；不一致则覆盖
- **修改边界**：不修改 DWSIM 源码，仅通过 API 注册用户化合物
- **测试要求**：
  - 注册后，对每个化合物执行 flash 计算（给定 T=25K, P=1atm），验证相态判断合理（H2 在 25K 应为气相或超临界，D2 同理）
  - 预期：6 个化合物均能通过 flash 计算不报错
- **验收标准**：
  - ✅ 6 个化合物均在 DWSIM 中注册成功
  - ✅ 每个化合物的 Tc/Pc/omega 与 `aspen_params.json` 中的值一致（相对偏差 < 0.1%）
  - ✅ 对 H2 进行 T=30K, P=2atm 的 flash 计算不报 ConvergenceError
- **潜在风险**：HD/HT/DT 等异核分子的蒸汽压曲线不在常见数据库中，需要从文献拟合 Antoine 或 Wagner 系数

#### ✅ Task 3.2: 配置 SRK 物性包与 BIP
- **目标**：在 DWSIM 中创建 SRK 物性包并设置 15 对二元交互参数
- **依赖**：T3.1
- **修改内容**：
  - 在 `script/dwsim/build_dwsim_flowsheet.py` 中添加物性包配置逻辑
  - 通过 DWSIM API 设置 `SRK` 物性包的 kij 矩阵
- **修改边界**：不修改 DWSIM 源码
- **测试要求**：
  - 创建一个二元体系（H2+D2），计算 T=25K, P=1atm 下的 VLE，与文献值对比
  - 预期：相对挥发度 α(H2/D2) 在 1.5-2.5 范围内（文献典型值约 1.8@25K）
- **验收标准**：
  - ✅ DWSIM 的 SRK 物性包成功加载 15 对 BIP
  - ✅ H2/D2 二元 VLE 的相对挥发度在文献范围内
- **潜在风险**：DWSIM 的 SRK BIP 矩阵存储格式可能与 Aspen 的 `kij` 定义（是否对称、温度依赖项）不一致 → 需核查 DWSIM 源码中 `SRKPropertyPackage` 的 BIP 设置方法

#### ✅ Task 3.3: 构建三塔精馏流程
- **目标**：在 DWSIM 中构建与 Aspen T2-Threetowers4.bkp 等效的三塔（CD1/CD2/CD3）精馏流程
- **依赖**：T3.2, T2.1（塔规格数据）
- **修改内容**：
  - 完善 `script/dwsim/build_dwsim_flowsheet.py` — 添加三塔创建、流股连接、塔规格设置
  - 输出 `example/example_dwsim/T2_Threetowers4.dwxmz` — DWSIM 流程文件
- **修改边界**：不修改 Aspen .bkp，不修改项目现有文件
- **测试要求**：
  - 运行 `python script/dwsim/build_dwsim_flowsheet.py`
  - 预期输出：生成 `.dwxmz` 文件且无异常；文件大小 > 1KB
- **验收标准**：
  - ✅ `.dwxmz` 文件可被 DWSIM 加载（`LoadFlowsheet()` 不报错）
  - ✅ 流程包含 3 个 DistillationColumn 对象、≥6 个 MaterialStream 对象
  - ✅ 各塔的 stage count、feed stage 与 `aspen_params.json` 一致
- **潜在风险**：DWSIM DistillationColumn API 可能不支持某些 Aspen 特有的塔规格（如 Design Spec / Vary 对）→ 需用等效的 condenser/reboiler spec 替代

#### ✅ Task 3.4: 单点手动对照
- **目标**：对 TC1 工况执行 DWSIM 求解，与 Aspen 结果人工对比，确认模型构建正确
- **依赖**：T3.3, T2.2
- **修改内容**：
  - 新建 `script/dwsim/run_dwsim_point.py` — 加载 `.dwxmz`，设置进料组成，求解，输出 9 个结果值
- **修改边界**：不修改 `.dwxmz`（只读加载后修改内存中的流股参数）
- **测试要求**：
  - 运行 `python script/dwsim/run_dwsim_point.py --case TC1`
  - 手动检查 9 个输出值与 Aspen baseline TC1 的偏差
- **验收标准**：
  - ✅ DWSIM 求解收敛（无 ConvergenceError）
  - ✅ 9 个输出中至少 6 个的相对偏差 < 20%（单点粗验，精确验证在 Phase 4）
- **潜在风险**：精馏塔不收敛 — 这是 P0 最大技术风险。缓解：(a) 先用 shortcut column 验证物性包正确性，再切换 rigorous column；(b) 手动调整初始温度估计

### Phase 4: 自动化对照测试

#### ✅ Task 4.1: 编写对照测试框架
- **目标**：创建 pytest 测试，自动比较 Aspen 基准 vs DWSIM 结果
- **依赖**：T3.4（确认 DWSIM 能收敛后再做自动化）
- **修改内容**：
  - 新建 `test/dwsim/__init__.py`
  - 新建 `test/dwsim/test_aspen_dwsim_parity.py` — 参数化测试，遍历 5 个工况
  - 新建 `test/dwsim/conftest.py` — fixture 加载 baseline CSV + DWSIM flowsheet
- **修改边界**：不修改 `test/` 下现有测试文件，不修改 `conftest.py`（如果根目录有的话）
- **测试要求**：
  - 运行 `pytest test/dwsim/ -v`
  - 预期输出：5 个参数化用例，每个检查 9 个输出值
- **验收标准**：
  - ✅ 测试框架可运行且不报 import 错误
  - ✅ 每个用例输出偏差表格（变量名、Aspen 值、DWSIM 值、相对偏差、PASS/FAIL）
- **潜在风险**：DWSIM 单次求解慢（>30s）→ 5 个用例可能需要 >3min → 添加 `@pytest.mark.slow` 标记

#### ✅ Task 4.2: 生成 Go/No-Go 报告
- **目标**：汇总对照结果，输出结构化报告，给出明确的可行性结论
- **依赖**：T4.1
- **修改内容**：
  - 新建 `script/dwsim/generate_parity_report.py` — 读取测试输出，生成 Markdown 报告 + 对比图
  - 输出 `example/example_dwsim/parity_report.md`
- **修改边界**：不修改项目现有文件
- **测试要求**：
  - 运行 `python script/dwsim/generate_parity_report.py`
  - 预期输出：Markdown 报告包含汇总表 + Go/No-Go 判定
- **验收标准**：
  - ✅ 报告包含 5 工况 × 9 输出 = 45 个对比点的偏差统计
  - ✅ Go/No-Go 判定依据明确：Go = 45 个点中 ≥90% 满足阈值（相对偏差 <5% 或绝对偏差 <0.01 g/h）
  - ✅ 报告列出所有 FAIL 点的偏差值与可能原因
- **潜在风险**：若不满足 Go 条件，报告需给出 Root Cause 分析（物性包差异 vs 塔求解器差异 vs 参数迁移错误）

## Execution Wave（并行执行波次）

| Wave | 可并行 Task | 依赖已完成       |
| ---- | ----------- | ---------------- |
| W1   | T1.1, T2.1  | —                |
| W2   | T1.2, T2.2  | W1               |
| W3   | T3.1        | W2 (T1.2 + T2.1) |
| W4   | T3.2        | W3               |
| W5   | T3.3        | W4               |
| W6   | T3.4        | W5 + T2.2        |
| W7   | T4.1        | W6               |
| W8   | T4.2        | W7               |

> T1.x（Linux 环境）和 T2.x（Windows Aspen 端）可在不同机器上并行执行。关键路径为 T2.1 → T3.1 → T3.2 → T3.3 → T3.4 → T4.1 → T4.2。

## 回归检查清单

- [ ] 现有测试通过：`pytest -v test/` 无新增失败（P0 不修改现有代码，此项应天然通过）
- [ ] 无新增 lint 警告：`ruff check script/dwsim/ test/dwsim/`
- [ ] DWSIM 流程文件可复现：`build_dwsim_flowsheet.py` 在不同 Linux 机器上生成的 `.dwxmz` 可加载并求解
- [ ] 参数提取可复现：`extract_aspen_params.py` 在不同 Windows + Aspen 环境下输出相同的 JSON（精确到 6 位小数）
- [ ] 5 组基准数据已版本化：`test/dwsim/fixtures/aspen_baseline.csv` 已 commit

## 审查日志

| 轮次     | 聚焦                       | 发现问题数 | 已修正 | 剩余  |
| -------- | -------------------------- | ---------- | ------ | ----- |
| R1       | 结构完整性                 | 3          | 3      | 0     |
| R1.5     | 外部引用事实核查           | 2          | 2      | 0     |
| R2       | 可执行性（含脚本干跑）     | 2          | 2      | 0     |
| R3       | 风险与边缘（含跨轮一致性） | 1          | 1      | 0     |
| **终止** | **T1 — 收敛终止**          |            |        | **0** |

### Completion Summary

| 维度               | 结果                                                     |
| ------------------ | -------------------------------------------------------- |
| 背景与目标         | 完整                                                     |
| 技术方案           | 完整                                                     |
| Error & Rescue Map | 6 条路径，1 CRITICAL GAP（Aspen 环境不可用时有降级策略） |
| 执行计划           | 4 Phase、10 Task                                         |
| 回归检查清单       | 5 项（含项目特定检查）                                   |
| 已知局限           | 无                                                       |

### R1 Issues
- **Issue R1-1**: 缺少 Error & Rescue Map → 已补充 6 条失败路径 ✅ 已修正
- **Issue R1-2**: Task 2.2 缺少具体测试工况设计 → 已补充 5 组工况表 ✅ 已修正
- **Issue R1-3**: 缺少已有代码复用分析 → 已补充 `run_dummy_simulation` 等 4 项复用点 ✅ 已修正

### R1.5 Issues
- **Issue R1.5-1**: DWSIM Linux 支持声明需要验证 → [verified: GitHub README "Linux (64-bit x86) with .NET 8 Runtime or newer"] ✅ 已修正
- **Issue R1.5-2**: DWSIM DistillationColumn API（`SetNumberOfStages`, `ConnectFeed`, `SetCondenserSpec`）需要验证存在 → [verified: DWSIM.Automation.Tests.CSharp/distColumn.cs:L36-88，API 签名确认] ✅ 已修正

### R2 Issues
- **Issue R2-1**: Task 4.1 的验收标准中 "偏差表格" 格式未具体化 → 已细化为"变量名、Aspen 值、DWSIM 值、相对偏差、PASS/FAIL" ✅ 已修正
- **Issue R2-2**: Go/No-Go 阈值定义含糊 → 已明确为"45 个对比点中 ≥90% 满足阈值" ✅ 已修正

### R3 Issues
- **Issue R3-1**: 精馏塔不收敛是 P0 最大技术风险，但 Task 3.4 的风险缓解措施（shortcut column 降级）可能导致 Task 4.1 的对照测试使用不同塔模型 → 已在 T3.4 风险说明中明确：shortcut column 仅用于物性包验证，最终对照必须用 rigorous column ✅ 已修正

## Pre-Delivery Audit (Level: L1-Lite)

| §   | Check            | Status | Note                          |
| --- | ---------------- | ------ | ----------------------------- |
| 1   | Unit consistency | ✅ PASS | 全文统一使用 g/h, mol, K, atm |

Auditor: Plan Architect | Date: 2026-04-30
