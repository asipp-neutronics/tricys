import logging

import fmpy

from tricys.online_cosim.processor_base import AbstractTrackProcessor
from tricys.online_cosim.schema import (
    TrackProcessorContext,
    TrackResult,
    UnifiedStateVector,
)

logger = logging.getLogger(__name__)


class LocalMassCompensatorWrapper(AbstractTrackProcessor):
    """
    通用局部质量补偿包装器 (基于反射机制)。
    它拦截原 Processor 的输入输出，假定所有边界输入和输出均为质量流量。
    应用带死区的漏桶算法修正流量，从而保证局部质量严格守恒。
    """

    def __init__(
        self,
        delegate: AbstractTrackProcessor,
        deadband_g: float = 0.1,
        relaxation_time_h: float = 1.0,
        audit_indices: list[int] | None = None,
        in_ports: list[str] | None = None,
        out_ports: list[str] | None = None,
    ):
        self.delegate = delegate
        self.deadband_g = deadband_g
        self.relaxation_time_h = relaxation_time_h

        self.audit_indices = audit_indices if audit_indices is not None else [0]
        self.in_ports = in_ports
        self.out_ports = out_ports

        self.pending_error_pool: dict[int, float] = {
            idx: 0.0 for idx in self.audit_indices
        }
        self.last_inventory: dict[int, float] = {idx: 0.0 for idx in self.audit_indices}

    def _get_delegate_inventories(self) -> dict[int, float]:
        if hasattr(self.delegate, "get_component_inventories"):
            return self.delegate.get_component_inventories()
        # Fallback to legacy single-float inventory, mapped to the first audit index
        idx = self.audit_indices[0] if self.audit_indices else 0
        return {idx: self.delegate.get_mass_inventory()}

    def initialize(self, context: TrackProcessorContext) -> None:
        self.delegate.initialize(context)
        # Parse mass control config if present in context
        lmc = context.config.get("local_mass_control", {})
        if "audit_indices" in lmc:
            self.audit_indices = lmc["audit_indices"]
            self.pending_error_pool = {idx: 0.0 for idx in self.audit_indices}
        if "in_ports" in lmc:
            self.in_ports = lmc["in_ports"]
        if "out_ports" in lmc:
            self.out_ports = lmc["out_ports"]

        invs = self._get_delegate_inventories()
        for idx in self.audit_indices:
            self.last_inventory[idx] = invs.get(idx, 0.0)

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        dt = request_vector.dt_slow_h

        # 1. 统计白名单输入端口的质量流
        total_inflow = {idx: 0.0 for idx in self.audit_indices}
        for port_name, value in request_vector.boundary_inputs.items():
            if self.in_ports is not None and port_name not in self.in_ports:
                continue
            if self.in_ports is None and "cumulative" in port_name.lower():
                # 默认后备：无白名单时，跳过明显非流速的端口
                continue

            for idx in self.audit_indices:
                if isinstance(value, list) and len(value) > idx:
                    total_inflow[idx] += float(value[idx])
                elif not isinstance(value, list) and idx == 0:
                    total_inflow[idx] += float(value)

        # 2. 调用真实的 Processor (黑盒)
        result = self.delegate.process(request_vector)

        if getattr(result, "fallback_to_fmu", False):
            # 发生回退时，FMU自身物理方程保证质量守恒，挂起本补偿器并直通返回
            for idx in self.audit_indices:
                self.pending_error_pool[idx] = 0.0
            return result

        # 3. 提取真实的计算结果：原始总流出量与新滞留量
        total_outflow_raw = {idx: 0.0 for idx in self.audit_indices}
        for port_name, value in result.outputs.items():
            # 仅在未指定 out_ports 白名单时，或者在白名单内时，才计入流出
            if self.out_ports is not None and port_name not in self.out_ports:
                continue
            if self.out_ports is None and "cumulative" in port_name.lower():
                continue

            for idx in self.audit_indices:
                if isinstance(value, list) and len(value) > idx:
                    total_outflow_raw[idx] += float(value[idx])
                elif not isinstance(value, list) and idx == 0:
                    total_outflow_raw[idx] += float(value)

        # 修正: 加上没有在物理端口中输出但消耗了质量的衰变和泄漏
        # TODO: 衰变和泄漏如果有多个组分，这里默认加在 idx=0 上（Tritium 衰变）
        if "decay_rate" not in result.outputs and 0 in self.audit_indices:
            total_outflow_raw[0] += self.get_decay_rate()
        if "release_rate" not in result.outputs and 0 in self.audit_indices:
            total_outflow_raw[0] += self.get_release_rate()

        new_invs = self._get_delegate_inventories()

        # 4. === 漏桶算法核心 (多组分) ===
        correction_flow_rates = {idx: 0.0 for idx in self.audit_indices}
        for idx in self.audit_indices:
            new_inv = new_invs.get(idx, 0.0)
            local_error = (total_inflow[idx] - total_outflow_raw[idx]) * dt - (
                new_inv - self.last_inventory[idx]
            )
            self.pending_error_pool[idx] += local_error

            if self.pending_error_pool[idx] > self.deadband_g:
                excess = self.pending_error_pool[idx] - self.deadband_g
                correction_flow_rates[idx] = excess / self.relaxation_time_h
            elif self.pending_error_pool[idx] < -self.deadband_g:
                excess = self.pending_error_pool[idx] + self.deadband_g
                correction_flow_rates[idx] = excess / self.relaxation_time_h

            self.pending_error_pool[idx] -= correction_flow_rates[idx] * dt
            self.last_inventory[idx] = new_inv

        # 5. 将补偿流量分配给输出端口
        if result.outputs and any(abs(v) > 0.0 for v in correction_flow_rates.values()):
            # 取指定的第一个 out_port 或者字典里的第一个 port
            main_out_port = (
                self.out_ports[0] if self.out_ports else list(result.outputs.keys())[0]
            )
            val = result.outputs.get(main_out_port)
            if val is not None:
                if isinstance(val, list):
                    new_val = list(val)
                    for idx in self.audit_indices:
                        if len(new_val) > idx:
                            new_val[idx] = (
                                float(new_val[idx]) + correction_flow_rates[idx]
                            )
                    result.outputs[main_out_port] = new_val
                else:
                    if 0 in self.audit_indices:
                        result.outputs[main_out_port] = (
                            float(val) + correction_flow_rates[0]
                        )

                # 收集日志
                corrections_log = ", ".join(
                    f"idx[{idx}]={corr:.4e}"
                    for idx, corr in correction_flow_rates.items()
                    if abs(corr) > 0
                )
                logger.info(
                    f"[{self.delegate.__class__.__name__}] 局部补偿激活: 端口={main_out_port}, 修正量=[{corrections_log}] g/h"
                )

        return result

    def finalize(self) -> None:
        self.delegate.finalize()

    def get_mass_inventory(self) -> float:
        base_inv = self.delegate.get_mass_inventory()
        # The pending_error_pool represents mass that has been numerically buffered
        # by the compensator but not yet discharged to the physical network.
        # It must be reported to the Auditor, otherwise it appears as global mass drift.
        buffered_mass = self.pending_error_pool.get(0, 0.0)
        return base_inv + buffered_mass

    def get_decay_rate(self) -> float:
        if hasattr(self.delegate, "get_decay_rate"):
            return self.delegate.get_decay_rate()
        return 0.0

    def get_release_rate(self) -> float:
        if hasattr(self.delegate, "get_release_rate"):
            return self.delegate.get_release_rate()
        return 0.0


class ShadowFMUFallbackWrapper(AbstractTrackProcessor):
    """
    带独立原生 FMU 引擎的降级容错包装器。
    在内存中挂载一个与主链隔离的“影子 FMU”，利用 fmpy 独立推演物理状态。
    当内部的 HighPrecisionProcessor 发生异常崩溃时，直接输出影子 FMU 的原生结果。
    """

    def __init__(self, delegate: AbstractTrackProcessor):
        self.delegate = delegate
        self.is_crashed = False
        self.fmu_instance = None
        self.fmu_model_description = None
        self.input_vars = {}
        self.output_vars = {}

    def initialize(self, context: TrackProcessorContext) -> None:
        try:
            self.delegate.initialize(context)
        except Exception as e:
            logger.error(
                f"[{self.__class__.__name__}] Delegate initialization failed: {e}"
            )
            self.is_crashed = True

        fmu_path = context.config.get("shadow_fmu_path")
        if not fmu_path:
            logger.error(
                f"[{self.__class__.__name__}] 'shadow_fmu_path' not provided in config. Shadow FMU will not be available."
            )
            return

        try:
            self.fmu_model_description = fmpy.read_model_description(fmu_path)
            unzipdir = fmpy.extract(fmu_path)
            self.fmu_instance = fmpy.instantiate_fmu(
                unzipdir=unzipdir, model_description=self.fmu_model_description
            )
            self.fmu_instance.instantiate()
            self.fmu_instance.setupExperiment(startTime=0.0)
            self.fmu_instance.enterInitializationMode()
            self.fmu_instance.exitInitializationMode()

            # 建立变量名称到 Variable 对象的映射
            for var in self.fmu_model_description.modelVariables:
                if var.causality == "input":
                    self.input_vars[var.name] = var
                elif var.causality == "output":
                    self.output_vars[var.name] = var

            logger.info(
                f"[{self.__class__.__name__}] Successfully loaded shadow FMU from {fmu_path}"
            )
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Failed to load shadow FMU: {e}")
            self.fmu_instance = None

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        # 1. 独立步进影子 FMU (伴随式推演)
        if self.fmu_instance:
            try:
                # 注入边界输入
                vrs = []
                vals = []
                for name, value in request_vector.boundary_inputs.items():
                    if isinstance(value, list):
                        for i, v in enumerate(value):
                            arr_name = f"{name}[{i+1}]"
                            if arr_name in self.input_vars:
                                vrs.append(self.input_vars[arr_name].valueReference)
                                vals.append(float(v))
                    else:
                        if name in self.input_vars:
                            vrs.append(self.input_vars[name].valueReference)
                            vals.append(float(value))

                if vrs:
                    self.fmu_instance.setReal(vrs, vals)

                # 执行步进 (此处 CFEDR 模型内部方程的 time 实际上是小时，所以不能乘 3600)
                current_time_s = float(request_vector.current_time_h)
                dt_s = float(request_vector.dt_slow_h)
                self.fmu_instance.doStep(
                    currentCommunicationPoint=current_time_s, communicationStepSize=dt_s
                )
            except Exception as e:
                logger.error(f"[{self.__class__.__name__}] Shadow FMU step failed: {e}")

        # 2. 尝试执行主链的高精度计算
        if not self.is_crashed:
            try:
                result = self.delegate.process(request_vector)

                # ==== 强制状态同步 (State Synchronization) ====
                # 将高精度计算完毕后的连续状态（如 inventory）强行覆写回影子 FMU，
                # 以彻底消除两者并行推演带来的状态漂移。
                if self.fmu_instance:
                    try:
                        ai_inventory = self.delegate.get_mass_inventory()
                        inventory_vr = None
                        for var in self.fmu_model_description.modelVariables:
                            # 模糊匹配寻找核心连续状态变量
                            if var.name.endswith("inventory") or var.name.endswith(
                                "inventory[1]"
                            ):
                                inventory_vr = var.valueReference
                                break

                        if inventory_vr is not None:
                            self.fmu_instance.setReal(
                                [inventory_vr], [float(ai_inventory)]
                            )
                            logger.debug(
                                f"[{self.__class__.__name__}] Force synced shadow FMU inventory to {ai_inventory:.4f}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[{self.__class__.__name__}] Failed to force sync state to shadow FMU: {e}"
                        )

                return result
            except Exception as e:
                logger.error(
                    f"[{self.__class__.__name__}] Delegate '{self.delegate.__class__.__name__}' "
                    f"crashed at t={request_vector.current_time_h:.4f}h! "
                    f"Triggering shadow FMU fallback. Error: {e}",
                    exc_info=True,
                )
                self.is_crashed = True

        # 3. 发生降级：提取并返回影子 FMU 的原生计算结果
        outputs = {}
        if self.fmu_instance:
            try:
                vrs = [var.valueReference for var in self.output_vars.values()]
                names = list(self.output_vars.keys())
                if vrs:
                    vals = self.fmu_instance.getReal(vrs)
                    for name, val in zip(names, vals):
                        if "[" in name and name.endswith("]"):
                            base_name, idx_str = name.split("[")
                            idx = int(idx_str[:-1])
                            if base_name not in outputs:
                                outputs[base_name] = {}
                            outputs[base_name][idx] = val
                        else:
                            outputs[name] = val

                    # Convert dicts of arrays back to lists
                    for k, v in list(outputs.items()):
                        if isinstance(v, dict):
                            max_idx = max(v.keys())
                            arr = [0.0] * max_idx
                            for i, val in v.items():
                                arr[i - 1] = val
                            outputs[k] = arr

                logger.warning(
                    f"[{self.__class__.__name__}] Formatted outputs: {list(outputs.keys())}"
                )
            except Exception as e:
                logger.error(
                    f"[{self.__class__.__name__}] Shadow FMU get outputs failed: {e}"
                )

        logger.warning(
            f"[{self.__class__.__name__}] Operating in fallback mode. Handing over to isolated Shadow FMU."
        )
        # 注意：此处返回正常的 TrackResult（不带 fallback_to_fmu=True），
        # 因为我们自己算出了原生结果，将其冒充正常结果交给主链，
        # 外层的 Compensator 可以继续对状态跳变进行平滑和补偿。
        return TrackResult(outputs=outputs)

    def finalize(self) -> None:
        try:
            self.delegate.finalize()
        except Exception as e:
            logger.error(f"[{self.__class__.__name__}] Finalize failed: {e}")

        if self.fmu_instance:
            try:
                self.fmu_instance.terminate()
                self.fmu_instance.freeInstance()
            except Exception as e:
                logger.error(
                    f"[{self.__class__.__name__}] Shadow FMU cleanup failed: {e}"
                )

    def get_mass_inventory(self) -> float:
        try:
            if not self.is_crashed:
                return self.delegate.get_mass_inventory()
        except Exception:
            pass

        # 尝试从影子 FMU 模糊匹配核心状态变量 (例如 sds.inventory[1] 或 inventory)
        if self.fmu_instance and hasattr(self, "fmu_model_description"):
            try:
                for var in self.fmu_model_description.modelVariables:
                    if var.name.endswith("inventory") or var.name.endswith(
                        "inventory[1]"
                    ):
                        vr = var.valueReference
                        return float(self.fmu_instance.getReal([vr])[0])
            except Exception:
                pass
        return 0.0

    def get_decay_rate(self) -> float:
        try:
            if not self.is_crashed and hasattr(self.delegate, "get_decay_rate"):
                return self.delegate.get_decay_rate()
        except Exception:
            pass
        return 0.0

    def get_release_rate(self) -> float:
        try:
            if not self.is_crashed and hasattr(self.delegate, "get_release_rate"):
                return self.delegate.get_release_rate()
        except Exception:
            pass
        return 0.0


class RuntimeTimeScaleCoordinatorWrapper(AbstractTrackProcessor):
    """
    双向时序协同包装器。
    接管主链(Runtime)与子组件之间的不同时间尺度步进，实现：
    - 主链低频、组件高频：输入FOH插值，输出积分累加ZOH。
    - 主链高频、组件低频：输入积分累加，输出FOH插值。
    """

    def __init__(
        self,
        delegate: AbstractTrackProcessor,
        component_step_size_h: float,
    ):
        self.delegate = delegate
        self.component_step_size_h = max(float(component_step_size_h), 1e-12)
        self.output_template = {}

        # 场景一状态
        self.last_inputs: dict | None = None

        # 场景二状态
        self.accumulated_mass_in: dict = {}
        self.sub_step_count: int = 0
        self.accumulation_start_time_h: float = 0.0
        self.last_outputs: dict | None = None
        self.current_outputs: dict | None = None
        self.last_output_time_h: float | None = None
        self.current_output_time_h: float | None = None

    def initialize(self, context: TrackProcessorContext) -> None:
        self.delegate.initialize(context)
        output_bindings = context.config.get("output_bindings", {})
        for out_name, binding_info in output_bindings.items():
            width = binding_info.get("width", 1)
            self.output_template[out_name] = [0.0] * width if width > 1 else 0.0

    def process(self, request_vector: UnifiedStateVector) -> TrackResult:
        dt_slow_h = request_vector.dt_slow_h

        # 设置容差，防止浮点数精度问题
        if abs(dt_slow_h - self.component_step_size_h) < 1e-9:
            # 步长一致，直接透传
            return self.delegate.process(request_vector)

        if dt_slow_h > self.component_step_size_h:
            return self._process_main_slow_comp_fast(request_vector)
        else:
            return self._process_main_fast_comp_slow(request_vector)

    def _process_main_slow_comp_fast(
        self, request_vector: UnifiedStateVector
    ) -> TrackResult:
        dt_slow_h = request_vector.dt_slow_h
        dt_comp_h = self.component_step_size_h
        m = max(int(dt_slow_h / dt_comp_h), 1)
        remainder_h = dt_slow_h - m * dt_comp_h

        logger.info(
            f"[{self.delegate.__class__.__name__} TimeScale] Main(Slow) dt={dt_slow_h}h -> Comp(Fast) dt={dt_comp_h}h. Executing {m} sub-steps at T={request_vector.current_time_h}h."
        )

        current_inputs = request_vector.boundary_inputs
        if self.last_inputs is None:
            self.last_inputs = current_inputs

        accumulated_out = {}
        total_accumulated_h = 0.0

        # Main sub-steps at component step size
        for i in range(1, m + 1):
            fraction = float(i) / m
            interp_inputs = {}
            for k, v in current_inputs.items():
                last_v = self.last_inputs.get(k, v)
                if isinstance(v, list):
                    interp_inputs[k] = [
                        last_val + fraction * (curr_val - last_val)
                        for last_val, curr_val in zip(last_v, v)
                    ]
                else:
                    interp_inputs[k] = last_v + fraction * (v - last_v)

            sub_req = UnifiedStateVector(
                component_name=request_vector.component_name,
                step_id=request_vector.step_id,
                seq_id=request_vector.seq_id,
                current_time_h=request_vector.current_time_h + (i - 1) * dt_comp_h,
                dt_slow_h=dt_comp_h,
                boundary_inputs=interp_inputs,
                extra_state=request_vector.extra_state,
                global_mass_error=request_vector.global_mass_error,
            )

            res = self.delegate.process(sub_req)
            if getattr(res, "fallback_to_fmu", False):
                self.last_inputs = current_inputs
                return res

            for k, v in res.outputs.items():
                if k not in accumulated_out:
                    accumulated_out[k] = [0.0] * len(v) if isinstance(v, list) else 0.0
                if isinstance(v, list):
                    for j in range(len(v)):
                        accumulated_out[k][j] += float(v[j]) * dt_comp_h
                else:
                    accumulated_out[k] += float(v) * dt_comp_h
            total_accumulated_h += dt_comp_h

        # Remainder sub-step to exactly cover dt_slow_h (fixes rounding gap)
        if remainder_h > 1e-12:
            interp_inputs = {}
            for k, v in current_inputs.items():
                last_v = self.last_inputs.get(k, v)
                if isinstance(v, list):
                    interp_inputs[k] = list(v)  # At fraction=1.0
                else:
                    interp_inputs[k] = v

            sub_req = UnifiedStateVector(
                component_name=request_vector.component_name,
                step_id=request_vector.step_id,
                seq_id=request_vector.seq_id,
                current_time_h=request_vector.current_time_h + m * dt_comp_h,
                dt_slow_h=remainder_h,
                boundary_inputs=interp_inputs,
                extra_state=request_vector.extra_state,
                global_mass_error=request_vector.global_mass_error,
            )

            res = self.delegate.process(sub_req)
            if not getattr(res, "fallback_to_fmu", False):
                for k, v in res.outputs.items():
                    if k not in accumulated_out:
                        accumulated_out[k] = (
                            [0.0] * len(v) if isinstance(v, list) else 0.0
                        )
                    if isinstance(v, list):
                        for j in range(len(v)):
                            accumulated_out[k][j] += float(v[j]) * remainder_h
                    else:
                        accumulated_out[k] += float(v) * remainder_h
                total_accumulated_h += remainder_h

        self.last_inputs = current_inputs

        avg_outputs = {}
        effective_dt = total_accumulated_h if total_accumulated_h > 0 else dt_slow_h
        for k, v in accumulated_out.items():
            if isinstance(v, list):
                avg_outputs[k] = [val / effective_dt for val in v]
            else:
                avg_outputs[k] = v / effective_dt

        return TrackResult(outputs=avg_outputs)

    def _process_main_fast_comp_slow(
        self, request_vector: UnifiedStateVector
    ) -> TrackResult:
        dt_slow_h = request_vector.dt_slow_h
        dt_comp_h = self.component_step_size_h
        m = int(round(dt_comp_h / dt_slow_h))
        if m == 0:
            m = 1

        current_inputs = request_vector.boundary_inputs
        logger.info(
            f"[{self.delegate.__class__.__name__} TimeScale] Main(Fast) dt={dt_slow_h}h -> Comp(Slow) dt={dt_comp_h}h. Accumulating step {self.sub_step_count + 1}/{m} at T={request_vector.current_time_h}h."
        )

        if self.sub_step_count == 0:
            self.accumulation_start_time_h = request_vector.current_time_h

        for k, v in current_inputs.items():
            if k not in self.accumulated_mass_in:
                self.accumulated_mass_in[k] = (
                    [0.0] * len(v) if isinstance(v, list) else 0.0
                )
            if isinstance(v, list):
                for j in range(len(v)):
                    self.accumulated_mass_in[k][j] += float(v[j]) * dt_slow_h
            else:
                self.accumulated_mass_in[k] += float(v) * dt_slow_h

        self.sub_step_count += 1

        if self.sub_step_count >= m:
            avg_inputs = {}
            for k, v in self.accumulated_mass_in.items():
                if isinstance(v, list):
                    avg_inputs[k] = [val / dt_comp_h for val in v]
                else:
                    avg_inputs[k] = v / dt_comp_h

            sub_req = UnifiedStateVector(
                component_name=request_vector.component_name,
                step_id=request_vector.step_id,
                seq_id=request_vector.seq_id,
                current_time_h=self.accumulation_start_time_h,
                dt_slow_h=dt_comp_h,
                boundary_inputs=avg_inputs,
                extra_state=request_vector.extra_state,
                global_mass_error=request_vector.global_mass_error,
            )

            res = self.delegate.process(sub_req)
            if getattr(res, "fallback_to_fmu", False):
                return res

            self.last_outputs = (
                self.current_outputs
                if self.current_outputs is not None
                else res.outputs
            )
            self.current_outputs = res.outputs

            logger.info(
                f"[{self.delegate.__class__.__name__} TimeScale] Fired component step at T={request_vector.current_time_h}h. Interpolating outputs for next {m} main steps."
            )

            self.last_output_time_h = self.current_output_time_h
            self.current_output_time_h = request_vector.current_time_h

            self.sub_step_count = 0
            self.accumulated_mass_in = {}

        if self.current_outputs is None:
            empty_out = {
                k: (list(v) if isinstance(v, list) else v)
                for k, v in self.output_template.items()
            }
            return TrackResult(outputs=empty_out)

        if (
            self.current_output_time_h is None
            or self.last_output_time_h is None
            or self.current_output_time_h == self.last_output_time_h
        ):
            fraction = 1.0
        else:
            fraction = (request_vector.current_time_h - self.last_output_time_h) / (
                self.current_output_time_h - self.last_output_time_h
            )

        interp_outputs = {}
        for k, v in self.current_outputs.items():
            last_v = self.last_outputs.get(k, v) if self.last_outputs else v
            if isinstance(v, list):
                interp_outputs[k] = [
                    last_val + fraction * (curr_val - last_val)
                    for last_val, curr_val in zip(last_v, v)
                ]
            else:
                interp_outputs[k] = last_v + fraction * (v - last_v)

        return TrackResult(outputs=interp_outputs)

    def finalize(self) -> None:
        self.delegate.finalize()

    def get_mass_inventory(self) -> float:
        delegate_inv = 0.0
        if hasattr(self.delegate, "get_mass_inventory"):
            delegate_inv = self.delegate.get_mass_inventory()
        # Include mass buffered in accumulated_mass_in (scene 2: main fast, comp slow)
        # This mass has been "received" from the main chain but not yet forwarded
        # to the delegate component. Without this, Auditor cannot see it.
        buffered_mass = 0.0
        for k, v in self.accumulated_mass_in.items():
            if isinstance(v, list):
                # Sum first component (tritium convention) for mass audit
                if len(v) > 0:
                    buffered_mass += float(v[0])
            else:
                buffered_mass += float(v)
        return delegate_inv + buffered_mass

    def get_decay_rate(self) -> float:
        if hasattr(self.delegate, "get_decay_rate"):
            return self.delegate.get_decay_rate()
        return 0.0

    def get_release_rate(self) -> float:
        if hasattr(self.delegate, "get_release_rate"):
            return self.delegate.get_release_rate()
        return 0.0
