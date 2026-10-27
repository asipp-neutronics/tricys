import io

import pandas as pd
import pytest

from tricys.visualizer.report.exporter import ReportExporter
from tricys.visualizer.report.templates import TEMPLATES_CATALOG, render_report


def test_templates_catalog_completeness():
    assert len(TEMPLATES_CATALOG) == 4
    template_ids = [t["id"] for t in TEMPLATES_CATALOG]
    assert "simulate" in template_ids
    assert "sweep" in template_ids
    assert "sensitivity" in template_ids
    assert "optimize" in template_ids

    from tricys.visualizer.report.templates import normalize_template_id

    # Default without template specified -> matches task.type
    assert normalize_template_id(None, "simulate") == "simulate"
    assert normalize_template_id("", "sweep") == "sweep"
    assert normalize_template_id(None, "sensitivity") == "sensitivity"
    assert normalize_template_id(None, "optimize") == "optimize"
    # Legacy alias mappings
    assert normalize_template_id("sweep_startup_self_sufficiency") == "sweep"
    assert normalize_template_id("simulate_pulsed_transient") == "simulate"
    assert normalize_template_id("sensitivity_sobol_gsa") == "sensitivity"
    assert normalize_template_id("nested_bisection") == "optimize"
    assert normalize_template_id("dual_basis") == "optimize"
    assert normalize_template_id("bisection_critical_tbr") == "optimize"
    assert normalize_template_id("kpi_metrics") == "sweep"


def test_render_all_templates():
    sample_config = {
        "model": {"modelName": "CFEDR.TritiumCycle"},
        "scenario": {"stopTime": 5000.0, "stepSize": 2.0},
        "task": {
            "type": "sweep",
            "method": "sobol",
            "samples": 64,
            "parameters": {"blanket.TBR": [1.05, 1.25]},
        },
    }
    sample_jobs = [{"id": 1, "blanket.TBR": 1.1}, {"id": 2, "blanket.TBR": 1.2}]
    sample_metrics_df = pd.DataFrame(
        [
            {"job_id": 1, "peak_inventory": 450.2, "settling_time": 1820.0},
            {"job_id": 2, "peak_inventory": 390.5, "settling_time": 1640.0},
        ]
    )

    for t in TEMPLATES_CATALOG:
        t_id = t["id"]
        rendered_md = render_report(
            template_id=t_id,
            config=sample_config,
            jobs=sample_jobs,
            metrics_df=sample_metrics_df,
            ai_insight="AI Recommendation: Maintain TBR above 1.15.",
        )
        assert isinstance(rendered_md, str)
        assert len(rendered_md) > 50
        assert "# TRICYS" in rendered_md


def test_render_optimize_template():
    sample_config = {
        "model": {"modelName": "example_model.Cycle"},
        "task": {
            "type": "optimize",
            "method": "nested_bisection",
            "optimizer": {
                "inner": {
                    "parameter": "blanket.TBR",
                    "bounds": [1.0, 1.4],
                    "tolerance": 0.005,
                    "target_metric": "I_reserve_req",
                    "threshold": 5000,
                },
                "outer": {
                    "parameter": "sds.I_S_0",
                    "bounds": [2000, 10000],
                    "tolerance": 100,
                    "target_metric": "Required_TBR",
                },
            },
        },
    }
    sample_jobs = [
        {"job_id": 1, "sds.I_S_0": 2000.0, "blanket.TBR": 1.35, "Required_TBR": 1.35},
        {"job_id": 2, "sds.I_S_0": 5000.0, "blanket.TBR": 1.25, "Required_TBR": 1.25},
        {"job_id": 3, "sds.I_S_0": 8000.0, "blanket.TBR": 1.18, "Required_TBR": 1.18},
    ]
    sample_metrics = pd.DataFrame(sample_jobs)

    rendered_md = render_report(
        template_id=None,
        config=sample_config,
        jobs=sample_jobs,
        metrics_df=sample_metrics,
        ai_insight="Abdou 2021 self-sufficiency frontier verified.",
    )
    assert "OPTIMIZE" in rendered_md
    assert "嵌套二分" in rendered_md or "nested_bisection" in rendered_md
    assert "Required TBR" in rendered_md
    assert "1.18" in rendered_md
    assert "Abdou 2021" in rendered_md


def test_export_markdown():
    text = "# Test Title\n\nSome paragraph text."
    md_output = ReportExporter.export_markdown(text)
    assert md_output.startswith("# Test Title")
    assert md_output.endswith("\n")


def test_export_docx():
    docx = pytest.importorskip("docx")

    md_text = """# TRICYS Executive Report
## 1. Summary
- Bullet item 1 with **bold text**
- Bullet item 2 with `code text`

| Param | Value |
| :--- | :--- |
| TBR | 1.15 |
| Peak | 450.0 |
"""
    docx_bytes = ReportExporter.export_docx(md_text, title="Test Report")
    assert isinstance(docx_bytes, bytes)
    assert len(docx_bytes) > 500

    # Verify reloading docx
    doc = docx.Document(io.BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    assert any("TRICYS Executive Report" in p for p in paragraphs)
    assert len(doc.tables) == 1
    assert len(doc.tables[0].rows) == 3


def test_export_pdf():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF / fitz is not installed")

    md_text = """# TRICYS Engineering PDF Report
## 1. General Info
This is an automated engineering PDF test report.

| Variable | Threshold | Status |
| :--- | :--- | :--- |
| Storage | 500.0 | Normal |

- Safety verification passed
"""
    pdf_bytes = ReportExporter.export_pdf(md_text, title="PDF Test Report")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500

    # Verify reloading with fitz
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert len(pdf_doc) >= 1
    first_page_text = pdf_doc[0].get_text()
    assert "TRICYS Engineering PDF Report" in first_page_text


def test_export_pdf_with_cjk_characters():
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF / fitz is not installed")

    cjk_md = """# TRICYS 聚变堆全厂氚循环工程仿真报告
## 1. 运行概要
本报告评估聚变堆全流程在役氚滞留量与自持增殖平衡。

| 参数指标 | 设计值 | 状态评估 |
| :--- | :--- | :--- |
| 增殖比 TBR | 1.15 | 合格 |
| 在役储氚 | 450.0 mol | 安全 |

- 质量守恒误差小于 1.0e-5，系统运行平稳。
"""
    pdf_bytes = ReportExporter.export_pdf(cjk_md)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500

    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    first_page_text = pdf_doc[0].get_text()
    # Ensure Chinese characters are not turned into dots
    assert "聚变堆全厂氚循环工程仿真报告" in first_page_text
    assert "在役储氚" in first_page_text


def test_render_simulate_with_required_tbr_optimization():
    config = {
        "model": {"modelName": "CFEDR.TritiumCycle"},
        "scenario": {"stopTime": 5000.0, "stepSize": 1.0},
        "task": {
            "type": "simulate",
            "metrics": {
                "peak_inventory": {"source": "sds.I[1]", "evaluate": "peak_value"},
                "Required_TBR": {
                    "evaluate": "bisection_search",
                    "parameter": "blanket.TBR",
                    "search_bounds": [1.0, 1.4],
                    "target_metric": {
                        "source": "sds.I[1]",
                        "evaluate": "doubling_time",
                        "threshold": 63072000.0,
                    },
                },
            },
        },
    }
    jobs = [{"job_id": 1, "blanket.TBR": 1.12, "Required_TBR": 1.152}]
    metrics_df = pd.DataFrame(
        [{"job_id": 1, "peak_inventory": 8420.0, "Required_TBR": 1.152}]
    )

    md = render_report(
        template_id="simulate", config=config, jobs=jobs, metrics_df=metrics_df
    )
    assert "任务目标性能与参数寻优指标清单" in md
    assert "Required_TBR" in md
    assert "1.152" in md


def test_render_sweep_with_embedded_optimization():
    config = {
        "model": {"modelName": "example_model.Cycle"},
        "task": {
            "type": "sweep",
            "optimization": {
                "track": "track2_bisection",
                "algorithm": "nested_bisection",
                "outer": {
                    "parameter": "sds.I_S_0",
                    "search_bounds": [1000.0, 5000.0],
                    "tolerance": 20.0,
                    "target_metric": {
                        "source": "sds.I[1]",
                        "evaluate": "min_value",
                        "threshold": 1000.0,
                    },
                },
                "inner": {
                    "metric_name": "Required_TBR",
                    "parameter": "blanket.TBR",
                    "search_bounds": [1.05, 1.25],
                    "tolerance": 0.005,
                    "target_metric": {
                        "source": "sds.I[1]",
                        "evaluate": "doubling_time",
                        "threshold": 2500.0,
                    },
                },
            },
        },
    }
    jobs = [
        {"job_id": 1, "plasma.fb": 0.03, "sds.I_S_0": 3500.0, "Required_TBR": 1.22},
        {"job_id": 2, "plasma.fb": 0.05, "sds.I_S_0": 2800.0, "Required_TBR": 1.15},
        {"job_id": 3, "plasma.fb": 0.08, "sds.I_S_0": 2200.0, "Required_TBR": 1.09},
    ]
    metrics_df = pd.DataFrame(jobs)

    md = render_report(
        template_id="sweep", config=config, jobs=jobs, metrics_df=metrics_df
    )
    assert "批量内嵌寻优数学架构与自持准则" in md
    assert "内层寻优" in md
    assert "外层寻优" in md
    assert "批量寻优指标图与多工况响应说明" in md


def test_generate_optimization_figures_with_2d_parameter_group(tmp_path):
    from tricys.visualizer.report.figures import generate_all_report_figures

    config = {
        "task": {
            "type": "sweep",
            "optimization": {
                "outer": {"parameter": "sds.I_S_0"},
                "inner": {"metric_name": "Required_TBR", "parameter": "blanket.TBR"},
            },
        }
    }
    # 2D Parameter Group: primary is plasma.fb, secondary is comp9_ISS.tau_9
    jobs_data = [
        {
            "job_id": 1,
            "plasma.fb": 0.03,
            "comp9_ISS.tau_9": 1.0,
            "sds.I_S_0": 3800.0,
            "Required_TBR": 1.24,
        },
        {
            "job_id": 2,
            "plasma.fb": 0.05,
            "comp9_ISS.tau_9": 1.0,
            "sds.I_S_0": 3000.0,
            "Required_TBR": 1.16,
        },
        {
            "job_id": 3,
            "plasma.fb": 0.08,
            "comp9_ISS.tau_9": 1.0,
            "sds.I_S_0": 2400.0,
            "Required_TBR": 1.10,
        },
        {
            "job_id": 4,
            "plasma.fb": 0.03,
            "comp9_ISS.tau_9": 12.0,
            "sds.I_S_0": 4200.0,
            "Required_TBR": 1.28,
        },
        {
            "job_id": 5,
            "plasma.fb": 0.05,
            "comp9_ISS.tau_9": 12.0,
            "sds.I_S_0": 3300.0,
            "Required_TBR": 1.20,
        },
        {
            "job_id": 6,
            "plasma.fb": 0.08,
            "comp9_ISS.tau_9": 12.0,
            "sds.I_S_0": 2600.0,
            "Required_TBR": 1.13,
        },
    ]
    jobs_df = pd.DataFrame(jobs_data)
    metrics_df = jobs_df.copy()

    figs = generate_all_report_figures(
        jobs_df=jobs_df,
        metrics_df=metrics_df,
        task_type="sweep",
        output_dir=str(tmp_path),
        task_config=config,
    )
    fig_titles = [f["title"] for f in figs]
    assert any("Required TBR" in t for t in fig_titles)
    assert any("最优初始启动库存" in t for t in fig_titles)
    # Check that output files were generated
    fig_files = list(tmp_path.glob("**/*.png"))
    assert len(fig_files) >= 2
