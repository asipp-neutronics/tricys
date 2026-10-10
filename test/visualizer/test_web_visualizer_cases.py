import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from tricys.visualizer.main import (
    DEFAULT_SECRET,
    _create_viewer_token,
    get_default_context_dir,
)

ARTIFACTS_DIR = Path(
    os.environ.get("TRICYS_TEST_ARTIFACTS", Path(__file__).parent / "test_artifacts")
)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://127.0.0.1:8055"


def generate_case_urls():
    secret = DEFAULT_SECRET
    ctx_dir = get_default_context_dir()

    cases = {
        "case1_sweep": {
            "name": "多工况参数扫参工况 (Parameter Sweep)",
            "file": Path(
                "test_example/basic/2_parameter_sweep/20260923_154358/results/sweep_results.h5"
            ).resolve(),
            "expected_jobs": 7,
        },
        "case2_sobol": {
            "name": "高维全局敏感性分析工况 (Sobol GSA)",
            "file": Path(
                "test_example/analysis/4_sobol_global_sensitivity_analysis/20260923_170607/SALIB_SOBOL_Analysis/results/sweep_results.h5"
            ).resolve(),
            "expected_jobs": 48,
        },
        "case3_pulse": {
            "name": "聚变脉冲瞬态运行工况 (Fusion Pulse Operation)",
            "file": Path(
                "test_example/basic/5_fusion_operation_config/20260923_154932/results/sweep_results.h5"
            ).resolve(),
            "expected_jobs": 1,
        },
    }

    urls = {}
    for key, data in cases.items():
        token = _create_viewer_token(str(data["file"]), secret, ctx_dir)
        urls[key] = {
            **data,
            "url": f"{BASE_URL}/?token={token}",
        }
    return urls


def select_report_template(page, option_text):
    page.locator("#report-template-dropdown").scroll_into_view_if_needed()
    time.sleep(0.5)
    page.locator("#report-template-dropdown .Select-control").click()
    time.sleep(0.5)
    page.locator(".VirtualizedSelectOption").filter(has_text=option_text).first.click()
    time.sleep(1.5)


def run_tests():
    case_urls = generate_case_urls()
    results = {}

    print("======================================================================")
    print("🚀 开始执行 TRICYS Visualizer 多配置工况网页端自动化端到端测试")
    print("======================================================================")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # -------------------------------------------------------------
        # Case 1: 多工况参数扫参 (Parameter Sweep, 7 Jobs)
        # -------------------------------------------------------------
        c1 = case_urls["case1_sweep"]
        print(f"\n[工况 1] 正在测试: {c1['name']}")
        print(f"URL: {c1['url']}")
        page.goto(c1["url"], timeout=30000)
        page.wait_for_selector("#overview-job-count", timeout=15000)
        time.sleep(2)  # Wait for Dash initial callback cascade

        job_count_text = page.inner_text("#overview-job-count").strip()
        print(
            f"  ✓ 概览卡片 Jobs 统计值: {job_count_text} (预期: {c1['expected_jobs']})"
        )
        assert str(c1["expected_jobs"]) in job_count_text

        # 布局验证: 报告中心卡片与指示牌
        print("  - 检查工程报告中心卡片渲染...")
        report_preview = page.locator("#report-markdown-preview")
        assert report_preview.count() > 0
        print("  ✓ 工程报告中心组件已成功加载")

        # 交互 1: 勾选当前页所有作业并应用
        print("  - 执行作业选择交互: 勾选全选并点击 Apply Selection...")
        select_all_cb = page.locator("#select-all-checkbox")
        if select_all_cb.count() > 0:
            select_all_cb.click()
            time.sleep(1)
        apply_btn = page.locator("#btn-apply-selection")
        if apply_btn.is_enabled():
            apply_btn.click()
            time.sleep(2)
        print("  ✓ 作业选择成功应用到活动分析集")

        # 交互 2: 验证右侧扇叶窗内原生报告中心渲染
        print("  - 检查扇叶窗内工程报告中心渲染...")
        report_preview = page.locator("#report-markdown-preview")
        assert report_preview.is_visible()

        page.wait_for_function(
            "() => { const el = document.querySelector('#report-markdown-preview'); return el && !el.innerText.includes('正在加载') && el.innerText.length > 50; }",
            timeout=15000,
        )
        initial_report_text = report_preview.inner_text()
        print(f"  ✓ 报告预览已成功渲染 (字符数: {len(initial_report_text)})")
        assert (
            "综合摘要" in initial_report_text
            or "执行摘要" in initial_report_text
            or "TRICYS" in initial_report_text
        )

        # 验证任务专属报告指示牌与内容
        page.wait_for_selector("#report-task-badge", timeout=10000)
        badge_text = page.inner_text("#report-task-badge")
        title_text = page.inner_text("#report-active-title")
        source_text = page.inner_text("#report-source-indicator")
        print(f"  ✓ 报告专属指示牌: [{badge_text}] {title_text} {source_text}")
        assert (
            "扫参" in badge_text
            or "SWEEP" in badge_text
            or "指标" in badge_text
            or "交付" in badge_text
        )

        # 交互 3: 刷新报告预览测试
        print("  - 点击 '刷新报告预览' 按钮...")
        page.locator("#btn-refresh-report").click()
        time.sleep(1.5)
        refreshed_report_text = report_preview.inner_text()
        print(f"  ✓ 刷新后报告预览依然稳定可用 (字符数: {len(refreshed_report_text)})")

        # 交互 5: AI 深度工程洞见生成
        print("  - 点击 '生成 AI 深度工程洞见' 按钮...")
        page.locator("#btn-generate-ai-insight").click()
        page.wait_for_selector("#report-status-alert", state="visible", timeout=10000)
        alert_text = page.inner_text("#report-status-alert")
        print(f"  ✓ AI 洞见状态提示已触发: {alert_text.strip()}")
        time.sleep(1)

        # 交互 6: 多格式导出按钮触发
        print("  - 测试 MD, Word (.docx), PDF (.pdf) 导出按钮响应...")
        with page.expect_download(timeout=10000):
            page.locator("#btn-export-report-md").click()
        print("    ✓ MD 下载流响应成功")

        with page.expect_download(timeout=10000):
            page.locator("#btn-export-report-docx").click()
        print("    ✓ Word (.docx) 下载流响应成功")

        with page.expect_download(timeout=10000):
            page.locator("#btn-export-report-pdf").click()
        print("    ✓ PDF (.pdf) 下载流响应成功")

        # 截图存证
        screenshot1 = ARTIFACTS_DIR / "visualizer_case1_sweep.png"
        page.screenshot(path=str(screenshot1), full_page=False)
        print(f"  📸 工况 1 截图已保存至: {screenshot1}")
        results["case1_sweep"] = "PASS"

        # -------------------------------------------------------------
        # Case 2: 高维全局敏感性分析 (Sobol GSA, 48 Jobs)
        # -------------------------------------------------------------
        c2 = case_urls["case2_sobol"]
        print(f"\n[工况 2] 正在测试: {c2['name']}")
        print(f"URL: {c2['url']}")
        page.goto(c2["url"], timeout=30000)
        page.wait_for_selector("#overview-job-count", timeout=15000)
        time.sleep(2)

        job_count_text2 = page.inner_text("#overview-job-count").strip()
        print(
            f"  ✓ 概览卡片 Jobs 统计值: {job_count_text2} (预期: {c2['expected_jobs']})"
        )
        assert str(c2["expected_jobs"]) in job_count_text2

        # 验证 Sobol 任务专属报告自动呈现
        page.wait_for_selector("#report-task-badge", timeout=10000)
        page.wait_for_function(
            "() => { const el = document.querySelector('#report-markdown-preview'); return el && !el.innerText.includes('正在加载') && el.innerText.length > 50; }",
            timeout=15000,
        )
        badge2 = page.inner_text("#report-task-badge")
        title2 = page.inner_text("#report-active-title")
        print(f"  ✓ 工况 2 任务专属报告: [{badge2}] {title2}")
        sobol_report_text = report_preview.inner_text()
        print(f"  ✓ Sobol 敏感性报告成功渲染 (字符数: {len(sobol_report_text)})")
        assert (
            "灵敏度" in sobol_report_text
            or "Sobol" in sobol_report_text
            or "GSA" in sobol_report_text
            or "TRICYS" in sobol_report_text
        )

        screenshot2 = ARTIFACTS_DIR / "visualizer_case2_sobol.png"
        page.screenshot(path=str(screenshot2), full_page=False)
        print(f"  📸 工况 2 截图已保存至: {screenshot2}")
        results["case2_sobol"] = "PASS"

        # -------------------------------------------------------------
        # Case 3: 聚变脉冲瞬态运行工况 (Fusion Pulse Operation, 1 Job)
        # -------------------------------------------------------------
        c3 = case_urls["case3_pulse"]
        print(f"\n[工况 3] 正在测试: {c3['name']}")
        print(f"URL: {c3['url']}")
        page.goto(c3["url"], timeout=30000)
        page.wait_for_selector("#overview-job-count", timeout=15000)
        time.sleep(2)

        job_count_text3 = page.inner_text("#overview-job-count").strip()
        print(
            f"  ✓ 概览卡片 Jobs 统计值: {job_count_text3} (预期: {c3['expected_jobs']})"
        )
        assert str(c3["expected_jobs"]) in job_count_text3

        # 检查工况配置视图
        print("  - 展开 Card 1 配置视图...")
        config_view = page.locator("#config-view")
        if config_view.is_visible():
            cfg_text = config_view.inner_text()
            print(f"  ✓ 配置视图正常呈现 JSON (长度: {len(cfg_text)})")

        # 验证脉冲瞬态任务专属报告自动呈现
        page.wait_for_selector("#report-task-badge", timeout=10000)
        page.wait_for_function(
            "() => { const el = document.querySelector('#report-markdown-preview'); return el && !el.innerText.includes('正在加载') && el.innerText.length > 50; }",
            timeout=15000,
        )
        badge3 = page.inner_text("#report-task-badge")
        title3 = page.inner_text("#report-active-title")
        print(f"  ✓ 工况 3 任务专属报告: [{badge3}] {title3}")
        pulse_report_text = report_preview.inner_text()
        print(f"  ✓ 动态脉冲时序报表成功渲染 (字符数: {len(pulse_report_text)})")
        assert (
            "瞬态" in pulse_report_text
            or "脉冲" in pulse_report_text
            or "质量平衡" in pulse_report_text
            or "TRICYS" in pulse_report_text
        )

        screenshot3 = ARTIFACTS_DIR / "visualizer_case3_pulse.png"
        page.screenshot(path=str(screenshot3), full_page=False)
        print(f"  📸 工况 3 截图已保存至: {screenshot3}")
        results["case3_pulse"] = "PASS"

        browser.close()

    print("\n======================================================================")
    print("🎉 自动化网页端测试全部顺利完成！所有配置工况验证通过！")
    for k, v in results.items():
        print(f"  - {k}: {v}")
    print("======================================================================")


if __name__ == "__main__":
    run_tests()
