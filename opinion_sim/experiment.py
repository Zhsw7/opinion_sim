"""
实验运行器：封装4类对比实验，供 app.py 调用。
每个函数都返回 {label: DataFrame} 字典，可直接传给 charts.py。
"""

import numpy as np
from model import OpinionSpreadModel


def run_single(params: dict, steps=100):
    """运行单次仿真，返回时序 DataFrame"""
    model = OpinionSpreadModel(**params)
    return model.run(steps)


def compare_intervention_timing(base_params: dict, timing_steps: list, steps=100):
    """
    实验一：对比不同干预时刻的效果。
    timing_steps: 干预触发步数列表，例如 [10, 20, 40]
    """
    results = {}

    # 无干预基准
    no_cfg = {**base_params, "intervention_config": None}
    results["无干预"] = run_single(no_cfg, steps)

    # 不同时机
    for t in timing_steps:
        cfg = dict(base_params.get("intervention_config") or {})
        cfg["trigger_step"] = t
        params = {**base_params, "intervention_config": cfg}
        results[f"第{t}步干预"] = run_single(params, steps)

    return results


def sensitivity_analysis(base_params: dict, param_name: str, param_values: list, steps=100):
    """
    实验二：单参数敏感性分析。
    param_name:   intervention_config 中的键，如 "delta0" / "lambda_" / "alpha"
    param_values: 该参数的取值列表
    """
    results = {}
    for val in param_values:
        cfg = dict(base_params.get("intervention_config") or {})
        cfg[param_name] = val
        params = {**base_params, "intervention_config": cfg}
        results[round(val, 3)] = run_single(params, steps)
    return results


def compare_network_types(base_params: dict, network_types: list, steps=100):
    """实验三：对比不同网络拓扑下干预效果的差异"""
    results = {}
    for ntype in network_types:
        params = {**base_params, "network_type": ntype}
        results[ntype] = run_single(params, steps)
    return results


def multi_intervention_decay(base_params: dict, intervals: list, steps=100):
    """
    实验四：多次干预叠加的边际递减效应。
    intervals: 每次干预间隔步数列表，例如 [15, 10, 10] 表示第15步第一次干预，之后每隔10步一次
    
    修复：正确计算触发点的累加
    """
    from model import OpinionSpreadModel

    cfg = dict(base_params.get("intervention_config") or {})
    # 关闭自动触发，由外部手动触发
    cfg.pop("trigger_step", None)
    cfg.pop("auto_threshold", None)

    params = {**base_params, "intervention_config": cfg}
    model = OpinionSpreadModel(**params)

    # 修复：正确计算触发点
    trigger_points = []
    t = 0
    for gap in intervals:
        t += gap
        trigger_points.append(t)

    step_count = 0
    trigger_idx = 0
    while step_count < steps and model.running:
        if trigger_idx < len(trigger_points) and step_count == trigger_points[trigger_idx]:
            model.intervention.trigger(step_count)
            trigger_idx += 1
        model.step()
        step_count += 1

    df = model.datacollector.get_model_vars_dataframe()
    df.index.name = "step"
    df = df.reset_index()
    return df, trigger_points


def extract_metrics(df) -> dict:
    """从时序 DataFrame 提取关键指标"""
    total = df[["S", "E", "I", "R"]].iloc[0].sum()
    peak_I = int(df["I"].max())
    peak_step = int(df["I"].idxmax())
    final_R = int(df["R"].iloc[-1])
    return {
        "peak_infected": peak_I,
        "peak_step": peak_step,
        "final_recovered": final_R,
        "attack_rate": round(final_R / total, 4) if total > 0 else 0,
    }
