import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）



feature_ranges = {
    # 数值型特征（numerical）：包含临床常规范围及合理默认值
    "UA": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 300.0},  # UA：尿酸（μmol/L），常规范围150-440
    "BMI": {"type": "numerical", "min": 10.0, "max": 50.0, "default": 24.5},   # BMI：身体质量指数，同原格式范围
    "ALP": {"type": "numerical", "min": 0.0, "max": 1000.0, "default": 80.0},  # ALP：碱性磷酸酶（U/L），常规范围40-150
    "CysC": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 1.0},    # CysC：胱抑素C（mg/L），常规范围0.6-2.5
    "LDH": {"type": "numerical", "min": 0.0, "max": 2000.0, "default": 180.0}, # LDH：乳酸脱氢酶（U/L），常规范围109-245
    "FPG": {"type": "numerical", "min": 2.0, "max": 30.0, "default": 5.5},     # FPG：空腹血糖（mmol/L），常规范围3.9-6.1
    "TG": {"type": "numerical", "min": 0.0, "max": 20.0, "default": 1.2},      # TG：甘油三酯（mmol/L），常规范围0.45-1.69
    "FT4": {"type": "numerical", "min": 5.0, "max": 50.0, "default": 16.0},    # FT4：游离甲状腺素（pmol/L），常规范围12-22
    "FT3": {"type": "numerical", "min": 1.0, "max": 20.0, "default": 4.5},     # FT3：游离三碘甲状腺原氨酸（pmol/L），常规范围3.1-6.8
    "CRP": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 3.0},    # CRP：C反应蛋白（mg/L），常规范围0-10（炎症时升高）
    "WBC": {"type": "numerical", "min": 0.0, "max": 50.0, "default": 7.0},     # WBC：白细胞计数（×10^9/L），常规范围4-10
    "LY": {"type": "numerical", "min": 0.0, "max": 10.0, "default": 2.5},      # LY：淋巴细胞计数（×10^9/L），常规范围0.8-4
    "Albumin": {"type": "numerical", "min": 10.0, "max": 60.0, "default": 40.0},# Albumin：白蛋白（g/L），补充笔误“c”，常规范围35-50
    "A/B": {"type": "numerical", "min": 0.5, "max": 5.0, "default": 1.8},      # A/B：白蛋白/球蛋白比值，常规范围1.2-2.5
    
    # 分类变量（categorical）：options按“无/有”逻辑设为[0,1]
    "Hypertension": {"type": "categorical", "options": [0, 1], "default": 0},  # Hypertension：高血压（0=无，1=有）
    "IVF": {"type": "categorical", "options": [0, 1], "default": 0}            # IVF：辅助生殖技术（0=未使用，1=使用）
}


# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
