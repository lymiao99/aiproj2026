import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 設定網頁標題與圖示 (必須是第一個 Streamlit 指令)
st.set_page_config(
    page_title="Wine AI Engine | 科技感預測系統",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入科技感自定義 CSS
st.markdown("""
    <style>
    /* 全域背景與文字 */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #f0f0f0 !important;
    }
    
    /* 強制所有標籤為明亮色 */
    label, .stMarkdown, p, span {
        color: #ffffff !important;
    }
    
    /* 側邊欄樣式 */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 12, 41, 0.9) !important;
        border-right: 1px solid #00f2fe;
        backdrop-filter: blur(15px);
    }
    
    /* 側邊欄內的文字與標籤 */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: #00f2fe !important;
        font-weight: bold;
    }
    
    /* 標題與字體 */
    h1, h2, h3, h4 {
        color: #00f2fe !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.5);
    }
    
    /* 卡片式容器 (磨砂玻璃) */
    div.stDataFrame, div.stTable, [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }
    
    /* 指標標籤顏色 */
    [data-testid="stMetricLabel"] p {
        color: #ffffff !important;
    }
    
    /* 資料表格文字顏色 */
    [data-testid="stTable"] td, [data-testid="stTable"] th {
        color: #ffffff !important;
    }
    
    /* 下拉選單文字顏色 */
    .stSelectbox div[data-baseweb="select"] > div {
        color: #00f2fe !important;
        background-color: rgba(0, 0, 0, 0.3) !important;
    }
    
    /* 按鈕樣式 (發光效果) */
    .stButton > button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.6) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 242, 254, 0.9) !important;
    }
    
    /* 指標樣式 */
    [data-testid="stMetricValue"] {
        color: #00f2fe !important;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* 側邊欄資訊框 */
    .stAlert {
        background: rgba(0, 242, 254, 0.1) !important;
        border: 1px solid #00f2fe !important;
        color: #00f2fe !important;
    }
    
    /* 移除頂部裝飾條 */
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 模型目錄路徑
MODEL_DIR = "wine_prediction_app"

# 模型檔案對照表
MODEL_FILES = {
    "KNN": "knn_model.joblib",
    "羅吉斯迴歸": "logistic_model.joblib",
    "XGBoost": "xgb_model.joblib",
    "隨機森林": "rf_model.joblib",
    "高斯貝耶斯": "gnb_model.joblib"
}

# 1. 載入資料集
@st.cache_data
def get_wine_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = get_wine_data()

# 2. 左側 Sidebar
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>SYSTEM CONFIG</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 下拉選單選擇模型
    model_option = st.selectbox(
        "🧠 選擇 AI 核心引擎：",
        tuple(MODEL_FILES.keys())
    )

    st.markdown("---")
    st.markdown("<h3>DATASET ANALYTICS</h3>", unsafe_allow_html=True)
    st.info(f"""
    **[STATUS: ONLINE]**
    - 🔢 樣本總數: {df.shape[0]}
    - 🧪 特徵屬性: {df.shape[1] - 1}
    - 🏷️ 類別數量: {len(wine_data.target_names)}
    """)
    st.caption("Target Types: " + ", ".join(wine_data.target_names).upper())

# 3. 右側 Main 區
st.markdown("<h1>🍷 WINE AI PREDICTION ENGINE</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #00f2fe; opacity: 0.8;'>Advanced Machine Learning Interface for Wine Classification</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 DATA STREAM RECAP")
    st.dataframe(df.head(), use_container_width=True)

with col2:
    st.markdown("### 📈 FEATURE STATISTICS")
    stats = df.describe().T[['mean', 'std', 'min', 'max']]
    st.write(stats)

st.markdown("---")

# 4. 預測邏輯
st.markdown("### ⚙️ CORE EXECUTION")

button_col1, button_col2 = st.columns([1, 3])
with button_col1:
    predict_btn = st.button("🚀 EXECUTE PREDICTION")

if predict_btn:
    # 資料準備
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_option])
    
    if os.path.exists(model_path):
        with st.spinner('🔄 正在同步 AI 權重...'):
            try:
                model = joblib.load(model_path)
                
                # 部分模型需要標準化 (KNN, 羅吉斯迴歸)
                if model_option in ["KNN", "羅吉斯迴歸"]:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                else:
                    y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                
                # 產出詳細分類報告 (字典格式)
                report_dict = classification_report(y_test, y_pred, target_names=wine_data.target_names, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                
                # 顯示結果
                st.markdown(f"<div style='border: 1px solid #00f2fe; padding: 20px; border-radius: 15px; background: rgba(0, 242, 254, 0.05);'>", unsafe_allow_html=True)
                st.markdown(f"#### ✅ PREDICTION COMPLETE")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("ACTIVE ENGINE", model_option)
                res_col2.metric("SOURCE FILE", MODEL_FILES[model_option])
                res_col3.metric("ENGINE ACCURACY", f"{accuracy:.2%}")
                
                # 顯示詳細分類報告
                st.markdown("#### 📄 CLASSIFICATION REPORT (DETAILED ANALYTICS)")
                st.dataframe(report_df.style.format(precision=2), use_container_width=True)
                
                # 顯示部分預測比較
                st.markdown("#### 🔍 SAMPLE LOG COMPARISON")
                comparison_df = pd.DataFrame({
                    "ACTUAL TYPE": [wine_data.target_names[i].upper() for i in y_test[:10]],
                    "PREDICTED TYPE": [wine_data.target_names[i].upper() for i in y_pred[:10]]
                })
                st.table(comparison_df)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"ENGINE ERROR: {e}")
    else:
        st.error(f"MISSING CORE FILE: {model_path}")
else:
    st.markdown("<div style='text-align: center; padding: 50px; border: 1px dashed rgba(0, 242, 254, 0.3); border-radius: 20px;'>系統待命中... 請點擊執行按鈕啟動預測序列</div>", unsafe_allow_html=True)
