import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import KronosPredictor, KronosTokenizer, Kronos
from datetime import datetime, timedelta
import baostock as bs # <--- 数据源已更换为 BaoStock

# --- 配置 ---
STOCK_CODE = 'sh.600977' # 预测的股票代码（格式已改为 BaoStock 的 'sh.600977' 或 'sz.000001'）
PRED_DAYS = 5 # 预测未来 5 个交易日
LOOKBACK_DAYS = 256 # 用于预测的历史数据长度
DEVICE = 'cpu' # 优先使用 CPU，如果有高性能显卡，可以改为 'cuda:0'
# ----------------------------------------------


# --- 初始化 Kronos 模型 (只运行一次) ---
# 已移除 @st.cache_resource 装饰器，强制重新加载模型
def initialize_models():
    """初始化 BaoStock 登录和 Kronos 模型"""
    
    # BaoStock 登录 (无需 Token，免费)
    try:
        bs.login()
        st.write("BaoStock 初始化成功。")
        print("--- DEBUG: BaoStock 初始化成功。")
    except Exception as e:
        st.error(f"BaoStock 登录失败: {e}")
        print(f"--- ERROR: BaoStock 登录失败: {e}")
        return None, None
    
    st.write("正在加载 Kronos-small 模型 (首次运行会自动下载)...")
    print("--- DEBUG: 正在尝试加载 Kronos 模型... ---")
    try:
        # 模型加载，依赖于终端设置的 HF_ENDPOINT 环境变量进行稳定下载
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(DEVICE)
        
        predictor = KronosPredictor(
            model=model, 
            tokenizer=tokenizer, 
            device=DEVICE, 
            max_context=512 
        )
        print("--- DEBUG: Kronos 模型加载成功。---")
        return predictor, tokenizer
    except Exception as e:
        st.error(f"Kronos 模型加载失败，请检查网络连接或依赖安装: {e}")
        print(f"--- ERROR: Kronos 模型加载失败: {e}")
        return None, None

# --- 数据获取与预测函数 ---
def get_and_predict_stock(predictor, stock_code, lookback, pred_len):
    """从 BaoStock 获取数据并进行预测"""
    st.write(f"正在获取 {stock_code} 最新历史数据 (BaoStock)...")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=500)).strftime('%Y-%m-%d')

    # 从 BaoStock 获取日线数据 (OHLCV)
    rs = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d", # 日线
        adjustflag="3" # 1：前复权
    )
    
    # 结果处理
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
        
    if not data_list:
        st.warning(f"未找到 {stock_code} 的历史数据，请检查代码格式是否为 'sh.600977'。")
        return None, None
        
    df_raw = pd.DataFrame(data_list, columns=rs.fields)
    
    # BaoStock 数据预处理，转换为 Kronos 需要的格式
    df_raw = df_raw.rename(columns={'date': 'timestamps'})
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        # BaoStock 返回的是字符串，需转换为数字
        df_raw[col] = pd.to_numeric(df_raw[col]) 
        
    df_raw['timestamps'] = pd.to_datetime(df_raw['timestamps'])
    df_raw = df_raw.set_index('timestamps').sort_index(ascending=True)

    # 截取历史数据作为输入 (lookback)
    x_df = df_raw.tail(lookback).reset_index(names='timestamps')
    x_df = x_df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    
    x_timestamp = x_df['timestamps']

    # 构造未来预测的时间戳 (简易模拟，假设预测的是工作日)
    last_date = x_timestamp.max()
    future_dates = []
    current_date = last_date + timedelta(days=1)
    while len(future_dates) < pred_len:
        # 仅预测未来的交易日（简单判断：跳过周末）
        if current_date.weekday() < 5: 
            future_dates.append(current_date)
        current_date += timedelta(days=1)
        
    y_timestamp = pd.to_datetime(future_dates)
    
    # 运行预测
    st.write(f"正在使用 Kronos 预测未来 {pred_len} 个交易日...")
    try:
        pred_df = predictor.predict(
            df=x_df[['open', 'high', 'low', 'close', 'volume', 'amount']],
            x_timestamp=pd.Series(x_timestamp),
            y_timestamp=pd.Series(y_timestamp),
            pred_len=pred_len,
            sample_count=1 # 只生成一条预测路径
        )
        # 将预测结果的时间戳索引设置为未来日期
        pred_df.index = y_timestamp 
        return x_df.set_index('timestamps'), pred_df
    except Exception as e:
        st.error(f"Kronos 预测失败: {e}")
        return x_df.set_index('timestamps'), None

# --- 可视化函数 (与原脚本相同) ---
def plot_candlestick(history_df, pred_df, stock_id):
    """绘制 K 线图"""
    fig = go.Figure()

    # 1. 绘制历史数据 (实线)
    fig.add_trace(go.Candlestick(
        x=history_df.index,
        open=history_df['open'],
        high=history_df['high'],
        low=history_df['low'],
        close=history_df['close'],
        name='历史价格 (OHLC)',
        increasing_line_color='red',
        decreasing_line_color='green'
    ))

    # 2. 绘制预测数据 (虚线)
    if pred_df is not None and not pred_df.empty:
        pred_index = pred_df.index
        
        fig.add_trace(go.Candlestick(
            x=pred_index,
            open=pred_df['open'],
            high=pred_df['high'],
            low=pred_df['low'],
            close=pred_df['close'],
            name=f'Kronos {PRED_DAYS}日预测 (虚线)',
            increasing_line_color='rgba(255, 0, 255, 0.5)', # 半透明紫色
            decreasing_line_color='rgba(0, 255, 255, 0.5)', # 半透明青色
            #line=dict(dash='dash') # 关键：使用虚线
        ))
    
    # 3. 布局设置
    fig.update_layout(
        title=f'A股 Kronos 预测可视化: {stock_id}',
        xaxis_title='日期',
        yaxis_title='价格 (元)',
        xaxis_rangeslider_visible=False, 
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)


# --- Streamlit 应用主函数 ---
def main():
    st.set_page_config(layout="wide")
    st.title("💡 A股 Kronos 实时预测工具 (BaoStock 数据源)")
    st.caption("基于 shiyu-coder/Kronos 基础模型构建。请检查终端 DEBUG 信息判断进度。")

    # --- 用户输入 ---
    col1, col2 = st.columns([1, 1])
    with col1:
        stock_code = st.text_input("输入股票代码 (如 sh.600977)", value=STOCK_CODE).lower() # BaoStock 要求小写市场代码
    with col2:
        pred_len = st.slider("预测交易天数", min_value=1, max_value=30, value=PRED_DAYS)
    
    # --- 初始化模型 ---
    predictor, _ = initialize_models()
    if predictor is None:
        return

    # --- 预测按钮 ---
    if st.button("开始预测"):
        st.markdown("---")
        
        with st.spinner(f"正在为 {stock_code} 运行预测..."):
            # 移除 Tushare API 参数
            history_df, pred_df = get_and_predict_stock(predictor, stock_code, LOOKBACK_DAYS, pred_len)

            if history_df is not None and not history_df.empty:
                st.success(f"预测完成！历史数据截止至 {history_df.index.max().strftime('%Y-%m-%d')}。")
                
                # 绘制图表
                plot_candlestick(history_df, pred_df, stock_code)
                
                st.subheader("📋 预测结果 (未来趋势)")
                if pred_df is not None and not pred_df.empty:
                    # 仅显示 OHLC
                    st.dataframe(pred_df[['open', 'high', 'low', 'close']].style.format("{:.2f}"))
                else:
                    st.warning("模型未成功生成预测数据。")
                
                st.subheader("📚 模型输入数据 (最近历史)")
                st.dataframe(history_df[['open', 'high', 'low', 'close']].tail(10).style.format("{:.2f}"))
            else:
                st.error("数据获取失败，请检查股票代码是否正确 (例如 sh.600977)。")
                
if __name__ == "__main__":
    main()