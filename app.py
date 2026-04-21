"""
校园与居民区周边声环境大数据监测分析及降噪优化决策研究
Python + Streamlit 主程序
运行方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="声环境大数据监测分析系统",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 自定义CSS样式
# ============================================================
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1a5276; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #5d6d7e; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   padding: 1.5rem; border-radius: 1rem; color: white; text-align: center; }
    .kpi-value { font-size: 2rem; font-weight: 700; }
    .kpi-label { font-size: 0.9rem; opacity: 0.9; }
    .info-box { background: #eaf2f8; border-left: 4px solid #2980b9; padding: 1rem; margin: 0.5rem 0; }
    .warning-box { background: #fef9e7; border-left: 4px solid #f39c12; padding: 1rem; margin: 0.5rem 0; }
    .danger-box { background: #fdedec; border-left: 4px solid #e74c3c; padding: 1rem; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 数据加载与缓存
# ============================================================
@st.cache_data
def load_data():
    """加载声环境监测数据"""
    data_path = os.path.join(os.path.dirname(__file__), "data", "noise_monitoring_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, encoding='utf-8-sig')
    else:
        df = generate_sample_data()
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def generate_sample_data():
    """生成样例数据（当没有真实数据时使用）"""
    np.random.seed(42)
    n_records = 8760  # 一年小时数据
    
    locations = ['校园教学楼A', '校园图书馆', '校园宿舍楼B', '居民区东门', 
                 '居民区中心花园', '居民区南门', '校园操场', '居民区北侧主干道']
    
    records = []
    start_date = datetime(2025, 1, 1)
    
    for i in range(n_records):
        dt = start_date + timedelta(hours=i)
        hour = dt.hour
        month = dt.month
        
        # 季节性噪声基线调整
        if month in [6, 7, 8]:  # 夏季噪声较高（开窗、空调外机）
            seasonal_base = 3
        elif month in [12, 1, 2]:  # 冬季噪声较低（关窗）
            seasonal_base = -2
        else:
            seasonal_base = 0
            
        for loc in locations:
            # 不同区域的噪声基线
            if '教学楼' in loc:
                base_la = 52 + seasonal_base
                base_leq = 55 + seasonal_base
            elif '图书馆' in loc:
                base_la = 42 + seasonal_base
                base_leq = 45 + seasonal_base
            elif '宿舍' in loc:
                base_la = 48 + seasonal_base
                base_leq = 51 + seasonal_base
            elif '操场' in loc:
                base_la = 58 + seasonal_base
                base_leq = 62 + seasonal_base
            elif '主干道' in loc:
                base_la = 68 + seasonal_base
                base_leq = 72 + seasonal_base
            elif '东门' in loc or '南门' in loc:
                base_la = 62 + seasonal_base
                base_leq = 66 + seasonal_base
            else:
                base_la = 50 + seasonal_base
                base_leq = 53 + seasonal_base
            
            # 时段调整
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 早晚高峰
                time_adj = 8
            elif 22 <= hour or hour <= 5:  # 深夜
                time_adj = -10
            elif 9 <= hour <= 17:  # 白天
                time_adj = 3
            else:
                time_adj = -3
            
            # 工作日/周末
            if dt.weekday() >= 5:
                weekend_adj = -2
            else:
                weekend_adj = 0
                
            la_eq = base_la + time_adj + weekend_adj + np.random.normal(0, 3)
            la_max = la_eq + np.random.uniform(8, 20)
            la_min = la_eq - np.random.uniform(5, 15)
            la_10 = la_eq + np.random.uniform(2, 6)
            la_50 = la_eq + np.random.normal(0, 1)
            la_90 = la_eq - np.random.uniform(2, 6)
            
            # 频带声压级
            freq_bands = {}
            for freq in [63, 125, 250, 500, 1000, 2000, 4000, 8000]:
                freq_bands[f'L{freq}'] = la_eq + np.random.uniform(-8, 5) + 10 * np.log10(freq / 1000)
            
            # 气象数据
            temp = 15 + 12 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3)
            humidity = 60 + 20 * np.cos(2 * np.pi * (month - 7) / 12) + np.random.normal(0, 5)
            wind_speed = np.random.exponential(2.5)
            
            # 偶尔插入异常值
            if np.random.random() < 0.005:
                la_eq += np.random.uniform(15, 30)
                la_max += np.random.uniform(20, 40)
            
            record = {
                'datetime': dt,
                'location': loc,
                'type': '校园' if '校园' in loc else '居民区',
                'LAeq': round(la_eq, 1),
                'LAmax': round(la_max, 1),
                'LAmin': round(max(la_min, 25), 1),
                'LA10': round(la_10, 1),
                'LA50': round(la_50, 1),
                'LA90': round(la_90, 1),
                'temperature': round(temp, 1),
                'humidity': round(np.clip(humidity, 20, 100), 1),
                'wind_speed': round(wind_speed, 1),
            }
            record.update({k: round(v, 1) for k, v in freq_bands.items()})
            records.append(record)
    
    df = pd.DataFrame(records)
    return df

# ============================================================
# 数据清洗模块
# ============================================================
@st.cache_data
def clean_data(df):
    """数据清洗：缺失值处理、异常值检测、数据类型转换"""
    df_clean = df.copy()
    
    # 缺失值统计
    missing_before = df_clean.isnull().sum().sum()
    
    # 数值列填充
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # 异常值检测（IQR + 3σ联合方法）
    outlier_info = {}
    for col in ['LAeq', 'LAmax', 'LAmin']:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        sigma_lower = df_clean[col].mean() - 3 * df_clean[col].std()
        sigma_upper = df_clean[col].mean() + 3 * df_clean[col].std()
        
        outlier_mask = (df_clean[col] < max(lower, sigma_lower)) | (df_clean[col] > min(upper, sigma_upper))
        outlier_count = outlier_mask.sum()
        outlier_info[col] = {
            'count': int(outlier_count),
            'lower_bound': round(max(lower, sigma_lower), 1),
            'upper_bound': round(min(upper, sigma_upper), 1),
            'percentage': round(outlier_count / len(df_clean) * 100, 2)
        }
        
        # 用滑动窗口中位数替换异常值
        df_clean.loc[outlier_mask, col] = np.nan
        df_clean[col] = df_clean[col].interpolate(method='linear')
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    return df_clean, outlier_info, missing_before

# ============================================================
# 噪声评价标准
# ============================================================
NOISE_STANDARDS = {
    '0类（疗养区）': {'day': 50, 'night': 40},
    '1类（居民文教区）': {'day': 55, 'night': 45},
    '2类（商业居住混合区）': {'day': 60, 'night': 50},
    '3类（工业区）': {'day': 65, 'night': 55},
    '4a类（交通干线两侧）': {'day': 70, 'night': 55},
    '4b类（铁路干线两侧）': {'day': 70, 'night': 60},
}

def get_standard_exceeded(rate, standard_val):
    """判断是否超标"""
    if rate > standard_val:
        return "⚠️ 超标", "red"
    elif rate > standard_val - 3:
        return "接近限值", "orange"
    else:
        return "✅ 达标", "green"

# ============================================================
# 降噪优化建议引擎
# ============================================================
def generate_noise_reduction_advice(df_filtered):
    """基于数据分析生成降噪优化建议"""
    advices = []
    
    for location in df_filtered['location'].unique():
        loc_data = df_filtered[df_filtered['location'] == location]
        day_data = loc_data[loc_data['datetime'].dt.hour.between(6, 22)]
        night_data = loc_data[loc_data['datetime'].dt.hour.between(22, 6) | (loc_data['datetime'].dt.hour < 6)]
        
        day_leq = day_data['LAeq'].mean() if len(day_data) > 0 else 0
        night_leq = night_data['LAeq'].mean() if len(night_data) > 0 else 0
        
        # 判断区域类型对应的标准
        if '教学楼' in location or '图书馆' in location:
            std_type = '1类（居民文教区）'
        elif '主干道' in location:
            std_type = '4a类（交通干线两侧）'
        elif '东门' in location or '南门' in location:
            std_type = '2类（商业居住混合区）'
        else:
            std_type = '1类（居民文教区）'
        
        day_std = NOISE_STANDARDS[std_type]['day']
        night_std = NOISE_STANDARDS[std_type]['night']
        
        advice = {
            'location': location,
            'std_type': std_type,
            'day_leq': round(day_leq, 1),
            'night_leq': round(night_leq, 1),
            'day_std': day_std,
            'night_std': night_std,
            'day_exceed': day_leq > day_std,
            'night_exceed': night_leq > night_std,
            'day_over': round(max(0, day_leq - day_std), 1),
            'night_over': round(max(0, night_leq - night_std), 1),
        }
        
        # 生成具体建议
        measures = []
        if advice['day_exceed'] or advice['night_exceed']:
            exceed_db = max(advice['day_over'], advice['night_over'])
            if exceed_db > 10:
                measures.append("🚨 严重超标：建议设置声屏障+双层隔音窗+绿化带组合降噪方案")
                measures.append(f"  预估降噪效果：声屏障(8-12dB) + 隔音窗(25-35dB) + 绿化带(3-5dB)")
            elif exceed_db > 5:
                measures.append("⚠️ 中度超标：建议设置声屏障或种植宽幅绿化带")
                measures.append(f"  预估降噪效果：声屏障(5-10dB) 或 绿化带(3-8dB)")
            else:
                measures.append("💡 轻微超标：建议增加绿化植被或安装隔音窗")
                measures.append(f"  预估降噪效果：绿化带(2-5dB) 或 隔音窗(20-30dB)")
            
            if '主干道' in location or '东门' in location or '南门' in location:
                measures.append("🚗 交通噪声为主：建议限速管控+禁鸣喇叭+低噪路面铺设")
            if '操场' in location:
                measures.append("🏃 活动噪声为主：建议调整活动时段+设置活动噪声限制")
        
        advice['measures'] = measures
        advices.append(advice)
    
    return advices

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/noise-pollution.png", width=80)
    st.title("🔊 声环境监测系统")
    st.markdown("---")
    
    # 数据加载
    df_raw = load_data()
    df_clean, outlier_info, missing_count = clean_data(df_raw)
    
    # 筛选器
    st.subheader("📊 数据筛选")
    
    # 时间范围
    min_date = df_clean['datetime'].min().date()
    max_date = df_clean['datetime'].max().date()
    date_range = st.date_input("📅 选择时间范围", [min_date, max_date])
    
    # 监测点
    all_locations = df_clean['location'].unique().tolist()
    selected_locations = st.multiselect("📍 选择监测点", all_locations, default=all_locations)
    
    # 区域类型
    selected_type = st.multiselect("🏗️ 区域类型", ['校园', '居民区'], default=['校园', '居民区'])
    
    # 时段筛选
    time_period = st.selectbox("🕐 时段", ['全天', '白天(6:00-22:00)', '夜间(22:00-6:00)'])
    
    # 噪声标准选择
    st.subheader("📋 噪声评价标准")
    selected_std = st.selectbox("选择适用标准", list(NOISE_STANDARDS.keys()), index=1)
    
    st.markdown("---")
    st.caption("© 2026 声环境大数据监测分析系统")

# ============================================================
# 数据筛选
# ============================================================
df_filtered = df_clean.copy()

if len(date_range) == 2:
    mask = (df_filtered['datetime'].dt.date >= date_range[0]) & (df_filtered['datetime'].dt.date <= date_range[1])
    df_filtered = df_filtered[mask]

if selected_locations:
    df_filtered = df_filtered[df_filtered['location'].isin(selected_locations)]

if selected_type:
    df_filtered = df_filtered[df_filtered['type'].isin(selected_type)]

if time_period == '白天(6:00-22:00)':
    df_filtered = df_filtered[df_filtered['datetime'].dt.hour.between(6, 21)]
elif time_period == '夜间(22:00-6:00)':
    df_filtered = df_filtered[(df_filtered['datetime'].dt.hour >= 22) | (df_filtered['datetime'].dt.hour < 6)]

# ============================================================
# 主界面
# ============================================================
st.markdown('<p class="main-title">校园与居民区周边声环境大数据监测分析系统</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于大数据技术的噪声监测、评估与降噪优化决策支持平台</p>', unsafe_allow_html=True)

# Tab布局
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 总览仪表板", "📈 时序分析", "🗺️ 空间对比", "🔧 数据处理", 
    "📉 频谱分析", "💡 降噪决策"
])

# ============================================================
# Tab 1: 总览仪表板
# ============================================================
with tab1:
    st.subheader("实时监测总览")
    
    # KPI卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_leq = df_filtered['LAeq'].mean()
        st.metric("平均等效声级", f"{avg_leq:.1f} dB(A)", 
                   delta=f"{'超标' if avg_leq > NOISE_STANDARDS[selected_std]['day'] else '达标'}")
    
    with col2:
        max_leq = df_filtered['LAmax'].max()
        st.metric("最大声级", f"{max_leq:.1f} dB(A)")
    
    with col3:
        exceed_rate = (df_filtered['LAeq'] > NOISE_STANDARDS[selected_std]['day']).mean() * 100
        st.metric("昼间超标率", f"{exceed_rate:.1f}%", delta_color="inverse")
    
    with col4:
        st.metric("监测数据量", f"{len(df_filtered):,} 条")
    
    st.markdown("---")
    
    # 各监测点概览
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📍 各监测点LAeq均值对比")
        loc_stats = df_filtered.groupby('location')['LAeq'].agg(['mean', 'std', 'min', 'max']).reset_index()
        loc_stats = loc_stats.sort_values('mean', ascending=True)
        
        fig_bar = px.bar(loc_stats, y='location', x='mean', orientation='h',
                         error_x='std', color='mean',
                         color_continuous_scale='RdYlGn_r',
                         labels={'mean': 'LAeq dB(A)', 'location': '监测点'},
                         title='各监测点等效声级均值')
        fig_bar.add_vline(x=NOISE_STANDARDS[selected_std]['day'], line_dash="dash", 
                          line_color="red", annotation_text=f"昼间限值 {NOISE_STANDARDS[selected_std]['day']}dB")
        fig_bar.add_vline(x=NOISE_STANDARDS[selected_std]['night'], line_dash="dash",
                          line_color="orange", annotation_text=f"夜间限值 {NOISE_STANDARDS[selected_std]['night']}dB")
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col_right:
        st.subheader("🕐 24小时声级变化曲线")
        hourly_avg = df_filtered.groupby(df_filtered['datetime'].dt.hour)['LAeq'].mean().reset_index()
        hourly_avg.columns = ['小时', 'LAeq']
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Scatter(x=hourly_avg['小时'], y=hourly_avg['LAeq'],
                                         mode='lines+markers', name='LAeq均值',
                                         line=dict(color='#2196F3', width=2)))
        fig_hourly.add_hrect(y0=NOISE_STANDARDS[selected_std]['night'], 
                             y1=NOISE_STANDARDS[selected_std]['day'],
                             fillcolor="yellow", opacity=0.1, annotation_text="过渡区")
        fig_hourly.add_hline(y=NOISE_STANDARDS[selected_std]['day'], line_dash="dash",
                             line_color="red", annotation_text=f"昼间限值")
        fig_hourly.add_hline(y=NOISE_STANDARDS[selected_std]['night'], line_dash="dash",
                             line_color="orange", annotation_text=f"夜间限值")
        fig_hourly.update_layout(xaxis_title='小时', yaxis_title='LAeq dB(A)',
                                  xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # 达标率统计
    st.subheader("📋 各监测点达标情况")
    compliance_data = []
    for loc in df_filtered['location'].unique():
        loc_data = df_filtered[df_filtered['location'] == loc]
        day_data = loc_data[loc_data['datetime'].dt.hour.between(6, 21)]
        night_data = loc_data[(loc_data['datetime'].dt.hour >= 22) | (loc_data['datetime'].dt.hour < 6)]
        
        day_comply = (day_data['LAeq'] <= NOISE_STANDARDS[selected_std]['day']).mean() * 100 if len(day_data) > 0 else 100
        night_comply = (night_data['LAeq'] <= NOISE_STANDARDS[selected_std]['night']).mean() * 100 if len(night_data) > 0 else 100
        
        compliance_data.append({
            '监测点': loc,
            '昼间达标率(%)': round(day_comply, 1),
            '夜间达标率(%)': round(night_comply, 1),
            '昼间均值(dB)': round(day_data['LAeq'].mean(), 1) if len(day_data) > 0 else '-',
            '夜间均值(dB)': round(night_data['LAeq'].mean(), 1) if len(night_data) > 0 else '-',
        })
    
    df_compliance = pd.DataFrame(compliance_data)
    st.dataframe(df_compliance, use_container_width=True, hide_index=True)

# ============================================================
# Tab 2: 时序分析
# ============================================================
with tab2:
    st.subheader("噪声时序变化分析")
    
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        selected_loc_ts = st.selectbox("选择监测点", df_filtered['location'].unique())
        show_trend = st.checkbox("显示趋势线", value=True)
        show_range = st.checkbox("显示声级范围", value=True)
    
    with col_t2:
        loc_ts_data = df_filtered[df_filtered['location'] == selected_loc_ts].sort_values('datetime')
        
        fig_ts = go.Figure()
        
        # LAeq主线
        fig_ts.add_trace(go.Scatter(
            x=loc_ts_data['datetime'], y=loc_ts_data['LAeq'],
            mode='lines', name='LAeq', line=dict(color='#1976D2', width=1.5)
        ))
        
        if show_range:
            fig_ts.add_trace(go.Scatter(
                x=loc_ts_data['datetime'], y=loc_ts_data['LAmax'],
                mode='lines', name='LAmax', line=dict(color='rgba(255,82,82,0.3)', width=1)
            ))
            fig_ts.add_trace(go.Scatter(
                x=loc_ts_data['datetime'], y=loc_ts_data['LAmin'],
                mode='lines', name='LAmin', line=dict(color='rgba(76,175,80,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(33,150,243,0.1)'
            ))
        
        if show_trend:
            # 7日移动平均趋势
            daily_avg = loc_ts_data.set_index('datetime')['LAeq'].resample('D').mean().dropna()
            trend = daily_avg.rolling(window=7, center=True).mean()
            fig_ts.add_trace(go.Scatter(
                x=trend.index, y=trend.values,
                mode='lines', name='7日趋势', line=dict(color='#FF9800', width=2, dash='dash')
            ))
        
        # 标准线
        fig_ts.add_hline(y=NOISE_STANDARDS[selected_std]['day'], line_dash="dash",
                         line_color="red", annotation_text="昼间限值")
        fig_ts.add_hline(y=NOISE_STANDARDS[selected_std]['night'], line_dash="dash",
                         line_color="orange", annotation_text="夜间限值")
        
        fig_ts.update_layout(
            title=f'{selected_loc_ts} - 噪声时序变化',
            xaxis_title='时间', yaxis_title='声级 dB(A)',
            height=500
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    
    # 日历热力图
    st.subheader("📅 日均声级热力图")
    daily_by_loc = df_filtered.groupby(['location', df_filtered['datetime'].dt.date])['LAeq'].mean().reset_index()
    daily_by_loc.columns = ['location', 'date', 'LAeq']
    
    selected_loc_hm = st.selectbox("选择监测点(热力图)", df_filtered['location'].unique(), key='heatmap_loc')
    hm_data = daily_by_loc[daily_by_loc['location'] == selected_loc_hm]
    
    if len(hm_data) > 0:
        hm_data['date'] = pd.to_datetime(hm_data['date'])
        hm_pivot = hm_data.set_index('date')['LAeq']
        
        fig_cal = px.scatter(hm_data, x=hm_data['date'].dt.day, y=hm_data['date'].dt.month,
                             color='LAeq', size_max=15,
                             color_continuous_scale='RdYlGn_r',
                             labels={'x': '日', 'y': '月', 'LAeq': 'LAeq dB(A)'},
                             title=f'{selected_loc_hm} - 月日均声级分布')
        st.plotly_chart(fig_cal, use_container_width=True)
    
    # 统计摘要
    st.subheader("📊 时序统计摘要")
    ts_summary = df_filtered.groupby('location').agg(
        样本数=('LAeq', 'count'),
        LAeq均值=('LAeq', 'mean'),
        LAeq标准差=('LAeq', 'std'),
        LAeq最大值=('LAeq', 'max'),
        LAeq最小值=('LAeq', 'min'),
        LA90背景噪声=('LA90', 'mean'),
    ).round(1)
    st.dataframe(ts_summary, use_container_width=True)

# ============================================================
# Tab 3: 空间对比
# ============================================================
with tab3:
    st.subheader("监测点空间对比分析")
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        # 雷达图对比
        st.subheader("🎯 多维噪声特征对比")
        compare_locs = st.multiselect("选择对比监测点(至少2个)", 
                                       df_filtered['location'].unique(),
                                       default=df_filtered['location'].unique()[:3],
                                       key='radar_locs')
        
        if len(compare_locs) >= 2:
            radar_data = []
            for loc in compare_locs:
                loc_data = df_filtered[df_filtered['location'] == loc]
                radar_data.append({
                    '监测点': loc,
                    'LAeq均值': loc_data['LAeq'].mean(),
                    'LAmax均值': loc_data['LAmax'].mean(),
                    'LA10': loc_data['LA10'].mean(),
                    'LA50': loc_data['LA50'].mean(),
                    'LA90': loc_data['LA90'].mean(),
                    '昼夜差': loc_data[loc_data['datetime'].dt.hour.between(6, 21)]['LAeq'].mean() - 
                              loc_data[(loc_data['datetime'].dt.hour >= 22) | (loc_data['datetime'].dt.hour < 6)]['LAeq'].mean()
                              if len(loc_data[loc_data['datetime'].dt.hour.between(6, 21)]) > 0 and 
                                 len(loc_data[(loc_data['datetime'].dt.hour >= 22) | (loc_data['datetime'].dt.hour < 6)]) > 0 else 0,
                })
            
            df_radar = pd.DataFrame(radar_data)
            
            categories = ['LAeq均值', 'LAmax均值', 'LA10', 'LA50', 'LA90', '昼夜差']
            fig_radar = go.Figure()
            
            for _, row in df_radar.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row[c] for c in categories],
                    theta=categories,
                    fill='toself',
                    name=row['监测点'],
                    opacity=0.6
                ))
            
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                     title='多维噪声特征雷达图')
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col_s2:
        # 箱线图
        st.subheader("📦 各监测点声级分布")
        fig_box = px.box(df_filtered, x='location', y='LAeq', color='type',
                         labels={'LAeq': 'LAeq dB(A)', 'location': '监测点', 'type': '区域类型'},
                         title='各监测点LAeq分布箱线图')
        fig_box.add_hline(y=NOISE_STANDARDS[selected_std]['day'], line_dash="dash",
                          line_color="red", annotation_text="昼间限值")
        st.plotly_chart(fig_box, use_container_width=True)
    
    # 校园 vs 居民区对比
    st.subheader("🏘️ 校园与居民区噪声对比")
    type_comparison = df_filtered.groupby('type').agg({
        'LAeq': ['mean', 'std', 'max'],
        'LAmax': 'mean',
        'LA90': 'mean'
    }).round(1)
    type_comparison.columns = ['LAeq均值', 'LAeq标准差', 'LAeq最大值', 'LAmax均值', 'LA90均值']
    st.dataframe(type_comparison, use_container_width=True)
    
    # 统计检验结果
    from scipy import stats
    campus_data = df_filtered[df_filtered['type'] == '校园']['LAeq']
    residential_data = df_filtered[df_filtered['type'] == '居民区']['LAeq']
    
    if len(campus_data) > 0 and len(residential_data) > 0:
        t_stat, p_value = stats.ttest_ind(campus_data, residential_data)
        st.info(f"**独立样本t检验**: t = {t_stat:.3f}, p = {p_value:.6f} → {'差异显著' if p_value < 0.05 else '差异不显著'} (α=0.05)")

# ============================================================
# Tab 4: 数据处理
# ============================================================
with tab4:
    st.subheader("数据清洗与预处理")
    
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        st.subheader("📊 数据概览")
        st.write(f"原始数据量: **{len(df_raw):,}** 条记录")
        st.write(f"清洗后数据量: **{len(df_clean):,}** 条记录")
        st.write(f"缺失值数量: **{missing_count:,}**")
        st.write(f"监测点数量: **{df_clean['location'].nunique()}** 个")
        st.write(f"时间跨度: **{df_clean['datetime'].min()}** 至 **{df_clean['datetime'].max()}**")
        
        # 数据字段说明
        st.subheader("📋 数据字段说明")
        field_desc = pd.DataFrame({
            '字段名': ['datetime', 'location', 'type', 'LAeq', 'LAmax', 'LAmin', 'LA10', 'LA50', 'LA90',
                       'L63-L8000', 'temperature', 'humidity', 'wind_speed'],
            '说明': ['监测时间', '监测点名称', '区域类型(校园/居民区)', '等效连续A声级',
                     '最大A声级', '最小A声级', '统计声级L10(峰值)', '统计声级L50(中位数)',
                     '统计声级L90(背景)', '倍频程频带声压级', '温度(℃)', '湿度(%)', '风速(m/s)'],
            '单位': ['-', '-', '-', 'dB(A)', 'dB(A)', 'dB(A)', 'dB(A)', 'dB(A)', 'dB(A)',
                     'dB', '℃', '%', 'm/s']
        })
        st.dataframe(field_desc, use_container_width=True, hide_index=True)
    
    with col_d2:
        st.subheader("⚠️ 异常值检测结果")
        for col, info in outlier_info.items():
            st.markdown(f"**{col}**: 检出异常值 **{info['count']}** 个 ({info['percentage']}%)，"
                       f"阈值范围 [{info['lower_bound']}, {info['upper_bound']}] dB(A)")
        
        st.subheader("📈 清洗前后对比")
        fig_compare = make_subplots(rows=1, cols=2, subplot_titles=('清洗前LAeq分布', '清洗后LAeq分布'))
        
        fig_compare.add_trace(go.Histogram(x=df_raw['LAeq'].dropna(), nbinsx=50, 
                                            name='清洗前', marker_color='rgba(255,82,82,0.7)'),
                              row=1, col=1)
        fig_compare.add_trace(go.Histogram(x=df_clean['LAeq'], nbinsx=50,
                                            name='清洗后', marker_color='rgba(33,150,243,0.7)'),
                              row=1, col=2)
        fig_compare.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_compare, use_container_width=True)
    
    # 原始数据浏览
    st.subheader("🗂️ 数据浏览")
    show_cols = st.multiselect("选择显示列", df_filtered.columns.tolist(),
                                default=['datetime', 'location', 'type', 'LAeq', 'LAmax', 'LAmin', 'LA10', 'LA50', 'LA90'])
    st.dataframe(df_filtered[show_cols].head(100), use_container_width=True, hide_index=True)

# ============================================================
# Tab 5: 频谱分析
# ============================================================
with tab5:
    st.subheader("噪声频谱分析")
    
    freq_cols = ['L63', 'L125', 'L250', 'L500', 'L1000', 'L2000', 'L4000', 'L8000']
    freq_labels = ['63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz', '8kHz']
    
    selected_loc_freq = st.selectbox("选择监测点", df_filtered['location'].unique(), key='freq_loc')
    freq_data = df_filtered[df_filtered['location'] == selected_loc_freq]
    
    col_f1, col_f2 = st.columns(2)
    
    with col_f1:
        # 频谱曲线
        freq_means = freq_data[freq_cols].mean()
        
        fig_freq = go.Figure()
        fig_freq.add_trace(go.Scatter(
            x=freq_labels, y=freq_means.values,
            mode='lines+markers', name='平均频谱',
            line=dict(color='#9C27B0', width=2),
            marker=dict(size=8)
        ))
        
        # 添加不同时段的频谱对比
        day_freq = freq_data[freq_data['datetime'].dt.hour.between(6, 21)][freq_cols].mean()
        night_freq = freq_data[(freq_data['datetime'].dt.hour >= 22) | (freq_data['datetime'].dt.hour < 6)][freq_cols].mean()
        
        fig_freq.add_trace(go.Scatter(
            x=freq_labels, y=day_freq.values,
            mode='lines+markers', name='白天频谱',
            line=dict(color='#FF9800', width=1.5, dash='dash')
        ))
        fig_freq.add_trace(go.Scatter(
            x=freq_labels, y=night_freq.values,
            mode='lines+markers', name='夜间频谱',
            line=dict(color='#3F51B5', width=1.5, dash='dot')
        ))
        
        fig_freq.update_layout(
            title=f'{selected_loc_freq} - 噪声频谱分析',
            xaxis_title='中心频率', yaxis_title='声压级 dB',
            height=450
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col_f2:
        # 频谱热力图（按小时）
        hourly_freq = freq_data.groupby(freq_data['datetime'].dt.hour)[freq_cols].mean()
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=hourly_freq.values,
            x=freq_labels,
            y=[f'{h}:00' for h in hourly_freq.index],
            colorscale='RdYlGn_r',
            colorbar=dict(title='dB')
        ))
        fig_heatmap.update_layout(
            title='时段-频率声压级热力图',
            xaxis_title='中心频率', yaxis_title='时段',
            height=450
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 噪声源识别
    st.subheader("🔍 基于频谱特征的噪声源识别")
    
    for loc in df_filtered['location'].unique():
        loc_freq_mean = df_filtered[df_filtered['location'] == loc][freq_cols].mean()
        peak_freq_idx = loc_freq_mean.idxmax()
        peak_freq = freq_labels[freq_cols.index(peak_freq_idx)]
        
        # 简化的噪声源识别逻辑
        if peak_freq in ['63Hz', '125Hz']:
            source = "低频为主 → 可能来源：交通振动、大型机械设备、空调外机"
        elif peak_freq in ['250Hz', '500Hz']:
            source = "中低频为主 → 可能来源：交通噪声、施工噪声、人群低语"
        elif peak_freq in ['1kHz', '2kHz']:
            source = "中高频为主 → 可能来源：人声交谈、广播、教学活动"
        else:
            source = "高频为主 → 可能来源：哨声、金属摩擦、电子设备"
        
        st.markdown(f"**{loc}**: 主频 {peak_freq} → {source}")

# ============================================================
# Tab 6: 降噪决策
# ============================================================
with tab6:
    st.subheader("💡 降噪优化决策支持")
    
    # 生成建议
    advices = generate_noise_reduction_advice(df_filtered)
    
    # 超标总览
    exceed_locs = [a for a in advices if a['day_exceed'] or a['night_exceed']]
    
    st.markdown(f"### 📊 达标评估总览（适用标准：{selected_std}）")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("监测点总数", len(advices))
    with col_e2:
        st.metric("超标监测点", len(exceed_locs), delta=f"{len(exceed_locs)/len(advices)*100:.0f}%")
    with col_e3:
        if exceed_locs:
            max_over = max(max(a['day_over'], a['night_over']) for a in exceed_locs)
            st.metric("最大超标量", f"{max_over:.1f} dB(A)")
        else:
            st.metric("最大超标量", "0 dB(A)", delta="全部达标")
    
    st.markdown("---")
    
    # 各监测点详细评估
    for advice in advices:
        loc_name = advice['location']
        
        with st.expander(f"📍 {loc_name}（适用标准：{advice['std_type']}）", expanded=advice['day_exceed'] or advice['night_exceed']):
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                day_status, day_color = get_standard_exceeded(advice['day_leq'], advice['day_std'])
                st.markdown(f"**昼间**: LAeq = {advice['day_leq']} dB(A) / 限值 = {advice['day_std']} dB(A)")
                st.markdown(f"状态: :{day_color}[{day_status}]" + (f"（超标 {advice['day_over']} dB）" if advice['day_exceed'] else ""))
            
            with col_a2:
                night_status, night_color = get_standard_exceeded(advice['night_leq'], advice['night_std'])
                st.markdown(f"**夜间**: LAeq = {advice['night_leq']} dB(A) / 限值 = {advice['night_std']} dB(A)")
                st.markdown(f"状态: :{night_color}[{night_status}]" + (f"（超标 {advice['night_over']} dB）" if advice['night_exceed'] else ""))
            
            if advice['measures']:
                st.markdown("#### 🔧 降噪建议")
                for m in advice['measures']:
                    st.markdown(f"- {m}")
    
    # 降噪方案对比
    st.markdown("---")
    st.subheader("📊 常见降噪措施效果对比")
    
    reduction_data = pd.DataFrame({
        '降噪措施': ['声屏障(4m高)', '双层隔音窗', '绿化带(30m宽)', '低噪路面', '建筑外保温', '限速管控'],
        '降噪效果(dB)': [8, 30, 5, 3, 4, 2],
        '适用场景': ['交通干线', '临街建筑', '开放空间', '道路', '建筑外墙', '小区道路'],
        '经济成本': ['中', '中', '低', '中', '高', '低'],
        '实施难度': ['中', '低', '低', '高', '中', '低']
    })
    st.dataframe(reduction_data, use_container_width=True, hide_index=True)
    
    fig_reduction = px.bar(reduction_data, x='降噪措施', y='降噪效果(dB)',
                           color='经济成本', title='常见降噪措施效果对比',
                           color_discrete_map={'低': '#4CAF50', '中': '#FF9800', '高': '#F44336'})
    st.plotly_chart(fig_reduction, use_container_width=True)

# ============================================================
# 页脚
# ============================================================
st.markdown("---")
st.caption("校园与居民区周边声环境大数据监测分析及降噪优化决策研究 | 2026年中国大学生计算机设计大赛 | 大数据应用赛道")
