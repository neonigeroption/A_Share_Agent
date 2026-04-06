import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 导入底层工具
from market_tool import get_today_top_stocks, get_today_bottom_stocks, get_single_stock_quote, get_stock_history
# 向量 RAG 模块（FAISS + embedding-3）
from vector_rag import rag_retrieve
# 🤖 ReAct Agent 模块（Function Calling）
from agent_tools import run_agent, TOOL_DEFINITIONS
# 💼 模拟持仓模块
from portfolio import load_portfolio, add_position, remove_position, get_portfolio_with_pnl

st.set_page_config(page_title="A_Share_Agent 游资终端", page_icon="🚀", layout="wide")

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ================= 侧边栏：手机外网演示入口 =================
import os
if os.path.exists("qr_tunnel.png"):
    with st.sidebar:
        st.markdown("### 📱 扫码手机体验")
        st.image("qr_tunnel.png", use_container_width=True)
        st.markdown("**公共演示网址:**")
        st.code("https://a0fe583b0a2e88.lhr.life")
        st.caption("提示：使用 5G 流量访问，无需连接校园内网。")
# ================= Plotly 可视化组件 =================
def render_momentum_gauge(value, title="市场势能", suffix="%"):
    """
    资金势能仪表盘：用 Plotly Indicator (Gauge) 展示涨跌幅强度。
    
    刻度逻辑：
        - 绿色区 (0~3%)：温和上涨，观望为主
        - 黄色区 (3~7%)：资金开始活跃，可留意
        - 红色区 (7%+)：极端情绪，涨停/跌停级别
    """
    # 根据正负值决定颜色主题
    if value >= 0:
        bar_color = "#FF4136"       # 涨 → 红色（A股惯例）
        gauge_range = [0, 12]
        steps = [
            {"range": [0, 3], "color": "#2d2d2d"},
            {"range": [3, 7], "color": "#3d1f00"},
            {"range": [7, 12], "color": "#5c1a1a"},
        ]
        threshold_val = 10          # 涨停警戒线
    else:
        bar_color = "#2ECC40"       # 跌 → 绿色（A股惯例）
        gauge_range = [-12, 0]
        steps = [
            {"range": [-12, -7], "color": "#1a3d1a"},
            {"range": [-7, -3], "color": "#1f2d00"},
            {"range": [-3, 0], "color": "#2d2d2d"},
        ]
        threshold_val = -10         # 跌停警戒线

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": suffix, "font": {"size": 36, "color": "white"}},
        title={"text": title, "font": {"size": 16, "color": "#aaa"}},
        gauge={
            "axis": {"range": gauge_range, "tickcolor": "#666", "tickfont": {"color": "#999"}},
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "#1a1a1a",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": "#FFD700", "width": 3},
                "thickness": 0.8,
                "value": threshold_val,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def render_bar_chart(df, title="涨跌幅分布"):
    """
    个股涨跌幅柱状图：红涨绿跌，直观展示资金攻击方向。
    """
    names = df['名称'].tolist()
    changes = df['涨跌幅'].tolist()
    
    colors = ["#FF4136" if v >= 0 else "#2ECC40" for v in changes]
    
    fig = go.Figure(go.Bar(
        x=names,
        y=changes,
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in changes],
        textposition="outside",
        textfont={"color": "white", "size": 11},
    ))
    fig.update_layout(
        title={"text": title, "font": {"size": 14, "color": "#aaa"}, "x": 0.5},
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"tickfont": {"color": "#ccc", "size": 10}, "showgrid": False},
        yaxis={"tickfont": {"color": "#999"}, "gridcolor": "#333", "zerolinecolor": "#555",
               "title": {"text": "涨跌幅(%)", "font": {"color": "#999"}}},
        font={"color": "white"},
    )
    return fig


def render_kline_chart(hist_df, stock_name):
    """
    K 线图 + MA 均线 + 成交量 + MACD 四合一图表。
    
    包含：
        - 上方：日 K 线蜡烛图 + MA5/MA10/MA20 均线
        - 中间：成交量柱状图
        - 下方：MACD 指标（DIF + DEA + MACD 柱）
    """
    df = hist_df.copy()
    
    # ========== 计算技术指标 ==========
    # MA 均线
    df['MA5'] = df['收盘'].rolling(window=5).mean()
    df['MA10'] = df['收盘'].rolling(window=10).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    
    # MACD 计算
    ema12 = df['收盘'].ewm(span=12, adjust=False).mean()
    ema26 = df['收盘'].ewm(span=26, adjust=False).mean()
    df['DIF'] = ema12 - ema26
    df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean()
    df['MACD'] = 2 * (df['DIF'] - df['DEA'])
    
    dates = df['日期'].astype(str).tolist()
    
    # ========== 创建三行子图 ==========
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=[f"{stock_name} 日 K 线图", "成交量", "MACD"]
    )
    
    # --- 第 1 行：K 线蜡烛图 ---
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df['开盘'], high=df['最高'],
        low=df['最低'], close=df['收盘'],
        increasing_line_color='#FF4136',
        decreasing_line_color='#2ECC40',
        increasing_fillcolor='#FF4136',
        decreasing_fillcolor='#2ECC40',
        name='K线',
    ), row=1, col=1)
    
    # MA 均线
    fig.add_trace(go.Scatter(x=dates, y=df['MA5'], name='MA5',
        line=dict(color='#FFD700', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df['MA10'], name='MA10',
        line=dict(color='#00BFFF', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=df['MA20'], name='MA20',
        line=dict(color='#FF69B4', width=1)), row=1, col=1)
    
    # --- 第 2 行：成交量 ---
    vol_colors = ['#FF4136' if df['收盘'].iloc[i] >= df['开盘'].iloc[i]
                  else '#2ECC40' for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=dates, y=df['成交量'],
        marker_color=vol_colors, name='成交量',
        showlegend=False,
    ), row=2, col=1)
    
    # --- 第 3 行：MACD ---
    macd_colors = ['#FF4136' if v >= 0 else '#2ECC40' for v in df['MACD']]
    fig.add_trace(go.Bar(
        x=dates, y=df['MACD'],
        marker_color=macd_colors, name='MACD柱',
        showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=df['DIF'], name='DIF',
        line=dict(color='#FFD700', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=df['DEA'], name='DEA',
        line=dict(color='#00BFFF', width=1.5)), row=3, col=1)
    
    # ========== 全局样式 ==========
    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#1a1a1a',
        font=dict(color='white', size=10),
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center',
                    font=dict(size=9)),
        xaxis_rangeslider_visible=False,
    )
    
    for i in range(1, 4):
        fig.update_xaxes(showgrid=False, row=i, col=1)
        fig.update_yaxes(gridcolor='#333', row=i, col=1)
    
    return fig


def display_rag_and_generate_report(df_targets, query_suffix, prompt_template, sys_sys_prompt, success_msg, report_title):
    """
    通用抽象组件：执行向量 RAG 检索并流式生成 AI 研报。
    消除各模式中大量重复的 RAG 查询、UI 展开栏与大模型流式调用代码。
    """
    st.subheader(report_title)
    
    # 构建缓存防抖键值（提取出该次扫描的标志）
    target_names = "_".join(df_targets['名称'].tolist())
    cache_key = f"scan_cache_{report_title}_{target_names}"
    
    if cache_key in st.session_state:
        # 如果缓存中存在，则直接从本地渲染，避免重复消耗 API
        cache_data = st.session_state[cache_key]
        rag_context = cache_data["rag_context"]
        all_chunks = cache_data["all_chunks"]
        full_report = cache_data["report"]
        
        with st.expander(f"👁️ 查看 FAISS 向量检索结果 (共命中 {len(all_chunks)} 个语义片段) [缓存]"):
            if all_chunks:
                for i, c in enumerate(all_chunks, 1):
                    st.markdown(f"**片段{i}** (相似度距离: `{c['distance']:.2f}`)")
                    st.caption(c['text'])
                    st.markdown("---")
            else:
                st.info("未检索到有效片段，降级为纯技术面分析。")
        
        st.markdown(full_report)
        st.success(f"{success_msg} (⚡ 极速缓存回放)")
        return

    # 1. 向量 RAG 检索
    with st.spinner("🔍 启动 FAISS 向量 RAG 链路：深挖底层逻辑..."):
        rag_context = ""
        all_chunks = []
        for index, row in df_targets.iterrows():
            name = row['名称']
            context, chunks = rag_retrieve(
                name, 
                query=f"{name}{query_suffix}"
            )
            rag_context += f"--- 【{name}】向量检索命中 ---\n{context}\n"
            all_chunks.extend(chunks)
        
        with st.expander(f"👁️ 查看 FAISS 向量检索结果 (共命中 {len(all_chunks)} 个语义片段)"):
            if all_chunks:
                for i, c in enumerate(all_chunks, 1):
                    st.markdown(f"**片段{i}** (相似度距离: `{c['distance']:.2f}`)")
                    st.caption(c['text'])
                    st.markdown("---")
            else:
                st.info("未检索到有效片段，将降级为纯技术面分析。")

    # 2. 组装 Prompt 并调用 LLM
    prompt = prompt_template.format(
        data=df_targets.to_string(index=False),
        rag_context=rag_context
    )
    
    report_placeholder = st.empty()
    try:
        stream = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": sys_sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=ai_temp, stream=True
        )
        
        def stream_and_collect():
            full_text = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_text += content
                    yield content
            # 流式传输完成后，连同 RAG 一起写入持久化内存
            st.session_state[cache_key] = {
                "rag_context": rag_context,
                "all_chunks": all_chunks,
                "report": full_text
            }
            
        report_placeholder.write_stream(stream_and_collect)
        st.success(success_msg)
    except Exception as e:
        st.error(f"大脑异常: {e}")


# ================= 侧边栏：终端控制台 =================
with st.sidebar:
    st.header("⚙️ 战术控制台")
    st.markdown("配置 Agent 的底层行为逻辑。")
    
    analysis_mode = st.radio(
        "选择侦察维度", 
        ["🚀 涨幅榜异动狙击", "💣 跌幅榜恐慌排雷", "🔍 单股深度扫描", "🤖 Agent 自主决策"]
    )
    
    st.markdown("---")
    ai_temp = st.slider(
        "🧠 游资激进程度 (Temperature)", 
        min_value=0.1, max_value=1.0, value=0.7, step=0.1,
        help="数值越高，AI 的推演越激进、越具发散性；数值越低，结论越保守。"
    )
    
    st.markdown("---")
    st.caption("🔧 RAG 引擎: FAISS + embedding-3")
    st.caption("📊 可视化: Plotly Gauge + Bar")
    st.caption("🤖 Agent: ReAct + Function Calling")
    
    # 💼 持仓概览
    st.markdown("---")
    st.header("💼 模拟持仓")
    portfolio = load_portfolio()
    if portfolio:
        st.metric("持仓数量", f"{len(portfolio)} 只")
        # 简单展示持仓列表
        for code, pos in portfolio.items():
            st.caption(f"{pos['name']}({code}) {pos['shares']}股 @{pos['buy_price']}")
    else:
        st.info("暂无持仓\n在“单股深度扫描”中模拟买入")

# ================= 主界面 UI 框架 =================
st.title(f"{analysis_mode.split()[0]} A_Share_Agent 量化决策终端")
st.markdown("基于 `GLM-4` + `FAISS 向量 RAG` + `AKShare` 实时舆情检索的短线博弈推演系统。")
st.markdown("---")

# 提示可能存在的全局持仓消息
if "portfolio_msg" in st.session_state:
    st.success(st.session_state.portfolio_msg)
    del st.session_state.portfolio_msg

# 建立左右分栏（Agent 模式不用分栏）
if analysis_mode != "🤖 Agent 自主决策":
    col1, col2 = st.columns([1, 1.5])

# ================= 逻辑分支 1：涨幅榜 =================
if analysis_mode == "🚀 涨幅榜异动狙击":
    if st.button("⚡ 启动涨幅榜扫描", type="primary"):
        with col1:
            st.subheader("📊 实时感知层 (涨幅榜 Top 10)")
            with st.spinner("抓取多头盘面数据中..."):
                df = get_today_top_stocks()
            if isinstance(df, str):
                st.error(df)
                st.stop()
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 🔴 Plotly 资金势能仪表盘
            st.subheader("📈 资金势能仪表盘")
            avg_change = df['涨跌幅'].mean()
            max_change = df['涨跌幅'].max()
            
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    render_momentum_gauge(avg_change, "榜单平均涨幅"),
                    use_container_width=True, key="gauge_top_avg"
                )
            with g2:
                st.plotly_chart(
                    render_momentum_gauge(max_change, "龙头涨幅"),
                    use_container_width=True, key="gauge_top_max"
                )
            
            # 柱状图
            st.plotly_chart(
                render_bar_chart(df, "🚀 涨幅榜 Top 10 资金攻击方向"),
                use_container_width=True, key="bar_top"
            )

        with col2:
            prompt_tpl = "【今日量价战报】:\n{data}\n【FAISS 向量 RAG 语义检索情报】:\n{rag_context}\n你是一个顶级游资。请结合情报：\n1. 识别前两名中的'真龙'与'杂毛'。\n2. 预测涨幅第一标的明早竞价表现。\n3. 给出具体的防守撤退线。要求：冷酷、专业，绝不废话。"
            sys_prompt = "你是一个言辞犀利、眼里只有资金进出逻辑的顶级游资。"
            display_rag_and_generate_report(
                df_targets=df.head(2),
                query_suffix="涨停的原因、利好消息和资金逻辑",
                prompt_template=prompt_tpl,
                sys_sys_prompt=sys_prompt,
                success_msg="✅ 多头盘面推演完成！",
                report_title="🧠 游资总舵主推演"
            )

# ================= 逻辑分支 2：跌幅榜 =================
elif analysis_mode == "💣 跌幅榜恐慌排雷":
    if st.button("💣 启动跌幅榜扫描", type="primary"):
        with col1:
            st.subheader("📊 实时感知层 (跌幅榜 Top 10)")
            with st.spinner("抓取空头盘面数据中..."):
                df = get_today_bottom_stocks()
            if isinstance(df, str):
                st.error(df)
                st.stop()
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 🔴 Plotly 资金势能仪表盘（跌幅版）
            st.subheader("💀 恐慌势能仪表盘")
            avg_change = df['涨跌幅'].mean()
            min_change = df['涨跌幅'].min()
            
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(
                    render_momentum_gauge(avg_change, "榜单平均跌幅"),
                    use_container_width=True, key="gauge_bot_avg"
                )
            with g2:
                st.plotly_chart(
                    render_momentum_gauge(min_change, "最惨个股跌幅"),
                    use_container_width=True, key="gauge_bot_min"
                )
            
            # 柱状图
            st.plotly_chart(
                render_bar_chart(df, "💣 跌幅榜 Top 10 杀伤力分布"),
                use_container_width=True, key="bar_bot"
            )

        with col2:
            prompt_tpl = "【今日最惨战报】:\n{data}\n【FAISS 向量 RAG 语义检索情报】:\n{rag_context}\n你是一个精通抄底与避险的顶级游资。请结合情报：\n1. 分析前两名暴跌的真实原因，是主力恶意洗盘，还是基本面彻底爆雷？\n2. 对于这种大跌票，现在是'别人恐惧我贪婪'的抄底时机，还是'君子不立危墙之下'的绝对禁区？\n3. 给出散户的逃命操作建议。要求：警醒、毒舌，打消散户幻想。"
            sys_prompt = "你是一个言辞犀利、精通危机处理的顶级游资。"
            display_rag_and_generate_report(
                df_targets=df.head(2),
                query_suffix="暴跌的原因、利空消息和风险因素",
                prompt_template=prompt_tpl,
                sys_sys_prompt=sys_prompt,
                success_msg="✅ 恐慌盘排雷完成！",
                report_title="🧠 游资总舵主推演 (危机处理)"
            )

# ================= 逻辑分支 3：单股扫描 =================
elif analysis_mode == "🔍 单股深度扫描":
    search_keyword = st.text_input("🎯 锁定目标 (输入股票代码或简称，如: 贵州茅台 或 600519)")
    
    if st.button("🔍 执行深度扫描", type="primary"):
        if not search_keyword:
            st.warning("⚠️ 请先输入目标代码或名称。")
            st.stop()
        # 将搜索目标存入 session_state，防止内部的"买入"按钮由于嵌套限制失效
        st.session_state.target_keyword = search_keyword
            
    if st.session_state.get("target_keyword"):
        kw = st.session_state.target_keyword
        with col1:
            st.subheader(f"📊 目标雷达: {kw}")
            with st.spinner("锁定目标盘口数据..."):
                df = get_single_stock_quote(kw)
            if df is None:
                st.error("未能找到该股票，请检查输入是否有误（暂不支持停牌或退市股票）。")
                st.stop()
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            stock_name = df.iloc[0]['名称']
            stock_change = df.iloc[0]['涨跌幅']
            
            # 🔴 Plotly 单股势能仪表盘
            st.subheader("🎯 目标势能扫描")
            st.plotly_chart(
                render_momentum_gauge(stock_change, f"{stock_name} 实时涨跌幅"),
                use_container_width=True, key="gauge_single"
            )
            
            # 🔴 K 线图 + MA + MACD 量化图表
            stock_code = df.iloc[0]['代码']
            st.subheader("📉 量化技术分析 (K线/MA/MACD)")
            with st.spinner("加载 60 日 K 线数据..."):
                hist_df = get_stock_history(stock_code, days=60)
            if hist_df is not None and not hist_df.empty:
                st.plotly_chart(
                    render_kline_chart(hist_df, stock_name),
                    use_container_width=True, key="kline_chart"
                )
            else:
                st.warning("未能获取历史 K 线数据，可能是新股或接口异常。")
            
        with col2:
            prompt_tpl = "【目标现价数据】:\n{data}\n【FAISS 向量 RAG 语义检索情报】:\n{rag_context}\n你是一个独立视角的顶级游资。请对该票进行定向体检：\n1. 结合今日量价和近期舆情，给该股近期的主力资金意图定性（吸筹、洗盘、拉高、出货？）。\n2. 如果我准备明早重仓买入，给我泼一盆冷水，指出最大的风险点。\n3. 给出一个严格的止损或止盈位建议。"
            sys_prompt = "你是一个客观冷酷、一针见血的独立交易员。"
            display_rag_and_generate_report(
                df_targets=df.head(1),
                query_suffix="近期的主力资金动向、核心事件和业绩变化",
                prompt_template=prompt_tpl,
                sys_sys_prompt=sys_prompt,
                success_msg="✅ 深度靶向扫描完成！",
                report_title=f"🧠 {stock_name} 独立研报"
            )
            
            # 💼 模拟买入按钮
            st.markdown("---")
            st.subheader("💼 模拟交易")
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_shares = st.number_input("买入股数", min_value=100, value=100, step=100, key="sim_shares")
            with sim_col2:
                current_price = df.iloc[0]['最新价']
                st.metric("当前价格", f"¥{current_price}")
            
            sim_cost = current_price * sim_shares
            st.caption(f"预估成本: ¥{sim_cost:,.2f}")
            
            def handle_buy_click(code, name, price):
                # 从 session_state 获取最新输入的股数，防止由于未按回车直接点击按钮导致的旧值绑定的 Bug
                shares = st.session_state.sim_shares
                res = add_position(code, name, price, shares)
                st.session_state.portfolio_msg = f"✅ 模拟买入成功！{name} {shares}股 @¥{price}，成本均价 ¥{res['buy_price']}"
            
            st.button(f"✅ 模拟买入 {stock_name}", type="primary", key="sim_buy_btn", 
                      on_click=handle_buy_click, args=(stock_code, stock_name, float(current_price)))

# ================= 逻辑分支 4：Agent 自主决策 =================
elif analysis_mode == "🤖 Agent 自主决策":
    st.markdown("""
    > 🤖 **Agent 模式**：不再需要点按钮，直接用自然语言提问。
    > Agent 会自主决定调用哪些工具（行情查询、RAG 检索），最终给出分析结论。
    
    示例提问：
    - "帮我看看今天涨幅榜，分析一下谁是真龙"
    - "查一下贵州茅台的最新情况"
    - "今天跌得最惨的股票是什么，有没有抄底机会"
    """)
    
    # 初始化对话历史（存在 session_state 中实现多轮对话）
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []
    
    # 展示历史对话
    for msg in st.session_state.agent_messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
    
    # 用户输入
    if user_input := st.chat_input("输入你的问题，Agent 会自主决定调用哪些工具..."):
        # 显示用户消息
        st.session_state.agent_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
        
        # Agent 推理
        with st.chat_message("assistant", avatar="🤖"):
            # 工具调用进度展示区
            tool_status = st.status("🧠 Agent 正在推理中...", expanded=True)
            tool_calls_display = []
            
            def on_tool_call(name, args):
                """Agent 每次调用工具时的回调，实时展示进度"""
                tool_labels = {
                    "get_today_top_stocks": "🚀 查询涨幅榜",
                    "get_today_bottom_stocks": "💣 查询跌幅榜",
                    "get_single_stock_quote": "🔍 查询个股行情",
                    "rag_retrieve": "📰 RAG 舆情检索",
                }
                label = tool_labels.get(name, name)
                tool_status.update(label=f"🔧 正在调用: {label}")
                tool_status.write(f"✅ {label} `{args}`")
                tool_calls_display.append(f"{label}")
            
            # 执行 ReAct Agent 循环
            with st.spinner(""):
                response, updated_history, tool_log = run_agent(
                    user_input,
                    history=st.session_state.agent_history,
                    temperature=ai_temp,
                    on_tool_call=on_tool_call
                )
            
            # 更新状态标签
            if tool_log:
                tool_summary = "、".join([t['tool'] for t in tool_log])
                tool_status.update(
                    label=f"✅ Agent 完成推理（调用了 {len(tool_log)} 个工具）",
                    state="complete", expanded=False
                )
            else:
                tool_status.update(
                    label="✅ Agent 直接回答（未调用工具）",
                    state="complete", expanded=False
                )
            
            # 展示最终回答
            st.markdown(response)
        
        # 保存到历史
        st.session_state.agent_messages.append({"role": "assistant", "content": response})
        # 只保留 user/assistant 消息给下一轮（不保留工具调用过程的中间消息）
        st.session_state.agent_history.append({"role": "user", "content": user_input})
        st.session_state.agent_history.append({"role": "assistant", "content": response})
    
    # 清空对话按钮
    if st.session_state.agent_messages:
        if st.button("🗑️ 清空对话历史"):
            st.session_state.agent_messages = []
            st.session_state.agent_history = []
            st.rerun()

# ================= 全局：模拟持仓管理面板 =================
st.markdown("---")
portfolio_data = load_portfolio()
if portfolio_data:
    st.header("💼 模拟持仓盈亏面板")
    
    # 获取实时价格的辅助函数
    def get_price_for_portfolio(code):
        """从缓存的全市场数据中查询当前价格"""
        try:
            from market_tool import fetch_all_market_data
            all_data = fetch_all_market_data()
            match = all_data[all_data['代码'].str.contains(code)]
            if not match.empty:
                row = match.iloc[0]
                return (row['最新价'], row['涨跌幅'])
        except Exception:
            pass
        return None
    
    positions, total_cost, total_value, total_pnl, total_pnl_pct = get_portfolio_with_pnl(
        get_price_for_portfolio
    )
    
    # 总览指标
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("总成本", f"¥{total_cost:,.2f}")
    with m2:
        st.metric("总市值", f"¥{total_value:,.2f}")
    with m3:
        pnl_color = "🔴" if total_pnl >= 0 else "🟢"
        st.metric("总浮动盈亏", f"¥{total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.2f}%", delta_color="inverse")
    
    # 持仓明细表
    if positions:
        st.subheader("📋 持仓明细")
        for pos in positions:
            with st.container():
                pc1, pc2, pc3, pc4, pc5 = st.columns([2, 1.5, 1.5, 1.5, 1])
                with pc1:
                    st.markdown(f"**{pos['name']}** `{pos['code']}`")
                    st.caption(f"{pos['shares']}股 | 买入: ¥{pos['buy_price']}")
                with pc2:
                    st.metric("现价", f"¥{pos['current_price']}", delta=f"{pos['today_change']:+.2f}%", delta_color="inverse")
                with pc3:
                    st.metric("市值", f"¥{pos['market_value']:,.0f}")
                with pc4:
                    pnl_str = f"¥{pos['pnl']:+,.2f}"
                    st.metric("浮动盈亏", pnl_str, delta=f"{pos['pnl_pct']:+.2f}%", delta_color="inverse")
                with pc5:
                    def handle_sell_click(code, name):
                        removed = remove_position(code)
                        if removed:
                            st.session_state.portfolio_msg = f"🔻 已清仓 {name}"
                            
                    st.button("🔻 清仓", key=f"sell_{pos['code']}", 
                              on_click=handle_sell_click, args=(pos['code'], pos['name']))
                st.markdown("---")
        
        # 持仓分布饼图
        if len(positions) > 1:
            st.subheader("📊 持仓分布")
            fig = go.Figure(go.Pie(
                labels=[p['name'] for p in positions],
                values=[p['market_value'] for p in positions],
                hole=0.4,
                marker=dict(colors=['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9', '#B10DC9']),
                textinfo='label+percent',
                textfont=dict(color='white'),
            ))
            fig.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="portfolio_pie")