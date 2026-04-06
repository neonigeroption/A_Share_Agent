import os
import akshare as ak
import streamlit as st

# 清空代理（必须保留）
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = '*'

# 🔴 核心全局装甲：回归新浪源，保留半小时缓存（1800秒）
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_market_data():
    print("🌐 正在向新浪请求全市场底层数据 (半小时内绝对安全)...")
    # 换回你之前唯一成功过的新浪接口！
    return ak.stock_zh_a_spot()

def get_today_top_stocks():
    print("🚀 从内存抽取涨幅榜...")
    try:
        # 直接从内存全集中切片，不走网络！
        df = fetch_all_market_data()
        top_10 = df.sort_values(by='涨跌幅', ascending=False).head(10)
        return top_10[['代码', '名称', '最新价', '涨跌幅', '成交额']]
    except Exception as e:
        return f"❌ 获取数据失败: {e}"

def get_today_bottom_stocks():
    print("💣 从内存抽取跌幅榜...")
    try:
        # 直接从内存全集中切片，不走网络！
        df = fetch_all_market_data()
        bottom_df = df.sort_values(by="涨跌幅", ascending=True).head(10)
        return bottom_df[['代码', '名称', '最新价', '涨跌幅', '成交额']]
    except Exception as e:
        return f"跌幅榜获取失败: {e}"

def get_single_stock_quote(keyword):
    print(f"🔍 从内存检索 [{keyword}] ...")
    try:
        # 直接从内存全集中切片，不走网络！
        df = fetch_all_market_data()
        res = df[(df['代码'].str.contains(keyword)) | (df['名称'].str.contains(keyword))]
        if res.empty:
            return None
        return res.head(1)[['代码', '名称', '最新价', '涨跌幅', '成交额']]
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_stock_history(symbol, days=60):
    """
    获取单只股票的历史 K 线数据（日线），用于画 K 线图 / MA / MACD。
    
    参数：
        symbol: 6 位股票代码，如 "600519"
        days: 获取最近多少个交易日的数据
    
    返回：
        DataFrame，包含: 日期, 开盘, 收盘, 最高, 最低, 成交量
        如果失败返回 None
    """
    print(f"📈 获取 [{symbol}] 最近 {days} 日 K 线数据...")
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime("%Y%m%d")
        # 多取一些天数，因为有非交易日
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
        
        # 采用新浪源 ak.stock_zh_a_daily 代替东方财富源，穿透对方对 IP 刷新频控造成的 ConnectionError
        # 新浪源要求 symbol 必须带 sh/sz 前缀，因此直接传入 symbol 即可
        df = ak.stock_zh_a_daily(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"        # 前复权
        )
        
        if df is None or df.empty:
            return None
            
        # 兼容处理：将新浪接口返回的英文字段映射为统一的中文列名供 Plotly 画图使用
        df = df.rename(columns={
            "date": "日期",
            "open": "开盘",
            "high": "最高",
            "low": "最低",
            "close": "收盘",
            "volume": "成交量"
        })
        
        # 将字符串全量转为浮点型，避免计算 MA 均线时出错
        for col in ["开盘", "最高", "最低", "收盘", "成交量"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # 只取最近 days 个交易日
        return df.tail(days)
    except Exception as e:
        print(f"⚠️ K 线数据获取失败: {e}")
        return None

if __name__ == "__main__":
    print(get_today_top_stocks())