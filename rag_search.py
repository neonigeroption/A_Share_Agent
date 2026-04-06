"""
rag_search.py — 新闻原始数据抓取层

职责：从东方财富抓取个股新闻原文。
    → 这是 RAG 管线的第一步（R = Retrieve），负责"把数据捞回来"。
    → 向量化、索引、检索等后续步骤在 vector_rag.py 中完成。
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import akshare as ak
import time


def get_realtime_news(stock_name, count=10):
    """
    从东方财富新闻库抓取指定股票的最新新闻。

    参数：
        stock_name: 股票名称，如 "贵州茅台"
        count: 抓取新闻条数（默认10条，给向量检索提供更多素材）
    
    返回：
        拼接好的新闻文本字符串
    """
    print(f"🌐 [RAG] 正在从东财新闻库抽调 {stock_name} 的最新 {count} 条情报...")
    time.sleep(1)   # 礼貌延时，避免被反爬
    
    try:
        # 清洗股票名称：去掉 N / *ST / ST 这些前缀，否则搜不到
        clean_name = stock_name.replace("N", "").replace("*ST", "").replace("ST", "")
        news_df = ak.stock_news_em(symbol=clean_name)
        
        if news_df is None or news_df.empty:
            return f"未检索到 {clean_name} 的突发消息，大概率为纯资金博弈或情绪炒作。"
        
        # 取前 count 条新闻
        top_news = news_df.head(count)
        news_context = ""
        for i, (_, row) in enumerate(top_news.iterrows(), 1):
            news_context += f"情报{i}: 【{row['新闻标题']}】\n{row['新闻内容']}\n\n"
        
        if not news_context.strip():
            return f"未检索到 {clean_name} 的突发消息，大概率为纯资金博弈或情绪炒作。"
        
        return news_context
    except Exception as e:
        print(f"⚠️ 搜索底层报错: {e}")
        return "RAG 检索线路受阻，降级为纯技术面分析。"


if __name__ == "__main__":
    print("测试搜索 贵州茅台:")
    print(get_realtime_news("贵州茅台", count=5))