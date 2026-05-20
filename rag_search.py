"""
rag_search.py — 新闻原始数据抓取层

职责：从东方财富抓取个股新闻原文。
    → 这是 RAG 管线的第一步（R = Retrieve），负责"把数据捞回来"。
    → 向量化、索引、检索等后续步骤在 vector_rag.py 中完成。

注意：akshare 1.18.x 在新版 pandas/pyarrow 环境下有正则兼容 bug，
      这里直接内置修复版实现，无需修改 akshare 源码，云端部署同样生效。
"""

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import json
import time
import pandas as pd

try:
    from curl_cffi import requests as cffi_requests
    _USE_CFFI = True
except ImportError:
    import requests as _requests
    _USE_CFFI = False


def _fetch_stock_news_em(symbol: str) -> pd.DataFrame:
    """
    内置修复版：直接调用东方财富搜索 API 抓取个股新闻。
    修复了 akshare 1.18.x 中 r'\\u3000' 导致的 pyarrow 正则兼容性崩溃。
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    inner_param = {
        "uid": "",
        "keyword": symbol,
        "type": ["cmsArticleWebOld"],
        "client": "web",
        "clientType": "web",
        "clientVersion": "curr",
        "param": {
            "cmsArticleWebOld": {
                "searchScope": "default",
                "sort": "default",
                "pageIndex": 1,
                "pageSize": 10,
                "preTag": "<em>",
                "postTag": "</em>",
            }
        },
    }
    params = {
        "cb": "jQuery35101792940631092459_1764599530165",
        "param": json.dumps(inner_param, ensure_ascii=False),
        "_": "1764599530176",
    }
    headers = {
        "accept": "*/*",
        "accept-language": "en,zh-CN;q=0.9,zh;q=0.8",
        "cache-control": "no-cache",
        "host": "search-api-web.eastmoney.com",
        "referer": "https://so.eastmoney.com/news/s?keyword=603777",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }

    if _USE_CFFI:
        r = cffi_requests.get(url, params=params, headers=headers, timeout=10)
    else:
        r = _requests.get(url, params=params, headers=headers, timeout=10)

    cb = "jQuery35101792940631092459_1764599530165"
    data_text = r.text.strip()
    if data_text.startswith(cb):
        data_text = data_text[len(cb):]
    if data_text.startswith("("):
        data_text = data_text[1:]
    if data_text.endswith(")"):
        data_text = data_text[:-1]

    data_json = json.loads(data_text)
    items = data_json.get("result", {}).get("cmsArticleWebOld", [])
    if not items:
        return pd.DataFrame()

    temp_df = pd.DataFrame(items)
    temp_df["url"] = "http://finance.eastmoney.com/a/" + temp_df["code"].astype(str) + ".html"
    temp_df.rename(columns={
        "date": "发布时间",
        "mediaName": "文章来源",
        "code": "-",
        "title": "新闻标题",
        "content": "新闻内容",
        "url": "新闻链接",
        "image": "-",
    }, inplace=True)
    temp_df["关键词"] = symbol
    cols = [c for c in ["关键词", "新闻标题", "新闻内容", "发布时间", "文章来源", "新闻链接"] if c in temp_df.columns]
    temp_df = temp_df[cols]

    # 清洗 HTML 标签 — 用 regex=False 字面量替换，彻底规避 pyarrow 正则兼容问题
    for col in ["新闻标题", "新闻内容"]:
        if col not in temp_df.columns:
            continue
        for tag in ["(<em>", "</em>)", "<em>", "</em>"]:
            temp_df[col] = temp_df[col].str.replace(tag, "", regex=False)
    if "新闻内容" in temp_df.columns:
        temp_df["新闻内容"] = temp_df["新闻内容"].str.replace("\u3000", "", regex=False)
        temp_df["新闻内容"] = temp_df["新闻内容"].str.replace("\r\n", " ", regex=False)

    return temp_df


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
        clean_name = stock_name.replace("N", "").replace("*ST", "").replace("ST", "").strip()
        news_df = _fetch_stock_news_em(symbol=clean_name)

        if news_df is None or news_df.empty:
            return f"未检索到 {clean_name} 的突发消息，大概率为纯资金博弈或情绪炒作。"

        # 取前 count 条新闻
        top_news = news_df.head(count)
        news_context = ""
        for i, (_, row) in enumerate(top_news.iterrows(), 1):
            title = row.get('新闻标题', '')
            content = row.get('新闻内容', '')
            news_context += f"情报{i}: 【{title}】\n{content}\n\n"

        if not news_context.strip():
            return f"未检索到 {clean_name} 的突发消息，大概率为纯资金博弈或情绪炒作。"

        return news_context
    except Exception as e:
        print(f"⚠️ 搜索底层报错: {e}")
        return "RAG 检索线路受阻，降级为纯技术面分析。"


if __name__ == "__main__":
    print("测试搜索 贵州茅台:")
    print(get_realtime_news("贵州茅台", count=5))
