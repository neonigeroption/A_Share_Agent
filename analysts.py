"""
analysts.py — 多智能体分析引擎

将原有的单一通用 Agent 拆分为 4 个专业分析师，各自独立调用 LLM 生成结构化信号：
    1. 技术面分析师 — MA/MACD/RSI/布林带 → 趋势信号
    2. 舆情分析师   — RAG 检索 → 情绪信号
    3. 基本面分析师 — 成交额/涨跌幅/量价关系 → 基本面信号
    4. 风险管理师   — 波动率/最大回撤/仓位建议

每个分析师统一返回：
    {
        "signal": "bullish" | "bearish" | "neutral",
        "confidence": 0.0 ~ 1.0,
        "reasoning": "...",
        "metrics": { ... }
    }
"""

import json
import math
from config import get_openai_client
client = get_openai_client()

# LLM 模型配置
MODEL = "glm-4-air"


def _call_analyst_llm(system_prompt, user_prompt):
    """
    通用：调用 LLM 并解析 JSON 信号。
    如果 LLM 返回的不是合法 JSON，则降级为 neutral 信号。
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        # 尝试提取 JSON（LLM 有时会在 JSON 外面加 markdown 代码块）
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)
        # 校验必要字段
        assert result.get("signal") in ("bullish", "bearish", "neutral")
        assert 0 <= result.get("confidence", 0) <= 1
        return result
    except Exception as e:
        return {
            "signal": "neutral",
            "confidence": 0.3,
            "reasoning": f"分析师调用异常，降级为中性信号: {str(e)}",
            "metrics": {}
        }


# ==================== 1. 技术面分析师 ====================

def run_technical_analyst(stock_name, stock_code, history_data):
    """
    技术面分析师：基于 K 线历史数据计算指标并判断趋势。
    
    参数:
        stock_name: 股票名称
        stock_code: 股票代码
        history_data: pandas DataFrame，含 ['日期', '开盘', '收盘', '最高', '最低', '成交量'] 列
    """
    # 本地计算技术指标，减少 LLM 幻觉
    metrics = _calculate_technical_indicators(history_data)
    
    system_prompt = """你是一位专业的 A 股技术面分析师。
你的任务是根据提供的技术指标数据，给出明确的交易信号。

必须以 JSON 格式返回：
{
    "signal": "bullish" 或 "bearish" 或 "neutral",
    "confidence": 0.0到1.0之间的数字,
    "reasoning": "详细的技术分析逻辑，引用具体指标数值",
    "metrics": {"trend": "up/down/sideways", "key_indicator": "指标名"}
}"""

    user_prompt = f"""请分析 {stock_name}({stock_code}) 的技术面：

近期技术指标摘要：
{json.dumps(metrics, ensure_ascii=False, indent=2)}

请基于以上指标给出你的判断。"""

    return _call_analyst_llm(system_prompt, user_prompt)


def _calculate_technical_indicators(df):
    """从 K 线数据中计算关键技术指标"""
    if df is None or df.empty or len(df) < 5:
        return {"error": "数据不足，无法计算技术指标"}
    
    try:
        close = df["收盘"].astype(float)
        volume = df["成交量"].astype(float) if "成交量" in df.columns else None

        latest_price = float(close.iloc[-1])
        
        # MA 均线
        ma5 = float(close.tail(5).mean()) if len(close) >= 5 else latest_price
        ma10 = float(close.tail(10).mean()) if len(close) >= 10 else latest_price
        ma20 = float(close.tail(20).mean()) if len(close) >= 20 else latest_price
        
        # 涨跌幅
        pct_change_1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0
        pct_change_5d = float((close.iloc[-1] / close.iloc[-5] - 1) * 100) if len(close) >= 5 else 0
        
        # RSI (14日)
        rsi_14 = _calculate_rsi(close, 14)
        
        # 布林带 (20日)
        if len(close) >= 20:
            bb_mid = ma20
            bb_std = float(close.tail(20).std())
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
        else:
            bb_upper = bb_lower = bb_mid = latest_price
        
        # 成交量趋势
        vol_trend = "N/A"
        if volume is not None and len(volume) >= 5:
            vol_ma5 = float(volume.tail(5).mean())
            vol_latest = float(volume.iloc[-1])
            vol_trend = "放量" if vol_latest > vol_ma5 * 1.2 else ("缩量" if vol_latest < vol_ma5 * 0.8 else "平量")

        return {
            "最新价": round(latest_price, 2),
            "MA5": round(ma5, 2),
            "MA10": round(ma10, 2),
            "MA20": round(ma20, 2),
            "均线排列": "多头排列" if ma5 > ma10 > ma20 else ("空头排列" if ma5 < ma10 < ma20 else "交叉缠绕"),
            "1日涨跌幅%": round(pct_change_1d, 2),
            "5日涨跌幅%": round(pct_change_5d, 2),
            "RSI_14": round(rsi_14, 2),
            "RSI状态": "超买(>70)" if rsi_14 > 70 else ("超卖(<30)" if rsi_14 < 30 else "中性"),
            "布林上轨": round(bb_upper, 2),
            "布林下轨": round(bb_lower, 2),
            "价格位置": "突破上轨" if latest_price > bb_upper else ("跌破下轨" if latest_price < bb_lower else "通道内运行"),
            "成交量趋势": vol_trend,
        }
    except Exception as e:
        return {"error": f"指标计算异常: {str(e)}"}


def _calculate_rsi(series, period=14):
    """计算 RSI 指标"""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = float(gain.tail(period).mean())
    avg_loss = float(loss.tail(period).mean())
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ==================== 2. 舆情分析师 ====================

def run_sentiment_analyst(stock_name, stock_code, rag_context=None):
    """
    舆情分析师：基于 RAG 检索到的新闻上下文进行情感分析。
    
    参数:
        stock_name: 股票名称
        stock_code: 股票代码
        rag_context: RAG 检索到的新闻文本（可选）
    """
    system_prompt = """你是一位专业的 A 股市场情绪分析师。
你的任务是根据提供的新闻资讯，分析市场对该股票的情绪倾向。

必须以 JSON 格式返回：
{
    "signal": "bullish" 或 "bearish" 或 "neutral",
    "confidence": 0.0到1.0之间的数字,
    "reasoning": "详细的情绪分析，引用具体新闻内容",
    "metrics": {"sentiment_score": -1到1之间的数字, "news_count": 新闻条数}
}"""

    news_text = rag_context if rag_context else "暂无相关新闻数据"
    
    user_prompt = f"""请分析 {stock_name}({stock_code}) 的市场情绪：

相关新闻资讯：
{news_text[:3000]}

请基于以上信息给出情绪判断。"""

    return _call_analyst_llm(system_prompt, user_prompt)


# ==================== 3. 基本面分析师 ====================

def run_fundamental_analyst(stock_name, stock_code, market_data=None):
    """
    基本面分析师：基于实时行情数据中的成交额、量比、换手率等指标分析。
    
    参数:
        stock_name: 股票名称
        stock_code: 股票代码
        market_data: dict，含成交额、换手率、量比等字段
    """
    system_prompt = """你是一位专业的 A 股基本面分析师。
你的任务是根据提供的行情基本面数据，分析该股票的短期价值与资金面状况。

必须以 JSON 格式返回：
{
    "signal": "bullish" 或 "bearish" 或 "neutral",
    "confidence": 0.0到1.0之间的数字,
    "reasoning": "详细的基本面分析，引用具体数据",
    "metrics": {"volume_status": "放量/缩量/平量", "capital_flow": "流入/流出/平衡"}
}"""

    data_text = json.dumps(market_data, ensure_ascii=False, indent=2) if market_data else "暂无基本面数据"

    user_prompt = f"""请分析 {stock_name}({stock_code}) 的基本面：

行情数据：
{data_text}

请基于以上数据给出你的判断。"""

    return _call_analyst_llm(system_prompt, user_prompt)


# ==================== 4. 风险管理师 ====================

def run_risk_manager(stock_name, stock_code, history_data, portfolio_info=None):
    """
    风险管理师：评估风险并给出仓位建议。
    
    参数:
        stock_name: 股票名称
        stock_code: 股票代码
        history_data: K 线历史数据
        portfolio_info: 当前持仓信息（可选）
    """
    # 本地计算风险指标
    risk_metrics = _calculate_risk_metrics(history_data)

    system_prompt = """你是一位专业的风险管理师。
你的任务是评估股票的风险水平，并给出仓位建议。

必须以 JSON 格式返回：
{
    "signal": "bullish"(风险可控可加仓) 或 "bearish"(风险过高应减仓) 或 "neutral"(风险适中维持),
    "confidence": 0.0到1.0之间的数字,
    "reasoning": "详细的风险评估",
    "metrics": {"risk_level": "低/中/高", "suggested_position": "0-100%之间", "max_drawdown": "最大回撤%"}
}"""

    portfolio_text = json.dumps(portfolio_info, ensure_ascii=False) if portfolio_info else "当前无持仓"
    
    user_prompt = f"""请评估 {stock_name}({stock_code}) 的风险：

风险指标：
{json.dumps(risk_metrics, ensure_ascii=False, indent=2)}

当前持仓：
{portfolio_text}

请给出风险评估和仓位建议。"""

    return _call_analyst_llm(system_prompt, user_prompt)


def _calculate_risk_metrics(df):
    """计算风险指标"""
    if df is None or df.empty or len(df) < 5:
        return {"error": "数据不足"}
    
    try:
        close = df["收盘"].astype(float)
        returns = close.pct_change().dropna()
        
        # 波动率（年化）
        daily_vol = float(returns.std())
        annual_vol = daily_vol * math.sqrt(252) * 100
        
        # 最大回撤
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax
        max_drawdown = float(drawdown.min()) * 100
        
        # VaR (95%)
        if len(returns) >= 20:
            var_95 = float(returns.quantile(0.05)) * 100
        else:
            var_95 = daily_vol * -1.645 * 100
        
        # 连续下跌天数
        neg_streak = 0
        temp_streak = 0
        for r in returns:
            if r < 0:
                temp_streak += 1
                neg_streak = max(neg_streak, temp_streak)
            else:
                temp_streak = 0
                
        return {
            "日波动率%": round(daily_vol * 100, 2),
            "年化波动率%": round(annual_vol, 2),
            "最大回撤%": round(max_drawdown, 2),
            "VaR_95%": round(var_95, 2),
            "最长连跌天数": neg_streak,
            "近5日均涨幅%": round(float(returns.tail(5).mean()) * 100, 2),
            "风险等级": "高" if annual_vol > 40 else ("中" if annual_vol > 25 else "低"),
        }
    except Exception as e:
        return {"error": f"风险计算异常: {str(e)}"}


# ==================== 编排函数 ====================

def run_all_analysts(stock_name, stock_code, history_data=None, 
                     rag_context=None, market_data=None, portfolio_info=None):
    """
    一键运行所有分析师，返回汇总结果。
    
    返回: dict，key 为分析师名称，value 为信号结果
    """
    results = {}
    
    results["technical"] = {
        "name": "技术面分析师",
        "result": run_technical_analyst(stock_name, stock_code, history_data)
    }
    
    results["sentiment"] = {
        "name": "舆情分析师",
        "result": run_sentiment_analyst(stock_name, stock_code, rag_context)
    }
    
    results["fundamental"] = {
        "name": "基本面分析师",
        "result": run_fundamental_analyst(stock_name, stock_code, market_data)
    }
    
    results["risk"] = {
        "name": "风险管理师",
        "result": run_risk_manager(stock_name, stock_code, history_data, portfolio_info)
    }
    
    return results
