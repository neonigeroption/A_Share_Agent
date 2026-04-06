"""
portfolio.py — 模拟持仓管理模块

功能：
    - 模拟买入 / 卖出股票
    - 根据实时行情计算浮动盈亏
    - 持仓数据持久化到本地 JSON 文件
    - 生成持仓概览 Plotly 图表

注意：这是纯模拟系统，不涉及真实交易。
"""

import json
import os
from datetime import datetime


# 持仓数据文件路径
PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), "portfolio_data.json")


def load_portfolio():
    """
    从本地 JSON 文件加载持仓数据。
    
    数据格式：
    {
        "600519": {
            "code": "600519",
            "name": "贵州茅台",
            "buy_price": 1800.00,
            "shares": 100,
            "buy_time": "2026-04-06 12:00:00"
        },
        ...
    }
    """
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_portfolio(portfolio):
    """把持仓数据保存到本地 JSON 文件。"""
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def add_position(code, name, price, shares):
    """
    模拟买入：新增或加仓一只股票。
    
    如果已持有该股票，采用成本均价法：
        新均价 = (原成本 + 新成本) / (原股数 + 新股数)
    """
    portfolio = load_portfolio()
    
    if code in portfolio:
        # 加仓：计算新的平均成本
        old = portfolio[code]
        old_total = old["buy_price"] * old["shares"]
        new_total = price * shares
        total_shares = old["shares"] + shares
        avg_price = (old_total + new_total) / total_shares
        
        portfolio[code]["buy_price"] = round(avg_price, 2)
        portfolio[code]["shares"] = total_shares
    else:
        # 新建仓
        portfolio[code] = {
            "code": code,
            "name": name,
            "buy_price": round(price, 2),
            "shares": shares,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    save_portfolio(portfolio)
    return portfolio[code]


def remove_position(code):
    """模拟卖出：清仓指定股票。"""
    portfolio = load_portfolio()
    if code in portfolio:
        removed = portfolio.pop(code)
        save_portfolio(portfolio)
        return removed
    return None


def get_portfolio_with_pnl(get_current_price_func):
    """
    计算所有持仓的浮动盈亏。
    
    参数：
        get_current_price_func: 获取当前价格的函数，
            接收股票代码，返回 (当前价, 涨跌幅) 或 None
    
    返回：
        positions: 带盈亏的持仓列表
        total_cost: 总成本
        total_value: 总市值
        total_pnl: 总浮动盈亏
        total_pnl_pct: 总盈亏百分比
    """
    portfolio = load_portfolio()
    
    if not portfolio:
        return [], 0, 0, 0, 0
    
    positions = []
    total_cost = 0
    total_value = 0
    
    for code, pos in portfolio.items():
        cost = pos["buy_price"] * pos["shares"]
        total_cost += cost
        
        # 尝试获取当前价格
        price_info = get_current_price_func(code)
        if price_info:
            current_price, today_change = price_info
            market_value = current_price * pos["shares"]
            pnl = market_value - cost
            pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        else:
            current_price = pos["buy_price"]  # 获取不到就用买入价
            market_value = cost
            pnl = 0
            pnl_pct = 0
            today_change = 0
        
        total_value += market_value
        
        positions.append({
            "code": pos["code"],
            "name": pos["name"],
            "shares": pos["shares"],
            "buy_price": pos["buy_price"],
            "current_price": current_price,
            "today_change": today_change,
            "cost": round(cost, 2),
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "buy_time": pos.get("buy_time", ""),
        })
    
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
    
    return positions, round(total_cost, 2), round(total_value, 2), round(total_pnl, 2), round(total_pnl_pct, 2)
