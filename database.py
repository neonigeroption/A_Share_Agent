"""
database.py — SQLite 持久化层

从 JSON 文件升级为关系型数据库，统一管理：
    - 模拟持仓数据（替代 portfolio_data.json）
    - 分析历史记录（多智能体分析报告存档）
    - 回测结果（策略绩效指标）
    - Agent 工具调用日志

为什么选 SQLite？
    → Python 内置 sqlite3 模块，零安装、零配置
    → Streamlit Cloud 直接可用，不需要额外云数据库
    → 单文件数据库，方便备份和迁移
"""

import sys
# 强制标准输出/标准错误使用 UTF-8 编码并对无法编码的字符安全替换，从根本上解决 Windows 下 Emoji 导致的 UnicodeEncodeError
try:
    if sys.platform.startswith('win'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import sqlite3
import json
import os
from datetime import datetime

# 数据库文件路径（和项目代码放在同一目录）
DB_PATH = os.path.join(os.path.dirname(__file__), "agent_data.db")


def get_connection():
    """获取数据库连接（自动创建表结构）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果可以用列名访问
    conn.execute("PRAGMA journal_mode=WAL")  # 提升并发性能
    _init_tables(conn)
    return conn


def _init_tables(conn):
    """初始化所有表（幂等操作，重复调用不会出错）"""
    conn.executescript("""
        -- 模拟持仓表（替代 portfolio_data.json）
        CREATE TABLE IF NOT EXISTS portfolio (
            code        TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            buy_price   REAL NOT NULL,
            shares      INTEGER NOT NULL,
            buy_time    TEXT NOT NULL
        );

        -- 分析历史表（存储每次多智能体分析的完整结果）
        CREATE TABLE IF NOT EXISTS analysis_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code      TEXT NOT NULL,
            stock_name      TEXT NOT NULL,
            analysis_type   TEXT NOT NULL,
            analyst_signals TEXT,
            bull_argument   TEXT,
            bear_argument   TEXT,
            final_signal    TEXT,
            final_confidence REAL,
            final_report    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        -- 回测结果表
        CREATE TABLE IF NOT EXISTS backtest_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code      TEXT NOT NULL,
            stock_name      TEXT NOT NULL,
            start_date      TEXT NOT NULL,
            end_date        TEXT NOT NULL,
            initial_capital REAL NOT NULL,
            -- Agent 策略指标
            agent_return    REAL,
            agent_sharpe    REAL,
            agent_max_drawdown REAL,
            agent_win_rate  REAL,
            agent_trades    INTEGER,
            -- Buy & Hold 基准指标
            bh_return       REAL,
            -- 净值序列（JSON 格式）
            agent_equity_curve TEXT,
            bh_equity_curve    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );

        -- Agent 工具调用日志
        CREATE TABLE IF NOT EXISTS agent_logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT,
            round_num   INTEGER,
            tool_name   TEXT NOT NULL,
            tool_args   TEXT,
            result_preview TEXT,
            created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
        );
    """)
    conn.commit()


# ==================== 持仓操作（兼容旧 portfolio.py 接口）====================

def db_load_portfolio():
    """从 SQLite 加载持仓数据，返回格式与旧 JSON 一致"""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM portfolio").fetchall()
    conn.close()
    
    portfolio = {}
    for row in rows:
        portfolio[row["code"]] = {
            "code": row["code"],
            "name": row["name"],
            "buy_price": row["buy_price"],
            "shares": row["shares"],
            "buy_time": row["buy_time"],
        }
    return portfolio


def db_save_portfolio(portfolio):
    """将持仓数据写入 SQLite（全量覆盖）"""
    conn = get_connection()
    conn.execute("DELETE FROM portfolio")
    for code, pos in portfolio.items():
        conn.execute(
            "INSERT INTO portfolio (code, name, buy_price, shares, buy_time) VALUES (?, ?, ?, ?, ?)",
            (pos["code"], pos["name"], pos["buy_price"], pos["shares"], pos.get("buy_time", ""))
        )
    conn.commit()
    conn.close()


def migrate_json_to_sqlite():
    """一次性迁移：把旧的 portfolio_data.json 导入 SQLite"""
    json_path = os.path.join(os.path.dirname(__file__), "portfolio_data.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            if old_data and not db_load_portfolio():
                db_save_portfolio(old_data)
                try:
                    print("✅ 已将 portfolio_data.json 数据迁移到 SQLite")
                except UnicodeEncodeError:
                    print("Success: migrated portfolio_data.json data to SQLite")
        except Exception as e:
            try:
                print(f"⚠️ JSON 迁移跳过: {e}")
            except UnicodeEncodeError:
                print(f"Warning: JSON migration skipped: {e}")


# ==================== 分析历史 ====================

def save_analysis(stock_code, stock_name, analysis_type, analyst_signals=None,
                  bull_argument=None, bear_argument=None,
                  final_signal=None, final_confidence=None, final_report=None):
    """保存一次分析结果"""
    conn = get_connection()
    conn.execute("""
        INSERT INTO analysis_history 
        (stock_code, stock_name, analysis_type, analyst_signals, 
         bull_argument, bear_argument, final_signal, final_confidence, final_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stock_code, stock_name, analysis_type,
        json.dumps(analyst_signals, ensure_ascii=False) if analyst_signals else None,
        bull_argument, bear_argument,
        final_signal, final_confidence, final_report
    ))
    conn.commit()
    conn.close()


def get_analysis_history(limit=20):
    """获取最近的分析记录"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM analysis_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ==================== 回测结果 ====================

def save_backtest(stock_code, stock_name, start_date, end_date, initial_capital,
                  agent_return, agent_sharpe, agent_max_drawdown, agent_win_rate, agent_trades,
                  bh_return, agent_equity_curve=None, bh_equity_curve=None):
    """保存回测结果"""
    conn = get_connection()
    conn.execute("""
        INSERT INTO backtest_results 
        (stock_code, stock_name, start_date, end_date, initial_capital,
         agent_return, agent_sharpe, agent_max_drawdown, agent_win_rate, agent_trades,
         bh_return, agent_equity_curve, bh_equity_curve)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        stock_code, stock_name, start_date, end_date, initial_capital,
        agent_return, agent_sharpe, agent_max_drawdown, agent_win_rate, agent_trades,
        bh_return,
        json.dumps(agent_equity_curve) if agent_equity_curve else None,
        json.dumps(bh_equity_curve) if bh_equity_curve else None,
    ))
    conn.commit()
    conn.close()


def get_backtest_history(limit=10):
    """获取最近的回测记录"""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM backtest_results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ==================== Agent 日志 ====================

def log_agent_call(session_id, round_num, tool_name, tool_args, result_preview):
    """记录 Agent 工具调用"""
    conn = get_connection()
    conn.execute("""
        INSERT INTO agent_logs (session_id, round_num, tool_name, tool_args, result_preview)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, round_num, tool_name, tool_args, result_preview))
    conn.commit()
    conn.close()


def get_agent_logs(limit=100):
    """查询最近的 Agent 决策日志"""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, session_id, round_num, tool_name, tool_args, result_preview, created_at
        FROM agent_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ==================== 初始化 ====================
# 模块加载时自动迁移旧数据
migrate_json_to_sqlite()
