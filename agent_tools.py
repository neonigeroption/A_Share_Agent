"""
agent_tools.py — Function Calling 工具注册与 ReAct Agent 循环

什么是 Function Calling？
    → 传统方式：我们写死代码决定"先调涨幅榜，再调 RAG，最后让 LLM 分析"
    → Function Calling：LLM 自己决定要调什么工具。
      用户说"帮我看看茅台"，LLM 会自主决定：
        1. 先调 get_single_stock_quote("贵州茅台") 拿行情
        2. 再调 rag_retrieve("贵州茅台", "近期资金动向") 拿舆情
        3. 最后综合分析出结论
    
    这才是真正的 "Agent"——有自主决策能力，而不是执行写死的脚本。

什么是 ReAct？
    → ReAct = Reasoning + Acting（推理 + 行动）
    → Agent 的思考循环：想一步 → 做一步 → 看结果 → 再想 → 再做 ... 直到得出答案
    → 这个文件实现的就是这个循环。
"""

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# 导入实际的工具函数
from market_tool import get_today_top_stocks, get_today_bottom_stocks, get_single_stock_quote
from vector_rag import rag_retrieve

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)


# ========== 第 1 步：定义工具清单（告诉 LLM 它有哪些工具可用）==========
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_today_top_stocks",
            "description": "获取今日 A 股涨幅榜 Top 10 的股票数据，包括代码、名称、最新价、涨跌幅、成交额。适用于用户想了解今天什么股票涨得好、市场热点在哪的场景。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_today_bottom_stocks",
            "description": "获取今日 A 股跌幅榜 Top 10 的股票数据。适用于用户想了解哪些股票跌得最惨、有没有黑天鹅事件的场景。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_single_stock_quote",
            "description": "查询指定股票的实时行情数据（最新价、涨跌幅、成交额等）。通过股票代码（如 600519）或名称（如 贵州茅台）进行模糊匹配。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "股票代码或名称关键字，例如 '贵州茅台' 或 '600519'"
                    }
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": "对指定股票进行 FAISS 向量 RAG 舆情检索。从东方财富抓取该股最新新闻，通过向量化和语义检索，找出与查询意图最相关的新闻片段。适用于用户想深入了解某只股票的利好/利空消息、异动原因的场景。",
            "parameters": {
                "type": "object",
                "properties": {
                    "stock_name": {
                        "type": "string",
                        "description": "股票名称，如 '贵州茅台'"
                    },
                    "query": {
                        "type": "string",
                        "description": "检索意图描述，如 '近期的利好消息和资金动向'"
                    }
                },
                "required": ["stock_name", "query"]
            }
        }
    }
]


# ========== 第 2 步：工具执行器（LLM 说要调什么工具，这里负责真正执行）==========
def execute_tool(tool_name, arguments_json):
    """
    根据 LLM 返回的工具名和参数，执行对应的函数，返回结果字符串。
    
    这就像一个调度中心：
        LLM 说："我要调 get_single_stock_quote，参数是 keyword=贵州茅台"
        → 这个函数就真的去调 get_single_stock_quote("贵州茅台")
        → 把结果转成字符串返回给 LLM
    """
    try:
        # 解析 LLM 传来的 JSON 参数
        args = json.loads(arguments_json) if arguments_json else {}
    except json.JSONDecodeError:
        args = {}
    
    try:
        if tool_name == "get_today_top_stocks":
            result = get_today_top_stocks()
            if isinstance(result, str):
                return result   # 错误信息
            return result.to_string(index=False)
        
        elif tool_name == "get_today_bottom_stocks":
            result = get_today_bottom_stocks()
            if isinstance(result, str):
                return result
            return result.to_string(index=False)
        
        elif tool_name == "get_single_stock_quote":
            keyword = args.get("keyword", "")
            result = get_single_stock_quote(keyword)
            if result is None:
                return f"未找到股票: {keyword}"
            return result.to_string(index=False)
        
        elif tool_name == "rag_retrieve":
            stock_name = args.get("stock_name", "")
            query = args.get("query", "该股票的最新消息")
            context, chunks = rag_retrieve(stock_name, query)
            return context
        
        else:
            return f"未知工具: {tool_name}"
    
    except Exception as e:
        return f"工具执行异常: {e}"


# ========== 第 3 步：ReAct Agent 循环 ==========
def run_agent(user_message, history=None, temperature=0.7, on_tool_call=None):
    """
    ReAct Agent 主循环：让 LLM 自主决定调用工具，循环执行直到得出最终答案。
    
    工作流程（ReAct 循环）：
        1. 把用户问题 + 工具清单发给 LLM
        2. LLM 返回两种可能：
           a) tool_calls → 它想调工具 → 执行工具 → 把结果喂回去 → 回到第 1 步
           b) 纯文本 → 它觉得信息够了 → 输出最终答案 → 结束
        3. 设一个最大轮次防止死循环
    
    参数：
        user_message: 用户的自然语言问题
        history: 历史对话记录（实现多轮对话）
        temperature: LLM 的 temperature
        on_tool_call: 回调函数，每次调用工具时触发（用于 UI 展示进度）
    
    返回：
        final_response: LLM 的最终文本回答
        messages: 完整的对话记录（含工具调用过程）
        tool_log: 工具调用日志（用于 UI 展示）
    """
    # 系统提示词：告诉 LLM 它是一个有工具的 Agent
    system_prompt = """你是 A_Share_Agent，一个专业的 A 股量化决策 Agent。

你拥有以下能力：
1. 查询实时行情数据（涨幅榜、跌幅榜、单股查询）
2. 通过 FAISS 向量 RAG 检索个股舆情新闻

工作原则：
- 收到用户问题后，先思考需要什么数据，主动调用工具获取
- 拿到数据后，结合舆情进行专业分析
- 分析风格：犀利、专业、不废话，像一个经验丰富的游资
- 永远给出具体的操作建议（止损位、仓位建议等），不说空话"""

    # 构建消息列表
    messages = [{"role": "system", "content": system_prompt}]
    
    # 加入历史对话（实现多轮）
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    tool_log = []       # 记录工具调用日志
    max_rounds = 5      # 最大推理轮次，防止死循环
    
    for round_num in range(max_rounds):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                tools=TOOL_DEFINITIONS,
                temperature=temperature,
            )
        except Exception as e:
            return f"❌ Agent 推理异常: {e}", messages, tool_log
        
        choice = response.choices[0]
        assistant_message = choice.message
        
        # 情况 A：LLM 想调用工具
        if assistant_message.tool_calls:
            # 把 LLM 的回复（含 tool_calls）加入消息历史
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                log_entry = {
                    "round": round_num + 1,
                    "tool": func_name,
                    "args": func_args,
                }
                
                # 回调通知 UI（显示进度）
                if on_tool_call:
                    on_tool_call(func_name, func_args)
                
                # 执行工具
                result = execute_tool(func_name, func_args)
                log_entry["result_preview"] = result[:200] + "..." if len(result) > 200 else result
                tool_log.append(log_entry)
                
                # 把工具结果喂回给 LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        
        # 情况 B：LLM 给出了最终回答（没有 tool_calls）
        else:
            final_response = assistant_message.content or "Agent 未返回内容。"
            messages.append({"role": "assistant", "content": final_response})
            return final_response, messages, tool_log
    
    # 超过最大轮次
    return "⚠️ Agent 推理轮次已达上限，请尝试更具体的问题。", messages, tool_log


# ========== 独立测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 A_Share_Agent ReAct Agent 测试")
    print("=" * 60)
    
    def print_tool_call(name, args):
        print(f"  🔧 调用工具: {name}({args})")
    
    test_query = "帮我看看今天涨幅榜前几名，分析一下谁是真龙"
    print(f"\n用户: {test_query}\n")
    
    response, msgs, logs = run_agent(
        test_query,
        on_tool_call=print_tool_call
    )
    
    print(f"\n🤖 Agent 回复:\n{response}")
    print(f"\n📋 工具调用日志 ({len(logs)} 次):")
    for log in logs:
        print(f"  第{log['round']}轮 → {log['tool']}({log['args']})")
