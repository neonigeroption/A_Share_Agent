"""
agent_brain.py — CLI 版 Agent 大脑 (命令行独立运行)

适用场景：不启动 Streamlit，直接在终端跑分析。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from market_tool import get_today_top_stocks
# 🔴 使用向量 RAG（替代旧的文本拼接）
from vector_rag import rag_retrieve

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

def analyze_stocks():
    # 1. 获取感知层数据（量价数据）
    df = get_today_top_stocks()
    if isinstance(df, str):
        print(df)
        return

    stock_list_str = df.to_string(index=False)

    # 2. 🔴 启动向量 RAG 检索（FAISS + embedding-3）
    print("🕵️ 启动 FAISS 向量 RAG 检索增强链路...")
    rag_context = ""
    for index, row in df.head(2).iterrows():
        name = row['名称']
        context, chunks = rag_retrieve(
            name,
            query=f"{name}涨停的原因、利好消息和资金逻辑"
        )
        rag_context += f"--- 【{name}】向量检索命中 ---\n{context}\n"
        print(f"  ✅ {name}: 命中 {len(chunks)} 个语义片段")

    # 3. 构造 RAG 增强提示词
    prompt = f"""
    【今日量价战报】:
    {stock_list_str}

    【FAISS 向量 RAG 语义检索情报】（这是你判断真伪龙头的核心依据）:
    {rag_context}

    你现在是一个在大连、深圳活跃多年的"游资总舵主"。
    请结合上方的【量价战报】和【向量检索情报】，用最犀利、不留情面的视角执行以下任务：

    1. **结合舆情识别"真龙"与"杂毛"**：根据向量检索命中的新闻片段，这前两名谁是真有实质性利好（真龙），谁是蹭概念（杂毛）？
    2. **明天竞价预判**：针对涨幅第一的标的，预测明早 9:15-9:25 的竞价表现。
    3. **实战操盘口令**：
        - 如果我手里有票，明天的"撤退水位线"在哪里？
        - 如果我空仓，明天有没有值得"打板"的目标？
    
    要求：拒绝废话。必须结合向量检索命中的具体事件来分析。
    """

    print("🧠 游资总舵主正在融合情报，研判盘面...")
    try:
        response = client.chat.completions.create(
            model="glm-4-flash", 
            messages=[
                {"role": "system", "content": "你是一个言辞犀利、眼里只有资金进出和核心逻辑的顶级游资。"},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ 决策模块报错: {e}"

if __name__ == "__main__":
    result = analyze_stocks()
    print("\n" + "🚀 A_Share_Agent 市场研报（向量 RAG 增强版）")
    print("-" * 50)
    print(result)