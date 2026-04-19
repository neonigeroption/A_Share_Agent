"""
vector_rag.py — 向量 RAG 核心模块 (FAISS + 智谱 Embedding)

这个模块实现了完整的 RAG (检索增强生成) 管线：
    1. 抓取新闻原文 (Retrieve)
    2. 把长文本切成小块 (Chunk)
    3. 用智谱 embedding-3 把文本变成向量 (Embed)
    4. 存进 FAISS 向量数据库 (Index)
    5. 用户提问时，找出最相关的几段话 (Search)
    6. 把找到的内容塞给 LLM 生成回答 (Augment + Generate)

为什么不用 LangChain？
    → 手写每一步，答辩时能讲清楚原理，不被问"你只是调了个库吧？"
"""

import numpy as np
import faiss

# 统一配置管理（st.secrets 优先，dotenv 兜底）
from config import get_openai_client

# 复用现有的新闻抓取函数
from rag_search import get_realtime_news

# ========== 初始化 ==========
client = get_openai_client()


# ========== 第 1 步：文本分块 (Chunking) ==========
def chunk_text(text, chunk_size=200, overlap=50):
    """
    把一大段新闻文本切成小块，每块约 chunk_size 个字符。
    
    为什么要分块？
        → LLM 的上下文窗口有限，而且一条新闻里不是每句话都有用。
        → 分成小块后，向量检索能精准定位到"最相关的那几句话"，而不是整篇文章。

    为什么要 overlap（重叠）？
        → 防止关键信息刚好被切断在两个块的边界。
        → 比如"公司获得3亿订单"这句话如果被切成"公司获得3"和"亿订单"就废了。

    参数：
        text: 原始长文本
        chunk_size: 每块的目标长度（字符数）
        overlap: 相邻块之间的重叠字符数
    
    返回：
        chunks: 文本块列表，如 ["第一块内容...", "第二块内容...", ...]
    """
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap   # 往前退 overlap 个字符，制造重叠
    
    return chunks


# ========== 第 2 步：文本 → 向量 (Embedding) ==========
def get_embedding(text):
    """
    调用智谱 embedding-3 模型，把一段文字变成一个向量（一串数字）。
    
    什么是 Embedding？
        → 把文字的"语义"编码成一个数字数组（比如 2048 维的浮点数组）。
        → 语义相近的文本，转出来的向量在空间中距离更近。
        → 比如"股价暴涨"和"涨停封板"的向量距离会很近，
          但和"天气预报"的向量距离很远。
    
    参数：
        text: 要转换的文本
    
    返回：
        一个 numpy 数组，形如 [0.01, -0.03, 0.15, ...] (2048个数字)
    """
    response = client.embeddings.create(
        model="embedding-3",       # 智谱的 embedding 模型
        input=text                 # 要编码的文本
    )
    # response.data[0].embedding 是一个 Python list，转成 numpy 数组方便后续计算
    return np.array(response.data[0].embedding, dtype=np.float32)


def get_embeddings_batch(texts):
    """
    批量版本：一次把多段文字全部转成向量。
    比循环调用 get_embedding() 快，因为只发一次网络请求。
    """
    if not texts:
        return np.array([])
    
    response = client.embeddings.create(
        model="embedding-3",
        input=texts                # 传一个列表进去，一次全转
    )
    # 按顺序取出每个文本的向量
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype=np.float32)


# ========== 第 3 步：构建 FAISS 向量索引 (Indexing) ==========
def build_faiss_index(chunks):
    """
    把所有文本块的向量存进 FAISS 索引。
    
    什么是 FAISS？
        → Facebook 开源的向量检索库，能在百万级向量中毫秒级找到最近邻。
        → 这里用最简单的 IndexFlatL2（暴力搜索 + L2 距离），
          对我们十几条新闻的体量来说绰绰有余。
    
    参数：
        chunks: 文本块列表
    
    返回：
        index: FAISS 索引对象（后续用它来检索）
        vectors: 所有块的向量矩阵（备用）
    """
    print(f"📦 正在对 {len(chunks)} 个文本块进行 Embedding...")
    vectors = get_embeddings_batch(chunks)
    
    # 获取向量维度（embedding-3 通常是 2048 维）
    dimension = vectors.shape[1]
    
    # 创建 FAISS 索引：IndexFlatL2 = 最基础的暴力搜索（对小数据量足够）
    index = faiss.IndexFlatL2(dimension)
    
    # 把所有向量加入索引
    index.add(vectors)
    
    print(f"✅ FAISS 索引构建完成！维度={dimension}, 向量数={index.ntotal}")
    return index, vectors


# ========== 第 4 步：语义检索 (Semantic Search) ==========
def search_relevant_chunks(query, index, chunks, top_k=5):
    """
    用户提了一个问题，我们把问题也变成向量，
    然后在 FAISS 里找出距离最近的 top_k 个文本块。
    
    这就是 RAG 的核心价值：
        → 旧方法：把 10 条新闻全塞给 LLM，LLM 自己猜哪条有用（浪费 token，容易迷失）
        → 新方法：先用向量检索精准定位最相关的 5 段话，只把这 5 段给 LLM（精准、省 token）
    
    参数：
        query: 用户的查询/分析意图，如"该股票涨停的原因"
        index: 之前建好的 FAISS 索引
        chunks: 原始文本块列表（和索引一一对应）
        top_k: 返回最相关的前 K 个块
    
    返回：
        results: [(文本块内容, 相似度距离), ...] 按相关性排序
    """
    # 把查询文本也变成向量
    query_vector = get_embedding(query)
    query_vector = query_vector.reshape(1, -1)  # FAISS 要求输入是二维数组
    
    # 在索引中搜索最近的 top_k 个向量
    # distances: 距离数组（越小越相似）
    # indices: 对应的块编号数组
    distances, indices = index.search(query_vector, min(top_k, len(chunks)))
    
    # 把编号转换回文本内容
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):   # 防止越界
            results.append((chunks[idx], float(distances[0][i])))
    
    return results


# ========== 封装：一键完成 RAG 检索 ==========
def rag_retrieve(stock_name, query, news_count=10, top_k=5):
    """
    完整的 RAG 管线封装：
        抓新闻 → 分块 → 建索引 → 检索 → 返回最相关的文本
    
    参数：
        stock_name: 股票名称，如 "贵州茅台"
        query: 分析意图，如 "该股票涨停的核心原因和资金逻辑"
        news_count: 抓取新闻条数（越多，检索效果越好）
        top_k: 返回最相关的前 K 个文本块
    
    返回：
        rag_context: 拼接好的相关文本（直接塞进 Prompt）
        retrieved_chunks: 检索到的原始块列表（用于 UI 展示）
    """
    # 第 0 步：抓取原始新闻
    print(f"🌐 [RAG] 正在抓取 {stock_name} 的新闻情报...")
    raw_news = get_realtime_news(stock_name, count=news_count)
    
    if not raw_news or "未检索到" in raw_news or "受阻" in raw_news:
        return raw_news or "无可用情报", []
    
    # 第 1 步：分块
    chunks = chunk_text(raw_news, chunk_size=200, overlap=50)
    if not chunks:
        return "新闻内容为空，无法进行向量检索。", []
    
    print(f"✂️ 文本已切分为 {len(chunks)} 个块")
    
    # 第 2+3 步：Embedding + 建索引
    try:
        index, _ = build_faiss_index(chunks)
    except Exception as e:
        print(f"⚠️ 向量索引构建失败: {e}")
        return raw_news, []     # 降级：返回原始新闻文本
    
    # 第 4 步：语义检索
    try:
        results = search_relevant_chunks(query, index, chunks, top_k=top_k)
    except Exception as e:
        print(f"⚠️ 向量检索失败: {e}")
        return raw_news, []     # 降级：返回原始新闻文本
    
    # 拼接检索结果
    rag_context = ""
    retrieved_chunks = []
    for i, (chunk, dist) in enumerate(results, 1):
        rag_context += f"[检索片段{i}] (相似度距离: {dist:.2f})\n{chunk}\n\n"
        retrieved_chunks.append({"text": chunk, "distance": dist})
    
    print(f"🎯 从 {len(chunks)} 个块中检索到 {len(results)} 个最相关片段")
    return rag_context, retrieved_chunks


# ========== 独立测试 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🧪 向量 RAG 管线独立测试")
    print("=" * 60)
    
    # 测试：搜索一只股票的相关新闻
    test_stock = "贵州茅台"
    test_query = "该股票近期的业绩表现和资金动向"
    
    context, chunks = rag_retrieve(test_stock, test_query)
    
    print("\n📋 RAG 检索结果：")
    print("-" * 40)
    print(context)
    
    if chunks:
        print(f"\n✅ 成功检索到 {len(chunks)} 个相关片段")
        for c in chunks:
            print(f"  距离={c['distance']:.2f} | 内容前50字: {c['text'][:50]}...")
    else:
        print("⚠️ 未检索到有效片段")
