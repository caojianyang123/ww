# 导入操作系统相关功能
import os
# 导入Chroma向量数据库（LangChain 1.0+使用langchain_chroma）
from langchain_chroma import Chroma
# 导入阿里云百炼的Embedding模型
from langchain_community.embeddings import DashScopeEmbeddings
# 导入文本文件加载器（用于加载txt文档）
from langchain_community.document_loaders import TextLoader
# 导入文本分割器（用于将长文档分割为小块）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 定义Chroma向量数据库的存储路径（项目目录下的chroma_db文件夹）
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
# 定义知识库文档的存储路径（项目目录下的knowledge_base文件夹）
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base")


def get_retriever():
    """
    获取向量检索器
    
    功能：
    1. 创建Embedding模型（阿里云百炼text-embedding-v2）
    2. 加载Chroma向量数据库
    3. 返回检索器对象（用于查询相似文档）
    
    Returns:
        向量检索器对象（Retriever）
    """
    # 创建Embedding模型（文本向量化）
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 加载Chroma向量数据库
    # persist_directory: 指定数据库存储路径
    # embedding_function: 指定向量化模型
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    # 将向量数据库转换为检索器
    # search_kwargs={"k": 3}: 每次检索返回最相似的3个文档
    return vector_store.as_retriever(search_kwargs={"k": 3})


def add_documents(filepaths):
    """
    添加文档到向量数据库
    
    功能：
    1. 加载txt文档
    2. 将文档分割为小块（chunk）
    3. 将小块向量化并存入Chroma
    
    Args:
        filepaths: 文件路径列表
        
    Returns:
        生成的向量块数量
    """
    # 存储加载的文档
    documents = []
    
    # 遍历所有文件路径
    for filepath in filepaths:
        # 只处理txt文件
        if filepath.endswith(".txt"):
            # 创建文本加载器（指定UTF-8编码）
            loader = TextLoader(filepath, encoding="utf-8")
            # 加载文档并添加到列表
            documents.extend(loader.load())

    # 创建文本分割器
    # chunk_size=500: 每个块最多500个字符
    # chunk_overlap=50: 相邻块重叠50个字符（保证上下文连贯）
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # 将文档分割为小块
    chunks = text_splitter.split_documents(documents)

    # 创建Embedding模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 加载或创建Chroma向量数据库
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    
    # 将分割后的文档块添加到向量数据库
    vector_store.add_documents(chunks)
    
    # 返回生成的向量块数量
    return len(chunks)


def rebuild_vector_store():
    """
    重建向量索引（重新处理所有知识库文档）
    
    功能：
    1. 扫描knowledge_base文件夹中的所有txt文件
    2. 重新加载、分割、向量化所有文档
    3. 覆盖原有的向量数据库
    
    Returns:
        生成的向量块数量（如果知识库为空则返回0）
    """
    # 如果知识库文件夹不存在，返回0
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return 0

    # 存储加载的文档
    documents = []
    
    # 遍历知识库文件夹中的所有文件
    for filename in os.listdir(KNOWLEDGE_BASE_PATH):
        # 只处理txt文件
        if filename.endswith(".txt"):
            # 拼接完整文件路径
            filepath = os.path.join(KNOWLEDGE_BASE_PATH, filename)
            # 创建文本加载器
            loader = TextLoader(filepath, encoding="utf-8")
            # 加载文档
            documents.extend(loader.load())

    # 如果没有文档，返回0
    if not documents:
        return 0

    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # 分割文档
    chunks = text_splitter.split_documents(documents)

    # 创建Embedding模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    
    # 使用from_documents创建新的向量数据库（覆盖原有数据）
    # 注意：此方法会重新创建数据库，删除所有旧数据
    vector_store = Chroma.from_documents(
        documents=chunks,               # 文档块
        embedding=embeddings,           # 向量化模型
        persist_directory=CHROMA_DB_PATH  # 存储路径
    )
    
    # 返回生成的向量块数量
    return len(chunks)


def get_knowledge_base_files():
    """
    获取知识库中的所有文件名
    
    Returns:
        文件名列表（只包含txt文件）
    """
    # 如果知识库文件夹不存在，返回空列表
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        return []
    # 返回所有txt文件的文件名
    return [f for f in os.listdir(KNOWLEDGE_BASE_PATH) if f.endswith(".txt")]