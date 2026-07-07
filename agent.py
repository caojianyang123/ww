# 导入操作系统相关功能
import os
# 导入JSON解析模块
import json
# 导入UUID生成器，用于生成唯一对话ID
import uuid
# 导入Pydantic，用于定义结构化输出模型
from pydantic import BaseModel, Field
# 导入LangChain的LLM初始化函数
from langchain.chat_models import init_chat_model
# 导入工具装饰器，用于定义Agent可用的工具
from langchain_core.tools import tool
# 导入create_agent函数，用于创建智能体
from langchain.agents import create_agent
# 导入HumanMessage，用于构建用户消息
from langchain.messages import HumanMessage
# 导入InMemorySaver，用于存储对话记忆（内存级）
from langgraph.checkpoint.memory import InMemorySaver
# 导入检索器，用于从知识库中检索相关文档
from rag.retriever import get_retriever


# 定义结构化输出模型，确保回答格式统一
class CustomerServiceResponse(BaseModel):
    """
    客服回答的结构化输出模型
    包含三个字段：回答内容、来源、置信度
    """
    # 对用户问题的回答内容
    answer: str = Field(description="对用户问题的回答")
    # 回答来源（知识库文件名或通用知识）
    source: str = Field(description="回答来源，如果来自知识库请列出文件名，否则填'通用知识'")
    # 回答置信度（0-1之间）
    confidence: float = Field(description="回答置信度，0-1之间")


# 使用@tool装饰器定义一个工具函数，Agent可以调用它来检索知识库
@tool
def search_knowledge_base(query: str) -> str:
    """
    检索知识库中的相关信息（Agent可调用的工具）
    
    Args:
        query: 用户查询的关键词
        
    Returns:
        检索到的相关知识片段，包含来源文件名
    """
    # 获取向量检索器
    retriever = get_retriever()
    # 调用检索器查询相关文档（LangChain 1.0使用invoke方法）
    docs = retriever.invoke(query)
    # 整理检索结果
    results = []
    for doc in docs:
        # 从文档元数据中提取来源文件名
        source = doc.metadata.get("source", "unknown").split("\\")[-1]
        # 格式化结果，包含来源和内容
        results.append(f"【来源: {source}】\n{doc.page_content}")
    # 如果有结果则拼接返回，否则返回未找到提示
    return "\n\n".join(results) if results else "未找到相关知识"


# 定义Agent的系统提示词，指导Agent如何工作
system_prompt = """你是一个智能知识库助手，专门回答用户关于产品和服务的问题。

工作流程：
1. 判断用户问题是否需要查询知识库
2. 如果需要，调用 search_knowledge_base 工具检索相关知识
3. 如果不需要，可以直接回答

回答要求：
- 回答要准确、简洁、友好
- 如果来自知识库，要在回答中注明来源文件名
- 不要编造知识"""


# 定义RAG智能体类
class RAGAgent:
    """
    RAG智能体类
    负责初始化LLM、工具、记忆，并处理用户提问
    """
    
    def __init__(self):
        """
        初始化智能体
        1. 创建LLM模型（阿里云百炼Qwen-Plus）
        2. 初始化工具列表
        3. 创建对话记忆（InMemorySaver）
        4. 创建Agent实例
        5. 生成唯一对话ID（thread_id）
        """
        # 使用init_chat_model创建LLM模型
        # 通过OpenAI兼容模式调用阿里云百炼API
        self.llm = init_chat_model(
            model="qwen-plus",                    # 模型名称
            model_provider="openai",              # 使用OpenAI兼容模式
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云百炼API地址
            api_key=os.getenv("DASHSCOPE_API_KEY")  # 从环境变量读取API密钥
        )
        
        # 定义Agent可用的工具列表
        self.tools = [search_knowledge_base]
        
        # 创建内存级的对话记忆存储（InMemorySaver）
        # 当Agent调用invoke时，会自动保存和恢复对话状态
        self.checkpointer = InMemorySaver()
        
        # 使用create_agent创建智能体实例
        # 传入模型、工具、系统提示词和记忆存储
        self.agent = create_agent(
            model=self.llm,                # 语言模型
            tools=self.tools,              # 可用工具
            system_prompt=system_prompt,   # 系统提示词
            checkpointer=self.checkpointer, # 记忆存储（让Agent具备存储能力）
        )
        
        # 生成唯一的对话ID（thread_id）
        # 同一个thread_id共享对话记忆，不同thread_id是独立对话
        self.thread_id = str(uuid.uuid4())

    def ask(self, question: str) -> CustomerServiceResponse:
        """
        处理用户提问
        
        Args:
            question: 用户输入的问题
            
        Returns:
            结构化的回答（CustomerServiceResponse对象）
        """
        # 构建对话配置，指定thread_id
        # 同一个thread_id会自动恢复之前的对话历史
        config = {"configurable": {"thread_id": self.thread_id}}
        
        # 调用Agent，传入用户消息和配置
        # HumanMessage用于构建用户消息
        response = self.agent.invoke(
            {"messages": [HumanMessage(question)]},  # 传入当前用户消息
            config=config                           # 传入对话配置（包含thread_id）
        )
        
        # 从响应中提取最后一条消息的内容（即Agent的回答）
        answer_text = response["messages"][-1].content
        
        # 尝试将回答转换为结构化输出（CustomerServiceResponse）
        try:
            # 使用with_structured_output将LLM包装为结构化输出模式
            structured_llm = self.llm.with_structured_output(CustomerServiceResponse)
            # 调用结构化LLM将回答转换为指定格式
            result = structured_llm.invoke([
                ("system", "请将以下回答转换为JSON格式，包含answer、source、confidence三个字段"),
                ("human", f"回答：{answer_text}")
            ])
            
            # 如果返回的是字典，转换为CustomerServiceResponse对象
            if isinstance(result, dict):
                return CustomerServiceResponse(**result)
            # 如果已经是对象，直接返回
            return result
            
        # 如果结构化转换失败，尝试手动解析JSON
        except Exception:
            try:
                # 去除首尾空格
                json_match = answer_text.strip()
                # 如果回答本身是JSON格式
                if json_match.startswith('{') and json_match.endswith('}'):
                    # 解析JSON
                    data = json.loads(json_match)
                    # 返回结构化对象
                    return CustomerServiceResponse(
                        answer=data.get("answer", answer_text),
                        source=data.get("source", "通用知识"),
                        confidence=data.get("confidence", 0.8)
                    )
            # JSON解析失败
            except json.JSONDecodeError:
                pass
            
            # 如果所有解析都失败，直接返回原始回答
            return CustomerServiceResponse(
                answer=answer_text,
                source="通用知识",
                confidence=0.7
            )

    def clear_history(self):
        """
        清空对话历史
        
        原理：生成新的thread_id，使Agent认为这是一个全新的对话
        之前的历史记录仍然保存在checkpointer中，但不再被访问
        """
        self.thread_id = str(uuid.uuid4())