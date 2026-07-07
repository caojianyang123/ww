# 导入RAGAgent类，用于创建智能体
from agent import RAGAgent


def main():
    """
    主函数：终端问答入口
    
    功能：
    1. 显示欢迎界面
    2. 初始化智能体
    3. 循环接收用户输入
    4. 调用智能体回答问题
    5. 支持退出和重置命令
    """
    # 打印欢迎界面
    print("=" * 60)
    print("           RAG智能知识库助手")
    print("=" * 60)
    print("欢迎使用智能助手！输入问题即可获取答案")
    print("输入 '退出' 或 'quit' 结束对话")
    print("输入 '重置' 清空对话记忆")
    print("=" * 60)

    # 尝试初始化智能体
    try:
        # 创建RAGAgent实例
        agent = RAGAgent()
        # 初始化成功提示
        print("✅ 智能助手已初始化完成\n")
    except Exception as e:
        # 初始化失败提示
        print(f"❌ 初始化失败: {e}")
        return

    # 循环接收用户输入
    while True:
        # 获取用户输入（去除首尾空格）
        query = input("\n用户: ").strip()

        # 如果输入为空，继续等待
        if not query:
            continue

        # 如果输入退出命令，结束程序
        if query in ["退出", "quit", "exit"]:
            print("👋 再见！")
            break

        # 如果输入重置命令，清空对话记忆
        if query in ["重置", "clear"]:
            agent.clear_history()
            print("🧹 对话记忆已清空")
            continue

        # 处理用户提问
        try:
            # 显示思考中提示
            print("🤔 正在思考...")
            # 调用Agent的ask方法获取回答
            response = agent.ask(query)

            # 打印结构化回答
            print("\n助手:")
            print(f"📝 回答: {response.answer}")    # 回答内容
            print(f"📄 来源: {response.source}")    # 回答来源
            print(f"📊 置信度: {response.confidence:.2f}")  # 置信度（保留两位小数）
        except Exception as e:
            # 回答失败提示
            print(f"❌ 回答失败: {e}")


# 如果是直接运行该文件，执行main函数
if __name__ == "__main__":
    main()