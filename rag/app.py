# 导入操作系统相关功能
import os
# 导入系统路径模块（用于添加项目路径）
import sys
# 导入Streamlit，用于构建Web界面
import streamlit as st

# 将项目根目录添加到Python路径（确保可以导入rag模块）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入retriever模块中的函数，用于文档操作
from rag.retriever import add_documents, rebuild_vector_store, get_knowledge_base_files, KNOWLEDGE_BASE_PATH

# 配置Streamlit页面
st.set_page_config(page_title="RAG知识库管理", page_icon="📚", layout="wide")

# 显示页面标题
st.title("📚 RAG知识库管理系统")

# 创建侧边栏，用于选择操作类型
with st.sidebar:
    # 操作选择下拉框：上传文档、查看文档、重建索引
    action = st.selectbox("选择操作", ["上传文档", "查看文档", "重建索引"])

# 如果选择"上传文档"操作
if action == "上传文档":
    # 显示子标题
    st.subheader("上传知识库文档")
    # 显示支持的格式提示
    st.write("支持 .txt 格式的文档")

    # 创建文件上传组件（支持多选）
    uploaded_files = st.file_uploader("选择文件", type=["txt"], accept_multiple_files=True)

    # 如果有文件被上传
    if uploaded_files:
        # 创建"上传并向量化"按钮
        if st.button("上传并向量化"):
            # 存储文件路径列表
            filepaths = []
            # 遍历上传的文件
            for uploaded_file in uploaded_files:
                # 拼接保存路径（保存到knowledge_base文件夹）
                save_path = os.path.join(KNOWLEDGE_BASE_PATH, uploaded_file.name)
                # 将文件写入本地
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # 添加到路径列表
                filepaths.append(save_path)

            # 显示加载动画（向量化过程）
            with st.spinner("正在向量化..."):
                # 调用add_documents函数将文档添加到向量数据库
                chunk_count = add_documents(filepaths)

            # 显示成功提示（包含文件数量和向量块数量）
            st.success(f"✅ 成功上传 {len(filepaths)} 个文件，生成 {chunk_count} 个向量块")

# 如果选择"查看文档"操作
elif action == "查看文档":
    # 显示子标题
    st.subheader("已上传文档列表")
    # 获取知识库中的所有文件名
    files = get_knowledge_base_files()

    # 如果有文档
    if files:
        # 遍历文档列表
        for i, filename in enumerate(files, 1):
            # 创建两列布局（文件名和删除按钮）
            col1, col2 = st.columns([4, 1])
            # 在第一列显示文件名（带序号）
            col1.write(f"{i}. {filename}")
            # 拼接文件完整路径
            filepath = os.path.join(KNOWLEDGE_BASE_PATH, filename)
            # 在第二列创建删除按钮（每个按钮有唯一key）
            if col2.button("删除", key=f"del_{filename}"):
                # 删除文件
                os.remove(filepath)
                # 重新加载页面
                st.rerun()
    else:
        # 如果没有文档，显示提示
        st.info("暂无文档，请先上传")

# 如果选择"重建索引"操作
elif action == "重建索引":
    # 显示子标题
    st.subheader("重建向量索引")
    # 创建"开始重建"按钮
    if st.button("开始重建"):
        # 显示加载动画
        with st.spinner("正在重建索引..."):
            # 调用rebuild_vector_store函数重建索引
            chunk_count = rebuild_vector_store()

        # 如果重建成功（有向量块）
        if chunk_count > 0:
            # 显示成功提示
            st.success(f"✅ 成功重建索引，共 {chunk_count} 个向量块")
        else:
            # 如果知识库为空，显示警告
            st.warning("⚠️ 知识库为空，请先上传文档")