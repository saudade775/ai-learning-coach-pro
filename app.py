import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatZhipuAI

from database import (
    init_db,
    save_profile, load_profile, has_profile, clear_profile,
    save_message, load_messages, clear_messages,
    save_file, load_files, clear_files,
    save_memory, load_memories, clear_questionnaire_memory,
    save_api_config, load_api_config,
    get_trial_count, increase_trial_count, reset_trial_count
)

init_db()

st.set_page_config(page_title="AI 学习陪跑顾问", page_icon="🧭", layout="centered")


# ============================================================
# 全局样式（深色主题 + 强制覆盖）
# ============================================================

st.markdown(
    """
    <style>
    /* 引入 Inter 字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* 隐藏 Streamlit 默认菜单和页脚 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 全局背景 */
    .stApp {
        background-color: #0E1117 !important;
    }

    /* 强制侧边栏变黑 */
    section[data-testid="stSidebar"] {
        background-color: #0E1117 !important;
        border-right: 1px solid #1E232B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #1E232B !important;
        color: #FAFAFA !important;
        border: 1px solid #2A2F3A !important;
    }

    /* 主区文字颜色 */
    .stMarkdown, .stText, p, h1, h2, h3, h4, li, span {
        color: #FAFAFA !important;
    }

    /* 标题字重和字间距 */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* 按钮圆角和悬停动效 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid #2A2F3A;
        background-color: #1E232B;
        color: #FAFAFA;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 139, 249, 0.25);
        border-color: #4F8BF9;
    }

    /* 聊天气泡圆角和阴影 */
    .stChatMessage {
        border-radius: 16px;
        border: 1px solid #1E232B;
        background-color: #161B22;
    }

    /* 输入框 */
    .stTextInput > div > div > input,
    .stChatInput > div > div > textarea {
        border-radius: 10px;
        background-color: #1E232B;
        color: #FAFAFA;
    }

    /* 容器卡片 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #161B22 !important;
        border-color: #1E232B !important;
    }

    /* 主容器最大宽度 */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9; margin-bottom: 0; font-size: 2.2rem;'>
        🧭 AI 学习陪跑顾问
    </h1>
    <p style='text-align: center; color: #8B949E; margin-top: 8px; font-size: 1.05rem;'>
        一个懂你的 AI 学习陪跑系统
    </p>
    """,
    unsafe_allow_html=True
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = load_messages()

FREE_TRIAL_LIMIT = 10


# ============================================================
# 模式管理
# ============================================================

def get_current_mode():
    return st.session_state.get("api_mode")


# ============================================================
# API 获取
# ============================================================

def get_api_key():
    mode = get_current_mode()
    if mode == "custom":
        config = load_api_config()
        return config.get("api_key", "")
    return os.environ.get("ZHIPU_API_KEY")


def get_model_name():
    mode = get_current_mode()
    if mode == "custom":
        config = load_api_config()
        return config.get("model") or "glm-3-turbo"
    return "glm-3-turbo"


def safe_invoke(chat_model, prompt):
    APPLY_GUIDE = (
        "\n\n---\n"
        "**📌 下一步怎么办：**\n\n"
        "1. 去智谱AI申请一个自己的 API Key："
        "[👉 点此前往申请](https://open.bigmodel.cn/apikey/platform)\n"
        "2. 注册后复制 Key\n"
        "3. 在左侧「⚙️ AI 设置」里粘贴 Key，点「保存 API 设置」\n"
        "4. 再回来提问即可\n\n"
        "**说明：** 智谱新用户有免费额度，超出后按量付费。"
    )

    try:
        response = chat_model.invoke(prompt)
        return response.content, None
    except Exception as e:
        err_msg = str(e)

        if "1302" in err_msg or "并发" in err_msg:
            return None, "⚠️ 当前体验人数较多，请稍后再试。" + APPLY_GUIDE
        elif "1113" in err_msg or "余额" in err_msg or "额度" in err_msg or "quota" in err_msg.lower():
            return None, "⚠️ 体验额度已用完。" + APPLY_GUIDE
        elif "401" in err_msg or "令牌" in err_msg or "auth" in err_msg.lower() or "invalid api key" in err_msg.lower():
            return None, "⚠️ API Key 无效或已过期，请检查后重试。" + APPLY_GUIDE
        else:
            return None, f"⚠️ 调用 AI 时出错：{err_msg[:200]}" + APPLY_GUIDE


# ============================================================
# 欢迎页
# ============================================================

if "welcome_done" not in st.session_state:
    st.session_state.welcome_done = False

if not has_profile() and not st.session_state.welcome_done:
    with st.container(border=True):
        st.markdown("### 👋 欢迎使用")
        st.markdown("这是一个**懂你的 AI 学习陪跑系统**。")
        st.caption("它会记住你的学习情况，陪着你一步步学 AI。")

    with st.container(border=True):
        st.markdown("### 📌 使用前你需要知道")
        st.markdown(
            """
            **1. 你需要准备一个 API Key（可选）**

            - 系统提供 **10 次免费体验**，你可以直接开始用。
            - 如果你打算长期使用，建议绑定自己的 API Key。
            - 推荐使用 [智谱AI](https://open.bigmodel.cn/apikey/platform)，注册送免费额度。

            **2. 接下来需要你花 2 分钟填一份问卷**
            """
        )

    if st.button("✅ 我准备好了，开始", use_container_width=True, type="primary"):
        st.session_state.welcome_done = True
        st.rerun()

    st.stop()


# ============================================================
# 问卷
# ============================================================

if not has_profile() and st.session_state.welcome_done:
    st.info("👋 第一次使用，先花 2 分钟让 AI 认识你。")

    with st.form("onboarding_form"):
        st.subheader("先花2分钟，让AI认识你")
        st.caption("这份问卷只需要填写一次。")

        identity = st.text_input("我的身份/职业", placeholder="例如：大学生、职场新人")
        doing = st.text_input("我平时主要在做什么", placeholder="例如：写文案、做表格")
        ai_time = st.selectbox("我接触AI的时间", ["还没正式用过", "不到1个月", "1~6个月", "超过半年"])
        tools = st.text_input("我用过的AI工具", placeholder="例如：豆包、ChatGPT；没用过写“无”")
        tried = st.text_input("我已经尝试过的事情", placeholder="例如：问问题、写文章")
        state = st.selectbox("我觉得自己更接近哪种状态", [
            "完全零基础，不知道从哪里开始",
            "用过一些，但主要是简单问答",
            "能完成简单任务，但结果不稳定",
            "经常使用，希望提升效率或解决复杂问题"
        ])
        purpose = st.multiselect("我最主要的学习目的（可多选）", [
            "提高工作效率", "辅助学习与研究", "提升内容创作能力",
            "学会编程或开发应用", "探索转行、副业或业务机会",
            "跟上技术变化，建立基础认知", "其他"
        ])
        one_thing = st.text_input("如果AI现在只能帮我解决一件事，我最希望是")
        goal_3m = st.text_input("未来1~3个月，我最想做到的是")
        deliverable = st.text_input("我希望做出的具体成果")
        difficulties = st.multiselect("最困扰我的问题（最多选3项）", [
            "工具太多，不知道选哪个", "教程太多，不知道先学什么",
            "不知道怎么向AI提问", "AI的回答经常不符合需求",
            "不知道怎么判断AI回答是否可靠", "听得懂教程，但自己做不出来",
            "不知道如何用到真实工作或学习中", "缺少时间，难以坚持", "其他"
        ])
        hours_week = st.text_input("每周可投入约多少小时")
        learn_style = st.multiselect("我更喜欢的学习方式", [
            "先讲清楚，再动手", "直接跟着案例做",
            "围绕我的真实任务边做边学", "给我任务，我先尝试，再获得反馈"
        ])
        support = st.multiselect("我最需要的支持（最多选2项）", [
            "制定学习路线", "筛选工具和资料",
            "拆解任务、带着实操", "解答问题、检查成果",
            "定期提醒、帮助坚持", "阶段复盘、调整计划"
        ])

        submitted = st.form_submit_button("🚀 开始学习", use_container_width=True, type="primary")

        if submitted:
            profile_text = f"""
身份：{identity}
平时在做：{doing}
接触AI时间：{ai_time}
用过的AI工具：{tools}
尝试过：{tried}
当前状态：{state}
学习目的：{"、".join(purpose)}
最想解决的一件事：{one_thing}
3个月目标：{goal_3m}
具体成果：{deliverable}
困难：{"、".join(difficulties)}
每周可投入：{hours_week}小时
学习方式：{"、".join(learn_style)}
需要的支持：{"、".join(support)}
"""
            save_profile(
                name=identity or "未填写",
                level=state or "未填写",
                direction="、".join(purpose) if purpose else "未填写",
                time_per_day=hours_week if hours_week else "未填写",
                preference="、".join(learn_style) if learn_style else "未填写"
            )
            save_memory("用户问卷信息：" + profile_text)
            st.toast("✅ 档案已保存", icon="🎉")
            st.rerun()

    st.stop()


# ============================================================
# 选择使用方式
# ============================================================

if "api_mode" not in st.session_state:
    st.session_state["api_mode"] = None

if "skip_key_check" not in st.session_state:
    st.session_state["skip_key_check"] = False

if st.session_state.get("_reenter_custom"):
    st.session_state["api_mode"] = "custom"
    st.session_state["_reenter_custom"] = False
    st.session_state["skip_key_check"] = True

if st.session_state["api_mode"] is None:
    st.markdown("## 🔧 开始前，请选择使用方式")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown("#### 🎁 体验模式")
            st.caption("提供 10 次免费体验，先感受一下。")
            if st.button("用体验模式", use_container_width=True):
                st.session_state["api_mode"] = "trial"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("#### 🔑 用自己的 API Key")
            st.caption("长期使用推荐，费用透明。")
            if st.button("用我自己的 Key", use_container_width=True):
                st.session_state["api_mode"] = "custom"
                st.rerun()

    st.stop()


# ============================================================
# custom 模式但没填 Key
# ============================================================

if st.session_state["api_mode"] == "custom":
    config = load_api_config()
    if not config.get("api_key") and not st.session_state.get("skip_key_check"):
        st.warning(
            "你选择了「使用自己的 API Key」，但还没有填写。"
            "请到左侧「⚙️ AI 设置」里填写你的智谱 API Key。"
        )
        st.markdown(
            "[👉 点此前往智谱AI申请 API Key](https://open.bigmodel.cn/apikey/platform)\n\n"
            "**说明：** 智谱新用户有免费额度，超出后按量付费。"
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 我已经申请好了，进入应用", use_container_width=True, type="primary"):
                st.session_state["_reenter_custom"] = True
                st.rerun()

        with col2:
            if st.button("🔄 返回选择使用方式", use_container_width=True):
                st.session_state["api_mode"] = None
                st.rerun()

        st.stop()


# ============================================================
# trial 模式检查次数
# ============================================================

if st.session_state["api_mode"] == "trial":
    used = get_trial_count()
    if used >= FREE_TRIAL_LIMIT:
        st.error("🎁 你的 10 次免费体验已经用完。")
        st.markdown(
            """
            要继续使用，请去智谱申请自己的 API Key：
            [👉 点此前往智谱AI申请](https://open.bigmodel.cn/apikey/platform)

            智谱新用户有免费额度，超出后按量付费。
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("✅ 我已经申请好了，进入应用", use_container_width=True, type="primary"):
                st.session_state["_reenter_custom"] = True
                st.rerun()

        with col2:
            if st.button("🔄 重新选择使用方式", use_container_width=True):
                st.session_state["api_mode"] = None
                st.rerun()

        st.stop()
    else:
        remaining = FREE_TRIAL_LIMIT - used
        st.info(f"🎁 体验模式：还剩 {remaining} 次免费体验。")


# ============================================================
# 文档加载
# ============================================================

def load_document(file_path, suffix):
    suffix = suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8").load()
    elif suffix == ".pptx":
        return UnstructuredPowerPointLoader(file_path).load()
    elif suffix == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    return []


@st.cache_resource
def build_retriever(file_info):
    docs = []
    for file_path, suffix in file_info:
        try:
            docs.extend(load_document(file_path, suffix))
        except Exception as e:
            st.warning(f"读取 {file_path} 出错：{e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    api_key = get_api_key()
    if not api_key:
        return None

    embeddings = ZhipuAIEmbeddings(api_key=api_key, model="embedding-2")

    BATCH_SIZE = 60
    db = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        if db is None:
            db = Chroma.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)

    return db.as_retriever(search_kwargs={"k": 3})


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown("### 📊 你的数据")

    saved_files = load_files()
    if saved_files:
        with st.expander(f"📁 已上传 {len(saved_files)} 个文件", expanded=False):
            for filename in saved_files:
                st.write(f"📄 {filename}")
    else:
        st.caption("还没有上传文件")

    if st.session_state.messages:
        with st.expander(f"💬 最近 {min(10, len(st.session_state.messages))} 条对话", expanded=True):
            for msg in st.session_state.messages[-10:]:
                role_label = "你" if msg["role"] == "user" else "AI"
                content = msg["content"]
                if len(content) > 40:
                    content = content[:40] + "..."
                st.caption(f"**{role_label}：** {content}")
    else:
        st.caption("还没有对话")

    st.markdown("---")
    st.markdown("### ⚙️ 设置")

    if st.button("🗑️ 清空对话", use_container_width=True):
        clear_messages()
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ 清空所有文件", use_container_width=True):
        clear_files()
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        st.warning(f"删除 {filename} 失败：{e}")
        build_retriever.clear()
        st.toast("✅ 文件已清除", icon="🎉")
        st.rerun()

    if st.session_state["api_mode"] == "custom":
        with st.expander("⚙️ AI 设置", expanded=True):
            api_config = load_api_config()

            api_key_input = st.text_input(
                "你的 API Key",
                value=api_config.get("api_key", ""),
                type="password"
            )
            model_input = st.text_input(
                "模型名",
                value=api_config.get("model", "") or "glm-3-turbo"
            )

            if st.button("保存 API 设置"):
                save_api_config("智谱", api_key_input, model_input)
                st.session_state["skip_key_check"] = False
                st.toast("✅ 设置已保存", icon="🎉")
                st.rerun()

    if st.button("🔄 切换使用方式"):
        st.session_state["api_mode"] = None
        st.rerun()

    with st.expander("ℹ️ 关于"):
        st.markdown("""
        **AI 学习陪跑顾问 v1.0**

        - 开源项目：[GitHub 链接]
        - 作者：[你的名字]
        - 反馈：[你的邮箱/微信]
        """)


# ============================================================
# 准备 Retriever
# ============================================================

file_info = []
saved_files = load_files()
if saved_files:
    for filename in saved_files:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(file_path):
            file_info.append((file_path, os.path.splitext(filename)[1]))

retriever = None
if file_info:
    with st.spinner("🧠 正在加载知识库..."):
        retriever = build_retriever(tuple(file_info))


# ============================================================
# 显示历史聊天
# ============================================================

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.write(msg["content"])


# ============================================================
# 示例问题
# ============================================================

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown("#### 💡 你可以这样问")
        st.markdown("- 我是零基础，该从哪开始学 AI？")
        st.markdown("- 我想做一个 RAG 项目，需要准备什么？")
        st.markdown("- 上传资料后，帮我总结重点。")


# ============================================================
# 上传文件
# ============================================================

with st.container(border=True):
    uploaded_files = st.file_uploader(
        "📎 上传学习资料（支持 PDF / TXT / PPTX / DOCX，可多选）",
        type=["pdf", "txt", "pptx", "docx"],
        accept_multiple_files=True,
        key="main_uploader"
    )


# ============================================================
# 提问栏
# ============================================================

user_question = st.chat_input("今天学了什么？卡在哪了？")


# ============================================================
# 处理用户输入
# ============================================================

if user_question or uploaded_files:

    if st.session_state["api_mode"] == "trial":
        if get_trial_count() >= FREE_TRIAL_LIMIT:
            st.error("🎁 体验次数已用完，请切换为「使用自己的 API Key」。")
            st.stop()

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            save_file(uploaded_file.name)
        build_retriever.clear()

    if not user_question and uploaded_files:
        user_question = """
请分析我刚刚上传的文件。

请告诉我：

1. 文件主要讲了什么
2. 核心知识点是什么
3. 对我来说哪些内容最重要
4. 我接下来应该学习什么
"""

    display_message = ""
    if uploaded_files:
        for uploaded_file in uploaded_files:
            display_message += f"📎 {uploaded_file.name}\n"
        display_message += "\n"
    if user_question:
        display_message += user_question

    if not display_message:
        st.stop()

    with st.chat_message("user", avatar="🧑"):
        st.markdown(display_message)

    st.session_state.messages.append({"role": "user", "content": display_message})
    save_message("user", display_message)

    file_info = []
    saved_files = load_files()
    if saved_files:
        for filename in saved_files:
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.exists(file_path):
                file_info.append((file_path, os.path.splitext(filename)[1]))

    retriever = None
    if file_info:
        with st.spinner("🧠 正在准备知识库..."):
            retriever = build_retriever(tuple(file_info))

    context = ""
    results = []
    if retriever:
        try:
            results = retriever.invoke(user_question)
            context = "\n\n".join([r.page_content for r in results])
        except Exception as e:
            st.warning(f"知识库检索失败：{e}")

    memories = load_memories(10)
    memory_text = "\n".join([f"- {m}" for m in memories]) if memories else "暂无"

    profile = load_profile()
    user_profile = f"""
用户信息：
- 名字/身份：{profile.get('name', '')}
- 当前学习状态：{profile.get('level', '')}
- 想学方向：{profile.get('direction', '')}
- 每周可投入时间：{profile.get('time_per_day', '')}
- 学习偏好：{profile.get('preference', '')}
"""

    if context:
        prompt = f"""你是一个 AI 学习陪跑顾问，主要帮助正在学习 AI、想做项目的学生或初学者。

回答要求：具体、可执行、不要泛泛而谈、尽量结合用户当前情况。

====================
用户档案
====================
{user_profile}

====================
用户记忆
====================
{memory_text}

====================
知识库资料
====================
{context}

====================
用户问题
====================
{user_question}

====================
回答规则
====================
1. 如果资料里有相关内容，优先严格根据资料回答。
2. 如果使用了资料，请在回答中说明参考了相关资料。
3. 如果资料里没有直接相关内容，不要编造资料中的事实。
4. 如果资料没有直接答案，明确说明："资料里没有直接相关内容，以下是我的建议："
5. 可以结合通用知识和用户档案给出学习建议。
"""
    else:
        prompt = f"""你是一个 AI 学习陪跑顾问，主要帮助正在学习 AI、想做项目的学生或初学者。

回答要求：具体、可执行、不要泛泛而谈、尽量结合用户当前情况。

====================
用户档案
====================
{user_profile}

====================
用户记忆
====================
{memory_text}

====================
用户问题
====================
{user_question}

用户目前没有上传资料，请基于通用知识和用户档案回答。
"""

    api_key = get_api_key()
    if not api_key:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning(
                "⚠️ 你还没有配置 API Key。\n\n"
                "请到左侧「⚙️ AI 设置」里粘贴你的智谱 API Key，点「保存 API 设置」后再提问。\n\n"
                "如果你还没有 Key，去这里申请："
                "[👉 点此前往申请](https://open.bigmodel.cn/apikey/platform)\n\n"
                "**说明：** 智谱新用户有免费额度，超出后按量付费。"
            )
        st.stop()

    chat_model = ChatZhipuAI(api_key=api_key, model=get_model_name())
    ai_answer, error = safe_invoke(chat_model, prompt)

    if error:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning(error)
        st.stop()

    if st.session_state["api_mode"] == "trial":
        increase_trial_count()

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_answer)

    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
    save_message("assistant", ai_answer)

    memory_prompt = f"""从下面这段对话里，提炼一条真正值得长期记住的用户信息。

要求：一句话、只记录对未来学习有帮助的信息、不要记录普通闲聊、如果没有值得记忆的信息，回复“无”。

用户：{user_question}

AI：{ai_answer}
"""
    memory_response, mem_error = safe_invoke(chat_model, memory_prompt)
    if not mem_error:
        memory = memory_response.strip()
        if memory and memory != "无":
            save_memory(memory)

    if results:
        with st.expander("📚 查看参考来源"):
            for i, r in enumerate(results):
                st.write(f"**片段 {i + 1}：**")
                st.caption(r.page_content[:300])