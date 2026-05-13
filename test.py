import asyncio
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from memory_agent import MemoryManager, MemoryConfig


# ─── 1. 定义 Agent 状态 ─────────────────────────────────

class AgentState(TypedDict):
    messages: List[dict]       # 当前对话历史
    session_id: str            # 会话标识
    user_input: str            # 用户最新输入
    memories: Optional[str]     # 检索到的相关记忆（注入到 prompt）
    response: Optional[str]     # Agent 回复


# ─── 2. 初始化 MemoryManager ───────────────────────────

config = MemoryConfig()
memory = MemoryManager(config)


# ─── 3. 定义 LangGraph 节点 ─────────────────────────────

async def recall_memories(state: AgentState) -> dict:
    """节点 1：检索相关记忆，注入上下文"""
    result = await memory.recall(
        query=state["user_input"],
        memory_type=None,           # 检索所有类型记忆
        session_id=state["session_id"],
        top_k=5,
    )

    # 将记忆拼接为上下文文本
    context_parts = []
    for item in result:
        context_parts.append(f"[{item.memory_type.value}] {item.content}")

    memories_text = "\n".join(context_parts) if context_parts else "(无相关历史记忆)"
    return {"memories": memories_text}


async def generate_response(state: AgentState) -> dict:
    """节点 2：结合记忆生成回复（这里用 DeepSeek 演示）"""
    system_prompt = f"""你是一个有记忆的 AI 助手。以下是与你之前对话相关的记忆：

{state["memories"]}

请根据这些记忆和当前对话上下文，自然地回复用户。如果用户提到之前的对话内容，你应该能关联起来。"""

    messages = [
        {"role": "system", "content": system_prompt},
        *state["messages"],
        {"role": "user", "content": state["user_input"]},
    ]

    # 使用 MemoryManager 内置的 DeepSeek 客户端
    response_text = await memory._llm_client.chat(
        messages=messages,
        temperature=0.7,
    )
    return {"response": response_text}


async def remember_this_turn(state: AgentState) -> dict:
    """节点 3：将本轮对话存入记忆"""
    user_msg = state["user_input"]
    assistant_msg = state.get("response", "")

    # 工作记忆：短期记住本轮对话要点
    await memory.remember(
        content=f"用户: {user_msg}",
        memory_type="working",
        session_id=state["session_id"],
    )

    # 情景记忆：存储完整交互
    await memory.remember(
        content=f"用户说: {user_msg}\n助手回复: {assistant_msg}",
        memory_type="episodic",
        session_id=state["session_id"],
    )

    return {}


# ─── 4. 构建 Graph ──────────────────────────────────────

def build_agent():
    builder = StateGraph(AgentState)

    builder.add_node("recall", recall_memories)
    builder.add_node("generate", generate_response)
    builder.add_node("remember", remember_this_turn)

    builder.set_entry_point("recall")
    builder.add_edge("recall", "generate")
    builder.add_edge("generate", "remember")
    builder.add_edge("remember", END)

    # 无需 checkpointer：ai-memory-hub 自身管理持久化记忆，LangGraph 只负责流程编排
    return builder.compile()


# ─── 5. 运行 ────────────────────────────────────────────

async def main():
    agent = build_agent()
    session_id = "user-session-001"

    print("=" * 60)
    print("  带记忆的 LangGraph 对话 Agent")
    print("  输入 'quit' 退出，输入 'consolidate' 触发记忆整合")
    print("=" * 60)

    messages_history = []

    while True:
        user_input = input("\n👤 你: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "consolidate":
            result = await memory.consolidate(session_id=session_id)
            print(f"\n📦 记忆整合完成：新增 {result.new_entities} 实体, "
                  f"{result.new_preferences} 偏好, {result.new_relations} 关系")
            continue

        initial_state: AgentState = {
            "messages": messages_history[-10:],  # 保留最近 10 轮
            "session_id": session_id,
            "user_input": user_input,
            "memories": None,
            "response": None,
        }

        result = await agent.ainvoke(initial_state)

        # 更新对话历史
        messages_history.append({"role": "user", "content": user_input})
        messages_history.append({"role": "assistant", "content": result["response"]})

        print(f"\n🤖 助手: {result['response']}")

        # 显示检索到的记忆
        if result["memories"] and result["memories"] != "(无相关历史记忆)":
            print(f"\n📝 关联记忆:\n{result['memories'][:200]}...")


if __name__ == "__main__":
    asyncio.run(main())