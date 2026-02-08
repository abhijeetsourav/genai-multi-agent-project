import asyncio
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from backend.agents import build_agents


async def run_review_analysis(user_query: str, api_key: str):

    retriever, analyzer, categorizer, decision_maker = build_agents(api_key)

    team = RoundRobinGroupChat(
        participants=[retriever, analyzer, categorizer, decision_maker],
        termination_condition=MaxMessageTermination(max_messages=6),
    )

    result = await team.run(
        task=f"""
Retrieve relevant customer reviews.
Analyze them.
Categorize issues.
Produce ONE final business summary in JSON.

User query:
{user_query}
"""
    )

    # Extract final JSON only
    for msg in reversed(result.messages):
        if msg.source == "DecisionMaker":
            return msg.content

    raise RuntimeError("DecisionMaker output not found")
