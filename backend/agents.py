from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from backend.vectorstore import retrieve_reviews


def build_agents(api_key: str):

    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    retriever = AssistantAgent(
        name="Retriever",
        model_client=model_client,
        tools=[retrieve_reviews],
        system_message="""
You retrieve customer reviews using the retrieve_reviews tool.
Return only raw review text. No analysis.
""",
    )

    analyzer = AssistantAgent(
        name="Analyzer",
        model_client=model_client,
        system_message="""
You analyze retrieved reviews.
Identify themes, pain points, and sentiment drivers.
No final decisions.
""",
    )

    categorizer = AssistantAgent(
        name="Categorizer",
        model_client=model_client,
        system_message="""
You categorize issues into:
- Performance
- UX
- Monetization
- Gameplay
- Stability
""",
    )

    decision_maker = AssistantAgent(
        name="DecisionMaker",
        model_client=model_client,
        system_message="""
You are the final decision authority.

Return EXACTLY one JSON object.
NO markdown. NO explanations.

Schema:
{
  "meta": {
    "topic": "",
    "analysis_scope": "",
    "confidence_level": "",
    "evidence_count": 0
  },
  "insight": {
    "primary_issue": "",
    "root_causes": []
  },
  "evidence": [
    {"text": "", "score": 0.0}
  ],
  "impact": [],
  "actions": []
}
""",
    )

    return retriever, analyzer, categorizer, decision_maker
