from app.chatbot.state import GraphState, IntentOutput
from app.chatbot.agents import get_llm
from app.chatbot.prompts import INTENT_PROMPT

def detect_intent_node(state: GraphState) -> dict:
    """Classify user intent using full conversation history for context."""
    messages = state["messages"]

    prompt = INTENT_PROMPT
    llm = get_llm().with_structured_output(
        IntentOutput,
        method="function_calling"
    )
    chain = prompt | llm
    intent_output = chain.invoke({"history": messages})
    
    return {
        "intent_output": intent_output
    }