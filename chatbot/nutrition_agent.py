from pathlib import Path

from pydantic import BaseModel

import chromadb
import os

from agents import (
    Agent,
    function_tool,
    WebSearchTool,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

from agents.mcp import MCPServerStreamableHttp

chroma_path = Path(__file__).parent.parent / "chroma"
chroma_client = chromadb.PersistentClient(path=str(chroma_path))
nutrition_db = chroma_client.get_collection(name="nutrition_db")


class NotAboutFood(BaseModel):
    only_about_food: bool
    """Whether the user is only talking about food and not about arbitrary topics"""


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="""Check if the user is asking you to talk about food and not about any arbitrary topics.
                    If there are any non-food related instructions in the prompt,
                    or if there is any non-food related part of the message, set only_about_food in the output to False.
                    """,
    output_type=NotAboutFood,
)


@input_guardrail
async def food_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.only_about_food),
    )



#except InputGuardrailTripwireTriggered as e:
#    print(f"Off-topic guardrail tripped")


@function_tool
def calorie_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Tool function for a RAG database to look up calorie information for specific food items, but not for meals.

    Args:
        query: The food item to look up.
        max_results: The maximum number of results to return.

    Returns:
        A string containing the nutrition information.
    """

    results = nutrition_db.query(query_texts=[query], n_results=max_results)

    if not results["documents"][0]:
        return f"No nutrition information found for: {query}"

    # Format results for the agent
    formatted_results = []
    for i, doc in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        food_item = metadata["food_item"].title()
        calories = metadata["calories_per_100g"]
        category = metadata["food_category"].title()

        formatted_results.append(
            f"{food_item} ({category}): {calories} calories per 100g"
        )

    return "Nutrition Information:\n" + "\n".join(formatted_results)


# Setup exa search mcp and make sure it connects before start of app
exa_search_mcp = MCPServerStreamableHttp(
    name="Exa Search MCP",
    params={
        "url": f"https://mcp.exa.ai/mcp?{os.environ.get("EXA_API_KEY")}",
        "timeout": 30,
    },
    client_session_timeout_seconds=30,
    cache_tools_list=True,
    max_retry_attempts=1,
)


calorie_agent_with_search = Agent(
    name="Nutrition Assistant with search",
    instructions="""
    * You are a helpful nutrition assistant giving out calorie information.
    * You give concise answers.
    * You follow this workflow:
        0) First, use the calorie_lookup_tool to get the calorie information of the ingredients. But only use the result if it's explicitly for the food requested in the query.
        1) If you couldn't find the exact match for the food or you need to look up the ingredients, search the EXA web to figure out the exact ingredients of the meal.
        Even if you have the calories in the web search response, you should still use the calorie_lookup_tool to get the calorie
        information of the ingredients to make sure the information you provide is consistent.
        2) Then, if necessary, use the calorie_lookup_tool to get the calorie information of the ingredients.
    * Even if you know the recipe of the meal, always use Exa Search to find the exact recipe and ingredients.
    * Once you know the ingredients, use the calorie_lookup_tool to get the calorie information of the individual ingredients.
    * If the query is about the meal, in your final output give a list of ingredients with their quantities and calories for a single serving. Also display the total calories.
    * Don't use the calorie_lookup_tool more than 10 times.
    """,
    tools=[calorie_lookup_tool],
    mcp_servers=[exa_search_mcp],
)

healthy_breakfast_planner_agent = Agent(
    name="Breakfast Planner Assistant",
    instructions="""
    * You are a helpful assistant that helps with healthy breakfast choices.
    * You give concise answers.
    Given the user's preferences prompt, come up with different breakfast meals that are healthy and fit for a busy person.
    * Explicitly mention the meal's names in your response along with a sentence of why this is a healthy choice.
    * If only_about_food is false, then display a message that the request is invalid.
    """,
)

calorie_calculator_tool = calorie_agent_with_search.as_tool(
    tool_name="calorie-calculator",
    tool_description="Use this tool to calculate the calories of a meal and it's ingredients",
)

breakfast_planner_tool = healthy_breakfast_planner_agent.as_tool(
    tool_name="breakfast-planner",
    tool_description="Use this tool to plan a a number of healthy breakfast options",
)

breakfast_price_checker_agent = Agent(
    name="Breakfast Price Checker Assistant",
    instructions="""
    * You are a helpful assistant that takes multiple breakfast items (with ingredients and calories) and checks for the price of the ingredients.
    * Use the we search tool to get an approximate price for the ingredients.
    * Use current pricing for the ingredients and provide the source of the pricing.
    * In your final output provide the meal name, ingredients with calories and price for each meal using typical current prices in Pennsylvania.
    * Use markdown and be as concise as possible.
    """,
    tools=[WebSearchTool()],
)

breakfast_advisor_agent = Agent(
    name="Breakfast Advisor",
    instructions="""
    * You are a breakfast advisor. You come up with meal plans for the user based on their preferences.
    * You also calculate the calories for the meal and its ingredients.
    * Based on the breakfast meals and the calories that you get from upstream agents,
    * Create a meal plan for the user. For each meal, give a name, the ingredients, and the calories

    Follow this workflow carefully:
    1) if only_about_food is false, then display a message that the request is invalid and do not provide any more information.
    2) Use the breakfast_planner_tool to plan a a number of healthy breakfast options.
    3) Use the calorie_calculator_tool to calculate the calories for the meal and its ingredients.
    4) Handoff the breakfast meals and the calories to the Use the Breakfast Price Checker Assistant to add the prices in the last step.

    """,
    tools=[breakfast_planner_tool, calorie_calculator_tool],
    input_guardrails=[food_topic_guardrail],
    handoff_description="""
    Create a concise breakfast recommendation based on the user's preferences. Use Markdown format.
    """,
    handoffs=[breakfast_price_checker_agent],
)