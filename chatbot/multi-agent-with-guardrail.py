import chainlit as cl
import dotenv
import os

dotenv.load_dotenv()

from openai.types.responses import ResponseTextDeltaEvent

from agents import Runner, SQLiteSession
from nutrition_agent import breakfast_advisor_agent, exa_search_mcp

@cl.on_chat_start
async def on_chat_start():
    session = SQLiteSession("historical_conversation")
    cl.user_session.set("for_each_agent_session", session)
    await exa_search_mcp.connect()

@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("for_each_agent_session")
    result = Runner.run_streamed(breakfast_advisor_agent, message.content, session=session)

    msg = cl.Message(content="")
    async for event in result.stream_events():
        # Stream final message text to screen
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)
            print(event.data.delta, end="", flush=True)

        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments
                print(
                    f"\nTool call: {
                        event.data.item.name} with args: {
                        event.data.item.arguments}"
                )

    await msg.update()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == (
        os.getenv("CHAINLIT_USERNAME"),
        os.getenv("CHAINLIT_PASSWORD"),
    ):
        return cl.User(
            identifier="Student",
            metadata={"role": "student", "provider": "credentials"},
        )
    else:
        return None