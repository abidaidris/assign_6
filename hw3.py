import json
from dataclasses import dataclass
from typing import cast

import chainlit as cl
import requests
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai.types.responses import ResponseTextDeltaEvent
from rich import print

from my_secrets import Secrets

secrets = Secrets()


@function_tool("student_info_tool")
@cl.step(type="student tool")
async def get_student_info(student_id: int) -> str:
    """
    Get information about a student by their ID.
    """
    students = {
        1: {"name": "John Doe", "age": 20, "major": "Computer Science"},
        2: {"name": "Jane Smith", "age": 22, "major": "Mathematics"},
        3: {"name": "Alice Johnson", "age": 21, "major": "Physics"},
        4: {"name": "Bob Brown", "age": 23, "major": "Chemistry"},
    }
    # Simulate fetching student data
    student_info = students.get(student_id)
    if student_info:
        return f"Student ID: {student_id}, Name: {student_info['name']}, Age: {student_info['age']}, Major: {student_info['major']}."
    else:
        return f"No student found with ID {student_id}."


@function_tool("prtx_tool")
@cl.step(type="practice tool")
async def get_practice_details() -> str:
    """
    Returns a message that this block is reserved for further additions and formats the author details from the context wrapper.

    Args:

    Returns:
        str: A formatted string containing the message for under development block.

    Example:
        "Sorry The developer is about to code this block"
    """
    thinking_msg = cl.Message(content="Comming Soon...")
    await thinking_msg.send()
    return f"The developer is going to implement it soon"


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="Get Student Info",
            message="Retrieve information about a student using their ID.",
            icon="/public/student.svg",
        ),
        cl.Starter(
            label="Explore General Questions",
            message="Find answers to the given questions.",
            icon="/public/question.svg",
        ),
        cl.Starter(
            label="Write an Essay",
            message="Generate 1000 words essay on a given topic.",
            icon="/public/article.svg",
        ),
        cl.Starter(
            label="practice",
            message="try....dashboard ",
            icon="/public/weather.svg",
        ),
    ]


@cl.on_chat_start
async def start():

    external_client = AsyncOpenAI(
        base_url=secrets.gemini_api_url,
        api_key=secrets.gemini_api_key,
    )
    set_tracing_disabled(True)

    essay_agent = Agent(
        name="Essay Writer",
        instructions="You are an expert essay writer. You can write 1000 word essays on various topics. even if only title is mentioned ",
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=secrets.gemini_api_model,
        ),
    )

    agent = Agent(
        name="Chatbot",
        instructions=""""
        You are a friendly and informative assistant. You can answer general questions and provide specific information.
        * For **student-related queries**, you can retrieve details using the student ID.
        * For **code updation and addition practice ** this code is available, .
        * For **essay writing**, you can retrieve an essay on a given topic by only mentioning topic.

        * Use tools **only when necessary**, not by default.
        * If a question falls outside essay writing, weather or student information, provide a helpful general response or ask for clarification.
        * If you're unsure of the answer, say "I don't know" or ask for more details.
        """,
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=secrets.gemini_api_model,
        ),
        tools=[
            get_student_info,
            get_practice_details,
            essay_agent.as_tool(
                tool_name="essay_writer_tool",
                tool_description="Write a 1000 word essay on any  given topic.",
            ),
        ],
    )

    cl.user_session.set("agent", agent)
    cl.user_session.set("chat_history", [])


@cl.on_message
async def main(message: cl.Message):
    
    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    agent = cast(Agent, cl.user_session.get("agent"))
    chat_history: list = cl.user_session.get("chat_history") or []
    chat_history.append(
        {
            "role": "user",
            "content": message.content,
        }
    )

    try:
        result = Runner.run_streamed(
            starting_agent=agent,
            input=chat_history,
        )

        response_message = cl.Message(
            content="",
        )
        first_response = True
        async for chunk in result.stream_events():
            if chunk.type == "raw_response_event" and isinstance(
                chunk.data, ResponseTextDeltaEvent
            ):
                if first_response:
                    await thinking_msg.remove()
                    await response_message.send()
                    first_response = False
                await response_message.stream_token(chunk.data.delta)

        chat_history.append(
            {
                "role": "assistant",
                "content": response_message.content,
            }
        )
        cl.user_session.set("chat_history", chat_history)
        await response_message.update()
    except Exception as e:
        response_message.content = (
            "An error occurred while processing your request. Please try again."
        )
        await response_message.update()
