import chainlit as cl
from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
)
from my_secrets import Secrets
from typing import cast
import json
from openai.types.responses import ResponseTextDeltaEvent
import requests
from rich import print

secrets = Secrets()


@function_tool("current_weather_tool")
@cl.step(type="weather tool")
async def get_current_weather(location: str) -> str:
    """
    Retrieves current weather information for a specified location.

    This function makes an asynchronous API call to fetch real-time weather data
    including temperature, weather conditions, wind speed, humidity, and UV index
    for the given location.

    Args:
        location (str): The location for which to fetch weather data. Can be a city name,
                       coordinates, or other location identifier supported by the weather API.

    Returns:
        str: A formatted string containing comprehensive weather information including:
             - Location details (name, region, country)
             - Current date and time
             - Temperature in Celsius and "feels like" temperature
             - Weather condition description
             - Wind speed (km/h) and direction
             - Humidity percentage
             - UV index

             If the API request fails, returns an error message indicating the failure.

    Raises:
        This function handles HTTP errors internally and returns error messages as strings
        rather than raising exceptions.

    Example:
        >>> weather = await get_current_weather("London")
        >>> print(weather)
        Current weather in London, England, United Kingdom as of 2023-10-15 14:30 is 18°C (Partly cloudy), feels like 17°C, wind 15 km/h SW, humidity 65% and UV index is 4.
    """
    result = requests.get(
        f"{secrets.weather_api_url}//current.json?key={secrets.weather_api_key}&q={location}"
    )
    if result.status_code != 200:
        return f"Error fetching weather data for {location}. Please try again later."
    data = result.json()
    return f"Current weather in {data['location']['name']}, {data['location']['region']}, {data['location']['country']} as of {data['location']['localtime']} is {data['current']['temp_c']}°C ({data['current']['condition']['text']}), feels like {data['current']['feelslike_c']}°C, wind {data['current']['wind_kph']} km/h {data['current']['wind_dir']}, humidity {data['current']['humidity']}% and UV index is {data['current']['uv']}."


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


@cl.set_starters
async def starters():
    return [
        cl.Starter(
            label="Get Current Weather",
            message="Fetch the current weather for a specified location.",
            icon="/public/weather.svg",
        ),
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
            message="Generate an 1000 words essay on a given topic.",
            icon="/public/article.svg",
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
        instructions="You are an expert essay writer. You can write 1000 word essays on various topics.",
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=secrets.gemini_api_model,
        ),
    )

    agent = Agent(
        name="Chatbot",
        instructions=""""
        You are a friendly and informative assistant. You can answer general questions and provide specific information.
        * For **weather inquiries**, you may fetch and share the current weather.
        * For **student-related queries**, you can retrieve details using the student ID.
        * For **essay writing**, you can retrieve an essay on a given topic.
        * Use tools **only when necessary**, not by default.
        * If a question falls outside essay writing, weather or student information, provide a helpful general response or ask for clarification.
        * If you're unsure of the answer, say "I don't know" or ask for more details.
        """,
        model=OpenAIChatCompletionsModel(
            openai_client=external_client,
            model=secrets.gemini_api_model,
        ),
        tools=[
            get_current_weather,
            get_student_info,
            essay_agent.as_tool(
                tool_name="essay_writer_tool",
                tool_description="Write a 1000 word essay on a given topic.",
            )
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


@cl.on_chat_end
def end():
    chat_history: list = cl.user_session.get("chat_history") or []
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
