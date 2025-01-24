import asyncio
import os
import random

import uvicorn

from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from .med_agent import MedAgent, format_user_request, latest_stage

MAX_AGENT_ATTEMPTS = int(os.environ.get('MAX_AGENT_ATTEMPTS', 3))


async def periodic_task(choice, stop_event, interval):
    """Periodic task to be executed every interval seconds."""
    while not stop_event.is_set():
        choice.append_content(' ')
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue  # Continue if the timeout is reached


class EchoApplication(ChatCompletion):
    async def chat_completion(
        self, request: Request, response: Response
    ) -> None:
        temperature = request.temperature
        user_request = request.messages[-1]
        messages_history = request.messages[:-1]

        with response.create_single_choice() as choice:
            # search_url = generate_search_url(user_request=last_user_message.content)
            # choice.append_content(search_url)
            inputs = format_user_request(
                messages={"user_request": user_request, 'previous_requests': messages_history}
            )
            agent = MedAgent(max_attempts=MAX_AGENT_ATTEMPTS)

            # choice.append_content('Processing your request')
            current_step = choice.create_stage('')  # Show the first empty stage with a spinner
            current_step.open()

            # Start the periodic task
            stop_event = asyncio.Event()  # event to signal when to stop the periodic task
            periodic = asyncio.create_task(periodic_task(choice, stop_event, interval=10))

            # Main flow (LangGraph agent)
            async for agent_state in agent.workflow.astream(inputs, stream_mode="values"):
                stage = latest_stage(agent_state)
                if not stage.get('name'):
                    continue

                current_step.append_name(stage['name'])
                current_step.append_content(stage['content'])
                current_step.close()
                current_step = choice.create_stage('')
                current_step.open()
            current_step.close()

            # # Fake streaming (just fancy, no real profit)
            choice.append_content("\n\n")
            for word in agent_state['final_answer'].split(' '):
                choice.append_content(f"{word} ")
                await asyncio.sleep(random.uniform(0.005, 0.05))


            # Wait for the periodic task to finish
            stop_event.set()
            await periodic

app = DIALApp()
app.add_chat_completion("ask", EchoApplication())


# if __name__ == "__main__":
#     uvicorn.run(app, port=8001, host="0.0.0.0", reload=True)
