import pdb
import json
import asyncio
import operator

from enum import Enum
from datetime import date
from dataclasses import dataclass
from typing import TypedDict, Annotated, Literal
from urllib.parse import quote

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph

from llm_adapter import BaseLlmAdapter, BaseTemplater
from mix import load_pubmed_page, load_researches


# default_model_name = 'anthropic.claude-v3-5-sonnet'
default_model_name = 'gpt-4o-2024-08-06'  # TODO: implement and use BaseModels

MAX_HISTORY_LENGTH = 10000


@dataclass
class Stage:
    title: str
    model_name: str = None

    def model(self, sql_agent):
        if self.model_name == 'default_model':
            model_name = default_model_name
        else:
            model_name = self.model_name or sql_agent.model_name
        return BaseLlmAdapter(model=model_name, temperature=sql_agent.llm_temperature)


class STAGES:
    search = Stage('Search Articles', model_name='default_model')
    enrich = Stage('Enrich Articles', model_name='default_model')
    final_answer = Stage('Final Answer', model_name='default_model')


class AgentState(TypedDict):  # TODO: check if it can be Pydantic class
    user_request: str
    previous_requests: str
    latest_search_result: str
    articles: list
    final_answer: str
    messages: Annotated[list, operator.add]


class MedAgent:
    def __init__(self, max_attempts: int, today: date = None):
        self.max_attempts = max_attempts
        self.attempts_left = max_attempts
        self.llm_temperature = 0
        self.today = today or date.today()

        graph = StateGraph(AgentState)

        graph.set_entry_point("search")
        graph.add_node("search", self._search_articles)
        graph.add_edge(start_key="search", end_key="enrich")
        graph.add_node("enrich", self._enrich_articles)
        graph.add_edge(start_key="enrich", end_key="final")

        # # Feedback loop
        # graph.add_node("validate", self._validation)
        # graph.add_conditional_edges("validate", self._should_continue) # Go to "search" node or to "final" end node.

        graph.add_node("final", self._final_answer)
        graph.set_finish_point("final")

        self.workflow = graph.compile()

    async def _search_articles(self, state: AgentState):
        current_stage = STAGES.search
        print(current_stage)

        prompt = BaseTemplater(file='prompts/pubmed/1_search_url_generator.prompt').format(
            safe_vars={"user_request": state['user_request'], "previous_requests": state['previous_requests']},
            results_count=50,
        )
        llm_answer = await current_stage.model(self).ainvoke(prompt, json_parse=True)
        search_url = llm_answer['url']
        search_result = load_pubmed_page(url=search_url)

        return {
            "latest_search_result": search_result,
            "messages": [AIMessage(
                content=f"#### Search URL:\n{search_url}\n\n#### Details:\n```json\n{json.dumps(llm_answer, indent=4)}\n```",
                additional_kwargs={'stage': current_stage}
            )]
        }

    async def _enrich_articles(self, state: AgentState):
        current_stage = STAGES.enrich
        print(current_stage)

        prompt = BaseTemplater(file='prompts/pubmed/2_search_process.prompt').format(
            safe_vars={"user_request": state['user_request'], "search_results": state['latest_search_result']}
        )
        llm_answer = await current_stage.model(self).ainvoke(prompt, json_parse=True)
        articles = load_researches(search_llm_result=llm_answer)

        return {
            "articles": state.get('articles', []) + articles,
            "messages": [AIMessage(
                content=f"#### Articles:\n```json\n{json.dumps(articles, indent=4)}\n```",
                additional_kwargs={'stage': current_stage}
            )]
        }

    # def _should_continue(self, state: AgentState) -> Literal["sql_generator", "final_answer"]:
    #     if self.attempts_left <= 0:
    #         print(f"Max cycles left ({self.attempts_left}/{self.max_attempts})")
    #         return 'final_answer'
    #
    #     if 'invalid' in state.get('self_validation', ''):
    #         print("I'm going to re-generate SQL")
    #         if self.target_mode == Mode.combined and self.current_mode == Mode.precise:
    #             self.current_mode = Mode.combined
    #         return 'sql_generator'
    #     else:
    #         print("I know the final answer")
    #         return 'final_answer'

    async def _final_answer(self, state: AgentState):
        current_stage = STAGES.final_answer
        print(current_stage)

        prompt = BaseTemplater(file='prompts/pubmed/3_researches_summary.prompt').format(
            safe_vars={"user_request": state['user_request']},
            researches=state['articles']
        )
        final_llm_answer = await current_stage.model(self).ainvoke(prompt)

        return {
            "final_answer": final_llm_answer,
            "messages": [AIMessage(content='', additional_kwargs={'stage': current_stage})]
        }


def format_executed_steps(state):
    return "\n".join([
        f"{message.additional_kwargs.get('stage').title}:\n{message.content}" for message in state['messages']
    ])


def latest_stage(state: AgentState):
    if state['messages']:
        last_message = state['messages'][-1]
        if last_message:
            content = last_message.content
            stage_name = last_message.additional_kwargs.get('stage', None).title
            return {'name': stage_name, 'content': content}
    return {}


# TODO: Ideally, should be moved to a separate file, as it's going to be used in other agents
def format_user_request(messages: dict):
    previous_requests = messages.get('previous_requests', [])
    user_request = messages['user_request']
    if hasattr(user_request, 'content'):
        user_request = user_request.content

    def question_answer_format(human: HumanMessage, ai: AIMessage):
        if hasattr(human, 'content'):
            human = human.content
        if hasattr(ai, 'content'):
            ai = ai.content
        return f"Question:\n{human}\nAnswer:\n{ai}"

    def get_last_words(text, words):
        words_list = text.split(' ')
        last_words = words_list[-words:]
        return ' '.join(last_words)

    human_requests = previous_requests[::2]
    ai_answers = previous_requests[1::2]
    history_string = "\n\n".join(
        question_answer_format(human, ai) for human, ai in zip(human_requests, ai_answers)
    )
    history_string = get_last_words(history_string, words=MAX_HISTORY_LENGTH)

    return {'user_request': user_request, 'previous_requests': history_string}


async def _run(user_request: str, messages_history: list[str]):
    inputs = format_user_request(
        messages={"user_request": user_request, 'previous_requests': messages_history}
    )
    agent = MedAgent(max_attempts=3)

    stage = {}
    async for agent_state in agent.workflow.astream(inputs, stream_mode="values"):
        stage = latest_stage(agent_state)
        if stage.get('name'):
            print(f"Stage: {stage['name']}")
            print(f"Message: {stage['content'][:50]}")

    if stage.get('final_answer'):
        print(f"Final result: {stage['final_answer']}")


if __name__ == '__main__':

    asyncio.run(_run(
        messages_history=[""],
        user_request="Find the most promising research in US in 2024 on diabetes prevention."
    ))
