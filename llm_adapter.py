import os
import pdb
import json

from urllib.parse import urlparse
from typing import Dict, Any

from tenacity import retry, stop_after_attempt
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="tmp/.langchain.tmp.db"))


class BaseLlmAdapter:
    DEFAULT_TEMPERATURE = 0

    def __init__(self, model: str, temperature: float = DEFAULT_TEMPERATURE):
        self.model = model
        self.temperature = temperature

    @retry(reraise=True, stop=stop_after_attempt(3))
    async def ainvoke(self, prompt: str, json_parse: bool = False):
        result = await self.llm().ainvoke(prompt)
        if json_parse:
            return JsonOutputParser().invoke(result)
        else:
            return result.content

    @retry(reraise=True, stop=stop_after_attempt(3))
    def invoke(self, prompt: str, json_parse: bool = False):
        result = self.llm().invoke(prompt)
        if json_parse:
            return JsonOutputParser().invoke(result)
        else:
            return result.content

    def llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            api_key=os.environ.get('OPENAI_ACCESS_TOKEN') or os.environ.get('AZURE_OPENAI_API_KEY'),
            azure_endpoint=self.endpoint(),
            openai_api_version="2023-03-15-preview",
            azure_deployment=self.model,
            temperature=self.temperature,
        )

    @staticmethod
    def endpoint() -> str:
        base_url = os.environ.get('OPENAI_URI_BASE')
        if base_url:
            parsed_url = urlparse(base_url)
            return f"{parsed_url.scheme}://{parsed_url.netloc}"
        else:
            return os.environ.get('AZURE_OPENAI_ENDPOINT')


class BaseTemplater:
    def __init__(self, file):
        self.file = file

    def format(self, safe_vars: dict = {}, **variables):
        prompt = PromptTemplate.from_file(
            template_file=self.file,
            template_format="jinja2"
        ).format(**variables)
        for name, value in safe_vars.items():
            prompt = prompt.replace(f"[{name}]", value)

        return prompt
