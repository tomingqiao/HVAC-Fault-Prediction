# -*- coding: utf-8 -*-
from langchain.tools.base import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from utils.configs import configs
from utils.fewshots import get_notes_few_shot
from utils.load_query import load_query
from utils.prompts import get_query_format
import openai


class StepThought(BaseTool):
    """Tool that adds the origin api."""
    name = "StepThought"
    description = (
        "适合需要一步一步完成的问题。"
    )
    dataset = 'aqua'
    few_shot = 0

    # def _run(self, query: str) -> str:
    #     """Use the tool."""

    #     query_json = load_query(query)
    #     query_json, final_query = get_query_format(self.dataset, query_json)
    #     step_thought = """Let's think step by step."""
    #     llm = OpenAI(temperature=0, model_name=configs['model_name'])
        
    #     if self.few_shot:
    #         templates_prefix = step_thought
    #         notes_few_shot = get_notes_few_shot(
    #             query_json, self.dataset, self.few_shot, templates_prefix)
    #         return llm(notes_few_shot + final_query + step_thought)
    #     return llm(final_query + step_thought)
    
    # 重写_run 原来这个llm用不了报错
    def _run(self, query: str) -> str:
        """Use the tool."""

        query_json = load_query(query)
        #生成模版
        query_json, final_query = get_query_format(self.dataset, query_json)
        step_thought = """让我们一步一步思考"""
    
        # 配置openai
        openai.api_base = "https://gtapi.xiaoerchaoren.com:8932/v1"
        openai.api_key = "sk-axvPUHxKJKxvmWvJC75e8d0e4c6c4eA3B70d97F438215830"
        
        if self.few_shot:
            templates_prefix = step_thought
            notes_few_shot = get_notes_few_shot(
                query_json, self.dataset, self.few_shot, templates_prefix)
            response = openai.ChatCompletion.create(
                model="SparkDesk",
                messages=[
                    {"role": "user", "content": notes_few_shot + final_query + step_thought}
                ],
                stream=False,
                stop=[] # 在此处添加自定义的stop words 例如ReAct prompting时需要增加： stop=["Observation:"]。
            )
            res = response.choices[0].message.content
            return res
        
        # 不使用流式回复的请求
        response = openai.ChatCompletion.create(
            model="SparkDesk",
            messages=[
                {"role": "user", "content": final_query + step_thought}
            ],
            stream=False,
            stop=[] # 在此处添加自定义的stop words 例如ReAct prompting时需要增加： stop=["Observation:"]。
        )
        res = response.choices[0].message.content
        return res

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("StepThought does not support async")
