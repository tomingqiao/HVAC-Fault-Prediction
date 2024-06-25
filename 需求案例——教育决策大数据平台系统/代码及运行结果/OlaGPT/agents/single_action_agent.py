# -*- coding: utf-8 -*-
import os
import re
from typing import List, Union, Dict, Tuple, Any, Optional
from langchain.agents import Tool, AgentExecutor, AgentOutputParser, BaseSingleActionAgent
from langchain.tools.base import BaseTool
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, GoogleSearchAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish

# import custom module
import sys
sys.path.append('.')
sys.path.append('..')
from utils.configs import configs
from utils.parser import get_arguments
from agents.custom_base_agent import LLMMultiActionAgent
from agent_template.AnalogyThought.tool import AnalogyThought
from agent_template.CombineSearchJudge.tool import CombineTool
from agent_template.DecompositionThought.tool import DecompositionThought
from agent_template.DIY.tool import DIYTool
from agent_template.Origin.tool import CustomOriginTool
from agent_template.PlanThought.tool import PlanThought
# from agent_template.Reflect.tool import ReactTool, ReactReflectTool
from agent_template.StepThought.tool import StepThought
from agent_template.ValidationThought.tool import ValidationThought
from agent_template.DisassembleThought.tool import DisassembleThought
from utils.evaluation import evaluation

args = get_arguments()
# 修改成使用自己的大模型
os.environ["OPENAI_API_BASE"] = "http://localhost:13000/v1"
os.environ["OPENAI_API_KEY"] = "sk-96GeVW4uPRlNGwO56eC90d799a664d79Ba624dFfC04a75Bc"

# 问题定位点：intermediate_steps为什么是空的=》导致tools_answer是空的
class SubAgent(BaseSingleActionAgent):
    """Master agent that controls all sub agents"""
    allowed_tools: Optional[List[str]] = None

    @property
    def input_keys(self):
        return ["input"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools

    # -> 一般出现在python函数定义的函数名后面，为函数添加元数据,描述函数的返回类型，也可以理解为给函数添加注解。
    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        print("***调用plan函数")
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        question = kwargs['input']
        if len(intermediate_steps) == 0:
            allowed_tools = self.get_allowed_tools()
            aac = AgentAction(tool=allowed_tools[0], tool_input=question, log="")
            return aac
        else:
            # get tool answer as final answer
            tools_answer = ""
            for action, observation in intermediate_steps:
                tools_answer += f'{action.tool}: {observation}\n'
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": tools_answer},
                log="",
            )

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        print("***调用aplan函数")
        question = kwargs['input']
        if len(intermediate_steps) == 0:
            allowed_tools = self.get_allowed_tools()
            return AgentAction(tool=allowed_tools[0], tool_input=question, log="")
        else:
            # get tool answer as final answer
            tools_answer = ""
            for action, observation in intermediate_steps:
                tools_answer += f'{action.tool}: {observation}\n'
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": tools_answer},
                log="",
            )
        
class EasyAgent(BaseSingleActionAgent):
    allowed_tools: Optional[List[str]] = None
    @property
    def input_keys(self):
        return ["input"]

    def get_allowed_tools(self) -> Optional[List[str]]:
        return self.allowed_tools
    def plan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": 'A'},
                log="",
            )
            

    async def aplan(
            self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": 'A'},
                log="",
            )

if __name__ == '__main__':
    # Define custom LLM
    model_name = configs['model_name']
    # llm = OpenAI(model_name=model_name, temperature=0)
    # 修改成使用自己的大模型
    llm = ChatOpenAI(
        openai_api_key="sk-96GeVW4uPRlNGwO56eC90d799a664d79Ba624dFfC04a75Bc",
        openai_api_base="http://localhost:3000/v1",
        model_name="SparkDesk",
        temperature=0
        )

    # Define which tools the agent can use to answer user queries
    name_dict = {
        'diy': DIYTool(dataset=args.dataset, few_shot=args.few_shot),
        'turbo': CustomOriginTool(dataset=args.dataset, few_shot=args.few_shot),
        'at': AnalogyThought(dataset=args.dataset, few_shot=args.few_shot),
        'dt': DecompositionThought(dataset=args.dataset, few_shot=args.few_shot),
        'pt': PlanThought(dataset=args.dataset, few_shot=args.few_shot),
        'st': StepThought(dataset=args.dataset, few_shot=args.few_shot),
        'vt': ValidationThought(dataset=args.dataset, few_shot=args.few_shot),
        'dst': DisassembleThought(dataset=args.dataset, few_shot=args.few_shot)
    }
    tools = [name_dict[args.model_name]]
    tool_names = [tool.name for tool in tools]
      

    # Define custom LLMMultiActionAgent
    agent = SubAgent(
        allowed_tools=tool_names
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
                        agent=agent, tools=tools, verbose=True, return_intermediate_steps=True
                    )


    # get args
    question = args.question
    is_eval = args.is_eval

    if is_eval:
        # correct_num、incorrect_num,accuracy,correct,incorrect
        result = evaluation(agent_executor, llm, args)
        for k, v in result.items():
            print(f'{k}: {v}')
    else:
        ans = agent_executor.run(question)
        print(ans)
