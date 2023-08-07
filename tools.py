from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from bash_tool import BashTool
from custom_classes import CustomPromptTemplate

# Initialize tools
bash_tool = BashTool()

tools = [
    Tool(
        name="Bash",
        func=bash_tool.run_command,
        description="Execute bash commands"
    )
]