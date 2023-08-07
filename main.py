from langchain import LLMChain, PromptTemplate
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from config import MAX_META_ITERS, TEMPERATURE, MODEL_NAME, TIMEOUT, STREAMING, CSV_FILE_PATH, CSV_COLUMNS, CONSTRAINTS, TIPS
from tools import tools
from callback_manager import CallbackManager
from custom_classes import CustomPromptTemplate, CustomOutputParser
from embedding_utils import generate_embedding
from parsing import get_new_instructions, get_relevant_actions
from prompts import get_init_prompt_template, get_meta_prompt_template, get_evaluation_prompt_template, get_extracted_actions_prompt_template
import os
import pandas as pd
import argparse

# Initialize parser and callback manager
output_parser = CustomOutputParser()
cb = CallbackManager()

# Initialize language learning model
agent_llm = ChatOpenAI(
    temperature=TEMPERATURE, 
    model_name=MODEL_NAME,
    request_timeout=TIMEOUT,
    streaming=STREAMING,
)

def initialize_agent(david_instantiation_prompt: str):
    """Initializes agent with provided prompt.

    Args:
        david_instantiation_prompt: The prompt for initializing the agent.

    Returns:
        Agent executor object.
    """
    prompt = CustomPromptTemplate(
        template=david_instantiation_prompt,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["input", "constraints", "tips", "intermediate_steps"]
    )
    # LLM chain consisting of the LLM and a prompt
    llm_chain = LLMChain(llm=agent_llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain, 
        output_parser=output_parser,
        stop=["\nObservation:"], 
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
    return agent_executor

def initialize_meta_chain():
    """Initializes and returns a language learning model chain."""

    meta_prompt = PromptTemplate(
        input_variables=["goal", "david_instantiation_prompt", "david_execution", "actions"], 
        template=get_meta_prompt_template()
    )

    meta_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=meta_prompt, 
        verbose=True, 
    )
    return meta_chain

def initialize_chain(prompt_template):
    """Initializes and returns a goal evaluation chain."""
    evaluation_prompt = PromptTemplate(
        input_variables=["execution_output", "goal"], 
        template=prompt_template
    )
    evaluation_chain = LLMChain(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4"), 
        prompt=evaluation_prompt, 
        verbose=True, 
    )
    return evaluation_chain

def main(goal, max_meta_iters=MAX_META_ITERS):
    """Main execution function.

    Args:
        goal: The goal for the AI.
        max_meta_iters: Maximum iterations for the meta AI.

    Returns:
        None.
    """
    try:
        df = load_or_create_dataframe()
        david_instantiation_prompt = get_init_prompt_template()
        evaluation_chain, extracted_actions_chain = initialize_chains()

        for i in range(max_meta_iters):
            print(f'[Episode {i+1}/{max_meta_iters}]')
            relevant_previous_actions = get_relevant_actions(goal)
            tips = TIPS.format(relevant_previous_actions)
            execution_output = run_agent(goal, david_instantiation_prompt, CONSTRAINTS, tips)
            evaluation_output = evaluation_chain.predict(execution_output=execution_output, goal=goal)

            print("evaluation Output:", evaluation_output)
            extracted_actions_output = extracted_actions_chain.predict(execution_output=execution_output, goal=goal)
            update_dataframe_based_on_evaluation(df, goal, david_instantiation_prompt, CONSTRAINTS, tips, execution_output, evaluation_output, extracted_actions_output)

            if 'yes' in evaluation_output.strip().lower():
                break

            update_meta_chain_and_variables(extracted_actions_output, goal, david_instantiation_prompt, execution_output)

    except Exception as e:
        print(f'Error: {e}')

def load_or_create_dataframe():
    if os.path.isfile(CSV_FILE_PATH):
        return pd.read_csv(CSV_FILE_PATH)
    else:
        return pd.DataFrame(columns=CSV_COLUMNS)

def initialize_chains():
    evaluation_prompt_template = get_evaluation_prompt_template()
    extracted_actions_prompt = get_extracted_actions_prompt_template()
    return initialize_chain(evaluation_prompt_template), initialize_chain(extracted_actions_prompt)

def run_agent(goal, david_instantiation_prompt, constraints, tips):
    cb.clear()
    agent = initialize_agent(david_instantiation_prompt)
    try:
        agent.run(input=goal, constraints=constraints, tips=tips, callbacks=[cb])
    except Exception as e:
        print(f'Exception: {e}')
        print('Continuing...')
    return ''.join(cb.last_execution)

def update_dataframe_based_on_evaluation(df, goal, david_instantiation_prompt, constraints, tips, execution_output, evaluation_output, extracted_actions_output):
    print("Extracted actions:", extracted_actions_output)
    actions_embedding = generate_embedding(input=extracted_actions_output)
    status = 'succeeded' if 'yes' in evaluation_output.strip().lower() else 'false'

    new_row = {
        'Goal': goal, 
        'InstantiationPrompt': david_instantiation_prompt, 
        'Constraints': constraints, 
        'Tips': tips, 
        'Actions': extracted_actions_output, 
        'embedding': actions_embedding, 
        'execution': execution_output, 
        'status': status
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE_PATH, index=False)
    return extracted_actions_output

def update_meta_chain_and_variables(extracted_actions_output, goal, david_instantiation_prompt, execution_output):
    meta_chain = initialize_meta_chain()
    temp_prompt = PromptTemplate(
        input_variables=["tool_names","tools","input","constraints","tips","agent_scratchpad"],
        template=david_instantiation_prompt
    )
    temp_prompt = temp_prompt.format(
        tools="Bash", 
        tool_names="Bash Tool", 
        input=goal, 
        constraints=CONSTRAINTS, 
        tips=TIPS, 
        agent_scratchpad=""
    )
    meta_output = meta_chain.predict(
        goal=goal, 
        david_instantiation_prompt=temp_prompt,
        david_execution=execution_output,
        actions=extracted_actions_output.replace('```','')
    )
    print(f'New Prompt: {meta_output}')
    constraints, tips = get_new_instructions(meta_output)
    cb.last_execution = []
    print(f'New Constraints: {constraints}')
    print(f'New Tips: {tips}')

# Create the argument parser
parser = argparse.ArgumentParser()

# Add the desired arguments
parser.add_argument("--goal", help="Specify the goal")

if __name__ == '__main__':
    """Entry point of the script.

    Here we set the goal and call the main function.
    """
    # smol_ai_repo = 'https://github.com/smol-ai/developer.git'
    # goal = f"Write a short story about an AI enthusiast. Send it to <email>."
    args = parser.parse_args()
    if args.goal.endswith('.txt'):
        with open(args.goal, 'r') as file:
            goal = file.read().strip()
    else:
        goal = args.goal
    main(goal)