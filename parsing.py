from embedding_utils import generate_embedding
from openai.embeddings_utils import get_embedding, cosine_similarity
import pandas as pd
import re
import os
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from prompts import get_process_actions_prompt_template 

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

def get_new_instructions(meta_output):
    """Extracts and returns new constraints and tips from meta output.

    Args:
        meta_output: Output from the meta-chain.

    Returns:
        Tuple containing new constraints and tips.
    """
    constraints_pattern = r"Constraints: ([^\n]*)(?=Tips:|\n|$)"
    tips_pattern = r"Tips: ([^\n]*)(?=Constraints:|\n|$)"
    
    constraints_match = re.search(constraints_pattern, meta_output)
    tips_match = re.search(tips_pattern, meta_output)
    
    constraints = constraints_match.group(1).strip() if constraints_match else None
    tips = tips_match.group(1).strip() if tips_match else None
    
    return constraints, tips
    
def get_relevant_actions(goal, n=1):
    """Returns relevant actions for the goal.

    Args:
        goal: The goal for the AI.

    Returns:
        List of relevant actions.
    """
    # return ""
    if not os.path.isfile('successful_invocations.csv'):
        return "None"
    process_actions_prompt = get_process_actions_prompt_template()
    similar_actions = None
    goal_embedding = generate_embedding(goal)
    df = pd.read_csv('successful_invocations.csv')
    df = df.dropna(subset=['embedding'])
    df['embedding'] = df.embedding.apply(eval).apply(list)
    df['similarities']=df.embedding.apply(lambda x: cosine_similarity(x, goal_embedding))
    df = df.sort_values('similarities', ascending=False)
    df_no_confirmation = df[~df['Actions'].str.contains('confirmation', case=False)]
    similar_actions = df_no_confirmation['Actions'].head(n).tolist()
    print(similar_actions)
    relevant_actions_chain = initialize_chain(process_actions_prompt)
    relevant_actions = relevant_actions_chain.predict(execution_output=similar_actions, goal=goal)
    return relevant_actions
