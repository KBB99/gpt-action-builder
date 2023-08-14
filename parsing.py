from embedding_utils import generate_embedding
# from openai.embeddings_utils import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import os
from prompts import get_process_actions_prompt_template 
from chains import initialize_chain
import argparse

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

def get_similar_actions(query, df, top_n=5):
    if df.empty:
        return "None"
    query_embedding = generate_embedding(query)
    
    # Compute similarities with all actions
    similarities = df['action_embedding'].apply(lambda emb: cosine_similarity([emb], [query_embedding])[0][0])
    
    # Sort by similarity
    top_indices = similarities.nlargest(top_n).index
    return df.iloc[top_indices]


def parse_arguments():
    """
    Parses command-line arguments.
    
    Returns:
        goal (str): The parsed goal from the command-line arguments or from a text file.
    """
    parser = argparse.ArgumentParser(description="AI Meta Iteration Script")
    parser.add_argument("--goal", help="Specify the goal", required=True)
    args = parser.parse_args()

    if args.goal.endswith('.txt'):
        with open(args.goal, 'r') as file:
            return file.read().strip()

    return args.goal