from config import MAX_META_ITERS, CONSTRAINTS, TIPS
from embedding_utils import load_or_create_dataframe, update_dataframe_based_on_evaluation
from parsing import get_relevant_actions
from prompts import get_init_prompt_template
from base_agent import run_agent
from meta_agent import update_meta_chain_and_variables
from chains import initialize_chains
import argparse

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

if __name__ == '__main__':
    """Entry point of the script.

    Here we set the goal and call the main function.
    """
    # smol_ai_repo = 'https://github.com/smol-ai/developer.git'
    # goal = f"Write a short story about an AI enthusiast. Send it to <email>."
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add the desired arguments
    parser.add_argument("--goal", help="Specify the goal")
    args = parser.parse_args()
    if args.goal.endswith('.txt'):
        with open(args.goal, 'r') as file:
            goal = file.read().strip()
    else:
        goal = args.goal
    main(goal)