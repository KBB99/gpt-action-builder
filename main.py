from config import MAX_META_ITERS, CONSTRAINTS, TIPS_TEMPLATE, SUCCESS_INDICATOR
from embedding_utils import load_or_create_dataframe, load_or_create_actions_dataframe, update_actions_dataframe
from parsing import parse_arguments, get_similar_actions
from prompts import get_init_prompt_template
from base_agent import run_agent
from meta_agent import update_meta_chain_and_variables
from chains import initialize_chains
import logging

def main(goal, max_meta_iters=MAX_META_ITERS, constraints=CONSTRAINTS, tips_template=TIPS_TEMPLATE):
    """
    Executes the main program flow.

    Args:
        goal (str): The goal for the AI.
        max_meta_iters (int): Maximum iterations for the meta AI.
        constraints (str): Constraints for the AI.
        tips_template (str): Tips template for the AI.

    Returns:
        None.
    """
    logging.info("Starting the main program execution.")
    
    try:
        actions_dataframe = load_or_create_actions_dataframe()
        david_prompt = get_init_prompt_template()
        evaluation_chain, actions_chain = initialize_chains()

        for iteration in range(max_meta_iters):
            logging.info(f'[Episode {iteration + 1}/{max_meta_iters}]')
            
            similar_actions = get_similar_actions(goal, actions_dataframe)

            print("Similar actions:", similar_actions)

            tips = tips_template.format(similar_actions)
            
            execution_output = run_agent(goal, david_prompt, constraints, tips)
            
            evaluation_output = evaluation_chain.predict(execution_output=execution_output, goal=goal)
            logging.info(f"Evaluation Output: {evaluation_output}")

            actions_output = actions_chain.predict(execution_output=execution_output, goal=goal)

            update_actions_dataframe(actions_dataframe, actions_output)

            if SUCCESS_INDICATOR in evaluation_output.strip().lower():
                break

            update_meta_chain_and_variables(actions_output, goal, david_prompt, execution_output)

    except Exception as e:
        logging.error(f"Error encountered: {e}", exc_info=True)  # exc_info logs the traceback

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # set logging level
    goal = parse_arguments()
    main(goal)
