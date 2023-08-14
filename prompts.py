def get_init_prompt_template():
    """Returns initial prompt for the Agent."""
    return """Your name is David.

    If something doesn't work twice in a row try something new.

    Never give up until you accomplish your goal.

    You have access to the following tool:

    {tools}

    Use the following format:

    Goal: the goal you are built to accomplish
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I have now completed my goal
    Final Summary: a final memo summarizing what was accomplished
    Constraints: {constraints}
    Tips: {tips}

    Begin!

    Goal: {input}
    {agent_scratchpad}"""

def get_meta_prompt_template(): 
    return """{{I want to instantiate an AI I'm calling David who successfully accomplishes my GOAL.}}

    #######
    MY GOAL
    #######

    {goal}

    ##############
    END OF MY GOAL
    ##############

    ############################
    DAVID'S INSTANTIATION PROMPT
    ############################

    {david_instantiation_prompt}

    ###################################
    END OF DAVID'S INSTANTIATION PROMPT
    ###################################

    #################
    DAVID'S EXECUTION
    #################

    {david_execution}

    ########################
    END OF DAVID'S EXECUTION
    ########################

    {{I do not count delegation back to myself as success.}}
    {{I will write an improved prompt specifying a new constraint and a new tip to instantiate a new David who hopefully gets closer to accomplishing my goal.}}
    {{Too bad I cannot add new tools, good thing bash is enough for someone to do anything.}}
    {{Even though David may think he did enough to complete goal I do not count it as success, lest I would not need to write a new prompt.}}
    {{I will make sure to work my notes into the tips and constraints ONLY if they seem useful. }}
    {{Notes: {actions} }}

    ###############
    IMPROVED PROMPT
    ###############

    """

def get_evaluation_prompt_template():
    return """
    {execution_output}
    Note that delegation does not count as success. Based on the above execution output, did David accomplish the goal of "{goal}"? (Yes/No)
    """

def get_extracted_actions_prompt_template():
    return """
    Goal: {goal}
    {execution_output}
    Given the above execution output, extract the actions from the execution output. 
    Place each action between the sequence '|||'. 
    Give the actions names like "Send an email" with a description like, "To send an email run ```mail -S <subject> < <data>```". 
    This information will be re-used in the subsequent iterations.
    |||
    """


def get_process_actions_prompt_template():
    return """
        Here is the goal we are trying to accomplish: `{goal}`.
        Here are the actions that are semantically similar to our goal:\n```{execution_output}```\n.
        Please find the actions that are relevant to our goal and write them down with ideas of how to use them to successfully accomplish our goal.
        If no actions are relevant then you must write: no relevant actions found.
        """