# Constants
MAX_META_ITERS = 3
TEMPERATURE = 0
MODEL_NAME = "gpt-4"
TIMEOUT = 9999
STREAMING = True
CSV_FILE_PATH = 'successful_invocations.csv'
CSV_COLUMNS = ['Goal', 'InstantiationPrompt', 'Constraints', 'Tips', 'Actions', 'embedding', 'execution', 'status']
CONSTRAINTS = "You cannot use the open command. Everything must be done in the terminal. You cannot use blocking commands like nano or vim."
TIPS_TEMPLATE = """You are in a mac zshell. 
To write to a file, use the echo command. 
You start in an empty playground directory. 
Relevant Actions: {}"""
SUCCESS_INDICATOR = "yes"
ACTIONS_CSV_FILE_PATH = 'actions.csv'
ACTIONS_CSV_COLUMNS = ['action', 'action_embedding']