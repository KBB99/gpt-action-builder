# GPT-Action-Builder

This project revolves around an advanced AI agent execution system. It systematically searches for and evaluates actions to achieve a given goal, adapting and learning in the process.

Modules:
config: Contains configurations such as maximum meta iterations, constraints, templates, and success indicators.
embedding_utils: Utilities for loading or creating dataframes related to actions and updating these dataframes.
parsing: Helps parse arguments and finds actions similar to a specified goal.
prompts: Manages initialization of prompts for the agent.
base_agent: The primary agent's run functionality.
meta_agent: Utilities for updating meta chains and other variables.
chains: Initiates chains used in the program.
How to Use:
Ensure you have all the necessary modules and configurations set in their respective files.
Set the logging.basicConfig(level=logging.INFO) to your desired logging level.
Pass the desired goal for the AI when you run the program.
The program will execute, searching for and evaluating actions, adapting as needed until it achieves the goal or reaches the maximum number of iterations.
Main Function:
The main() function is responsible for the core execution of the AI engine. Here's a brief overview of its operation:

Loads or creates an actions dataframe.
Initializes chains and prompts.
Iterates to find similar actions based on the goal.
Executes the agent and evaluates its output.
Updates dataframes and meta chains.
Checks for success and breaks the loop if successful.
Logging:
The program employs extensive logging to keep track of its progress. Error handling is also in place to catch and log exceptions for easier debugging.

To run the program follow the steps below:

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain
pip install openai
```

Set environment variable:
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

To see the gpt-action-builder in action create a goal and run:
```bash
python main.py --goal "Your goal"
```


Contribution:
If you wish to contribute to the project, please ensure to follow the module structure and keep the core functionality within the main() function. Any improvements or extensions to the existing functionalities should be discussed prior to implementation.

License:
MIT LICENSE

Author:
Kenton Blacutt

Related Medium article: [Coding the Future: Harnessing GPT-4 to Build AI Systems Capable of Achieving Any Objective](https://medium.com/@kenton_69720/coding-the-future-harnessing-gpt-4-to-build-develop-ai-systems-capable-of-achieving-any-objective-85d2f55dc052).