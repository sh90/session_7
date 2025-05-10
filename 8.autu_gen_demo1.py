#pip install pyautogen
from autogen import ConversableAgent
print("Imported successfully")
import data_info
configuration =  {"model": "gpt-4o-mini", "api_key": data_info.open_ai_key    }

# Configure a Conversable Agent that will never ask for our input and uses the config provided
agent = ConversableAgent(
    name="Ask_me_Anything_Bot", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    human_input_mode="NEVER", # Other options includes - ALWAYS or TERMINATE (Terminate based on a set of conditions)
)

#this step will send a  request (a prompt) to the agent!
#Now we will use the generate_reply() function which is tied to this agent, and we'll send it a message for now as a dictionary that must specify the content and the role keys. The user role means this is a request, we could use, as we did before, the system role to specify a system prompt for this agent (we'll re-do this later).

reply = agent.generate_reply(
    messages=[{"content": "What is politics?", "role": "user"}]
)
print(reply)

# Agents have no memory
reply_new = agent.generate_reply(
    messages=[{"content": "Repeat the previous task", "role": "user"}]
)
print(reply_new)

# Agents have  memory when added the previous context
reply = agent.generate_reply(
    messages=[{"content": reply + "\nRepeat the previous task", "role": "user"}]
)
print(reply)

