from autogen import ConversableAgent
import data_info
print("Imported successfully")

configuration =  {
                    "model": "gpt-4o-mini",
                    "api_key":  data_info.open_ai_key
                  }

# Configure a Conversable Agent that will never ask for our input and uses the config provided
nba_fan = ConversableAgent(
    name="nba", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    system_message="you are an nba fan",
)

soccer_fan = ConversableAgent(
    name="soccer", # Name of the agent
    llm_config=configuration, # The Agent will use the LLM config provided to answer
    system_message="you are a soccer fan",

)

# This time,  initiate_chat() function will be used and we can start a chat between them.
# We need to first set the receptent of the message.
# The initial message
# Number of turns after what the conversation will stop.
# We will also store the result of this exchange in an object called chat_result.

result = nba_fan.initiate_chat(
    recipient = soccer_fan,
    message="convince me that soccer is better than nba",
    max_turns=2 # The conversation will stop after each agent has spoken twice
)
print(result)

#
import pprint
pprint.pprint(result.chat_history)
pprint.pprint(result.cost)
# #send method
# print("----------------")
# nba_fan.send(message="What was the last statement we made?",recipient=soccer_fan)
#
#

#
result = nba_fan.initiate_chat(
     recipient = soccer_fan,
     message="convience me that soccer is better than nba",
     max_turns=1, # The conversation will stop after each agent has spoken twice
     summary_method="reflection_with_llm", # Can be "last_message" (DEFAULT) or "reflection_with_llm"
     summary_prompt="Summarize the conversation in less than 50 words", # We specify the prompt used to summarize the chat
 )
import pprint
pprint.pprint(result.summary)

##Termination
# We've used the argument max_turns=2 which will end the conversation after two turns.
# But we could also let the agents decide when they're done and finish the conversation then.
# To do that, first we will have to tell each agent which words they should use when they're done and we'll have to monitor their messages for those words.
# The conversation terminate, when autogen detects that the agent sent those words.



#
#
# reply = agent.generate_reply(
#     messages=[{"content": "Repeat the previous task", "role": "user"}]
# )
# print(reply)
