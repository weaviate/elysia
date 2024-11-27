import random
import dspy
import pandas as pd
import json

from tqdm.auto import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CreateSlackConversation(dspy.Signature):
    """
    Create a slack conversation from a given github issue.
    The conversation should simulate a real conversation that might have happened in a slack workspace related to the given issue.
    The conversation can be only loosely related to the issue, and can be completely fictional, or it could be explicitly related to the issue.
    """
    
    issue = dspy.InputField(description="The description of the issue to be converted into a slack conversation")
    conversation_length = dspy.InputField(description="The number of messages in the conversation to generate")

    previous_conversations = dspy.InputField(
        description="""
        A list of summaries of previous conversations. 
        You should avoid repeating the same structure and topics as these conversations. 
        Do not talk about the same content as the conversations in this list.
        I.e., the summary you output should not be similar to the summaries in this list.
        However, it should still be related to the issue. 
        If empty, you can ignore this.
        """.strip()
    )

    user_names = dspy.InputField(description="The list of names that can be used in the conversation, these _do not_ have to alternate.")

    summary = dspy.OutputField(
        description="""
        A short summary of the conversation. Around 1 sentence.
        """.strip()
    )

    conversation = dspy.OutputField(
        description="""
        The generated slack conversation. 
        You should generate {{conversation_length}} messages in the conversation.
        The conversation should be in the format of a list of dictionaries, where each dictionary contains the following keys:
        - "name": The name of the speaker in the conversation, these come from the {{user_names}} list.
        - "content": The content of the message.
        - "timestamp": The timestamp of the message in the format of YYYY-MM-DD HH:MM:SS. Add a random amount of delay between messages, between 1 and 20 minutes and a random amount of seconds.
        """.strip()
    )

class SlackConversationGenerator(dspy.Module):
    def __init__(self):
        super().__init__()

        self.create_slack_conversation = dspy.ChainOfThought(CreateSlackConversation)

    def forward(
            self, 
            issue: str, 
            conversation_length: int, 
            user_names: list[int], 
            previous_conversations: list[str], 
            idx: int = 0
        ) -> tuple[list[dict], str]:

        user_names = "[" + ", ".join(map(str, user_names)) + "]"
        previous_conversations = "[" + ", ".join(previous_conversations) + "]"

        prediction = self.create_slack_conversation(
            issue=issue, 
            conversation_length=conversation_length, 
            user_names=user_names, 
            previous_conversations=previous_conversations, 
            config = {"temperature": 0.7+0.005*idx}
        )

        return prediction["conversation"], prediction["summary"]


names = [
    "Danny",
    "Edward",
    "John",
    "Sofia",
    "Zara",
    "Kerrigan",
    "Dalinar",
    "Alice",
    "Xaden",
    "Ravi",
    "Fatima",
    "Vin",
    "Shallan",
    "Kaladin",
    "Tychus",
    "Wu Zi-nan",
    "Jaina"
]

if __name__ == "__main__":

    dspy.settings.configure(lm = dspy.LM(model="gpt-4o-mini"))

    df = pd.read_csv("../verba_github_issues.csv")
    generator = SlackConversationGenerator()

    slack_conversations = []
    num_issues = len(df)
    id = 0
    for i in tqdm(range(num_issues)):
        
        num_conversations = random.randint(0, 3)
        previous_conversations = []

        for idx in range(num_conversations):
            num_messages = random.randint(2, 10)
            user_names = random.sample(names, 3)

            generation, summary = generator(
                issue=df.iloc[i]["body"], 
                conversation_length=num_messages, 
                user_names=user_names, 
                previous_conversations=previous_conversations, 
                idx=idx
            )


            # try parsing as json
            try:
                generation = json.loads(generation)
            except:
                print(f"Failed to parse generation as json (i={i}, idx={idx}, num_messages={num_messages}, num_conversations={num_conversations})")
                continue

            # add metadata to the conversation
            conversation = {
                "issue_id": df.iloc[i]["id"].item(),
                "conversation_id": id,
                "conversation": generation,
                "summary": summary
            }

            slack_conversations.append(conversation)
            previous_conversations.append(summary)
            id += 1

    with open("verba_slack_conversations.jsonl", "w") as f:
        for conversation in slack_conversations:
            json.dump(conversation, f)
            f.write("\n")

    slack_conversations_df = pd.DataFrame(slack_conversations)
    slack_conversations_df.to_csv("verba_slack_conversations.csv", index=False)