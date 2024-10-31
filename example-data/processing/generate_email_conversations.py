import random
import dspy
import pandas as pd
import json

from tqdm.auto import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CreateEmailChain(dspy.Signature):
    """
    Create an email chain from a given github issue.
    The email conversation should simulate a real sequence of emails that might have happened in a workspace related to the given issue.
    Emails are slightly more formal than chat conversations, and are usually longer.
    The chain can be only loosely related to the issue, and can be completely fictional, or it could be explicitly related to the issue.
    More specifically, the emails should be _different to what the type of content expected in a slack conversation_, it should have different content (i.e. less likely to be discussing the issue directly).
    """
    
    issue = dspy.InputField(description="The description of the issue to be converted into a email chain")
    chain_length = dspy.InputField(description="The number of emails in the chain to generate")
    email_type = dspy.InputField(description="A description of the type of email chain to generate, be creative within this prompt")
        
    previous_chains = dspy.InputField(
        description="""
        A list of summaries of previous email chains. 
        You should avoid repeating the same structure and topics as these chains. 
        Do not talk about the same content as the chains in this list.
        I.e., the summary you output should not be similar to the summaries in this list.
        However, it should still be related to the issue. 
        If empty, you can ignore this.
        """.strip()
    )

    user_names = dspy.InputField(description="The list of names that can be used in the chain, these _do not_ have to alternate.")

    summary = dspy.OutputField(
        description="""
        A short summary of the email conversation. Around 1 sentence.
        """.strip()
    )

    chain = dspy.OutputField(
        description="""
        The generated email chain. 
        You should generate {{chain_length}} emails in the chain.
        The chain should be in the format of a list of dictionaries, where each dictionary contains the following keys:
        - "name": The name of the sender in the email, these come from the {{user_names}} list.
        - "content": The content of the email. These usually start with "Hi", "Hello", "Dear", etc., and sign off with "Best regards", "Regards", "Kind regards", etc. but do not have to, you should vary whether this is the case.
        - "timestamp": The timestamp of the email in the format of YYYY-MM-DD HH:MM:SS. Add a random amount of delay between emails, from 10 minutes to 6 hours and a random amount of seconds.
        """.strip()
    )

class EmailChainGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.create_email_chain = dspy.ChainOfThought(CreateEmailChain)

    def forward(
            self, 
            issue: str, 
            chain_length: int, 
            user_names: list[int], 
            previous_chains: list[str], 
            email_type: str,
            idx: int = 0
        ) -> tuple[list[dict], str]:

        user_names = "[" + ", ".join(map(str, user_names)) + "]"
        previous_chains = "[" + ", ".join(previous_chains) + "]"

        prediction = self.create_email_chain(
            issue=issue, 
            chain_length=chain_length, 
            user_names=user_names, 
            previous_chains=previous_chains, 
            email_type=email_type,
            config = {"temperature": 0.7+0.005*idx}
        )

        return prediction["chain"], prediction["summary"]


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

email_types = [
    "A sequence of emails between a customer and a support team (e.g. the customer is having an issue with the product, and is contacting support)",
    "A sequence of emails between a manager and their team, discussing the issue within a broader topic. (e.g., a general project update, and this issue is listed as a sub-task)",
    "A sequence of emails between a customer and a salesperson (e.g. the customer has heard about this issue and is looking for reassurance from sales)",
    "An automated email from a service, such as a newsletter, or a product update, which is not related to the issue but contains similar content to the issue",
    "A sequence of emails between a customer and a support team, where the customer is asking a question about the product, and the support team is answering the question. The issue may or may not be mentioned.",
    "Emails between colleagues discussing a project, where the issue may or may not be mentioned. If it is mentioned, it is mentioned in passing (e.g. we still have {{this issue}} to deal with).",
    "A sequence of emails between a customer and a support team, where the customer is escalating an issue to the support team, and the support team is trying to resolve the issue. The customer has been trying to resolve the issue themselves without success and has had previous contact with support and is getting angrier."
]

if __name__ == "__main__":

    dspy.settings.configure(lm = dspy.LM(model="gpt-4o-mini", max_tokens=10000))

    df = pd.read_csv("verba_github_issues.csv")
    generator = EmailChainGenerator()

    email_chains = []
    num_issues = 100
    for i in tqdm(range(num_issues)):
        
        num_chains = random.randint(0, 3)
        previous_chains = []

        for idx in range(num_chains):
            num_messages = random.randint(2, 10)
            user_names = random.sample(names, 3)
            email_type = random.choice(email_types)

            generation, summary = generator(
                issue=df.iloc[i]["body"], 
                chain_length=num_messages, 
                user_names=user_names, 
                previous_chains=previous_chains, 
                email_type=email_type,
                idx=idx
            )


            # try parsing as json
            try:
                generation = json.loads(generation)
            except:
                print(f"Failed to parse generation as json (i={i}, idx={idx}, num_messages={num_messages}, num_chains={num_chains})")
                continue

            # add metadata to the chain
            chain = {
                "issue_id": df.iloc[i]["id"].item(),
                "conversation_id": idx,
                "conversation": generation,
                "summary": summary
            }

            email_chains.append(chain)
            previous_chains.append(summary)

        error

    with open("verba_email_chains.jsonl", "w") as f:
        for chain in email_chains:
            json.dump(chain, f)
            f.write("\n")

    email_chains_df = pd.DataFrame(email_chains)
    email_chains_df.to_csv("verba_email_chains.csv", index=False)
