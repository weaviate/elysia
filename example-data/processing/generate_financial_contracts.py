import dspy
import random
import pandas as pd
import json
from tqdm.auto import tqdm

class CreateFinancialContract(dspy.Signature):
    """
    You are an expert at creating synthetic data.
    You will be given some input fields to create, relating to financial contracts.
    Do not use any placeholders, you should make something that looks like a real contract.
    These are for synthetic data purposes, so do not worry about the validity of the contract,
    and make sure it looks like a real contract.
    The financial documents are between a company (Weaviate) and a client/customer/employee or other.
    """

    contract_type = dspy.InputField(description="The type of contract to create, e.g. 'loan agreement', 'lease agreement', 'service agreement', etc.")
    author = dspy.InputField(description="The author of the contract, e.g. 'John Doe'")
    random_values = dspy.InputField(description="A list of random costs to include in the contract. You do not have to use all of them, but use them as a source of randomness.")
    random_names = dspy.InputField(description="A list of random names to include in the contract. You do not have to use all of them, but use them as a source of randomness.")

    date = dspy.OutputField(
        description="""
        The date of the contract, as a timestamp in the format of YYYY-MM-DD HH:MM:SS. 
        Pick this completely randomly, but make it realistic.
        """
    )
    contract_length: int = dspy.OutputField(description="The length of the contract in years")
    contract_content: str = dspy.OutputField(
        description="""
        The content of the contract, in plain text
        Do not use any placeholders for costs, amounts, parties, people, create random values
        """
    )

class FinancialContractGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.create_financial_contract = dspy.ChainOfThought(CreateFinancialContract)

    def forward(
        self, 
        contract_type: str, 
        author: str, 
        contract_length: int, 
        random_values: list[float],
        random_names: list[str],
        idx: int = 0
    ) -> dict:
        return self.create_financial_contract(
            contract_type=contract_type, 
            author=author, 
            contract_length=contract_length,
            random_values=random_values,
            random_names=random_names,
            config = {"temperature": 0.7+random.random()*0.1*idx}
        )

if __name__ == "__main__":
    lm = dspy.LM(model="gpt-4o-mini")
    dspy.settings.configure(lm=lm)

    names = [
        "John Williams", 
        "Hans Zimmer", 
        "Kaladin Stormblessed", 
        "Edward Elric", 
        "Arthur Penndragon",
        "Johnathan Smith",
        "Jane Doe",
        "Alice Johnson",
        "Bob Brown"
    ]

    random_names = [
        "Weaviate",
        "OpenAI",
        "Google",
        "Microsoft",
        "Mark Robson",
        "Danny Williams",
        "John Smith"
    ]

    contract_types = [
        "loan agreement",
        "lease agreement",
        "service agreement",
        "invoice",
        "purchase order",
        "sales agreement",
        "employment contract",
        "non-disclosure agreement",
        "partnership agreement"
    ]

    generator = FinancialContractGenerator()

    num_documents = 100
    documents = []
    for i in tqdm(range(num_documents), desc="Generating financial contracts"):  

        type = random.choice(contract_types)
        author = random.choice(names)
        length = random.randint(1, 10)

        random_values = [
            str(round(random.random()*400, 2)),
            str(round(random.random()*200, 2)),
            str(round(random.random()*100, 2)),
            str(round(random.random()*500, 2)),
            str(round(random.random()*1000, 2))
        ]
        
        prediction = generator.forward(
            contract_type=type, 
            random_values=random_values,
            random_names=random_names,
            author=author, 
            contract_length=length,
            idx=i/num_documents
        )

        documents.append({
            "contract_type": type,
            "author": author,
            "date": prediction["date"],
            "contract_length": prediction["contract_length"],
            "contract_text": prediction["contract_content"]
        })

    with open("financial_contracts.jsonl", "w") as f:
        for document in documents:
            json.dump(document, f)
            f.write("\n")

    documents_df = pd.DataFrame(documents)
    documents_df.to_csv("financial_contracts.csv", index=False)