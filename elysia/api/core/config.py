import spacy

class Settings:

    COLLECTION_NAMES = [
        "example_verba_github_issues", 
        "example_verba_email_chains", 
        "example_verba_slack_conversations", 
        "ecommerce",
        "financial_contracts",
        "weather"
    ]


    VERSION = "0.2.0"

settings = Settings()

nlp = spacy.load("en_core_web_sm")
