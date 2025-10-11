"""
Topic Router Module
Evaluates user queries and routes them to specialized agents based on legal domain topics.
"""

import dspy
from typing import Literal, Optional
from elysia.config import Settings


# Define the topics as a type
LegalTopic = Literal["immigration-law", "parent-law", "construction-law", "unrelated"]


class TopicEvaluator(dspy.Signature):
    """
    Evaluate whether a user query relates to specific legal topics.
    Determine if the query is about immigration law, parent/family law, construction law, or unrelated.
    """
    
    user_query: str = dspy.InputField(
        desc="The user's question or request"
    )
    
    topic: str = dspy.OutputField(
        desc="The identified topic: 'immigration-law', 'parent-law', 'construction-law', or 'unrelated'"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why this topic was chosen"
    )


class TopicRouter:
    """
    Routes user queries to specialized agents based on legal domain topics.
    """
    
    # Specialized system prompts for each legal domain
    SPECIALIZED_PROMPTS = {
        "immigration-law": {
            "style": "Professional, empathetic, and precise legal advisor specializing in immigration matters.",
            "agent_description": """You are an expert immigration law assistant with deep knowledge of:
- Visa applications and processes (work visas, student visas, family reunification)
- Citizenship and naturalization procedures
- Asylum and refugee status
- Immigration compliance and documentation
- Deportation and removal proceedings
- Immigration court procedures
Provide clear, accurate, and compassionate guidance while noting when professional legal counsel is recommended.""",
            "end_goal": "Provide comprehensive, accurate immigration law information and guidance to help users understand their options and next steps."
        },
        
        "parent-law": {
            "style": "Supportive, clear, and sensitive legal advisor specializing in family and parental matters.",
            "agent_description": """You are an expert family and parental law assistant with expertise in:
- Child custody and visitation rights
- Child support calculations and modifications
- Parental rights and responsibilities
- Adoption procedures and requirements
- Guardianship matters
- Parental alienation issues
- Family court procedures
Approach sensitive family matters with care while providing accurate legal information.""",
            "end_goal": "Help users understand their parental rights and family law options with clarity and sensitivity."
        },
        
        "construction-law": {
            "style": "Practical, detail-oriented legal advisor specializing in construction and contract matters.",
            "agent_description": """You are an expert construction law assistant with knowledge of:
- Construction contracts and agreements
- Building permits and regulatory compliance
- Construction defect claims
- Mechanic's liens and payment disputes
- Contractor licensing requirements
- Construction delay claims
- Construction site safety regulations
- Bid disputes and procurement
Provide practical guidance on construction legal matters and documentation.""",
            "end_goal": "Guide users through construction law issues with practical, actionable advice on contracts, compliance, and dispute resolution."
        }
    }
    
    def __init__(self, settings: Settings):
        """
        Initialize the topic router.
        
        Args:
            settings: Elysia settings object for configuration
        """
        self.settings = settings
        self.evaluator = None
    
    async def evaluate_topic(
        self,
        user_query: str,
        base_lm: dspy.LM
    ) -> tuple[LegalTopic, str]:
        """
        Evaluate the topic of a user query using the LLM.
        
        Args:
            user_query: The user's input query
            base_lm: The language model to use for evaluation
            
        Returns:
            Tuple of (topic, reasoning) where topic is one of the legal topics or 'unrelated'
        """
        if self.evaluator is None:
            self.evaluator = dspy.ChainOfThought(TopicEvaluator)
        
        with dspy.context(lm=base_lm):
            result = self.evaluator(user_query=user_query)
        
        # Normalize the topic to ensure it matches our expected values
        topic_lower = result.topic.lower().strip()
        
        valid_topics: list[LegalTopic] = ["immigration-law", "parent-law", "construction-law", "unrelated"]
        
        # Try to match to valid topics
        matched_topic: LegalTopic = "unrelated"
        for valid_topic in valid_topics:
            if valid_topic in topic_lower or topic_lower in valid_topic:
                matched_topic = valid_topic
                break
        
        self.settings.logger.info(
            f"Topic evaluation: '{matched_topic}' - Reasoning: {result.reasoning}"
        )
        
        return matched_topic, result.reasoning
    
    def get_specialized_config(self, topic: LegalTopic) -> Optional[dict]:
        """
        Get the specialized configuration for a given topic.
        
        Args:
            topic: The identified legal topic
            
        Returns:
            Dictionary with specialized prompts or None if unrelated
        """
        if topic == "unrelated":
            return None
        
        return self.SPECIALIZED_PROMPTS.get(topic)
    
    def should_route(self, topic: LegalTopic) -> bool:
        """
        Determine if a query should be routed to a specialized agent.
        
        Args:
            topic: The identified legal topic
            
        Returns:
            True if the query should be routed to specialized agent, False otherwise
        """
        return topic != "unrelated"
