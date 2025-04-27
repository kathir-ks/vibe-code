import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

class CapabilityValue(Enum):
    TRUE = "true"
    FALSE = "false"

@dataclass
class Skill:
    id: str
    name: str
    description: str
    tags: List[str]

@dataclass
class Capabilities:
    streaming: bool = False
    summarization: bool = False
    # Add other capabilities as needed

class AgentDiscoveryClient:
    def __init__(self, base_url: str):
        """
        Initialize the client with the base URL of the agent discovery service.
        
        Args:
            base_url (str): The base URL of the agent discovery service (e.g., 'http://localhost:8080')
        """
        self.base_url = base_url.rstrip('/')
        self.agents_endpoint = f"{self.base_url}/agents"

    def register_agent(
        self,
        name: str,
        description: str,
        skills: List[Skill],
        capabilities: Capabilities,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Register a new agent with the discovery service.
        
        Args:
            name (str): Name of the agent
            description (str): Description of the agent
            skills (List[Skill]): List of skills the agent has
            capabilities (Capabilities): Capabilities of the agent
            metadata (Optional[Dict]): Additional metadata about the agent
            
        Returns:
            Dict: The registered agent data including its ID
        """
        agent_data = {
            "name": name,
            "description": description,
            "skills": [{
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": skill.tags
            } for skill in skills],
            "capabilities": {
                "streaming": capabilities.streaming,
                "summarization": capabilities.summarization,
                # Add other capabilities as needed
            }
        }
        
        if metadata:
            agent_data["metadata"] = metadata

        response = requests.post(self.agents_endpoint, json=agent_data)
        response.raise_for_status()
        return response.json()

    def discover_agents(
        self,
        capability: Optional[str] = None,
        capability_value: Optional[bool] = None,
        skill_tag: Optional[str] = None,
        skill_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Discover agents based on various filters.
        
        Args:
            capability (Optional[str]): Filter by capability name
            capability_value (Optional[bool]): Value for the capability filter
            skill_tag (Optional[str]): Filter by skill tag
            skill_id (Optional[str]): Filter by skill ID
            
        Returns:
            List[Dict]: List of matching agents
        """
        params = {}
        
        if capability:
            params['capability'] = capability
            if capability_value is not None:
                params[f'{capability}_value'] = str(capability_value).lower()
        
        if skill_tag:
            params['skill_tag'] = skill_tag
            
        if skill_id:
            params['skill_id'] = skill_id

        response = requests.get(self.agents_endpoint, params=params)
        response.raise_for_status()
        return response.json() 