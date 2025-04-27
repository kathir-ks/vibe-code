from agent_discovery_client import AgentDiscoveryClient, Skill, Capabilities

def main():
    # Initialize the client
    client = AgentDiscoveryClient("https://agent-discovery-service-504268371434.us-central1.run.app")

    # Create skills
    skills = [
        Skill(
            id="summarize-doc-v1",
            name="Document Summarization",
            description="Summarizes documents",
            tags=["summarization", "document"]
        )
    ]

    # Create capabilities
    capabilities = Capabilities(
        streaming=True,
        summarization=True
    )

    # Register an agent
    agent = client.register_agent(
        name="Summarizer Agent",
        description="An agent that summarizes documents",
        skills=skills,
        capabilities=capabilities
    )
    print(f"Registered agent: {agent}")

    # Discover agents with specific capabilities
    agents = client.discover_agents(
        capability="streaming",
        capability_value=True,
        skill_tag="summarization"
    )
    print(f"Found {len(agents)} agents matching the criteria")

if __name__ == "__main__":
    main() 