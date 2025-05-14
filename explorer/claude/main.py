import argparse
from web_wanderer_agent import WebWandererAgent
import os

def main():
    parser = argparse.ArgumentParser(description='Run the Web Wanderer Agent')
    parser.add_argument('--start_url', type=str, default="https://news.ycombinator.com/",
                        help='URL to start exploration from')
    parser.add_argument('--max_pages', type=int, default=5,
                        help='Maximum number of pages to visit')
    parser.add_argument('--delay', type=float, default=2.0,
                        help='Delay between requests in seconds')
    parser.add_argument('--output', type=str, default="web_wanderer_discoveries.txt",
                        help='Output file for discoveries')
    args = parser.parse_args()

    # Get API key from environment variable or prompt
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google AI API key: ")
        
    print(f"Starting Web Wanderer from: {args.start_url}")
    print(f"Will visit up to {args.max_pages} pages with {args.delay}s delay between requests")
    
    # Create and start the agent
    agent = WebWandererAgent(api_key)
    agent.explore(args.start_url, max_pages=args.max_pages, delay=args.delay)
    agent.save_discoveries(args.output)
    
    print(f"Exploration complete! Results saved to {args.output}")
    
    # Print a summary of the most interesting discoveries
    print("\nTop Discoveries Summary:")
    for i, discovery in enumerate(agent.discoveries[:3], 1):
        print(f"\n{i}. {discovery['url']}")
        print(f"   {discovery['analysis'][:200]}...")

if __name__ == "__main__":
    main()