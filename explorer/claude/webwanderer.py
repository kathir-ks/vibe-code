import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
import random
import time
import os
from urllib.parse import urljoin, urlparse

class WebWandererAgent:
    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        """
        Initialize the Web Wanderer Agent.
        
        Args:
            api_key: Your Google AI API key
            model_name: The model to use (default: gemini-2.0-flash)
        """
        # Configure the Google Generative AI API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Agent state
        self.visited_urls = set()
        self.discoveries = []
        self.current_url = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_text_from_html(self, html_content):
        """Extract readable text content from HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def extract_links(self, html_content, base_url):
        """Extract all links from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('http'):
                links.append(href)
            elif not href.startswith('#') and not href.startswith('javascript:'):
                # Resolve relative URLs
                links.append(urljoin(base_url, href))
        
        return links

    def is_valid_url(self, url):
        """Check if a URL is valid and safe to visit."""
        # Basic validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Avoid visiting certain file types
        if any(url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip']):
            return False
            
        return True

    def analyze_content(self, text, url):
        """Use Gemini 2.0 Flash to analyze the content."""
        # Truncate text if too long
        if len(text) > 30000:  # Adjust based on model's context window
            text = text[:30000]
        
        prompt = f"""
        You are an AI assistant helping explore the web. You've just visited: {url}
        
        Here's the text content from that page:
        
        {text}
        
        Please do the following:
        1. Summarize the main content in 2-3 sentences
        2. Identify any notable or interesting information
        3. Rate how interesting this content is on a scale of 1-10
        4. Suggest 3 keywords or topics that this content relates to
        
        Format your response as a JSON object with fields:
        - summary
        - interesting_points
        - interest_rating
        - keywords
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return "Error analyzing content"

    def decide_next_url(self, links, current_content_analysis):
        """Use Gemini 2.0 Flash to decide which link to visit next."""
        if not links:
            return None
            
        # Filter out already visited URLs and invalid ones
        valid_links = [link for link in links if link not in self.visited_urls and self.is_valid_url(link)]
        
        if not valid_links:
            return None
            
        # If there are too many links, select a subset
        if len(valid_links) > 10:
            valid_links = random.sample(valid_links, 10)
            
        links_text = "\n".join([f"- {link}" for link in valid_links])
        
        prompt = f"""
        You are an AI web explorer. You've just visited a page and analyzed its content.
        
        Analysis of current page: {current_content_analysis}
        
        Here are links you can visit next:
        {links_text}
        
        Please choose which link seems most interesting to explore next based on:
        1. Likely to contain unique or valuable information
        2. Looks topically interesting based on the URL
        3. Seems to lead to a different type of content than what you've seen
        
        Return only the URL of the single link you want to visit, with no additional text.
        """
        
        try:
            response = self.model.generate_content(prompt)
            next_url = response.text.strip()
            
            # Validate the returned URL is in our list
            if next_url in valid_links:
                return next_url
            else:
                # If the model returned something invalid, just pick randomly
                return random.choice(valid_links)
        except Exception as e:
            print(f"Error deciding next URL: {e}")
            # Fallback to random selection
            return random.choice(valid_links) if valid_links else None

    def explore(self, starting_url, max_pages=10, delay=2):
        """
        Start exploring the web from a given URL.
        
        Args:
            starting_url: The URL to start from
            max_pages: Maximum number of pages to visit
            delay: Delay between requests in seconds
        """
        self.current_url = starting_url
        pages_visited = 0
        
        while pages_visited < max_pages and self.current_url:
            if self.current_url in self.visited_urls:
                print(f"Already visited {self.current_url}, finding a new link...")
                # This should be rare if decide_next_url is working correctly
                self.current_url = None
                continue
                
            print(f"\n[{pages_visited + 1}/{max_pages}] Visiting: {self.current_url}")
            
            try:
                # Add to visited set
                self.visited_urls.add(self.current_url)
                
                # Fetch the page
                response = requests.get(self.current_url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    # Extract text and links
                    text_content = self.extract_text_from_html(response.text)
                    links = self.extract_links(response.text, self.current_url)
                    
                    # Analyze the content
                    analysis = self.analyze_content(text_content, self.current_url)
                    print(f"\nAnalysis: {analysis}")
                    
                    # Save the discovery
                    self.discoveries.append({
                        "url": self.current_url,
                        "analysis": analysis,
                        "timestamp": time.time()
                    })
                    
                    # Decide where to go next
                    self.current_url = self.decide_next_url(links, analysis)
                    
                    if not self.current_url:
                        print("No valid links to follow. Exploration ended.")
                        break
                        
                else:
                    print(f"Failed to fetch page. Status code: {response.status_code}")
                    self.current_url = None
            
            except Exception as e:
                print(f"Error exploring {self.current_url}: {e}")
                self.current_url = None
            
            pages_visited += 1
            time.sleep(delay)  # Be polite to servers
            
        print(f"\nExploration complete. Visited {pages_visited} pages.")
        return self.discoveries
        
    def save_discoveries(self, filename="web_wanderer_discoveries.txt"):
        """Save all discoveries to a file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Web Wanderer Discoveries - {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, discovery in enumerate(self.discoveries, 1):
                f.write(f"Discovery #{i} - {discovery['url']}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(discovery['timestamp']))}\n")
                f.write(f"Analysis: {discovery['analysis']}\n")
                f.write("-" * 80 + "\n\n")
                
        print(f"Discoveries saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Get API key from environment variable or input
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = input("Enter your Google API key: ")
    
    # Create and start the agent
    agent = WebWandererAgent(api_key)
    
    # Start exploring from a given URL
    starting_url = "https://news.ycombinator.com/"  # Example: Hacker News
    discoveries = agent.explore(starting_url, max_pages=5)
    
    # Save the discoveries
    agent.save_discoveries()