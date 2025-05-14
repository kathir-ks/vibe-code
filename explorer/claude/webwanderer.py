import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
import random
import time
import os
import logging
from urllib.parse import urljoin, urlparse

class WebWandererAgent:
    def __init__(self, api_key, model_name="gemini-2.0-flash", logger=None):
        """
        Initialize the Web Wanderer Agent.
        
        Args:
            api_key: Your Google AI API key
            model_name: The model to use (default: gemini-2.0-flash)
            logger: Optional logger instance (will create one if not provided)
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        
        # Configure the Google Generative AI API
        self.logger.info(f"Initializing WebWandererAgent with model: {model_name}")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.logger.debug("Generative model initialized")
        
        # Agent state
        self.visited_urls = set()
        self.discoveries = []
        self.current_url = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.logger.debug("Agent state initialized")
    
    def extract_text_from_html(self, html_content):
        """Extract readable text content from HTML."""
        self.logger.debug("Extracting text from HTML content")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        script_count = 0
        for script in soup(["script", "style"]):
            script.decompose()
            script_count += 1
        self.logger.debug(f"Removed {script_count} script/style elements")
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        text_length = len(text)
        self.logger.debug(f"Extracted {text_length} characters of text")
        return text

    def extract_links(self, html_content, base_url):
        """Extract all links from HTML content."""
        self.logger.debug(f"Extracting links from {base_url}")
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.startswith('http'):
                links.append(href)
            elif not href.startswith('#') and not href.startswith('javascript:'):
                # Resolve relative URLs
                resolved_url = urljoin(base_url, href)
                links.append(resolved_url)
        
        self.logger.debug(f"Found {len(links)} links on the page")
        return links

    def is_valid_url(self, url):
        """Check if a URL is valid and safe to visit."""
        # Basic validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            self.logger.debug(f"Invalid URL (missing scheme or netloc): {url}")
            return False
        
        # Avoid visiting certain file types
        if any(url.endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip']):
            self.logger.debug(f"Skipping file URL: {url}")
            return False
        
        self.logger.debug(f"URL is valid: {url}")    
        return True

    def analyze_content(self, text, url):
        """Use Gemini 2.0 Flash to analyze the content."""
        self.logger.info(f"Analyzing content from: {url}")
        
        # Log original text length
        original_length = len(text)
        self.logger.debug(f"Original text length: {original_length} characters")
        
        # Truncate text if too long
        if len(text) > 30000:  # Adjust based on model's context window
            text = text[:30000]
            self.logger.debug(f"Text truncated to 30000 characters (from {original_length})")
        
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
        
        self.logger.debug("Sending content to Gemini for analysis")
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Gemini API response received in {elapsed_time:.2f} seconds")
            return response.text
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}", exc_info=True)
            return "Error analyzing content"

    def decide_next_url(self, links, current_content_analysis):
        """Use Gemini 2.0 Flash to decide which link to visit next."""
        self.logger.info("Deciding which URL to visit next")
        
        if not links:
            self.logger.warning("No links found to explore next")
            return None
            
        # Filter out already visited URLs and invalid ones
        valid_links = [link for link in links if link not in self.visited_urls and self.is_valid_url(link)]
        
        self.logger.debug(f"Found {len(valid_links)} valid links out of {len(links)} total links")
        
        if not valid_links:
            self.logger.warning("No valid links found to explore next")
            return None
            
        # If there are too many links, select a subset
        if len(valid_links) > 10:
            valid_links = random.sample(valid_links, 10)
            self.logger.debug(f"Sampled 10 links from {len(valid_links)} valid links")
            
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
        
        self.logger.debug("Asking Gemini to select next URL")
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            elapsed_time = time.time() - start_time
            self.logger.debug(f"Gemini API response received in {elapsed_time:.2f} seconds")
            
            next_url = response.text.strip()
            
            # Validate the returned URL is in our list
            if next_url in valid_links:
                self.logger.info(f"Selected next URL: {next_url}")
                return next_url
            else:
                self.logger.warning(f"Gemini returned invalid URL: '{next_url}', falling back to random selection")
                # If the model returned something invalid, just pick randomly
                random_url = random.choice(valid_links)
                self.logger.info(f"Randomly selected URL: {random_url}")
                return random_url
        except Exception as e:
            self.logger.error(f"Error deciding next URL: {e}", exc_info=True)
            # Fallback to random selection
            if valid_links:
                random_url = random.choice(valid_links)
                self.logger.info(f"Randomly selected URL after error: {random_url}")
                return random_url
            else:
                return None

    def explore(self, starting_url, max_pages=10, delay=2):
        """
        Start exploring the web from a given URL.
        
        Args:
            starting_url: The URL to start from
            max_pages: Maximum number of pages to visit
            delay: Delay between requests in seconds
        """
        self.logger.info(f"Starting web exploration from: {starting_url}")
        self.logger.info(f"Parameters: max_pages={max_pages}, delay={delay}s")
        
        self.current_url = starting_url
        pages_visited = 0
        
        while pages_visited < max_pages and self.current_url:
            if self.current_url in self.visited_urls:
                self.logger.warning(f"Already visited {self.current_url}, finding a new link...")
                # This should be rare if decide_next_url is working correctly
                self.current_url = None
                continue
                
            self.logger.info(f"\n[{pages_visited + 1}/{max_pages}] Visiting: {self.current_url}")
            
            try:
                # Add to visited set
                self.visited_urls.add(self.current_url)
                
                # Fetch the page
                self.logger.debug(f"Fetching URL: {self.current_url}")
                start_time = time.time()
                response = requests.get(self.current_url, headers=self.headers, timeout=10)
                elapsed_time = time.time() - start_time
                self.logger.debug(f"Page fetched in {elapsed_time:.2f} seconds, status code: {response.status_code}")
                
                if response.status_code == 200:
                    # Extract text and links
                    text_content = self.extract_text_from_html(response.text)
                    links = self.extract_links(response.text, self.current_url)
                    
                    # Analyze the content
                    analysis = self.analyze_content(text_content, self.current_url)
                    self.logger.info(f"Analysis complete for: {self.current_url}")
                    self.logger.debug(f"Analysis result: {analysis[:500]}...")
                    
                    # Save the discovery
                    self.discoveries.append({
                        "url": self.current_url,
                        "analysis": analysis,
                        "timestamp": time.time()
                    })
                    self.logger.debug(f"Discovery #{len(self.discoveries)} saved")
                    
                    # Decide where to go next
                    next_url = self.decide_next_url(links, analysis)
                    if next_url:
                        self.current_url = next_url
                    else:
                        self.logger.warning("No valid links to follow. Exploration ended.")
                        break
                        
                else:
                    self.logger.error(f"Failed to fetch page. Status code: {response.status_code}")
                    self.current_url = None
            
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error exploring {self.current_url}: {e}", exc_info=True)
                self.current_url = None
            except Exception as e:
                self.logger.error(f"Unexpected error exploring {self.current_url}: {e}", exc_info=True)
                self.current_url = None
            
            pages_visited += 1
            self.logger.debug(f"Waiting {delay} seconds before next request")
            time.sleep(delay)  # Be polite to servers
            
        self.logger.info(f"Exploration complete. Visited {pages_visited} pages.")
        self.logger.info(f"Discovery count: {len(self.discoveries)}")
        return self.discoveries
        
    def save_discoveries(self, filename="web_wanderer_discoveries.txt"):
        """Save all discoveries to a file."""
        self.logger.info(f"Saving discoveries to {filename}")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"Web Wanderer Discoveries - {timestamp}\n\n")
                
                for i, discovery in enumerate(self.discoveries, 1):
                    discovery_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                time.localtime(discovery['timestamp']))
                    
                    f.write(f"Discovery #{i} - {discovery['url']}\n")
                    f.write(f"Timestamp: {discovery_time}\n")
                    f.write(f"Analysis: {discovery['analysis']}\n")
                    f.write("-" * 80 + "\n\n")
                
            self.logger.info(f"Successfully saved {len(self.discoveries)} discoveries to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving discoveries to {filename}: {e}", exc_info=True)

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