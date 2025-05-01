import os
import quart # Ensure quart is installed: pip install quart
import google.generativeai as genai
from google.cloud import firestore
import logging
import random
import asyncio

from quart import Quart, jsonify, request, send_from_directory

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)

# --- Load API Key ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logging.warning("GOOGLE_API_KEY environment variable not set. Using placeholder (will likely fail).")
    GOOGLE_API_KEY = "YOUR_API_KEY_HERE" # Replace or set env var

try:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using 1.5 flash - good balance of capability and speed
    model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info(f"Gemini model '{model.model_name}' initialized successfully.")
except Exception as e:
    model = None
    logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)

# --- Initialize Firestore Client ---
try:
    db = firestore.Client()
    # Keep distinct collection name
    topics_collection_ref = db.collection('generated_topics_longform')
    logging.info("Firestore client initialized successfully.")
except Exception as e:
    db = None
    topics_collection_ref = None
    logging.error(f"Failed to initialize Firestore client: {e}", exc_info=True)

app = Quart(__name__, static_folder='static')

# --- Constants ---
# Examples of interesting areas, less restrictive than before
# Use these to guide the *kind* of topics, not as the sole source.
INTERESTING_AREAS_EXAMPLES = [
    "the intersection of philosophy and modern technology",
    "fundamental concepts in theoretical physics explained simply",
    "surprising connections in mathematical history",
    "the future of artificial intelligence and society",
    "deep dives into pivotal historical moments or figures in science/tech",
    "exploring paradoxes in science, logic, or mathematics",
    "the mysteries of cosmology, astrophysics, and the universe",
    "advancements in computer science with broad societal impact",
    "ethical dilemmas posed by scientific research or technological advancement",
    "the evolution of human understanding (e.g., time, space, consciousness)",
    "forgotten stories or underappreciated figures from scientific discovery",
    "applications of advanced mathematics in unexpected fields",
    "the cognitive science behind learning and creativity",
    "biohacking and the future of human enhancement",
    "complex systems theory applied to biology, economics, or social science"
]
MAX_TOPIC_GENERATION_ATTEMPTS = 5
TARGET_WORD_COUNT_MIN = 800
TARGET_WORD_COUNT_MAX = 2000
# How many recent topics to fetch from Firestore for uniqueness check
FIRESTORE_FETCH_LIMIT = 100
# How many of the most recent topics to include IN THE PROMPT to guide the LLM
MAX_RECENT_TOPICS_FOR_PROMPT = 20 # Adjust as needed based on prompt length limits/effectiveness

# --- Helper Functions ---

async def suggest_unique_topic() -> str | None:
    """
    Asks the LLM to suggest a unique, intriguing topic from a broad domain,
    explicitly providing recent topics to avoid and checking against Firestore history.
    """
    if not model:
        logging.error("Gemini model not available for topic suggestion.")
        return None

    existing_topics_map = {} # Store normalized -> original casing
    recent_topic_titles_for_prompt = []

    if topics_collection_ref:
        try:
            # Fetch recent topics for uniqueness check and prompt guidance
            docs = topics_collection_ref.order_by(
                "timestamp", direction=firestore.Query.DESCENDING
            ).limit(FIRESTORE_FETCH_LIMIT).stream()

            count = 0
            for doc in docs:
                doc_data = doc.to_dict()
                original_topic = doc_data.get('topic')
                normalized_topic = doc_data.get('normalized_topic', '').lower()
                if original_topic and normalized_topic:
                    if normalized_topic not in existing_topics_map:
                         existing_topics_map[normalized_topic] = original_topic
                         # Add the original-cased topic to the list for the prompt
                         if count < MAX_RECENT_TOPICS_FOR_PROMPT:
                            recent_topic_titles_for_prompt.append(original_topic)
                            count += 1

            logging.info(f"Fetched {len(existing_topics_map)} unique recent topics. Using {len(recent_topic_titles_for_prompt)} in prompt.")

        except Exception as e:
            logging.error(f"Error fetching existing topics: {e}", exc_info=True)
            # Proceed without uniqueness check if Firestore fails
            existing_topics_map = {}
            recent_topic_titles_for_prompt = []
    else:
        logging.warning("Firestore not available. Cannot check for topic uniqueness or provide recent topics to LLM.")


    # Prepare the list of recent topics for the prompt string
    recent_topics_str = "None"
    if recent_topic_titles_for_prompt:
        # Format nicely for the prompt
        recent_topics_str = "\n - " + "\n - ".join(f"'{t}'" for t in recent_topic_titles_for_prompt)


    for attempt in range(MAX_TOPIC_GENERATION_ATTEMPTS):
        try:
            # Broader prompt, guided by examples and explicit avoidance list
            prompt = (
                f"You are an expert curator of fascinating and thought-provoking ideas. "
                f"Suggest a single, compelling, and specific blog post topic (around 4-12 words).\n"
                f"**Domain:** The topic should be from the broad realms of science (physics, cosmology, biology, cognitive science), mathematics, computer science, technology, philosophy, logic, or the history/future of ideas.\n"
                f"**Style:** Aim for intriguing questions, surprising connections, explorations of paradoxes, or deep dives into specific concepts. Examples of interesting areas include: {random.sample(INTERESTING_AREAS_EXAMPLES, 3)}.\n" # Show a few random examples
                f"**Goal:** The topic should be suitable for a detailed article ({TARGET_WORD_COUNT_MIN}-{TARGET_WORD_COUNT_MAX} words).\n"
                f"**Uniqueness Requirement:** CRITICALLY IMPORTANT - Ensure the suggested topic is distinct and not substantially similar to these recently generated ones: {recent_topics_str}\n\n"
                f"**Output:** Respond with *only* the suggested topic title string, and nothing else. Do not add quotes or introductory text."
            )

            logging.debug(f"Topic Suggestion Prompt (Attempt {attempt+1}):\n{prompt[:500]}...") # Log start of prompt

            response = await model.generate_content_async(prompt)

            # Clean up potential LLM variations (quotes, extra spaces)
            potential_topic = response.text.strip().strip('"').strip("'").strip()

            if not potential_topic:
                logging.warning(f"Attempt {attempt+1}: LLM returned empty topic suggestion.")
                continue

            normalized_topic = potential_topic.lower()

            # Final check against the larger fetched list (in case LLM missed one or list was truncated)
            if normalized_topic not in existing_topics_map:
                logging.info(f"Attempt {attempt+1}: Suggested unique topic: '{potential_topic}'")
                # Store the *chosen* topic in Firestore
                if topics_collection_ref:
                    try:
                        doc_ref = topics_collection_ref.document()
                        doc_ref.set({
                            'topic': potential_topic,
                            'normalized_topic': normalized_topic,
                            # 'theme_inspiration': None, # Removed specific theme tracking
                            'timestamp': firestore.SERVER_TIMESTAMP
                        })
                        logging.info(f"Stored suggested topic '{potential_topic}' in Firestore.")
                    except Exception as e:
                        logging.error(f"Error storing topic '{potential_topic}' in Firestore: {e}", exc_info=True)
                        # Log error but proceed, return the topic anyway
                return potential_topic # Success!
            else:
                logging.warning(f"Attempt {attempt+1}: Suggested topic '{potential_topic}' is too similar to recent ones (found in Firestore check). Retrying...")

        except Exception as e:
            logging.error(f"Error during topic suggestion attempt {attempt+1}: {e}", exc_info=True)
            await asyncio.sleep(random.uniform(0.5, 1.5)) # Wait a bit longer with jitter before retry

    logging.error(f"Failed to suggest a unique topic after {MAX_TOPIC_GENERATION_ATTEMPTS} attempts.")
    # Fallback: Generate a topic without uniqueness constraint if all attempts fail
    try:
        fallback_prompt = (
            f"Suggest a single, concise, and intriguing blog post topic (4-12 words) "
            f"from the realms of science, mathematics, technology, or philosophy. "
            f"Output only the suggested topic title."
        )
        response = await model.generate_content_async(fallback_prompt)
        topic = response.text.strip().strip('"').strip("'").strip()
        logging.warning(f"Falling back to potentially non-unique topic suggestion: '{topic}'")
        # Optionally store this fallback topic too, marked differently? For now, just return.
        return topic if topic else "Exploration of Foundational Ideas"
    except Exception as fallback_e:
        logging.error(f"Error during fallback topic suggestion: {fallback_e}", exc_info=True)
        return "The Nature of Inquiry" # Absolute fallback


async def generate_long_article_content(topic: str) -> str | None:
    """
    Generates a longer, potentially structured article for the given topic.
    (No changes needed in this function based on the request)
    """
    if not model:
        logging.error("Gemini model not available for article generation.")
        return None
    if not topic:
        logging.error("Cannot generate article for empty topic.")
        return None

    logging.info(f"Starting long-form content generation for topic: '{topic}' (Target: {TARGET_WORD_COUNT_MIN}-{TARGET_WORD_COUNT_MAX} words)")
    word_target_mid = (TARGET_WORD_COUNT_MIN + TARGET_WORD_COUNT_MAX) // 2

    try:
        # Prompt for longer, structured content using Markdown
        prompt = (
            f"Write a detailed, engaging, and informative article exploring the topic: '{topic}'.\n\n"
            f"**Article Requirements:**\n"
            f"- **Length:** Aim for approximately {word_target_mid} words (within the range of {TARGET_WORD_COUNT_MIN}-{TARGET_WORD_COUNT_MAX} words).\n"
            f"- **Structure:** Organize the article logically. Use Markdown headings (`## Section Title`) for major sections to improve readability.\n"
            f"  - Include a compelling introduction that hooks the reader and states the article's purpose or main question.\n"
            f"  - Develop the topic across several well-reasoned body sections/paragraphs, providing explanations, examples, or evidence where appropriate. Explore nuances and different perspectives if relevant.\n"
            f"  - Conclude with a summary of key points, broader implications, or a thought-provoking final statement.\n"
            f"- **Tone:** Maintain an accessible yet insightful tone suitable for an intelligent general audience. Explain complex ideas clearly without oversimplifying critical details.\n"
            f"- **Formatting:** Use Markdown for headings and potentially bullet points (`* Item`) or numbered lists (`1. Item`) if it enhances clarity.\n\n"
            f"**Output:** Provide *only* the full article content in Markdown format. Do *not* include the main title '{topic}' again at the start of the content itself. Begin directly with the article's introduction."
        )

        # Configuration for generation - you might adjust temperature, top_p etc. if needed
        # generation_config = genai.types.GenerationConfig(
        #     temperature=0.75
        # )
        response = await model.generate_content_async(
            prompt,
            # generation_config=generation_config
            )

        content = response.text.strip()
        # Basic check if generation looks truncated or failed silently
        if not content or len(content) < 200: # Arbitrary short length check
             logging.warning(f"Generated content for '{topic}' seems very short or empty. Might indicate an issue.")
             # Optionally return an error here, or let the potentially short content pass
             # return f"## Generation Issue\n\nThere was a problem generating the full content for '{topic}'. The response was too short."


        logging.info(f"Finished long-form content generation for topic: '{topic}'. Word count approx: {len(content.split())}")
        return content

    except Exception as e:
        # Catch potential specific API errors if needed (e.g., BlockedError)
        # if isinstance(e, genai.types.BlockedPromptException):
        #    logging.error(f"Content generation blocked for topic '{topic}': {e}")
        #    return f"## Content Generation Blocked\n\nContent generation for the topic '{topic}' was blocked due to safety filters."
        logging.error(f"Error generating long article content for topic '{topic}': {e}", exc_info=True)
        # Provide a user-friendly error message
        return f"## Error Generating Content\n\nUnfortunately, there was an error generating the full article for the topic: '{topic}'. This could be due to a temporary issue or the nature of the topic. Please try reloading to get a different topic."


# --- API Route (Backend) ---
@app.route('/generate-post', methods=['GET'])
async def handle_generate_post():
    """API endpoint to suggest a unique topic and generate long-form content."""
    start_time = asyncio.get_event_loop().time()
    logging.info("Received request for /generate-post")

    if not model:
        return jsonify({"error": "LLM service not available"}), 503
    if not topics_collection_ref:
         logging.warning("Firestore connection not available, proceeding without uniqueness guarantees.")
         # Allow proceeding but maybe with a warning? Depends on requirements.

    # 1. Suggest a unique topic
    topic = await suggest_unique_topic()
    if not topic:
        # Specific error already logged in suggest_unique_topic
        return jsonify({"error": "Failed to suggest a suitable topic after multiple attempts."}), 500
    topic_time = asyncio.get_event_loop().time()
    logging.info(f"Topic suggestion finished in {topic_time - start_time:.2f} seconds: '{topic}'")


    # 2. Generate the long-form article content
    content = await generate_long_article_content(topic)
    if not content or content.startswith("## Error Generating Content") or content.startswith("## Generation Issue") or content.startswith("## Content Generation Blocked"):
        # Content generation failed or returned specific error message
        logging.error(f"Content generation failed or returned error state for topic: {topic}")
        # Return a generic server error, or potentially the error content itself?
        # Let's return a generic error to the client for consistency.
        return jsonify({"error": f"Failed to generate article content for the suggested topic: '{topic}'"}), 500
    content_time = asyncio.get_event_loop().time()
    logging.info(f"Content generation finished in {content_time - topic_time:.2f} seconds for topic: '{topic}'")


    # 3. Return the successful result
    total_time = asyncio.get_event_loop().time()
    logging.info(f"Total request processed in {total_time - start_time:.2f} seconds.")
    return jsonify({
        "topic": topic,
        "content": content # Content is expected to be Markdown
    })

# --- Static File Serving Routes (Frontend) ---
@app.route('/')
async def serve_index():
    """Serves the main index.html file."""
    return await send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:filename>')
async def serve_static(filename):
    """Serves other static files like CSS, JS."""
    return await send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logging.info(f"Starting Quart server on host 0.0.0.0 port {port}")
    # Use asyncio.run for the top-level await
    # Use app.run_task which integrates with asyncio event loop
    asyncio.run(app.run_task(host='0.0.0.0', port=port, debug=False)) # Turn debug=False for production