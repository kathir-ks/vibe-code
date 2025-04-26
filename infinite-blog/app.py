import os
import quart
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
    # Consider gemini-pro for potentially longer/more coherent generation if needed
    model = genai.GenerativeModel('gemini-1.5-flash')
    logging.info(f"Gemini model '{model.model_name}' initialized successfully.")
except Exception as e:
    model = None
    logging.error(f"Failed to initialize Gemini model: {e}", exc_info=True)

# --- Initialize Firestore Client ---
try:
    db = firestore.Client()
    topics_collection_ref = db.collection('generated_topics_longform') # Use a distinct collection
    logging.info("Firestore client initialized successfully.")
except Exception as e:
    db = None
    topics_collection_ref = None
    logging.error(f"Failed to initialize Firestore client: {e}", exc_info=True)

app = Quart(__name__, static_folder='static')

# --- Constants ---
# Broad themes for topic suggestion
BROAD_THEMES = [
    "the intersection of philosophy and modern technology",
    "fundamental concepts in theoretical physics explained simply",
    "surprising connections in mathematical history",
    "the future of artificial intelligence and society",
    "deep dives into pivotal historical moments or figures",
    "exploring paradoxes in science or logic",
    "the mysteries of cosmology and the universe",
    "advancements in computer science with societal impact",
    "ethical dilemmas in scientific research",
    "the evolution of human understanding of time or space",
    "forgotten stories from scientific discovery"
]
MAX_TOPIC_GENERATION_ATTEMPTS = 5
TARGET_WORD_COUNT_MIN = 800 # Lower bound for prompt
TARGET_WORD_COUNT_MAX = 2000 # Upper bound for prompt (adjust as needed)

# --- Helper Functions ---

async def suggest_unique_topic() -> str | None:
    """
    Asks the LLM to suggest a unique, intriguing topic based on broad themes,
    checking against recent Firestore history.
    """
    if not model:
        logging.error("Gemini model not available for topic suggestion.")
        return None
    if not topics_collection_ref:
        logging.warning("Firestore not available. Cannot check for topic uniqueness.")
        # Fallback: Simple suggestion without check
        try:
            theme = random.choice(BROAD_THEMES)
            prompt = f"Suggest a single, concise, and intriguing blog post topic (4-10 words) inspired by the theme: '{theme}'. The topic should pose a question or explore a specific concept. Output only the suggested topic title."
            response = await model.generate_content_async(prompt)
            topic = response.text.strip().strip('"').strip("'")
            logging.info(f"Suggested topic (no uniqueness check): {topic}")
            return topic if topic else None
        except Exception as e:
            logging.error(f"Error suggesting topic (no Firestore): {e}", exc_info=True)
            return None

    existing_topics = set()
    try:
        # Fetch recent topics for uniqueness check
        docs = topics_collection_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(75).stream() # Fetch a few more
        existing_topics = {doc.to_dict().get('normalized_topic', '').lower() for doc in docs if doc.to_dict().get('normalized_topic')}
        logging.info(f"Fetched {len(existing_topics)} existing topics for uniqueness check.")
    except Exception as e:
        logging.error(f"Error fetching existing topics: {e}", exc_info=True)

    for attempt in range(MAX_TOPIC_GENERATION_ATTEMPTS):
        try:
            theme = random.choice(BROAD_THEMES)
            # More detailed prompt asking for a *suggestion*
            prompt = (
                f"You are an expert curator of fascinating ideas. Suggest a single, compelling, and specific blog post topic "
                f"(around 4-10 words) inspired by the broad theme: '{theme}'.\n"
                f"The topic should be suitable for a detailed article ({TARGET_WORD_COUNT_MIN}-{TARGET_WORD_COUNT_MAX} words).\n"
                f"Focus on a specific question, paradox, connection, or implication within the theme.\n"
                # Optional: Mentioning avoided topics can help but makes prompt long
                # f"Avoid topics too similar to these recent ones: {', '.join(list(existing_topics)[:3])}\n"
                f"Output *only* the suggested topic title, nothing else."
            )

            response = await model.generate_content_async(prompt)
            potential_topic = response.text.strip().strip('"').strip("'").strip()

            if not potential_topic:
                logging.warning(f"Attempt {attempt+1}: LLM returned empty topic suggestion.")
                continue

            normalized_topic = potential_topic.lower()

            if normalized_topic not in existing_topics:
                logging.info(f"Attempt {attempt+1}: Suggested unique topic: '{potential_topic}'")
                # Store the *chosen* topic in Firestore
                try:
                    doc_ref = topics_collection_ref.document()
                    doc_ref.set({
                        'topic': potential_topic,
                        'normalized_topic': normalized_topic,
                        'theme_inspiration': theme,
                        'timestamp': firestore.SERVER_TIMESTAMP
                    })
                    logging.info(f"Stored suggested topic '{potential_topic}' in Firestore.")
                    return potential_topic # Success!
                except Exception as e:
                    logging.error(f"Error storing topic '{potential_topic}': {e}", exc_info=True)
                    return potential_topic # Return topic anyway
            else:
                logging.warning(f"Attempt {attempt+1}: Suggested topic '{potential_topic}' is too similar to recent ones. Retrying...")

        except Exception as e:
            logging.error(f"Error during topic suggestion attempt {attempt+1}: {e}", exc_info=True)
            await asyncio.sleep(0.5) # Wait before retry

    logging.error(f"Failed to suggest a unique topic after {MAX_TOPIC_GENERATION_ATTEMPTS} attempts.")
    # Fallback: generate a non-unique one if all attempts fail?
    try:
        theme = random.choice(BROAD_THEMES)
        prompt = f"Suggest a single, concise, and intriguing blog post topic (4-10 words) inspired by the theme: '{theme}'. Output only the suggested topic title."
        response = await model.generate_content_async(prompt)
        topic = response.text.strip().strip('"').strip("'")
        logging.warning(f"Falling back to non-unique topic suggestion: {topic}")
        return topic if topic else "A Random Thought"
    except Exception:
        return "Exploration of Ideas" # Absolute fallback


async def generate_long_article_content(topic: str) -> str | None:
    """
    Generates a longer, potentially structured article for the given topic.
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
            f"  - Develop the topic across several well-reasoned body sections/paragraphs, providing explanations, examples, or evidence where appropriate.\n"
            f"  - Conclude with a summary of key points, broader implications, or a thought-provoking final question.\n"
            f"- **Tone:** Maintain an accessible yet insightful tone suitable for an intelligent general audience. Explain complex ideas clearly.\n"
            f"- **Formatting:** Use Markdown for headings and potentially bullet points (`* Item`) or numbered lists (`1. Item`) if it enhances clarity.\n\n"
            f"**Output:** Provide *only* the full article content in Markdown format. Do *not* include the main title '{topic}' again at the start of the content itself."
        )

        # Increased timeout might be needed for the model call itself, though Gemini API handles this internally to some extent.
        # The primary timeout concern is the *Flask/Gunicorn request* timeout.
        response = await model.generate_content_async(prompt) # Add generation_config if needed (e.g., temperature)

        content = response.text.strip()
        logging.info(f"Finished long-form content generation for topic: '{topic}'. Word count approx: {len(content.split())}") # Rough estimate
        return content

    except Exception as e:
        logging.error(f"Error generating long article content for topic '{topic}': {e}", exc_info=True)
        # Provide a shorter fallback error message if generation fails
        return f"## Error Generating Content\n\nUnfortunately, there was an error generating the full article for the topic: '{topic}'. Please try reloading to get a different topic."


# --- API Route (Backend) ---
@app.route('/generate-post', methods=['GET'])
async def handle_generate_post():
    """API endpoint to suggest a unique topic and generate long-form content."""
    start_time = asyncio.get_event_loop().time()
    logging.info("Received request for /generate-post")

    if not model:
        return jsonify({"error": "LLM service not available"}), 503

    # 1. Suggest a unique topic
    topic = await suggest_unique_topic()
    if not topic:
        return jsonify({"error": "Failed to suggest a unique topic"}), 500
    topic_time = asyncio.get_event_loop().time()
    logging.info(f"Topic suggestion finished in {topic_time - start_time:.2f} seconds: '{topic}'")


    # 2. Generate the long-form article content
    content = await generate_long_article_content(topic)
    if not content:
        # Content generation failed, return error
        return jsonify({"error": f"Failed to generate article content for topic: {topic}"}), 500
    content_time = asyncio.get_event_loop().time()
    logging.info(f"Content generation finished in {content_time - topic_time:.2f} seconds for topic: '{topic}'")


    # 3. Return the successful result
    total_time = asyncio.get_event_loop().time()
    logging.info(f"Total request processed in {total_time - start_time:.2f} seconds.")
    return jsonify({
        "topic": topic,
        "content": content # Content is now expected to be Markdown
    })

# --- Static File Serving Routes (Frontend) ---
@app.route('/')
async def serve_index():
    """Serves the main index.html file."""
    return await send_from_directory(app.static_folder, 'index.html')

# Flask handles /style.css, /script.js etc automatically from static_folder
if __name__ == '__main__':
    import asyncio
    asyncio.run(app.run_task(host='0.0.0.0', port=int(os.environ.get('PORT', 8080))))
