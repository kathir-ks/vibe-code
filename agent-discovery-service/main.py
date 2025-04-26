import os
import json
from flask import Flask, request, jsonify
from google.cloud import firestore
from jsonschema import validate, ValidationError, RefResolver

# --- Configuration ---
# Environment variable for GCP Project ID. Cloud Run sets this automatically.
# For local development, set GOOGLE_CLOUD_PROJECT environment variable.
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT')
if not PROJECT_ID:
    # Try to get it from gcloud config if running locally after 'gcloud init'
    # or set it manually for local dev
    # Example: PROJECT_ID = "your-gcp-project-id"
    try:
        import subprocess
        PROJECT_ID = subprocess.check_output(
            ['gcloud', 'config', 'get-value', 'project'],
            text=True
        ).strip()
    except Exception:
        print("Warning: GOOGLE_CLOUD_PROJECT env var not set.")
        # Set a default or raise an error if needed for local dev
        # PROJECT_ID = "your-local-dev-project-id"


COLLECTION_NAME = "agents"

# --- Initialize App and Firestore ---
app = Flask(__name__)
try:
    # When deployed to Cloud Run/Functions/App Engine, it detects credentials automatically
    # For local dev, ensure you've run 'gcloud auth application-default login'
    db = firestore.Client(project=PROJECT_ID)
    print(f"Firestore client initialized for project: {PROJECT_ID}")
except Exception as e:
    print(f"Error initializing Firestore client: {e}")
    # Handle initialization error appropriately in a real app
    # Maybe exit or disable Firestore features
    db = None

# --- Load and Prepare Schema ---
schema_file_path = os.path.join(os.path.dirname(__file__), 'schema.json')
try:
    with open(schema_file_path, 'r') as f:
        full_schema = json.load(f)

    # We need a resolver to handle the internal $ref pointers like "#/$defs/AgentCard"
    resolver = RefResolver.from_schema(full_schema)
    agent_card_schema = full_schema["$defs"]["AgentCard"]

except FileNotFoundError:
    print(f"Error: schema.json not found at {schema_file_path}")
    full_schema = None
    agent_card_schema = None
except Exception as e:
    print(f"Error loading or parsing schema.json: {e}")
    full_schema = None
    agent_card_schema = None


# --- Helper Functions ---

def extract_query_fields(agent_data):
    """Extracts fields useful for querying from the agent data."""
    query_fields = {}
    # Extract skill tags for array-contains query
    skill_tags = set()
    if 'skills' in agent_data and agent_data['skills']:
        for skill in agent_data['skills']:
            if 'tags' in skill and skill['tags']:
                skill_tags.update(skill['tags'])
    if skill_tags:
        query_fields['_skill_tags'] = list(skill_tags) # Store as list for Firestore

    # Extract skill IDs for array-contains query
    skill_ids = set()
    if 'skills' in agent_data and agent_data['skills']:
        for skill in agent_data['skills']:
             if 'id' in skill:
                 skill_ids.add(skill['id'])
    if skill_ids:
        query_fields['_skill_ids'] = list(skill_ids)

    # Extract capabilities for direct query
    if 'capabilities' in agent_data:
        for cap_name, cap_value in agent_data['capabilities'].items():
             # Store boolean capabilities directly for easy filtering
             if isinstance(cap_value, bool):
                 query_fields[f'_capability_{cap_name}'] = cap_value

    return query_fields


# --- API Routes ---

@app.route('/agents', methods=['POST'])
def register_agent():
    """Registers a new agent."""
    if not db:
        return jsonify({"error": "Firestore client not initialized"}), 500
    if not agent_card_schema:
         return jsonify({"error": "AgentCard schema not loaded"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    agent_data = request.get_json()

    # 1. Validate against the AgentCard schema
    try:
        validate(instance=agent_data, schema=agent_card_schema, resolver=resolver)
    except ValidationError as e:
        return jsonify({
            "error": "Invalid AgentCard data",
            "message": e.message,
            "schema_path": list(e.path),
        }), 400
    except Exception as e:
         return jsonify({"error": "Schema validation failed", "details": str(e)}), 400

    # 2. Add extracted query fields
    query_fields = extract_query_fields(agent_data)
    agent_data_to_store = {**agent_data, **query_fields}

    # 3. Store in Firestore
    try:
        agents_ref = db.collection(COLLECTION_NAME)
        # Let Firestore auto-generate the ID
        update_time, doc_ref = agents_ref.add(agent_data_to_store)
        print(f"Agent registered with ID: {doc_ref.id} at {update_time}")
        # Return the ID and the stored data
        return jsonify({"id": doc_ref.id, **agent_data_to_store}), 201
    except Exception as e:
        print(f"Error adding agent to Firestore: {e}")
        return jsonify({"error": "Failed to register agent"}), 500


@app.route('/agents', methods=['GET'])
def discover_agents():
    """Discovers agents, optionally filtering by capability, skill tag, or skill ID."""
    if not db:
        return jsonify({"error": "Firestore client not initialized"}), 500

    try:
        agents_ref = db.collection(COLLECTION_NAME)
        query = agents_ref  # Start with the base collection reference

        # --- Filtering Logic ---
        capability_filter = request.args.get('capability')
        skill_tag_filter = request.args.get('skill_tag')
        skill_id_filter = request.args.get('skill_id')

        # Apply filters if provided
        if capability_filter:
            # Assumes capability is a boolean flag in the 'capabilities' object
            # Query the extracted field `_capability_<name>`
             query_field_name = f'_capability_{capability_filter}'
             # Simple check for 'true' or '1' for boolean capabilities
             capability_value = request.args.get(f'{capability_filter}_value', 'true').lower() in ['true', '1']
             query = query.where(filter=firestore.FieldFilter(query_field_name, "==", capability_value))
             # Example: /agents?capability=streaming -> filters where _capability_streaming == true
             # Example: /agents?capability=streaming&streaming_value=false -> filters where _capability_streaming == false

        if skill_tag_filter:
            # Use 'array-contains' for tags stored in the '_skill_tags' list
            query = query.where(filter=firestore.FieldFilter("_skill_tags", "array-contains", skill_tag_filter))
            # Example: /agents?skill_tag=summarization

        if skill_id_filter:
             # Use 'array-contains' for IDs stored in the '_skill_ids' list
             query = query.where(filter=firestore.FieldFilter("_skill_ids", "array-contains", skill_id_filter))
             # Example: /agents?skill_id=summarize-doc-v1

        # --- Execute Query and Format Results ---
        agents = []
        # Execute the query (or fetch all if no filters)
        for doc in query.stream():
            agent_data = doc.to_dict()
            agent_data['id'] = doc.id # Add the Firestore document ID
            agents.append(agent_data)

        return jsonify(agents), 200

    except Exception as e:
        print(f"Error querying agents from Firestore: {e}")
        # Be careful about leaking internal error details in production
        return jsonify({"error": "Failed to retrieve agents", "details": str(e)}), 500

# --- Run Flask App ---
# Gunicorn will run the app using the 'app' variable
if __name__ == '__main__':
    # This is for local development only
    # Use 'flask run' or 'python main.py'
    # Cloud Run uses gunicorn specified in Dockerfile/Procfile
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host='0.0.0.0', port=port)