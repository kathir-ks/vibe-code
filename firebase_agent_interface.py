import os
import asyncio
import flask
import functions_framework
import warnings
import logging
from typing import Optional, Dict, Any, List # Added for typing

# Firestore Client Library
from google.cloud import firestore # <-- Import Firestore

from google.adk.agents import Agent
from google.adk.tools import google_search
# --- Session Service Base/Interface (Assume an interface or define required methods) ---
# ADK doesn't explicitly export a base class, but session services need these methods.
# We define our own structure based on expected usage.
from google.adk.sessions import Session, SessionService as ADKSessionService # Use ADK's base if available, otherwise define methods

from google.adk.runners import Runner
from google.genai import types as genai_types

# --- Basic Configuration ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# --- Load API Keys/IDs from Environment Variables ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID")
GOOGLE_CSE_API_KEY = os.environ.get("GOOGLE_CSE_API_KEY", GOOGLE_API_KEY)

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY or ""
os.environ["GOOGLE_CSE_ID"] = GOOGLE_CSE_ID or ""
os.environ["GOOGLE_CSE_API_KEY"] = GOOGLE_CSE_API_KEY or ""
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

# --- Initialize Firestore Client Globally ---
try:
    db = firestore.Client()
    logging.info("Firestore client initialized successfully.")
except Exception as e:
    db = None
    logging.error(f"Failed to initialize Firestore client: {e}", exc_info=True)


# --- !!! START: Firestore Session Service Implementation (Skeleton) !!! ---
# This class needs to be fully implemented to actually persist sessions in Firestore.
# It needs to handle serialization/deserialization of ADK Session objects
# and interact with the Firestore database.

class FirestoreSessionService(ADKSessionService):
    """
    SessionService implementation using Google Cloud Firestore for persistence.

    *** NOTE: This is a SKELETON implementation. ***
    The methods need to be fully implemented to interact with Firestore,
    handle data serialization (e.g., ADK History, State), and manage errors.
    """
    def __init__(self, firestore_client: firestore.Client, collection_name: str = "adk_sessions"):
        """
        Initializes the FirestoreSessionService.

        Args:
            firestore_client: An initialized Firestore client instance.
            collection_name: The name of the Firestore collection to store sessions.
        """
        if not firestore_client:
            raise ValueError("Firestore client must be provided and initialized.")
        self._db = firestore_client
        self._collection_name = collection_name
        self._sessions_ref = self._db.collection(self._collection_name)
        logging.info(f"FirestoreSessionService initialized for collection '{collection_name}'.")

    def _get_doc_id(self, app_name: str, user_id: str, session_id: str) -> str:
        """Creates a unique Firestore document ID for the session."""
        # Use a composite key or hash if needed, ensure Firestore ID constraints are met.
        # Simple concatenation might be too long or contain invalid chars.
        # Example: Use '|' as separator (adjust if needed)
        return f"{app_name}|{user_id}|{session_id}" # Keep IDs clean

    def get_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> Optional[Session]:
        """
        Retrieves a session from Firestore.

        *** Needs Implementation: ***
        - Fetch the document using _get_doc_id.
        - Deserialize the Firestore data back into an ADK Session object
          (including history, state, etc.).
        - Handle cases where the document doesn't exist (return None).
        - Handle potential Firestore errors.
        """
        doc_id = self._get_doc_id(app_name, user_id, session_id)
        doc_ref = self._sessions_ref.document(doc_id)
        logging.info(f"[SKELETON] Attempting to get session from Firestore: {doc_ref.path}")
        try:
            doc = doc_ref.get()
            if doc.exists:
                logging.warning(f"[SKELETON] Document {doc_id} exists, but deserialization not implemented.")
                # !!! IMPLEMENT DESERIALIZATION HERE !!!
                # session_data = doc.to_dict()
                # history = deserialize_history(session_data.get('history')) # Requires helper
                # state = deserialize_state(session_data.get('state'))     # Requires helper
                # return Session(app_name=app_name, user_id=user_id, session_id=session_id, history=history, state=state)
                # For now, return None as if not found or cannot deserialize
                return None # Needs implementation
            else:
                logging.info(f"[SKELETON] Session document {doc_id} not found in Firestore.")
                return None
        except Exception as e:
            logging.error(f"Error getting session {doc_id} from Firestore: {e}", exc_info=True)
            return None # Or raise an error

    def create_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> Session:
        """
        Creates a new session and stores it in Firestore.

        *** Needs Implementation: ***
        - Create a new ADK Session object (likely with empty history/state).
        - Serialize the new Session object into a dictionary suitable for Firestore.
        - Write the dictionary to a new Firestore document using _get_doc_id.
        - Handle potential Firestore errors.
        """
        doc_id = self._get_doc_id(app_name, user_id, session_id)
        doc_ref = self._sessions_ref.document(doc_id)
        logging.info(f"[SKELETON] Attempting to create session in Firestore: {doc_ref.path}")

        # Create the basic ADK Session object
        new_session = Session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            history=[], # Start with empty history
            state={}    # Start with empty state
        )

        # !!! IMPLEMENT SERIALIZATION HERE !!!
        # session_data = {
        #    'app_name': app_name,
        #    'user_id': user_id,
        #    'session_id': session_id,
        #    'created_at': firestore.SERVER_TIMESTAMP,
        #    'updated_at': firestore.SERVER_TIMESTAMP,
        #    'history': serialize_history(new_session.history), # Requires helper
        #    'state': serialize_state(new_session.state)      # Requires helper
        # }
        # For now, just log and create an empty doc or log the intent
        try:
             # Example: Create a minimal doc to show it works structurally
             doc_ref.set({
                 'app_name': app_name,
                 'user_id': user_id,
                 'session_id': session_id,
                 'created_at': firestore.SERVER_TIMESTAMP,
                 'updated_at': firestore.SERVER_TIMESTAMP,
                 'placeholder': 'Serialization not implemented'
             })
             logging.warning(f"[SKELETON] Created placeholder document {doc_id}. Full session data not saved.")
        except Exception as e:
             logging.error(f"Error creating session {doc_id} in Firestore: {e}", exc_info=True)
             # Decide how to handle: raise error or return session anyway?
             # For now, return the session object created in memory.

        return new_session # Return the in-memory object

    def update_session(self, session: Session) -> None:
        """
        Updates an existing session in Firestore.

        *** Needs Implementation: ***
        - Serialize the provided Session object (history, state) into a dictionary.
        - Update the corresponding Firestore document using _get_doc_id.
        - Include an 'updated_at' timestamp.
        - Handle potential Firestore errors.
        """
        doc_id = self._get_doc_id(session.app_name, session.user_id, session.session_id)
        doc_ref = self._sessions_ref.document(doc_id)
        logging.info(f"[SKELETON] Attempting to update session in Firestore: {doc_ref.path}")

        # !!! IMPLEMENT SERIALIZATION HERE !!!
        # session_data = {
        #    'updated_at': firestore.SERVER_TIMESTAMP,
        #    'history': serialize_history(session.history), # Requires helper
        #    'state': serialize_state(session.state)      # Requires helper
        # }
        # For now, just update the timestamp or log the intent
        try:
            doc_ref.update({
                'updated_at': firestore.SERVER_TIMESTAMP,
                'placeholder_update': 'Session state/history update not implemented'
            })
            logging.warning(f"[SKELETON] Updated placeholder document {doc_id}. Full session data not saved.")
        except Exception as e:
            logging.error(f"Error updating session {doc_id} in Firestore: {e}", exc_info=True)
            # Decide if this should raise an error

    # Optional, but good practice:
    def delete_session(self, app_name: str, user_id: str, session_id: str) -> None:
        """
        Deletes a session from Firestore.

        *** Needs Implementation: ***
        - Delete the Firestore document using _get_doc_id.
        - Handle cases where the document doesn't exist.
        - Handle potential Firestore errors.
        """
        doc_id = self._get_doc_id(app_name, user_id, session_id)
        doc_ref = self._sessions_ref.document(doc_id)
        logging.info(f"[SKELETON] Attempting to delete session from Firestore: {doc_ref.path}")
        try:
            doc_ref.delete()
            logging.info(f"[SKELETON] Deleted session document {doc_id} (if it existed).")
        except Exception as e:
            logging.error(f"Error deleting session {doc_id} from Firestore: {e}", exc_info=True)
            # Decide if this should raise an error

# --- !!! END: Firestore Session Service Implementation (Skeleton) !!! ---


# --- Global ADK Setup ---
agent = None
runner = None
session_service = None

# Check for prerequisites including the Firestore client
if db and GOOGLE_API_KEY and GOOGLE_CSE_ID and GOOGLE_CSE_API_KEY:
    try:
        search_tool = google_search
        agent = Agent(
            name="search_assistant_cf_firestore", # Indicate Firestore intent
            model="gemini-2.0-flash", # Using a recommended model
            instruction="You are a helpful assistant running in a cloud function. "
                        "Answer user questions concisely based on your knowledge or by using the 'google_search' tool. "
                        "Try to remember our previous conversation based on the session.",
            description="An assistant that can search the web and aims to remember conversations using Firestore.",
            tools=[search_tool]
        )
        logging.info(f"Agent '{agent.name}' created successfully.")

        # --- Session Service Configuration ---
        # !!! NOW USING THE FirestoreSessionService SKELETON !!!
        try:
            session_service = FirestoreSessionService(firestore_client=db, collection_name="adk_sessions_cf") # Use specific collection
            logging.info("Using FirestoreSessionService (Skeleton) for persistence.")
            logging.warning("!!! FirestoreSessionService is a SKELETON. Actual persistence logic needs implementation. !!!")
        except ValueError as ve:
             logging.error(f"Failed to initialize FirestoreSessionService: {ve}", exc_info=True)
             session_service = None
        except Exception as e:
            logging.error(f"Unexpected error initializing FirestoreSessionService: {e}", exc_info=True)
            session_service = None

        # --- Fallback to InMemory if Firestore init failed ---
        # if not session_service:
        #     logging.warning("!!! Falling back to InMemorySessionService as FirestoreSessionService failed to initialize. Persistence is NOT active. !!!")
        #     from google.adk.sessions import InMemorySessionService
        #     session_service = InMemorySessionService()
        # --------------------------------------------------

        if session_service: # Check if session service was initialized (even if skeleton)
            runner = Runner(
                agent=agent,
                app_name="search_assistant_cf_firestore_app", # Indicate Firestore app
                session_service=session_service # Use the initialized service
            )
            logging.info(f"Runner created for agent '{runner.agent.name}' using {type(session_service).__name__}.")
        else:
             logging.error("Session service could not be initialized (Firestore or Fallback). Runner not created.")

    except Exception as e:
        logging.error(f"Error during ADK setup: {e}", exc_info=True)
        agent = None # Ensure agent is None if setup fails
        runner = None # Ensure runner is None if setup fails
else:
    missing = []
    if not db: missing.append("Firestore client init failed")
    if not GOOGLE_API_KEY: missing.append("GOOGLE_API_KEY")
    if not GOOGLE_CSE_ID: missing.append("GOOGLE_CSE_ID")
    if not GOOGLE_CSE_API_KEY: missing.append("GOOGLE_CSE_API_KEY")
    logging.error(f"Missing prerequisites: {', '.join(missing)}. ADK setup skipped.")


# --- Cloud Function Entry Point ---
@functions_framework.http
def app(request: flask.Request) -> flask.Response:
    """HTTP Cloud Function entry point."""
    # Make sure runner and session_service were successfully initialized globally
    if not runner or not session_service:
        logging.error("ADK runner or session service component not initialized during startup. Check setup logs.")
        return flask.jsonify({"error": "Agent service not configured correctly due to initialization failure"}), 500

    if request.method != "POST":
        return flask.jsonify({"error": "Method not allowed"}), 405

    request_json = request.get_json(silent=True)
    if not request_json or "query" not in request_json or "session_id" not in request_json:
        return flask.jsonify({"error": "Missing 'query' or 'session_id' in JSON payload"}), 400

    user_query = request_json["query"]
    session_id = request_json["session_id"] # Get session_id from request
    user_id = request_json.get("user_id", "cf_firestore_user") # User ID can also come from request

    if not session_id: # Basic validation
         return flask.jsonify({"error": "'session_id' cannot be empty"}), 400

    # Use runner's configured app_name
    app_name = runner.app_name
    logging.info(f"Received query: '{user_query}' for session: {session_id} (App: {app_name}, User: {user_id})")

    # --- Interaction Logic (async) ---
    async def get_agent_response(query: str, user_id_async: str, session_id_async: str, app_name_async: str) -> str:
        """Calls the ADK runner asynchronously, attempting to use the session via the configured SessionService."""
        current_session: Optional[Session] = None # Keep track of the session object

        # The runner *internally* uses session_service.get_session, create_session, update_session
        # We don't need to call them manually here if using runner.run_async correctly.
        # The logic below demonstrates the *intent* but runner handles it.
        # try:
        #     # This would call our FirestoreSessionService skeleton
        #     session = session_service.get_session(
        #         app_name=app_name_async,
        #         user_id=user_id_async,
        #         session_id=session_id_async
        #     )
        #     if not session:
        #         # This would call our FirestoreSessionService skeleton
        #         session = session_service.create_session(
        #             app_name=app_name_async,
        #             user_id=user_id_async,
        #             session_id=session_id_async
        #         )
        #         logging.info(f"Called create_session for {session_id_async} (using {type(session_service).__name__}).")
        #     else:
        #         logging.info(f"Called get_session for {session_id_async} (from {type(session_service).__name__}).")
        #     current_session = session # Store for potential debugging
        #
        # except Exception as e:
        #      logging.error(f"Error explicitly getting/creating session {session_id_async}: {e}", exc_info=True)
        #      return f"Error managing session: {e}. Cannot proceed."


        content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."

        try:
            # Runner uses the SessionService (our Firestore skeleton) to load history/state
            # and automatically calls update_session internally after processing.
            logging.info(f"Calling runner.run_async for session {session_id_async}...")
            async for event in runner.run_async(user_id=user_id_async, session_id=session_id_async, new_message=content):
                # Process events (optional, e.g., log intermediate steps)
                if event.session:
                    current_session = event.session # Update our local ref if needed

                if event.is_final_response():
                    if event.content and event.content.parts:
                        final_response_text = event.content.parts[0].text
                    elif event.actions and event.actions.escalate:
                        final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                    # Make sure session info is logged before breaking
                    if current_session:
                        logging.info(f"Agent final response generated for session {current_session.session_id} (History items: {len(current_session.history)}, State keys: {len(current_session.state)})")
                    else:
                        logging.info(f"Agent final response generated for session {session_id_async} (Session object not captured in event loop).")
                    break # Exit the loop once final response is found

            # Runner should have called session_service.update_session internally by now
            # You might want to add explicit logging inside update_session skeleton to confirm.

            return final_response_text

        except Exception as e:
            logging.error(f"Error during agent execution for session {session_id_async}: {e}", exc_info=True)
            # Log relevant session details if available
            session_details = f"ID={session_id_async}, User={user_id_async}, App={app_name_async}"
            if current_session:
                 session_details += f", History items: {len(current_session.history)}, State keys: {len(current_session.state)}"
            logging.error(f"Session context during error: {session_details}")
            return f"An error occurred during agent processing: {e}"

    # --- Run the async logic ---
    try:
        # Pass the runner's app_name to the async function
        response_text = asyncio.run(get_agent_response(user_query, user_id, session_id, app_name))
        return flask.jsonify({"response": response_text})
    except Exception as e:
         logging.error(f"Error running async task in Cloud Function: {e}", exc_info=True)
         return flask.jsonify({"error": f"Failed to process request: {e}"}), 500