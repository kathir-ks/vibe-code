# Agent Registration and Discovery API

This Flask application provides an API for registering and discovering agents using Google Cloud Firestore as a backend. It includes schema validation using JSON Schema for ensuring data consistency.

## Table of Contents

*   [Prerequisites](#prerequisites)
*   [Setup](#setup)
*   [Running the Application](#running-the-application)
*   [API Endpoints](#api-endpoints)
    *   [Registering an Agent (POST /agents)](#registering-an-agent-post-agents)
    *   [Discovering Agents (GET /agents)](#discovering-agents-get-agents)
*   [JSON Schema Validation](#json-schema-validation)
*   [Error Handling](#error-handling)
*   [Deployment to Google Cloud Run](#deployment-to-google-cloud-run)
*   [Local Development](#local-development)

## Prerequisites

*   **Google Cloud Account:** You'll need a Google Cloud account with billing enabled.
*   **Google Cloud SDK (gcloud):** Installed and configured.  [Installation Instructions](https://cloud.google.com/sdk/docs/install)
*   **Python 3.7+:** Make sure you have Python 3.7 or higher installed.
*   **Virtual Environment (Optional but Recommended):** Use a virtual environment to manage dependencies.
*   **Firestore Database:** A Firestore database instance set up in your Google Cloud project.

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your_repository_url>
    cd <your_repository_directory>
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Authentication:**

    *   **For Google Cloud Run (Deployment):** When deployed to Google Cloud Run, the application automatically detects the necessary credentials. No extra configuration is needed.

    *   **For Local Development:** Authenticate with your Google Cloud account using the `gcloud` CLI:

        ```bash
        gcloud auth application-default login
        ```

5.  **Configure Project ID:**

    *   **For Google Cloud Run (Deployment):** The application reads the `GOOGLE_CLOUD_PROJECT` environment variable, which is automatically set by Cloud Run.

    *   **For Local Development:** You have a few options:

        *   **Option 1 (Recommended):** Set the `GOOGLE_CLOUD_PROJECT` environment variable in your shell:

            ```bash
            export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
            ```

        *   **Option 2:** The code attempts to retrieve your project ID using `gcloud config get-value project`.  Ensure your `gcloud` CLI is initialized (using `gcloud init`) and the correct project is selected.

        *   **Option 3 (Less Recommended):**  Hardcode your project ID in the `main.py` file (look for the `# Example: PROJECT_ID = "your-gcp-project-id"` comment).  **Avoid this in production!**

6.  **Schema File (schema.json):** Ensure that the `schema.json` file is located in the same directory as `main.py`.  This file defines the structure of the AgentCard object.  Example structure:

    ```json
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "title": "Agent Schema",
      "type": "object",
      "$defs": {
        "AgentCard": {
          "type": "object",
          "properties": {
            "name": { "type": "string" },
            "description": { "type": "string" },
            "skills": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "id": { "type": "string" },
                  "name": { "type": "string" },
                  "tags": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["id", "name"]
              }
            },
            "capabilities": {
              "type": "object",
              "properties": {
                "streaming": { "type": "boolean" },
                "data_access": { "type": "boolean" }
              }
            }
          },
          "required": ["name", "description", "skills"]
        }
      }
    }
    ```
    **Note:** Adjust the schema based on the fields you want your `AgentCard` to have.
    It's important the schema is accurate, as it's used for validation before storing any data.

## Running the Application

*   **For Local Development:**

    ```bash
    python main.py
    ```

    Or, using the Flask CLI:

    ```bash
    flask run --host=0.0.0.0 --port=8080
    ```

    The application will be accessible at `http://localhost:8080`.  Make sure port 8080 is available.

*   **For Google Cloud Run:**

    Refer to the [Deployment to Google Cloud Run](#deployment-to-google-cloud-run) section.

## API Endpoints

### Registering an Agent (POST /agents)

This endpoint allows you to register a new agent by sending a JSON payload to the `/agents` endpoint using the POST method. The JSON payload must conform to the `AgentCard` schema defined in `schema.json`.

**Request:**

```
POST /agents
Content-Type: application/json

{
  "name": "My Awesome Agent",
  "description": "This agent does amazing things.",
  "skills": [
    {
      "id": "summarize-doc-v1",
      "name": "Document Summarization",
      "tags": ["summarization", "text processing"]
    },
    {
      "id": "translate-text-v2",
      "name": "Text Translation",
      "tags": ["translation", "text processing"]
    }
  ],
  "capabilities": {
    "streaming": true,
    "data_access": false
  }
}
```

**Response (Success - 201 Created):**

```json
{
  "id": "auto-generated-firestore-id",
  "name": "My Awesome Agent",
  "description": "This agent does amazing things.",
  "skills": [
    {
      "id": "summarize-doc-v1",
      "name": "Document Summarization",
      "tags": ["summarization", "text processing"]
    },
    {
      "id": "translate-text-v2",
      "name": "Text Translation",
      "tags": ["translation", "text processing"]
    }
  ],
  "capabilities": {
    "streaming": true,
    "data_access": false
  },
  "_skill_tags": ["summarization", "text processing", "translation"],
  "_skill_ids": ["summarize-doc-v1", "translate-text-v2"],
  "_capability_streaming": true,
  "_capability_data_access": false
}
```

**Response (Error - 400 Bad Request):**  (Example - Schema Validation Error)

```json
{
  "error": "Invalid AgentCard data",
  "message": "'skills' is a required property",
  "schema_path": []
}
```

### Discovering Agents (GET /agents)

This endpoint allows you to discover agents. You can optionally filter the results based on capabilities, skill tags, or skill IDs using query parameters.

**Request (No Filters):**

```
GET /agents
```

**Request (Filter by Capability):**

```
GET /agents?capability=streaming
```

This will return agents that have the `streaming` capability set to `true`. To filter for `false`:

```
GET /agents?capability=streaming&streaming_value=false
```

**Request (Filter by Skill Tag):**

```
GET /agents?skill_tag=summarization
```

This will return agents that have the "summarization" skill tag.

**Request (Filter by Skill ID):**

```
GET /agents?skill_id=summarize-doc-v1
```

This will return agents that have the "summarize-doc-v1" skill ID.

**Response (Success - 200 OK):**

```json
[
  {
    "id": "firestore-document-id-1",
    "name": "My Awesome Agent",
    "description": "This agent does amazing things.",
    "skills": [
      {
        "id": "summarize-doc-v1",
        "name": "Document Summarization",
        "tags": ["summarization", "text processing"]
      }
    ],
    "capabilities": {
      "streaming": true
    },
    "_skill_tags": ["summarization", "text processing"],
    "_skill_ids": ["summarize-doc-v1"],
    "_capability_streaming": true
  },
  {
    "id": "firestore-document-id-2",
    "name": "Another Great Agent",
    "description": "This agent translates text.",
    "skills": [
      {
        "id": "translate-text-v2",
        "name": "Text Translation",
        "tags": ["translation", "text processing"]
      }
    ],
    "capabilities": {
      "streaming": false
    },
    "_skill_tags": ["translation", "text processing"],
    "_skill_ids": ["translate-text-v2"],
    "_capability_streaming": false
  }
]
```

**Response (Error - 500 Internal Server Error):**

```json
{
  "error": "Failed to retrieve agents",
  "details": "Database connection error."
}
```

## JSON Schema Validation

The application uses JSON Schema to validate the incoming agent data against the `AgentCard` schema defined in `schema.json`. This ensures that the data is consistent and conforms to the expected structure.

*   The schema is loaded during application startup.
*   The `register_agent` function uses the `validate` function from the `jsonschema` library to validate the agent data.
*   If the data is invalid, a `400 Bad Request` error is returned with details about the validation errors.

## Error Handling

The application includes error handling to gracefully handle exceptions and provide informative error messages to the client.

*   If the Firestore client fails to initialize, a `500 Internal Server Error` is returned.
*   If the agent data fails schema validation, a `400 Bad Request` error is returned with details about the validation errors.
*   If an error occurs while adding an agent to Firestore, a `500 Internal Server Error` is returned.
*   If an error occurs while querying agents from Firestore, a `500 Internal Server Error` is returned.

## Deployment to Google Cloud Run

1.  **Create a Dockerfile:**

    Create a `Dockerfile` in the root of your project with the following content:

    ```dockerfile
    FROM python:3.9-slim-buster

    WORKDIR /app

    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    CMD gunicorn --bind :$PORT --workers 1 --threads 8 main:app
    ```

2.  **Build and Push the Docker Image:**

    ```bash
    gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/agent-api
    ```

3.  **Deploy to Cloud Run:**

    ```bash
    gcloud run deploy agent-api \
      --image gcr.io/$GOOGLE_CLOUD_PROJECT/agent-api \
      --platform managed \
      --region us-central1 \
      --allow-unauthenticated
    ```

    Replace `us-central1` with your desired region. Cloud Run will provide you with a URL after the deployment is complete.

## Local Development

When running locally, ensure you have:

*   `gcloud auth application-default login` executed to authenticate.
*   The `GOOGLE_CLOUD_PROJECT` environment variable set correctly.

Remember to update the `schema.json` file to reflect the actual structure of your agent data and adjust the filtering logic in the `discover_agents` function accordingly.
