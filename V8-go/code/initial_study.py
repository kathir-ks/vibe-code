#!/usr/bin/env python3

"""

The below code is generate by Claude 3.7 Sonnet
"""


"""
V8 Codebase Analyzer

This script walks through a V8 codebase, analyzes each C++ file using the Gemini 2.0 Flash model,
and extracts information about functions, classes, imports, and their logic.
The analysis is logged to individual files with metadata including file paths.
"""

import os
import argparse
import json
import logging
import time
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Any, Optional
import concurrent.futures
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("v8_analyzer.log"),
        logging.StreamHandler()
    ]
)

# Gemini API configuration
GEMINI_MODEL = "gemini-2.0-flash"

def setup_gemini_api(api_key: str) -> None:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logging.info(f"Configured Gemini API with model: {GEMINI_MODEL}")

def analyze_file(file_path: str, model) -> Dict[str, Any]:
    """
    Analyze a single C++ file using Gemini model to extract functions, classes, imports,
    and their logic descriptions.
    
    Args:
        file_path: Path to the C++ file
        model: Configured Gemini model
        
    Returns:
        Dictionary containing the analysis results
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()
        
        # Skip empty files or files that are too large
        if len(file_content) == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            return None
        
        if len(file_content) > 100000:  # Truncate large files to avoid token limits
            logging.warning(f"File {file_path} is large ({len(file_content)} chars). Truncating...")
            file_content = file_content[:100000] + "\n// ... [truncated due to size]"
        
        prompt = f"""
        Analyze this C++ file from the V8 JavaScript engine codebase:
        
        Path: {file_path}
        
        Code content:
        ```cpp
        {file_content}
        ```
        
        Extract and provide the following information in a structured JSON format:
        1. File path
        2. All imports or includes
        3. All classes defined (with their member functions and properties)
        4. All functions defined (with parameters and return types)
        5. For each function and class method, provide a brief explanation of its logic and purpose
        
        Format the output as a valid JSON object with the following structure:
        {{
            "file_path": "path/to/file.cc",
            "imports": ["include1", "include2", ...],
            "classes": [
                {{
                    "name": "ClassName",
                    "properties": ["property1", "property2", ...],
                    "methods": [
                        {{
                            "name": "methodName",
                            "parameters": ["param1", "param2", ...],
                            "return_type": "type",
                            "logic": "Brief explanation of what this method does"
                        }},
                        ...
                    ]
                }},
                ...
            ],
            "functions": [
                {{
                    "name": "functionName",
                    "parameters": ["param1", "param2", ...],
                    "return_type": "type",
                    "logic": "Brief explanation of what this function does"
                }},
                ...
            ]
        }}
        
        Please ensure the JSON output is valid and properly formatted.
        """

        # Add the random delay:
        delay = random.uniform(25, 40)  # Random delay between 15-20 seconds
        logging.info(f"Sleeping for {delay:.2f} seconds after processing {file_path}")
        time.sleep(delay)

        # Make API call to Gemini
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        result_text = response.text
        # Find JSON part in the response
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_text = result_text[json_start:json_end]
            try:
                result = json.loads(json_text)
                result["file_path"] = file_path  # Ensure file path is correct
                return result
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON for {file_path}: {e}")
                return {"file_path": file_path, "error": "JSON parsing error", "raw_response": result_text}
        else:
            logging.error(f"No valid JSON found in response for {file_path}")
            return {"file_path": file_path, "error": "No JSON in response", "raw_response": result_text}
            
    except Exception as e:
        logging.error(f"Error analyzing file {file_path}: {e}")
        return {"file_path": file_path, "error": str(e)}

def save_analysis(analysis_result: Dict[str, Any], output_dir: str) -> None:
    """
    Save the analysis result to a JSON file in the output directory.
    
    Args:
        analysis_result: The analysis result dictionary
        output_dir: Directory to save the output file
    """
    if not analysis_result:
        return
        
    file_path = analysis_result.get("file_path", "unknown_file")
    
    # Get just the path without any leading slashes or drive letters
    # This ensures we don't try to create absolute paths inside the output directory
    file_path = file_path.lstrip('/').lstrip('\\')
    if ':' in file_path:  # Handle Windows paths
        file_path = file_path.split(':', 1)[1].lstrip('/').lstrip('\\')
    
    # Create a relative path structure in the output directory
    relative_path = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, relative_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Create output filename
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_path, f"{file_name}.analysis.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2)
    
    logging.info(f"Saved analysis for {file_path} to {output_file}")

def find_cpp_files(source_dir: str) -> List[str]:
    """
    Find all C++ source files in the given directory.
    
    Args:
        source_dir: Directory to search for C++ files
        
    Returns:
        List of file paths
    """
    cpp_extensions = ['.cc', '.cpp', '.cxx', '.h', '.hpp', '.c']
    cpp_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in cpp_extensions):
                cpp_files.append(os.path.join(root, file))
    
    logging.info(f"Found {len(cpp_files)} C++ files in {source_dir}")
    return cpp_files

def analyze_v8_codebase(source_dir: str, output_dir: str, api_key: str, max_workers: int = 4, 
                        limit: Optional[int] = None) -> None:
    """
    Main function to analyze the V8 codebase.
    
    Args:
        source_dir: Directory containing the V8 source code
        output_dir: Directory to save analysis results
        api_key: Gemini API key
        max_workers: Maximum number of concurrent workers
        limit: Optional limit on the number of files to process
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Gemini API
    setup_gemini_api(api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    
    # Find all C++ files
    cpp_files = find_cpp_files(source_dir)
    
    if limit:
        cpp_files = cpp_files[:limit]
        logging.info(f"Limited processing to {limit} files")
    
    # Process files
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_file, file_path, model): file_path for file_path in cpp_files}
        
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result:
                    save_analysis(result, output_dir)
                    processed_count += 1
                else:
                    logging.warning(f"No result for {file_path}")
                    error_count += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                error_count += 1
            
            # Log progress
            total_processed = processed_count + error_count
            if total_processed % 10 == 0:
                logging.info(f"Progress: {total_processed}/{len(cpp_files)} files processed")
    
    logging.info(f"Analysis complete. Successfully processed {processed_count} files. Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser(description="Analyze V8 JavaScript Engine Codebase")
    parser.add_argument("--source", required=True, help="Path to V8 source code directory")
    parser.add_argument("--output", required=True, help="Path to output directory for analysis")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process (for testing)")
    
    args = parser.parse_args()
    
    logging.info(f"Starting V8 codebase analysis")
    start_time = time.time()
    
    analyze_v8_codebase(
        source_dir=args.source,
        output_dir=args.output,
        api_key=args.api_key,
        max_workers=args.workers,
        limit=args.limit
    )
    
    elapsed_time = time.time() - start_time
    logging.info(f"Analysis completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()