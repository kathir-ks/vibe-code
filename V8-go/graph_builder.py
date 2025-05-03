#!/usr/bin/env python3

"""
V8 C++ to Rust Converter

This script aims to build a graph representation of a V8 codebase using
an AI model to identify code blocks, and then use the Gemini model
to convert C++ code blocks to equivalent Rust code, potentially leveraging
the graph for context management in future iterations.
"""

import os
import argparse
import json
import logging
import time
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Any, Optional, Tuple
import concurrent.futures
import random
from dataclasses import dataclass, field
import re # Import regex for parsing AI output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("v8_converter.log"),
        logging.StreamHandler()
    ]
)

# Gemini API configuration
GEMINI_MODEL = "gemini-2.0-flash" # Using Flash for potentially faster/cheaper analysis

# --- Graph Node Data Structures ---

@dataclass
class CodeBlockNode:
    """Represents a specific code block (e.g., class, function, method)."""
    id: str # Unique identifier (e.g., file_path:line_number or file_path:block_name)
    description: str = ""
    nature: str = "" # e.g., "class", "method", "function", "template", "macro", "enum", "struct"
    code: str = "" # The extracted code snippet for this block
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    edges: List[str] = field(default_factory=list) # List of IDs of related CodeBlockNodes or FileNodes/FolderNodes for dependencies (placeholder)

@dataclass
class FileNode:
    """Represents a C++ file."""
    id: str # Unique identifier (e.g., relative_file_path)
    description: str = ""
    file_path: str = "" # Absolute path to the file
    language: str = "C++"
    edges: List[str] = field(default_factory=list) # List of IDs of CodeBlockNodes within this file

@dataclass
class FolderNode:
    """Represents a directory within the codebase."""
    id: str # Unique identifier (e.g., relative_folder_path)
    description: str = ""
    folder_path: str = "" # Absolute path to the folder
    edges: List[str] = field(default_factory=list) # List of IDs of FileNodes within this folder

@dataclass
class CodebaseNode:
    """Represents the entire codebase."""
    id: str = "root"
    description: str = "V8 C++ Codebase"
    metadata: Dict[str, Any] = field(default_factory=lambda: {"languages": ["C++"]})
    edges: List[str] = field(default_factory=list) # List of IDs of FolderNodes at the root level

@dataclass
class CodebaseGraph:
    """Holds the entire graph structure."""
    nodes: Dict[str, Any] = field(default_factory=dict) # Map node ID to node object
    root: Optional[CodebaseNode] = None

# --- Graph Building Logic ---

def setup_gemini_api(api_key: str) -> Any:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logging.info(f"Configured Gemini API with model: {GEMINI_MODEL}")
    return genai.GenerativeModel(GEMINI_MODEL)

def analyze_code_block_with_ai(file_content: str, file_path: str, model) -> List[CodeBlockNode]:
    """
    Analyze file content using the AI model to identify code blocks.
    """
    logging.info(f"Analyzing file {file_path} for code blocks using AI...")
    code_blocks: List[CodeBlockNode] = []
    lines = file_content.splitlines()

    # Truncate large files for AI analysis if necessary
    # The main conversion function will handle larger inputs for the actual conversion
    analysis_content = file_content
    if len(analysis_content) > 50000: # Adjust analysis limit
         logging.warning(f"File {file_path} content too large for analysis, truncating...")
         analysis_content = analysis_content[:50000] + "\n// ... [truncated for analysis]"


    prompt = f"""
    Analyze the following C++ code from the V8 codebase. Identify the major code blocks such as classes, structs, enums, functions, methods, global variables, and macros.

    For each identified block, provide the following information in a structured format, ideally JSON, but a clear list of key-value pairs is acceptable if JSON is not feasible:
    - type: The type of the block (e.g., "class", "function", "method", "enum", "struct", "macro", "global_variable")
    - name: The name of the block (e.g., "Isolate", "NewObject", "HeapObjectTag")
    - description: A brief explanation of its purpose.
    - start_line: The line number (1-based) where the block definition starts.
    - end_line: The line number (1-based) where the block definition ends.

    If a block spans multiple lines (like a function or class), provide the start and end lines of its definition, including the body. For single-line declarations (like global variables or simple macros), start_line and end_line can be the same.

    List the blocks found in the order they appear in the file.

    C++ Code:
    ```cpp
    {analysis_content}
    ```

    Provide the output as a list of blocks. Example format (JSON):
    [
      {{
        "type": "class",
        "name": "ExampleClass",
        "description": "A brief description.",
        "start_line": 10,
        "end_line": 50
      }},
      {{
        "type": "method",
        "name": "ExampleClass::doSomething",
        "description": "Does something important.",
        "start_line": 15,
        "end_line": 45
      }}
      // ... more blocks
    ]

    If JSON is not possible, use a clear list format like:
    - Block 1:
      type: class
      name: ExampleClass
      description: A brief description.
      start_line: 10
      end_line: 50
    - Block 2:
      type: method
      name: ExampleClass::doSomething
      ...
    """

    max_retries = 2 # Fewer retries for analysis to save time/cost
    retry_count = 0

    while retry_count < max_retries:
        try:
            # Add a small delay to space out analysis requests
            delay = random.uniform(5, 15)
            time.sleep(delay)

            response = model.generate_content(prompt)
            ai_output = response.text.strip()

            logging.debug(f"AI analysis output for {file_path}:\n{ai_output}")

            # Attempt to parse JSON first
            try:
                blocks_data = json.loads(ai_output)
                if not isinstance(blocks_data, list):
                     raise ValueError("AI output is not a JSON list")
            except json.JSONDecodeError:
                logging.warning(f"AI output for {file_path} is not valid JSON. Attempting heuristic parsing.")
                # Fallback to heuristic parsing if JSON fails
                blocks_data = parse_ai_output_heuristically(ai_output)
                if not blocks_data:
                     logging.warning(f"Heuristic parsing failed for {file_path}. Skipping code block analysis for this file.")
                     break # Exit retry loop if parsing fails

            # Process parsed block data
            for block_data in blocks_data:
                try:
                    block_type = block_data.get("type", "unknown")
                    block_name = block_data.get("name", "anonymous")
                    block_description = block_data.get("description", "")
                    start_line = block_data.get("start_line")
                    end_line = block_data.get("end_line")

                    if start_line is None or end_line is None:
                         logging.warning(f"Missing start/end lines for a block in {file_path}. Skipping.")
                         continue

                    start_line = int(start_line)
                    end_line = int(end_line)

                    if start_line < 1 or end_line > len(lines) or start_line > end_line:
                         logging.warning(f"Invalid start/end lines ({start_line}-{end_line}) for a block in {file_path}. Skipping.")
                         continue

                    # Extract the code snippet based on line numbers (0-based index for list slicing)
                    code_snippet_lines = lines[start_line - 1:end_line]
                    code_snippet = "\n".join(code_snippet_lines)

                    block_id = f"{file_path}:{block_type}:{block_name}:{start_line}" # More specific ID
                    code_blocks.append(CodeBlockNode(
                        id=block_id,
                        description=block_description,
                        nature=block_type,
                        code=code_snippet,
                        start_line=start_line,
                        end_line=end_line
                    ))

                except (ValueError, KeyError, TypeError) as e:
                    logging.warning(f"Error processing AI block data for {file_path}: {block_data} - {e}")
                    continue # Skip this block but continue with others

            logging.info(f"AI identified {len(code_blocks)} code blocks in {file_path}.")
            return code_blocks # Successfully analyzed

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 5 * retry_count # Linear backoff for analysis retries
                logging.warning(f"AI analysis attempt {retry_count} failed for {file_path}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} AI analysis attempts failed for file {file_path}: {e}")
                break # Exit retry loop after max attempts

    # Fallback: If AI analysis fails, create a single block for the whole file
    logging.warning(f"Falling back to single file block for {file_path} due to AI analysis failure.")
    block_id = f"{file_path}:file_content"
    code_blocks.append(CodeBlockNode(
        id=block_id,
        description=f"Entire content of file {os.path.basename(file_path)} (AI analysis failed)",
        nature="file_content_fallback",
        code=file_content
    ))
    return code_blocks


def parse_ai_output_heuristically(ai_output: str) -> List[Dict[str, Any]]:
    """
    Attempts to parse AI output if it's not valid JSON, looking for key-value pairs.
    This is a basic heuristic and may not work for all AI output formats.
    """
    blocks_data: List[Dict[str, Any]] = []
    current_block: Dict[str, Any] = {}
    block_marker = re.compile(r"^- Block \d+:") # Matches "- Block N:"

    lines = ai_output.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if block_marker.match(line):
            if current_block: # Save previous block if exists
                blocks_data.append(current_block)
            current_block = {} # Start a new block
        else:
            # Try to parse key: value
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                current_block[key] = value
            elif current_block:
                # Append lines to the last seen key's value if it seems like continuation
                # This is very fragile and needs improvement for complex descriptions
                last_key = list(current_block.keys())[-1] if current_block else None
                if last_key:
                    current_block[last_key] += "\n" + line # Append line if no key: value format

    if current_block: # Add the last block
        blocks_data.append(current_block)

    # Convert line numbers to integers
    for block in blocks_data:
        for key in ['start_line', 'end_line']:
            if key in block:
                try:
                    block[key] = int(block[key])
                except (ValueError, TypeError):
                    block[key] = None # Invalid line number

    logging.warning(f"Heuristic parsing extracted {len(blocks_data)} potential blocks.")
    return blocks_data


def build_codebase_graph(source_dir: str, model, limit: Optional[int] = None) -> CodebaseGraph:
    """
    Traverses the codebase directory structure and builds the initial graph
    using AI analysis for code blocks.
    """
    graph = CodebaseGraph()
    root_node = CodebaseNode()
    graph.root = root_node
    graph.nodes[root_node.id] = root_node

    cpp_extensions = ['.cc', '.cpp', '.cxx', '.h', '.hpp', '.c']
    processed_files_count = 0

    logging.info(f"Starting graph building from {source_dir} using AI analysis.")

    # Pre-filter files based on limit if specified
    all_cpp_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in cpp_extensions):
                all_cpp_files.append(os.path.join(root, file))

    if limit is not None:
        all_cpp_files = all_cpp_files[:limit]
        logging.info(f"Limiting graph building to {len(all_cpp_files)} files.")

    for file_path in all_cpp_files:
        if processed_files_count >= len(all_cpp_files):
             # This check is mostly for clarity with the limited list
             break

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                file_content = f.read()

            rel_file_path = os.path.relpath(file_path, source_dir)
            file_id = rel_file_path

            # Ensure parent folders exist in the graph
            current_dir = os.path.dirname(file_path)
            while current_dir != source_dir and current_dir != os.path.dirname(source_dir):
                rel_folder_path = os.path.relpath(current_dir, source_dir)
                folder_id = rel_folder_path if rel_folder_path != '.' else 'root_folder'
                if folder_id not in graph.nodes:
                     graph.nodes[folder_id] = FolderNode(
                         id=folder_id,
                         description=f"Directory: {rel_folder_path}",
                         folder_path=current_dir
                     )
                     # Link to parent folder or root
                     parent_dir = os.path.dirname(current_dir)
                     rel_parent_folder_path = os.path.relpath(parent_dir, source_dir)
                     parent_folder_id = rel_parent_folder_path if rel_parent_folder_path != '.' else 'root_folder'
                     if parent_folder_id not in graph.nodes:
                         # Create parent if it doesn't exist (should happen in walk, but defensive)
                         graph.nodes[parent_folder_id] = FolderNode(
                             id=parent_folder_id,
                             description=f"Directory: {rel_parent_folder_path}",
                             folder_path=parent_dir
                         )
                     graph.nodes[parent_folder_id].edges.append(folder_id)
                     # Ensure root links to top-level folders
                     if parent_folder_id == 'root_folder' and folder_id not in graph.nodes['root'].edges:
                          graph.nodes['root'].edges.append(folder_id)

                current_dir = os.path.dirname(current_dir)

            # Link root to top-level folders if not already linked
            top_level_folder_rel = os.path.dirname(rel_file_path)
            top_level_folder_id = top_level_folder_rel if top_level_folder_rel != '.' else 'root_folder'
            if top_level_folder_id != 'root_folder' and top_level_folder_id not in graph.nodes['root'].edges:
                 graph.nodes['root'].edges.append(top_level_folder_id)


            # Create FileNode
            file_node = FileNode(
                id=file_id,
                description=f"File: {rel_file_path}",
                file_path=file_path
            )
            graph.nodes[file_id] = file_node

            # Link FileNode to its parent FolderNode
            parent_folder_rel = os.path.dirname(rel_file_path)
            parent_folder_id = parent_folder_rel if parent_folder_rel != '.' else 'root_folder'
            if parent_folder_id in graph.nodes:
                graph.nodes[parent_folder_id].edges.append(file_id)
            else:
                 logging.warning(f"Parent folder node {parent_folder_id} not found for file {file_id}. Cannot link.")


            # Analyze file content to find code blocks using AI
            code_blocks = analyze_code_block_with_ai(file_content, file_path, model)
            for block in code_blocks:
                graph.nodes[block.id] = block
                file_node.edges.append(block.id) # Link CodeBlockNode to its parent FileNode

            processed_files_count += 1
            if processed_files_count % 10 == 0: # Log more frequently during AI analysis
                 logging.info(f"Graph building progress: Processed {processed_files_count}/{len(all_cpp_files)} files.")


        except Exception as e:
            logging.error(f"Error processing file {file_path} during graph building: {e}")
            # Even on error, increment count to track progress towards limit
            processed_files_count += 1


    logging.info(f"Graph building complete. Nodes created: {len(graph.nodes)}")
    return graph

def save_graph_to_json(graph: CodebaseGraph, output_path: str):
    """Saves the graph structure to a JSON file."""
    try:
        # Convert dataclass objects to dictionaries for JSON serialization
        # Need to handle the case where a node is a CodebaseNode, FolderNode, etc.
        def node_to_dict(node):
            if isinstance(node, (CodebaseNode, FolderNode, FileNode, CodeBlockNode)):
                return node.__dict__
            return node # Return as is if not a defined node type

        graph_data = {
            "root": graph.root.id if graph.root else None,
            "nodes": {node_id: node_to_dict(node) for node_id, node in graph.nodes.items()}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        logging.info(f"Graph saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving graph to JSON: {e}")

# --- Conversion Logic ---
# The conversion logic largely remains the same as the previous version,
# but it will now iterate over the CodeBlockNodes generated by the AI analysis.

# convert_code_block_with_context and save_code_block_conversion
# functions are used from the previous version.
# Pasting them here for completeness in the full script.

def convert_code_block_with_context(code_block_node: CodeBlockNode, graph: CodebaseGraph, model) -> Dict[str, Any]:
    """
    Convert a single code block to Rust using Gemini model, potentially using
    graph context (currently not fully implemented for context).
    """
    max_retries = 3
    retry_count = 0

    # In a future iteration, this would gather context from the graph
    # based on code_block_node.edges and potentially parent nodes (FileNode, FolderNode).
    # For now, we just use the code within the block node.
    code_to_convert = code_block_node.code
    block_description = code_block_node.description
    original_id = code_block_node.id # Use ID for tracking

    if not code_to_convert or len(code_to_convert.strip()) == 0:
        logging.warning(f"Skipping empty code block: {original_id}")
        return None

    # Truncate large code blocks for conversion if necessary
    if len(code_to_convert) > 50000: # Adjusted limit for code blocks
        logging.warning(f"Code block {original_id} is large ({len(code_to_convert)} chars). Truncating...")
        code_to_convert = code_to_convert[:50000] + "\n// ... [truncated due to size]"


    prompt = f"""
    Convert this C++ code block from the V8 JavaScript engine codebase to equivalent idiomatic Rust code:

    Original C++ code block description: {block_description}
    Original C++ code block ID: {original_id}

    Code content:
    ```cpp
    {code_to_convert}
    ```

    Please convert this C++ code to idiomatic Rust code following these guidelines:

    1. Use appropriate Rust crates for any C++ libraries used.
    2. Convert classes to Rust structs with impl blocks for methods.
    3. Handle memory management appropriately (raw pointers to Box, Arc, Rc, etc.).
    4. Convert C++ templates to Rust generics where applicable.
    5. Transform error handling to use Rust's Result type.
    6. Adapt any preprocessor macros to Rust macro_rules! or const values.
    7. Add appropriate Rust documentation comments.
    8. Include only the converted Rust code in your response, without explanations or notes.
    9. For code that cannot be directly translated, include a commented explanation of what's missing.

    Format your response as pure Rust code (no markdown code blocks, just the raw code).
    """

    while retry_count < max_retries:
        try:
            # Add random delay to prevent rate limiting
            delay = random.uniform(25, 40)  # Random delay between 25-40 seconds
            logging.info(f"Sleeping for {delay:.2f} seconds before processing code block {original_id}")
            time.sleep(delay)

            # Make API call to Gemini
            response = model.generate_content(prompt)
            rust_code = response.text

            # Clean up response to extract just the Rust code
            if "```rust" in rust_code:
                rust_blocks = []
                lines = rust_code.split('\n')
                in_code_block = False
                for line in lines:
                    if line.strip() == "```rust" or line.strip() == "```":
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        rust_blocks.append(line)
                rust_code = '\n'.join(rust_blocks)

            return {
                "original_id": original_id,
                "rust_code": rust_code,
                "success": True
            }

        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                logging.warning(f"Attempt {retry_count} failed for code block {original_id}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} attempts failed when converting code block {original_id}: {e}")
                return {
                    "original_id": original_id,
                    "error": str(e),
                    "success": False
                }

    return {
        "original_id": original_id,
        "error": "Unknown error occurred after all retry attempts",
        "success": False
    }


def save_code_block_conversion(conversion_result: Dict[str, Any], source_dir: str, output_dir: str, graph: CodebaseGraph) -> None:
    """
    Save the converted Rust code for a code block. Appends to the target file.
    """
    if not conversion_result or not conversion_result.get("success", False):
        return

    original_id = conversion_result.get("original_id", "")
    rust_code = conversion_result.get("rust_code", "")

    # Extract original file path from the code block ID (based on current simple ID format)
    # The ID format is now file_path:block_type:block_name:start_line
    parts = original_id.split(':')
    if len(parts) < 1:
         logging.error(f"Invalid code block ID format for saving: {original_id}")
         return
    rel_file_path = parts[0]

    # Create a relative path structure in the output directory
    output_path = os.path.dirname(os.path.join(output_dir, rel_file_path))
    os.makedirs(output_path, exist_ok=True)

    # Create output filename with .rs extension (per original file)
    file_name = os.path.basename(rel_file_path)
    base_name = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_path, f"{base_name}.rs")

    # Append converted blocks to the same file with separators.
    # A more sophisticated approach would collect all blocks for a file before writing.
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n// --- Start of converted block: {original_id} ---\n")
            f.write(rust_code)
            f.write(f"\n// --- End of converted block: {original_id} ---\n")

        logging.info(f"Saved Rust conversion for code block {original_id} (appended to {output_file})")
    except Exception as e:
        logging.error(f"Error saving converted code block {original_id}: {e}")


def create_rust_module_structure(graph: CodebaseGraph, source_dir: str, output_dir: str) -> None:
    """
    Creates the basic Rust module directory structure and mod.rs files
    based on the graph's FolderNodes and FileNodes.
    """
    logging.info("Creating Rust module structure based on graph...")
    processed_dirs = set() # To avoid processing the same directory multiple times

    for node_id, node in graph.nodes.items():
        if isinstance(node, FolderNode):
            rel_folder_path = os.path.relpath(node.folder_path, source_dir)
            output_folder_path = os.path.join(output_dir, rel_folder_path)

            if output_folder_path in processed_dirs:
                 continue
            processed_dirs.add(output_folder_path)

            os.makedirs(output_folder_path, exist_ok=True)

            mod_path = os.path.join(output_folder_path, "mod.rs")
            if not os.path.exists(mod_path):
                 try:
                    with open(mod_path, 'w', encoding='utf-8') as f:
                        dir_name = os.path.basename(output_folder_path)
                        f.write(f"// Module declarations for converted {dir_name} code\n\n")
                 except Exception as e:
                     logging.error(f"Error creating mod.rs at {mod_path}: {e}")

    # After creating directories, add module declarations to mod.rs
    for node_id, node in graph.nodes.items():
        if isinstance(node, FolderNode):
            rel_folder_path = os.path.relpath(node.folder_path, source_dir)
            output_folder_path = os.path.join(output_dir, rel_folder_path)
            mod_path = os.path.join(output_folder_path, "mod.rs")

            if os.path.exists(mod_path):
                module_declarations = []
                for edge_id in node.edges:
                    child_node = graph.nodes.get(edge_id)
                    if isinstance(child_node, FileNode):
                        file_name = os.path.basename(child_node.file_path)
                        module_name = os.path.splitext(file_name)[0]
                        module_declarations.append(f"pub mod {module_name};")

                try:
                    # Read existing content to avoid duplicating declarations if script is run multiple times
                    with open(mod_path, 'r+', encoding='utf-8') as f:
                        content = f.read()
                        f.seek(0, 0) # Go to the beginning of the file
                        for declaration in module_declarations:
                             if declaration not in content: # Only add if not already present
                                  f.write(declaration + "\n")
                        # If you need to append after existing content, seek to end instead
                        # f.seek(0, 2) # Go to the end
                        # for declaration in module_declarations:
                        #      if declaration not in content:
                        #           f.write(declaration + "\n")

                except Exception as e:
                    logging.error(f"Error adding module declarations to {mod_path}: {e}")


def convert_v8_codebase_graph(source_dir: str, output_dir: str, api_key: str, max_workers: int = 4,
                            limit: Optional[int] = None, graph_output_path: str = "codebase_graph.json") -> None:
    """
    Main function to build the graph using AI analysis and then convert code blocks to Rust.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup Gemini API for both analysis and conversion
    model = setup_gemini_api(api_key)

    # --- Step 1 & 2: Build the codebase graph using AI analysis ---
    graph = build_codebase_graph(source_dir, model, limit=limit)
    save_graph_to_json(graph, graph_output_path) # Save the built graph

    # --- Step 3: Create Rust module structure based on the graph ---
    create_rust_module_structure(graph, source_dir, output_dir)

    # --- Step 4: Convert code blocks using the graph ---
    logging.info("Starting code block conversion using graph...")
    code_block_nodes_to_convert = [
        node for node_id, node in graph.nodes.items()
        if isinstance(node, CodeBlockNode) and node.nature != "file_content_fallback" # Optionally skip fallback blocks
    ]

    logging.info(f"Found {len(code_block_nodes_to_convert)} code blocks to convert.")

    # Filter out blocks with no code or empty code after analysis
    code_block_nodes_to_convert = [
        block for block in code_block_nodes_to_convert if block.code and block.code.strip()
    ]
    logging.info(f"Converting {len(code_block_nodes_to_convert)} non-empty code blocks.")


    processed_count = 0
    error_count = 0

    # Using ThreadPoolExecutor to convert code blocks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to the original code block node IDs for tracking
        futures = {
            executor.submit(convert_code_block_with_context, block_node, graph, model): block_node.id
            for block_node in code_block_nodes_to_convert
        }

        for future in concurrent.futures.as_completed(futures):
            block_id = futures[future]
            try:
                result = future.result()
                if result and result.get("success", False):
                    # Find the original file path for saving based on the block ID
                    # The ID format is now file_path:block_type:block_name:start_line
                    parts = block_id.split(':')
                    if parts:
                         original_file_rel_path = parts[0]
                         original_file_node = graph.nodes.get(original_file_rel_path)

                         if original_file_node and isinstance(original_file_node, FileNode):
                              save_code_block_conversion(result, source_dir, output_dir, graph)
                              processed_count += 1
                         else:
                              logging.error(f"Could not find original FileNode for block ID {block_id}. Cannot save conversion.")
                              error_count += 1 # Count as error as saving failed
                    else:
                         logging.error(f"Invalid block ID format {block_id} during conversion result processing.")
                         error_count += 1


                else:
                    logging.warning(f"No successful conversion result for code block {block_id}")
                    error_count += 1

            except Exception as e:
                logging.error(f"Error processing code block {block_id} conversion result: {e}")
                error_count += 1

            # Log progress
            total_processed = processed_count + error_count
            if total_processed % 10 == 0:
                 logging.info(f"Conversion progress: {total_processed}/{len(code_block_nodes_to_convert)} code blocks processed")


    logging.info(f"Code block conversion complete. Successfully converted {processed_count} blocks. Errors: {error_count}")
    logging.warning("Note: The converted code blocks for each file are currently appended to a single output file per original C++ file, with separators.")


def main():
    parser = argparse.ArgumentParser(description="Convert V8 C++ Codebase to Rust using an AI-assisted Graph Representation")
    parser.add_argument("--source", required=True, help="Path to V8 C++ source code directory")
    parser.add_argument("--output", required=True, help="Path to output directory for Rust code")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers for conversion")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process during graph building and conversion (for testing)")
    parser.add_argument("--graph-output", default="codebase_graph.json", help="Path to save the generated graph JSON file")

    args = parser.parse_args()

    logging.info(f"Starting V8 C++ to Rust conversion with AI-assisted graph building")
    start_time = time.time()

    convert_v8_codebase_graph(
        source_dir=args.source,
        output_dir=args.output,
        api_key=args.api_key,
        max_workers=args.workers,
        limit=args.limit,
        graph_output_path=args.graph_output
    )

    elapsed_time = time.time() - start_time
    logging.info(f"Total process completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()