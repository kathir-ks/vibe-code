#!/usr/bin/env python3

"""
V8 C++ to Rust Converter

This script aims to build a graph representation of a V8 codebase and then use
the Gemini model to convert C++ code blocks to equivalent Rust code,
potentially leveraging the graph for context management in future iterations.
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
GEMINI_MODEL = "gemini-2.0-flash"

# --- Graph Node Data Structures ---

@dataclass
class CodeBlockNode:
    """Represents a specific code block (e.g., class, function, method)."""
    id: str # Unique identifier (e.g., file_path:line_number or file_path:block_name)
    description: str = ""
    nature: str = "" # e.g., "class", "method", "function", "template", "macro"
    code: str = ""
    edges: List[str] = field(default_factory=list) # List of IDs of related CodeBlockNodes or FileNodes/FolderNodes for dependencies

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

def analyze_code_block(file_content: str, file_path: str) -> List[CodeBlockNode]:
    """
    Analyze file content to identify code blocks.
    NOTE: This is a simplified implementation. A proper C++ parser is needed
    for accurate and detailed code block identification and analysis.
    """
    code_blocks: List[CodeBlockNode] = []
    # --- Placeholder Implementation ---
    # In a real scenario, this would use a C++ parser (e.g., libclang)
    # to find classes, functions, methods, etc. and extract their code.
    # For now, let's treat the entire file as a single 'file_content' block,
    # or maybe split by simple markers if possible (though unreliable).

    # Simple approach: Create one code block node per file for this iteration
    block_id = f"{file_path}:file_content"
    code_blocks.append(CodeBlockNode(
        id=block_id,
        description=f"Entire content of file {os.path.basename(file_path)}",
        nature="file_content",
        code=file_content
    ))
    # Add edges to other code blocks/files would go here if dependencies were parsed
    # code_blocks[-1].edges.extend(...)

    logging.debug(f"Identified {len(code_blocks)} code blocks in {file_path} (simplified)")
    return code_blocks

def build_codebase_graph(source_dir: str, limit: Optional[int] = None) -> CodebaseGraph:
    """
    Traverses the codebase directory structure and builds the initial graph.
    Identifies folders, files, and creates placeholder code block nodes.
    """
    graph = CodebaseGraph()
    root_node = CodebaseNode()
    graph.root = root_node
    graph.nodes[root_node.id] = root_node

    cpp_extensions = ['.cc', '.cpp', '.cxx', '.h', '.hpp', '.c']
    processed_files_count = 0

    logging.info(f"Starting graph building from {source_dir}")

    for root, dirs, files in os.walk(source_dir):
        # Create FolderNode for the current directory
        rel_folder_path = os.path.relpath(root, source_dir)
        folder_id = rel_folder_path if rel_folder_path != '.' else 'root_folder'
        folder_node = FolderNode(
            id=folder_id,
            description=f"Directory: {rel_folder_path}",
            folder_path=root
        )
        graph.nodes[folder_node.id] = folder_node

        # Link FolderNode to its parent (CodebaseNode or another FolderNode)
        if folder_id == 'root_folder':
            root_node.edges.append(folder_node.id)
        else:
            parent_folder_path = os.path.dirname(root)
            rel_parent_folder_path = os.path.relpath(parent_folder_path, source_dir)
            parent_folder_id = rel_parent_folder_path if rel_parent_folder_path != '.' else 'root_folder'
            if parent_folder_id in graph.nodes:
                graph.nodes[parent_folder_id].edges.append(folder_node.id)

        # Process files in the current directory
        for file in files:
            if limit is not None and processed_files_count >= limit:
                logging.info(f"Reached file limit ({limit}). Stopping graph building.")
                return graph # Stop processing if limit is reached

            if any(file.endswith(ext) for ext in cpp_extensions):
                file_path = os.path.join(root, file)
                rel_file_path = os.path.relpath(file_path, source_dir)
                file_id = rel_file_path

                file_node = FileNode(
                    id=file_id,
                    description=f"File: {rel_file_path}",
                    file_path=file_path
                )
                graph.nodes[file_node.id] = file_node
                folder_node.edges.append(file_node.id) # Link FileNode to its parent FolderNode

                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_content = f.read()

                    # Analyze file content to find code blocks
                    code_blocks = analyze_code_block(file_content, file_path)
                    for block in code_blocks:
                        graph.nodes[block.id] = block
                        file_node.edges.append(block.id) # Link CodeBlockNode to its parent FileNode

                    processed_files_count += 1
                    if processed_files_count % 100 == 0:
                         logging.info(f"Graph building progress: Processed {processed_files_count} files.")

                except Exception as e:
                    logging.error(f"Error reading or analyzing file {file_path} during graph building: {e}")

    logging.info(f"Graph building complete. Nodes created: {len(graph.nodes)}")
    return graph

def save_graph_to_json(graph: CodebaseGraph, output_path: str):
    """Saves the graph structure to a JSON file."""
    try:
        # Convert dataclass objects to dictionaries for JSON serialization
        graph_data = {
            "root": graph.root.id if graph.root else None,
            "nodes": {node_id: node.__dict__ for node_id, node in graph.nodes.items()}
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        logging.info(f"Graph saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving graph to JSON: {e}")


# --- Conversion Logic (Modified to conceptually use graph) ---

def setup_gemini_api(api_key: str) -> Any:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logging.info(f"Configured Gemini API with model: {GEMINI_MODEL}")
    return genai.GenerativeModel(GEMINI_MODEL)

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

    # Truncate large code blocks to avoid token limits
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
    Save the converted Rust code for a code block. This needs refinement
    to handle how code blocks from a file are reassembled into a Rust file.
    For now, it will save based on the file the block came from.
    """
    if not conversion_result or not conversion_result.get("success", False):
        return

    original_id = conversion_result.get("original_id", "")
    rust_code = conversion_result.get("rust_code", "")

    # Extract original file path from the code block ID (based on current simple ID format)
    parts = original_id.split(':')
    if not parts:
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

    # This needs improvement: appending converted blocks to the same file.
    # A better approach would be to collect all blocks for a file and write once.
    # For demonstration, we append with a separator.
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
    for node_id, node in graph.nodes.items():
        if isinstance(node, FolderNode):
            rel_folder_path = os.path.relpath(node.folder_path, source_dir)
            output_folder_path = os.path.join(output_dir, rel_folder_path)
            os.makedirs(output_folder_path, exist_ok=True)

            mod_path = os.path.join(output_folder_path, "mod.rs")
            if not os.path.exists(mod_path):
                 try:
                    with open(mod_path, 'w', encoding='utf-8') as f:
                        dir_name = os.path.basename(output_folder_path)
                        f.write(f"// Module declarations for converted {dir_name} code\n\n")
                        # Add module declarations for files within this folder later
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
                    with open(mod_path, 'a', encoding='utf-8') as f:
                        for declaration in module_declarations:
                            f.write(declaration + "\n")
                except Exception as e:
                    logging.error(f"Error adding module declarations to {mod_path}: {e}")


def convert_v8_codebase_graph(source_dir: str, output_dir: str, api_key: str, max_workers: int = 4,
                            limit: Optional[int] = None, graph_output_path: str = "codebase_graph.json") -> None:
    """
    Main function to build the graph and then convert code blocks to Rust.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1 & 2: Build the codebase graph ---
    graph = build_codebase_graph(source_dir, limit=limit)
    save_graph_to_json(graph, graph_output_path) # Save the built graph

    # Setup Gemini API
    model = setup_gemini_api(api_key)

    # --- Step 3: Create Rust module structure based on the graph ---
    # This happens BEFORE conversion so files exist to write to.
    create_rust_module_structure(graph, source_dir, output_dir)

    # --- Step 4: Convert code blocks using the graph ---
    logging.info("Starting code block conversion using graph...")
    code_block_nodes_to_convert = [
        node for node_id, node in graph.nodes.items()
        if isinstance(node, CodeBlockNode)
    ]

    logging.info(f"Found {len(code_block_nodes_to_convert)} code blocks to convert.")

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
                    # (This relies on the current simple ID format: file_path:...)
                    original_file_rel_path = block_id.split(':')[0]
                    original_file_node = graph.nodes.get(original_file_rel_path)

                    if original_file_node and isinstance(original_file_node, FileNode):
                         save_code_block_conversion(result, source_dir, output_dir, graph)
                         processed_count += 1
                    else:
                         logging.error(f"Could not find original FileNode for block ID {block_id}. Cannot save conversion.")
                         error_count += 1 # Count as error as saving failed

                else:
                    logging.warning(f"No conversion result for code block {block_id}")
                    error_count += 1

            except Exception as e:
                logging.error(f"Error processing code block {block_id}: {e}")
                error_count += 1

            # Log progress
            total_processed = processed_count + error_count
            if total_processed % 10 == 0:
                 logging.info(f"Conversion progress: {total_processed}/{len(code_block_nodes_to_convert)} code blocks processed")


    logging.info(f"Code block conversion complete. Successfully converted {processed_count} blocks. Errors: {error_count}")
    logging.warning("Note: The converted code blocks for each file are currently appended to a single output file per original C++ file, with separators.")


def main():
    parser = argparse.ArgumentParser(description="Convert V8 C++ Codebase to Rust using a Graph Representation")
    parser.add_argument("--source", required=True, help="Path to V8 C++ source code directory")
    parser.add_argument("--output", required=True, help="Path to output directory for Rust code")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers for conversion")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process during graph building and conversion (for testing)")
    parser.add_argument("--graph-output", default="codebase_graph.json", help="Path to save the generated graph JSON file")

    args = parser.parse_args()

    logging.info(f"Starting V8 C++ to Rust conversion with graph building")
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