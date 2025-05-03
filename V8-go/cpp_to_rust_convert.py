#!/usr/bin/env python3

"""
V8 C++ to Rust Converter

This script walks through a V8 codebase, takes each C++ file, and uses the Gemini 2.0 Flash model
to convert the C++ code to equivalent Rust code. The converted Rust code is saved to a different
destination folder while preserving the directory structure.
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

def setup_gemini_api(api_key: str) -> None:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logging.info(f"Configured Gemini API with model: {GEMINI_MODEL}")

def convert_file(file_path: str, model) -> Dict[str, Any]:
    """
    Convert a single C++ file to Rust using Gemini model with up to 3 retry attempts.
    
    Args:
        file_path: Path to the C++ file
        model: Configured Gemini model
        
    Returns:
        Dictionary containing the conversion results
    """
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
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
            
            # Determine if it's a header or implementation file
            is_header = file_path.endswith(('.h', '.hpp', '.hxx'))
            file_type = "header" if is_header else "implementation"
            
            prompt = f"""
            Convert this C++ file from the V8 JavaScript engine codebase to equivalent Rust code:
            
            Original C++ file path: {file_path}
            This is a C++ {file_type} file.
            
            Code content:
            ```cpp
            {file_content}
            ```
            
            Please convert this C++ code to idiomatic Rust code following these guidelines:
            
            1. Use appropriate Rust crates for any C++ libraries used
            2. For header files (.h, .hpp), create appropriate Rust module definitions and public interfaces
            3. Convert classes to Rust structs with impl blocks for methods
            4. Handle memory management appropriately (raw pointers to Box, Arc, Rc, etc.)
            5. Convert C++ templates to Rust generics where applicable
            6. Transform error handling to use Rust's Result type
            7. Adapt any preprocessor macros to Rust macro_rules! or const values
            8. Add appropriate Rust documentation comments
            9. Include only the converted Rust code in your response, without explanations or notes
            10. For code that cannot be directly translated, include a commented explanation of what's missing

            Format your response as pure Rust code (no markdown code blocks, just the raw code).
            """

            # Add random delay to prevent rate limiting
            delay = random.uniform(25, 40)  # Random delay between 25-40 seconds
            logging.info(f"Sleeping for {delay:.2f} seconds after processing {file_path}")
            time.sleep(delay)

            # Make API call to Gemini
            response = model.generate_content(prompt)
            rust_code = response.text
            
            # Clean up response to extract just the Rust code
            # Remove any markdown code blocks if present
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
                "original_path": file_path,
                "rust_code": rust_code,
                "success": True
            }
                
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # Exponential backoff: 2, 4, 8 seconds
                logging.warning(f"Attempt {retry_count} failed for {file_path}: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"All {max_retries} attempts failed when converting file {file_path}: {e}")
                return {
                    "original_path": file_path,
                    "error": str(e),
                    "success": False
                }
    
    # This should not be reached due to the return statements above, but added for completeness
    return {
        "original_path": file_path,
        "error": "Unknown error occurred after all retry attempts",
        "success": False
    }

def save_conversion(conversion_result: Dict[str, Any], source_dir: str, output_dir: str) -> None:
    """
    Save the converted Rust code to a file in the output directory.
    
    Args:
        conversion_result: The conversion result dictionary
        source_dir: Original source directory (for path calculations)
        output_dir: Directory to save the output file
    """
    if not conversion_result or not conversion_result.get("success", False):
        return
        
    original_path = conversion_result.get("original_path", "")
    
    # Create a relative path structure in the output directory
    rel_path = os.path.relpath(original_path, source_dir)
    output_path = os.path.dirname(os.path.join(output_dir, rel_path))
    os.makedirs(output_path, exist_ok=True)
    
    # Create output filename with .rs extension
    file_name = os.path.basename(original_path)
    base_name = os.path.splitext(file_name)[0]
    output_file = os.path.join(output_path, f"{base_name}.rs")
    
    # Save the Rust code
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(conversion_result.get("rust_code", ""))
    
    logging.info(f"Saved Rust conversion for {original_path} to {output_file}")

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

def convert_v8_codebase(source_dir: str, output_dir: str, api_key: str, max_workers: int = 4, 
                       limit: Optional[int] = None) -> None:
    """
    Main function to convert the V8 C++ codebase to Rust.
    
    Args:
        source_dir: Directory containing the V8 source code
        output_dir: Directory to save Rust conversion results
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
    
    # Create mod.rs files in output directories to ensure proper Rust module structure
    logging.info("Creating Rust module structure...")
    rust_dirs = set()
    for cpp_file in cpp_files:
        rel_path = os.path.relpath(cpp_file, source_dir)
        output_path = os.path.dirname(os.path.join(output_dir, rel_path))
        rust_dirs.add(output_path)
    
    for rust_dir in rust_dirs:
        os.makedirs(rust_dir, exist_ok=True)
        mod_path = os.path.join(rust_dir, "mod.rs")
        if not os.path.exists(mod_path):
            with open(mod_path, 'w', encoding='utf-8') as f:
                # Get the directory name
                dir_name = os.path.basename(rust_dir)
                f.write(f"// Module declarations for converted {dir_name} code\n\n")
                # We'll fill this with actual module declarations after conversion
    
    # Process files
    processed_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_file, file_path, model): file_path for file_path in cpp_files}
        
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                result = future.result()
                if result and result.get("success", False):
                    save_conversion(result, source_dir, output_dir)
                    processed_count += 1
                else:
                    logging.warning(f"No conversion result for {file_path}")
                    error_count += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                error_count += 1
            
            # Log progress
            total_processed = processed_count + error_count
            if total_processed % 10 == 0:
                logging.info(f"Progress: {total_processed}/{len(cpp_files)} files processed")
    
    # Update mod.rs files with actual module declarations
    logging.info("Updating Rust module structure...")
    for rust_dir in rust_dirs:
        mod_rs_path = os.path.join(rust_dir, "mod.rs")
        if os.path.exists(mod_rs_path):
            # Find all .rs files in the directory (excluding mod.rs)
            module_declarations = []
            for file in os.listdir(rust_dir):
                if file.endswith('.rs') and file != 'mod.rs':
                    module_name = os.path.splitext(file)[0]
                    module_declarations.append(f"pub mod {module_name};")
            
            # Update the mod.rs file with proper module declarations
            with open(mod_rs_path, 'a', encoding='utf-8') as f:
                for declaration in module_declarations:
                    f.write(declaration + "\n")
    
    logging.info(f"Conversion complete. Successfully converted {processed_count} files. Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser(description="Convert V8 C++ Codebase to Rust")
    parser.add_argument("--source", required=True, help="Path to V8 C++ source code directory")
    parser.add_argument("--output", required=True, help="Path to output directory for Rust code")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process (for testing)")
    
    args = parser.parse_args()
    
    logging.info(f"Starting V8 C++ to Rust conversion")
    start_time = time.time()
    
    convert_v8_codebase(
        source_dir=args.source,
        output_dir=args.output,
        api_key=args.api_key,
        max_workers=args.workers,
        limit=args.limit
    )
    
    elapsed_time = time.time() - start_time
    logging.info(f"Conversion completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()