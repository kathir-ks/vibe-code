#!/usr/bin/env python3

"""

The below code is generate by Claude 3.7 Sonnet
"""

"""
V8 Codebase Context Builder

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
import xml.etree.ElementTree as ET
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"v8_context_builder_{time.time()}.log"),
        logging.StreamHandler()
    ]
)

# Gemini API configuration
GEMINI_MODEL = "gemini-2.0-flash"

def setup_gemini_api(api_key: str) -> None:
    """Configure the Gemini API with the provided key."""
    genai.configure(api_key=api_key)
    logging.info(f"Configured Gemini API with model: {GEMINI_MODEL}")

def _parse_code_block_xml_element(element: ET.Element, file_path_for_logging: str) -> Dict[str, Any]:
    """
    Parses an XML element representing a code block (e.g., func, class, imports)
    into a dictionary with 'metadata' and 'code'.
    """
    data = {}
    metadata_el = element.find('metadata')
    if metadata_el is not None and metadata_el.text:
        try:
            # Ensure metadata text is stripped before parsing
            data['metadata'] = json.loads(metadata_el.text.strip())
        except json.JSONDecodeError as e:
            logging.warning(f"File {file_path_for_logging}: Failed to parse JSON in <metadata> of <{element.tag}>: {e}. Raw text: '{metadata_el.text.strip()[:100]}...'")
            data['metadata'] = {"error": "JSON parsing error in metadata", "raw_text": metadata_el.text.strip()}
    else:
        data['metadata'] = {}
        if metadata_el is None:
            logging.debug(f"File {file_path_for_logging}: No <metadata> tag found in <{element.tag}>.")
        elif metadata_el.text is None or not metadata_el.text.strip():
            logging.debug(f"File {file_path_for_logging}: Empty <metadata> tag in <{element.tag}>.")

    code_el = element.find('code')
    if code_el is not None and code_el.text:
        data['code'] = code_el.text.strip() # ElementTree handles CDATA; .text gives its content
    else:
        data['code'] = ""
        if code_el is None:
            logging.debug(f"File {file_path_for_logging}: No <code> tag found in <{element.tag}>.")
        elif code_el.text is None or not code_el.text.strip():
             logging.debug(f"File {file_path_for_logging}: Empty <code> tag in <{element.tag}>.")
    return data

def _parse_file_xml_element(file_element: ET.Element, file_path_for_logging: str) -> Dict[str, Any]:
    """
    Parses the root <file> XML element into a structured dictionary.
    """
    result = {}
    if file_element.tag != 'file':
        logging.warning(f"File {file_path_for_logging}: Expected root XML tag <file>, got <{file_element.tag}>. Attempting to process children directly.")
    
    for child in file_element:
        if child.tag == 'metadata':
            if child.text:
                try:
                    # Ensure metadata text is stripped before parsing
                    result['metadata'] = json.loads(child.text.strip())
                except json.JSONDecodeError as e:
                    logging.warning(f"File {file_path_for_logging}: Failed to parse JSON in <file><metadata>: {e}. Raw text: '{child.text.strip()[:100]}...'")
                    result['metadata'] = {"error": "JSON parsing error in file metadata", "raw_text": child.text.strip()}
            else:
                result['metadata'] = {}
                logging.debug(f"File {file_path_for_logging}: Empty <file><metadata> tag.")
        elif child.tag == 'dependencies':
            deps_list = []
            for dep_item_el in child: 
                item_dict = _parse_code_block_xml_element(dep_item_el, file_path_for_logging)
                item_dict['type'] = dep_item_el.tag 
                deps_list.append(item_dict)
            result['dependencies'] = deps_list
        elif child.tag == 'imports': 
            result['imports'] = _parse_code_block_xml_element(child, file_path_for_logging)
        elif child.tag == 'func':
            result.setdefault('functions', []).append(_parse_code_block_xml_element(child, file_path_for_logging))
        elif child.tag == 'class':
            result.setdefault('classes', []).append(_parse_code_block_xml_element(child, file_path_for_logging))
        elif child.tag == 'interface':
            result.setdefault('interfaces', []).append(_parse_code_block_xml_element(child, file_path_for_logging))
        elif child.tag == 'code': # Handle top-level code block not fitting other categories
            if child.text:
                result['file_level_code_content'] = child.text.strip()
            else:
                result['file_level_code_content'] = ""
            logging.debug(f"File {file_path_for_logging}: Found and parsed top-level <code> block under <file>.")
        else:
            logging.warning(f"File {file_path_for_logging}: Unexpected tag <{child.tag}> in <file> element. Skipping.")
    return result

def build_context(file_path: str, model) -> Optional[Dict[str, Any]]:
    """
    Analyze a single C++ file using Gemini model to extract functions, classes, imports,
    and their logic descriptions.
    
    Args:
        file_path: Path to the C++ file
        model: Configured Gemini model
        
    Returns:
        Dictionary containing the analysis results, or None if skipped.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()
        
        if len(file_content) == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            return None
        
        if len(file_content) > 100000: # Increased limit slightly, can be tuned
            logging.warning(f"File {file_path} is large ({len(file_content)} chars). Truncating to 100000 chars...")
            file_content = file_content[:100000] + "\n// ... [truncated due to size]"
        
        # MODIFIED PROMPT STARTS HERE
        prompt = f"""
                # Role and Purpose
                You are an AI assistant specialized in programming and software engineering. Your primary responsibility is to build comprehensive context for an LLM that performs code migration between programming languages. You will analyze source files, categorize code elements, and create structured representations with metadata.

                # File Analysis Structure
                For each file, create a structured representation using the following format:

                <file>
                    <metadata>
                    {{
                        "path": "path/to/file.ext",
                        "file_name": "file.ext",
                        "language": "language_name",
                        "purpose": "Brief description of the file's purpose and functionality"
                    }}
                    </metadata>
                    <dependencies>
                        // List dependent code elements that are referenced but not defined in this file.
                        // Use <class>, <func>, <interface> etc. tags for these, similar to main content blocks.
                        // Each dependency should have its own <metadata> and <code> block.
                        // The <code> block for dependencies should contain minimal structural information.
                        // Wrap the code in these <code> blocks with <![CDATA[...]]>
                    </dependencies>
                    
                    // File content blocks go here...
                </file>

                # Code Block Types and Structure
                Categorize code into the following block types.
                **Important:** For all `<code>` elements, the content (the actual source code)
                **MUST** be wrapped in a `<![CDATA[...]]>` section to correctly handle special
                characters like `<`, `>`, and `&`. For example:
                `<code><![CDATA[int main() {{ if (x < 10) {{ return 0; }} }}]]></code>`

                ## Import Blocks
                <imports>
                    <metadata>
                    {{
                        "language": "language_name",
                        "purpose": "What functionality these imports provide"
                    }}
                    </metadata>
                    <code><![CDATA[
                        // Import statements go here
                        // Example: #include <vector>
                    ]]></code>
                </imports>

                ## Function Blocks
                <func>
                    <metadata>
                    {{
                        "language": "language_name",
                        "type": "function", // or "method" if part of a class
                        "name": "functionName",
                        "parent": "ParentClass", // Optional: if it's a method
                        "about": "Brief description of what the function does",
                        "logic": "Detailed explanation of algorithm or implementation logic",
                        "parameters": [
                            {{
                                "name": "paramName",
                                "type": "paramType",
                                "purpose": "What this parameter is used for"
                            }}
                        ],
                        "return": {{
                            "type": "returnType",
                            "description": "What is returned"
                        }},
                        "dependencies": [ // List of names of functions, classes, types this function depends on
                            "DependencyName"
                        ]
                    }}
                    </metadata>
                    <code><![CDATA[
                        // Function code goes here
                        // Example: void foo(int x) {{ if (x > 0) return; }}
                    ]]></code>
                </func>

                ## Class Blocks
                <class>
                    <metadata>
                    {{
                        "language": "language_name",
                        "type": "class",
                        "name": "ClassName",
                        "extends": "ParentClass", // Optional
                        "implements": [ // Optional
                            "InterfaceName"
                        ],
                        "about": "Brief description of the class purpose",
                        "attributes": [ // Member variables
                            {{
                                "name": "attributeName",
                                "type": "attributeType",
                                "access": "public/private/protected",
                                "purpose": "What this attribute represents"
                            }}
                        ],
                        "dependencies": [ // List of names of classes, types this class depends on
                            "DependencyName"
                        ]
                    }}
                    </metadata>
                    <code><![CDATA[
                        // Class code goes here (declarations and definitions if applicable)
                        // Example: class MyClass {{ public: int member; MyClass(); }};
                    ]]></code>
                </class>

                ## Interface Blocks
                <interface>
                    <metadata>
                    {{
                        "language": "language_name",
                        "type": "interface", // or "struct" if it acts like one and language uses struct for interfaces
                        "name": "InterfaceName",
                        "extends": [ // Optional
                            "ParentInterface"
                        ],
                        "about": "Brief description of the interface purpose",
                        "methods": [ // Abstract methods or method signatures
                            {{
                                "name": "methodName",
                                "parameters": [
                                    {{
                                        "name": "paramName",
                                        "type": "paramType"
                                    }}
                                ],
                                "return": "returnType",
                                "purpose": "Method purpose"
                            }}
                        ],
                        "dependencies": [
                            "DependencyName"
                        ]
                    }}
                    </metadata>
                    <code><![CDATA[
                        // Interface code goes here (e.g., pure virtual functions in C++)
                        // Example: class IMyInterface {{ virtual void doSomething() = 0; }};
                    ]]></code>
                </interface>

                # Analysis Guidelines
                1. Identify each logical code unit (imports, functions, classes, interfaces, etc.) and place it in the appropriate block type.
                2. Provide detailed metadata as specified for each block type. Ensure JSON in metadata is valid.
                3. Document dependencies between components to facilitate migration. List names in the "dependencies" array within metadata.
                4. Include information about language-specific features that might require special handling (e.g., noted in "logic" or "about" fields).
                5. For complex algorithms, explain the implementation logic clearly in the "logic" field.
                6. Note any potential migration challenges for specific code constructs.
                7. **Crucially, all code content within any `<code>` tag (including those in `<dependencies>`) MUST be wrapped in `<![CDATA[...]]>` sections.** This is vital for correct XML parsing.

                # Example Output
                <file>
                    <metadata>
                    {{
                        "path": "src/compiler/frame.cpp",
                        "file_name": "frame.cpp",
                        "language": "cpp",
                        "purpose": "Implements stack frame management for the compiler"
                    }}
                    </metadata>
                    <dependencies>
                        <class>
                            <metadata>
                            {{
                                "language": "cpp",
                                "type": "class",
                                "name": "AlignedSlotAllocator",
                                "about": "Manages aligned memory allocation in slots"
                            }}
                            </metadata>
                            <code><![CDATA[
                                class AlignedSlotAllocator {{
                                public:
                                    static int NumSlotsForWidth(int width);
                                    void Align(int slot_count);
                                    int Size() const;
                                }};
                            ]]></code>
                        </class>
                    </dependencies>

                    <func>
                        <metadata>
                        {{
                            "language": "cpp",
                            "type": "method",
                            "name": "AlignFrame",
                            "parent": "Frame",
                            "about": "Aligns the stack frame to the specified memory boundary",
                            "logic": "Ensures both return slots and regular slots are aligned to the power-of-2 boundary by adding padding slots as necessary",
                            "parameters": [
                                {{
                                    "name": "alignment",
                                    "type": "int",
                                    "purpose": "The memory alignment boundary (must be a power of 2)"
                                }}
                            ],
                            "return": {{
                                "type": "void",
                                "description": "No return value"
                            }},
                            "dependencies": [
                                "AlignedSlotAllocator",
                                "base::bits"
                            ]
                        }}
                        </metadata>
                        <code><![CDATA[
                            void Frame::AlignFrame(int alignment) {{
                            #if DEBUG
                            spill_slots_finished_ = true;
                            frame_aligned_ = true;
                            #endif
                            // In the calculations below we assume that alignment is a power of 2.
                            DCHECK(base::bits::IsPowerOfTwo(alignment));
                            int alignment_in_slots = AlignedSlotAllocator::NumSlotsForWidth(alignment);

                            // We have to align return slots separately, because they are claimed
                            // separately on the stack.
                            const int mask = alignment_in_slots - 1;
                            int return_delta = alignment_in_slots - (return_slot_count_ & mask);
                            if (return_delta != alignment_in_slots) {{
                                return_slot_count_ += return_delta;
                            }}
                            int delta = alignment_in_slots - (slot_allocator_.Size() & mask);
                            if (delta != alignment_in_slots) {{
                                slot_allocator_.Align(alignment_in_slots);
                                if (spill_slot_count_ != 0) {{
                                spill_slot_count_ += delta;
                                }}
                            }}
                            }}
                        ]]></code>
                    </func>d
                </file>

        Analyze this C++ file from the V8 JavaScript engine codebase:
        
        Path: {file_path}
        
        Code content:
        ```cpp
        {file_content}
        ```
        """
        # MODIFIED PROMPT ENDS HERE

        delay = random.uniform(25, 40) # Consider making this configurable or reducing for faster models
        logging.info(f"Sleeping for {delay:.2f} seconds before processing {file_path}")
        time.sleep(delay)

        response = model.generate_content(prompt)
        # Check for empty or problematic response early
        if not response.candidates or not response.text:
            logging.error(f"No content received from API for {file_path}. Response: {response}")
            return {"file_path": file_path, "error": "No content in API response", "raw_response": str(response)}
        
        result_text = response.text
        
        parsed_result = None
        json_decode_error_str = None

        # Attempt 1: Parse as JSON (should ideally not happen if LLM follows XML format)
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text_candidate = result_text[json_start:json_end]
            try:
                # A simple check: if it looks like XML, don't even try JSON
                if not ("<" in json_text_candidate and ">" in json_text_candidate and ("<file>" in result_text or "<metadata>" in result_text)):
                    parsed_result = json.loads(json_text_candidate)
                    logging.info(f"Successfully parsed response as JSON for {file_path} (unexpected, LLM should return XML)")
                else:
                    # This is the expected path if JSON delimiters are found within XML (e.g. in metadata)
                    json_decode_error_str = "Skipped JSON parsing for whole response due to presence of XML tags; XML is primary."
            except json.JSONDecodeError as e:
                json_decode_error_str = str(e)
                logging.warning(f"Failed to parse extracted text as JSON for {file_path}: {e}. Extracted text: '{json_text_candidate[:200]}...' Attempting XML parsing.")
        else:
            logging.info(f"No top-level JSON object {{...}} found in response for {file_path}. Proceeding with XML parsing as expected.")


        # Attempt 2: Parse as XML (this is the primary expected path)
        if parsed_result is None: # parsed_result will be None unless it was pure JSON
            logging.info(f"Attempting XML parsing for {file_path}.")
            text_to_parse_xml = result_text.strip()
            
            # Regex to match ```optional_lang\nCONTENT\n``` or ```CONTENT```
            fence_match = re.match(r"^```(?:[a-zA-Z0-9_.-]*)?\s*(?:\n)?([\s\S]*?)\s*(?:\n)?```$", text_to_parse_xml, re.DOTALL)
            if fence_match:
                text_to_parse_xml = fence_match.group(1).strip()
                logging.info(f"Stripped markdown code fences for XML parsing for {file_path}")
            else:
                # Check if the whole response is wrapped in <file>...</file> without fences
                if not text_to_parse_xml.startswith("<file>"):
                    file_tag_start = text_to_parse_xml.find("<file>")
                    file_tag_end = text_to_parse_xml.rfind("</file>")
                    if file_tag_start != -1 and file_tag_end != -1 and file_tag_end > file_tag_start:
                        text_to_parse_xml = text_to_parse_xml[file_tag_start : file_tag_end + len("</file>")]
                        logging.info(f"Extracted content between first <file> and last </file> for {file_path}")
                    else:
                        logging.warning(f"XML content for {file_path} does not start with <file> tag and full <file>...</file> block not found after initial stripping. Trying to parse as is but might fail. Content starts with: {text_to_parse_xml[:200]}")
            
            # Sanitize control characters that are invalid in XML 1.0
            # (excluding tab, newline, carriage return which are valid)
            sanitized_xml_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text_to_parse_xml)
            
            if not sanitized_xml_text.strip().startswith("<"):
                logging.error(f"Response for {file_path} (after potential fence stripping and cleanup) is not JSON and does not start with '<', unlikely to be XML. Raw response snippet: {sanitized_xml_text[:200]}")
                return {"file_path": file_path, "error": "Response not JSON and not XML-like after cleanup", 
                        "json_error_if_any": json_decode_error_str, "raw_response": result_text}
            try:
                # Ensure the XML declaration is not present if we are parsing a fragment
                if sanitized_xml_text.startswith("<?xml"):
                    sanitized_xml_text = re.sub(r"^\s*<\?xml[^>]*>\s*", "", sanitized_xml_text)

                root_element = ET.fromstring(sanitized_xml_text)
                if root_element.tag == 'file':
                    parsed_result = _parse_file_xml_element(root_element, file_path)
                    logging.info(f"Successfully parsed response as XML for {file_path}")
                else:
                    logging.error(f"XML root tag is <{root_element.tag}>, expected <file> for {file_path}. Raw response snippet: {result_text[:500]}")
                    return {"file_path": file_path, "error": f"XML root tag not <file> (was <{root_element.tag}>)", 
                            "json_error_if_any": json_decode_error_str, "raw_response": result_text}
            except ET.ParseError as xml_e:
                logging.error(f"Failed to parse response as XML for {file_path}: {xml_e}. Sanitized XML tried: {sanitized_xml_text[:500]}")
                # Attempt to find common invalid characters if CDATA was not used properly
                if "<" in sanitized_xml_text and not "<![CDATA[" in sanitized_xml_text:
                     logging.warning(f"File {file_path}: XML parse error might be due to unescaped '<' or '>' in <code> blocks. LLM might not have used CDATA correctly.")
                return {"file_path": file_path, "error": "JSON and XML parsing failed", 
                        "json_error_if_any": json_decode_error_str, "xml_error": str(xml_e), "raw_response": result_text}
            except Exception as general_xml_e: 
                logging.error(f"An unexpected error occurred during XML processing for {file_path}: {general_xml_e}")
                return {"file_path": file_path, "error": "Unexpected error during XML processing", 
                        "json_error_if_any": json_decode_error_str, "xml_processing_error": str(general_xml_e), "raw_response": result_text}

        if parsed_result:
            if isinstance(parsed_result, dict):
                 parsed_result["file_path"] = file_path # Ensure the script's file_path is authoritative
            else: 
                 logging.error(f"Parsed result for {file_path} is not a dictionary. Type: {type(parsed_result)}")
                 return {"file_path": file_path, "error": "Parsed result not a dictionary", "raw_response": result_text}
            return parsed_result
        else:
            # This case should ideally not be reached if XML parsing is attempted and fails, as it would return an error dict.
            # It might be reached if the response was pure JSON and parsed_result was populated, but then an issue occurred.
            # Or if the initial JSON check decided not to parse as JSON, and XML parsing was also skipped for some reason (logic error).
            logging.error(f"Failed to parse response for {file_path} using any method or result was unexpectedly empty.")
            return {"file_path": file_path, "error": "No parseable content (JSON or XML) or empty result after parsing", 
                    "json_error_if_any": json_decode_error_str, "raw_response": result_text}
            
    except genai.types.generation_types.BlockedPromptException as bpe:
        logging.error(f"Prompt blocked for file {file_path}: {bpe}")
        return {"file_path": file_path, "error": "Prompt blocked by API", "details": str(bpe), "raw_response": "PROMPT_BLOCKED"}
    except genai.types.generation_types.StopCandidateException as sce:
        logging.error(f"Generation stopped unexpectedly for file {file_path}: {sce}")
        return {"file_path": file_path, "error": "Generation stopped by API", "details": str(sce), "raw_response": "GENERATION_STOPPED"}
    except AttributeError as ae: # Catching potential issues with response.text or response.candidates
        logging.error(f"Attribute error during API response handling for {file_path}: {ae}. Likely malformed API response object.")
        return {"file_path": file_path, "error": "Attribute error in API response", "details": str(ae), "raw_response": "ATTRIBUTE_ERROR_IN_RESPONSE_HANDLING"}
    except Exception as e:
        logging.error(f"Generic error analyzing file {file_path}: {e.__class__.__name__} - {e}")
        return {"file_path": file_path, "error": str(e), "raw_response": "GENERIC_ERROR"}


def save_analysis(analysis_result: Dict[str, Any], output_dir: str, file_path_orig: str) -> None:
    """
    Save the analysis result to a JSON file in the output directory.
    
    Args:
        analysis_result: The analysis result dictionary
        output_dir: Directory to save the output file
        file_path_orig: The original file path, used for constructing output path
    """
    if not analysis_result:
        return
    
    # Create a path relative to the source_dir if possible, otherwise use basename
    # This requires source_dir to be passed or known globally, or make file_path_orig relative first
    # For simplicity, current approach uses the original path structure
    
    path_to_use = file_path_orig
    # Attempt to remove drive letter and leading slashes for Windows/Unix paths
    path_to_use = re.sub(r"^[a-zA-Z]:", "", path_to_use)
    path_to_use = path_to_use.lstrip('/').lstrip('\\')
    
    relative_dir = os.path.dirname(path_to_use)
    output_subdir = os.path.join(output_dir, relative_dir)
    os.makedirs(output_subdir, exist_ok=True)
    
    file_name_base = os.path.basename(path_to_use)
    output_file = os.path.join(output_subdir, f"{file_name_base}.context") # Changed extension to .context
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2)
        logging.info(f"Saved analysis for {file_path_orig} to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save analysis for {file_path_orig} to {output_file}: {e}")


def find_cpp_files(source_dir: str) -> List[str]:
    """
    Find all C++ source files in the given directory.
    
    Args:
        source_dir: Directory to search for C++ files
        
    Returns:
        List of file paths
    """
    cpp_extensions = ['.cc', '.cpp', '.cxx', '.h', '.hpp', '.c'] # .c added
    cpp_files = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in cpp_extensions):
                file_path = os.path.join(root, file)
                try:
                    # Basic check for accessibility and if it's a normal file
                    if os.path.isfile(file_path) and os.access(file_path, os.R_OK):
                         # Optional: skip very large files early, though build_context also truncates
                        # if os.path.getsize(file_path) > 500000: # Example: 500KB
                        #    logging.warning(f"Skipping very large file (found by find_cpp_files): {file_path}")
                        #    continue
                        cpp_files.append(file_path)
                    else:
                        logging.warning(f"Skipping non-file or unreadable path: {file_path}")
                except OSError as e:
                    logging.warning(f"OSError while accessing {file_path}: {e}. Skipping.")

    logging.info(f"Found {len(cpp_files)} C++ files in {source_dir}")
    return cpp_files

def build_codebase_context(source_dir: str, output_dir: str, api_key: str, max_workers: int = 4, 
                        limit: Optional[int] = None) -> None:
    """
    Main function to build context codebase.
    
    Args:
        source_dir: Directory containing the V8 source code
        output_dir: Directory to save analysis results
        api_key: Gemini API key
        max_workers: Maximum number of concurrent workers
        limit: Optional limit on the number of files to process
    """
    abs_source_dir = os.path.abspath(source_dir)
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    setup_gemini_api(api_key)
    
    # Configure safety settings to be less restrictive if needed, e.g. for code analysis
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    model = genai.GenerativeModel(GEMINI_MODEL, safety_settings=safety_settings)
    # model = genai.GenerativeModel(GEMINI_MODEL) # Original, if safety settings are not an issue

    cpp_files = find_cpp_files(abs_source_dir)
    
    if not cpp_files:
        logging.warning(f"No C++ files found in {abs_source_dir}. Exiting.")
        return

    if limit:
        cpp_files = cpp_files[:limit]
        logging.info(f"Limited processing to {len(cpp_files)} files")
    
    processed_count = 0
    error_count = 0
    skipped_empty_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Pass abs_source_dir to save_analysis for consistent relative pathing of errors
        futures = {executor.submit(build_context, file_path, model): file_path for file_path in cpp_files}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            file_path_key = futures[future]
            try:
                result = future.result() # This can raise exceptions from build_context
                if result: # result is a dict
                    if "error" not in result:
                        save_analysis(result, abs_output_dir, file_path_key) # file_path_key is absolute here
                        processed_count += 1
                    else: # Error dictionary from build_context
                        logging.error(f"Analysis for {file_path_key} resulted in an error: {result.get('error')}. Details: {result.get('details', '')}. Raw response snippet: {str(result.get('raw_response', ''))[:200]}")
                        error_output_dir = os.path.join(abs_output_dir, "errors")
                        os.makedirs(error_output_dir, exist_ok=True)
                        
                        relative_path = os.path.relpath(file_path_key, abs_source_dir)
                        sanitized_error_filename = relative_path.replace(os.sep, '_').replace(':', '_')
                        error_file_path = os.path.join(error_output_dir, f"{sanitized_error_filename}.error.json")
                        
                        try:
                            with open(error_file_path, 'w', encoding='utf-8') as ef:
                               json.dump(result, ef, indent=2)
                            logging.info(f"Saved error report for {file_path_key} to {error_file_path}")
                        except Exception as e_save:
                            logging.error(f"Failed to save error report for {file_path_key} to {error_file_path}: {e_save}")
                        error_count += 1
                else: # build_context returned None (e.g., for an empty file)
                    logging.warning(f"No result for {file_path_key} (e.g. skipped empty file by build_context).")
                    skipped_empty_count +=1
            except Exception as e: # Exception from future.result() itself (e.g., unhandled in build_context)
                logging.error(f"Critical error processing result for {file_path_key}: {e.__class__.__name__} - {e}")
                error_count += 1
                # Save a basic error report for this case as well
                error_output_dir = os.path.join(abs_output_dir, "errors")
                os.makedirs(error_output_dir, exist_ok=True)
                relative_path = os.path.relpath(file_path_key, abs_source_dir)
                sanitized_error_filename = relative_path.replace(os.sep, '_').replace(':', '_')
                error_file_path = os.path.join(error_output_dir, f"{sanitized_error_filename}.future_error.json")
                try:
                    with open(error_file_path, 'w', encoding='utf-8') as ef:
                        json.dump({"file_path": file_path_key, "error": "Exception in future processing", "details": str(e)}, ef, indent=2)
                    logging.info(f"Saved future processing error report for {file_path_key} to {error_file_path}")
                except Exception as e_save_future:
                    logging.error(f"Failed to save future processing error report for {file_path_key}: {e_save_future}")

            
            total_attempted = i + 1
            if total_attempted % 10 == 0 or total_attempted == len(cpp_files):
                logging.info(f"Progress: {total_attempted}/{len(cpp_files)} files attempted (Successful: {processed_count}, Errors: {error_count}, Skipped Empty: {skipped_empty_count})")
    
    logging.info(f"Analysis complete. Total files found: {len(cpp_files)}. Successfully processed: {processed_count}. Errors: {error_count}. Skipped (empty/no result): {skipped_empty_count}")


def main():
    parser = argparse.ArgumentParser(description="Analyze V8 JavaScript Engine Codebase")
    parser.add_argument("--source", required=True, help="Path to V8 source code directory")
    parser.add_argument("--output", required=True, help="Path to output directory for analysis")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--workers", type=int, default=4, help="Maximum number of concurrent workers (default: 4)")
    parser.add_argument("--limit", type=int, help="Limit the number of files to process (for testing)")
    
    args = parser.parse_args()
    
    logging.info(f"Starting V8 codebase analysis with source: {args.source}, output: {args.output}")
    start_time = time.time()
    
    build_codebase_context(
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