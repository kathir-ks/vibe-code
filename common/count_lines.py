import os
import argparse

# --- Configuration ---
# Add or remove file extensions you want to count.
CODE_EXTENSIONS = {
    '.py', '.java', '.c', '.h', '.cpp', '.hpp', '.js', '.ts', '.html', '.css',
    '.scss', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.kts', '.scala',
    '.m', '.mm', '.sh', '.ps1', '.bat'
}

# Add directory names you want to completely ignore.
IGNORE_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.venv', 'build', 'dist',
    'target', 'out', 'bin'
}
# --- End Configuration ---

def count_lines_in_file(file_path):
    """Counts the number of non-empty lines in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Count lines that are not just whitespace
            return len([line for line in f if line.strip()])
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return 0
    except Exception as e:
        print(f"An unexpected error occurred with file {file_path}: {e}")
        return 0

def count_lines_in_repo(root_dir):
    """
    Recursively walks through a directory and counts non-empty lines in code files.
    """
    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found at '{root_dir}'")
        return

    total_line_count = 0
    total_file_count = 0
    
    print(f"Starting line count in '{os.path.abspath(root_dir)}'...")

    for root, dirs, files in os.walk(root_dir, topdown=True):
        # Modify dirs in-place to prevent `os.walk` from descending into them
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            # Check if the file has one of the desired extensions
            if any(file.endswith(ext) for ext in CODE_EXTENSIONS):
                file_path = os.path.join(root, file)
                line_count = count_lines_in_file(file_path)
                if line_count > 0:
                    total_line_count += line_count
                    total_file_count += 1

    print("\n--- Line Count Summary ---")
    print(f"Total files scanned: {total_file_count}")
    print(f"Total non-empty lines of code: {total_line_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the non-empty lines of code in a repository.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help="The root directory of the repository to scan.\n"
             "Defaults to the current directory if not provided."
    )
    args = parser.parse_args()
    
    count_lines_in_repo(args.directory)
