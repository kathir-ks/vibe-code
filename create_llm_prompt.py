import os

# 🔧 CONFIGURE THIS
root_dir = 'C:\\Users\\HP\\Documents\\Development\\adk-python'  # <- replace with your folder
output_file = 'combined_output.txt'
allowed_extensions = ['.py', '.txt', '.md'] 

# 📎 Separator between files
def file_separator(file_path):
    return f"\n\n===== START OF FILE: {file_path} =====\n\n"

def walk_and_write(root_dir, output_file, allowed_extensions=None):
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for foldername, subfolders, filenames in os.walk(root_dir):
            for filename in filenames:
                full_path = os.path.join(foldername, filename)

                # 🌐 Skip if file doesn't match allowed extensions
                if allowed_extensions and not any(filename.endswith(ext) for ext in allowed_extensions):
                    continue

                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        out_file.write(file_separator(full_path))
                        out_file.write(content)
                except Exception as e:
                    out_file.write(file_separator(full_path))
                    out_file.write(f"[Could not read file: {e}]")

    print(f"\n✅ Combined file created at: {os.path.abspath(output_file)}")

# 🏃 Run it!
walk_and_write(root_dir, output_file, allowed_extensions)
