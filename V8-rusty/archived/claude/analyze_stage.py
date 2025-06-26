# V8 Structure Analyzer
# Analyzes the structure of V8 codebase to prepare for C++ to Go migration

import os
import glob
import json
from pathlib import Path
import google.generativeai as genai
from tqdm import tqdm

def configure_gemini(api_key):
    """Configure the Gemini API with your key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

class V8StructureAnalyzer:
    def __init__(self, repo_path, output_dir="structure_analysis", api_key=None):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if not api_key:
            print("Warning: No API key provided. You'll need to set one before analysis.")
            self.model = None
        else:
            self.model = configure_gemini(api_key)
            
        self.file_structure = {}
        self.module_structure = {}
        
    def set_api_key(self, api_key):
        """Set or update the API key"""
        self.model = configure_gemini(api_key)
    
    def _get_file_stats(self):
        """Get statistics about the codebase"""
        extensions = {}
        directories = set()
        
        for file_path in self.repo_path.glob('**/*'):
            if file_path.is_file():
                ext = file_path.suffix
                if ext in extensions:
                    extensions[ext] += 1
                else:
                    extensions[ext] = 1
                    
                parent_dir = str(file_path.parent.relative_to(self.repo_path))
                directories.add(parent_dir)
        
        return {
            "extensions": extensions,
            "directory_count": len(directories),
            "directories": list(directories)
        }
    
    def _sample_files(self, extension, max_files=10):
        """Get a sample of files with a given extension"""
        files = list(self.repo_path.glob(f'**/*{extension}'))
        return files[:min(max_files, len(files))]
    
    def _read_file_content(self, file_path, max_size_mb=1):
        """Read file content if it's not too large"""
        try:
            # Check file size
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > max_size_mb:
                return f"File too large to process ({size_mb:.2f} MB)"
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _prompt_gemini(self, prompt, temperature=0.2):
        """Send a prompt to Gemini"""
        if not self.model:
            raise ValueError("API key not set. Use set_api_key() method.")
            
        try:
            response = self.model.generate_content(prompt, generation_config={
                'temperature': temperature,
                'max_output_tokens': 8192,
            })
            return response.text
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def analyze_file_structure(self):
        """Analyze the overall file structure of the repository"""
        print("Analyzing file structure...")
        
        stats = self._get_file_stats()
        
        # Focus on C++ files
        cpp_extensions = [ext for ext in stats["extensions"].keys() 
                         if ext in ['.cc', '.h', '.cpp', '.hpp']]
        
        cpp_files_count = sum(stats["extensions"].get(ext, 0) for ext in cpp_extensions)
        
        # Sample some key files
        header_samples = self._sample_files('.h', 5)
        impl_samples = self._sample_files('.cc', 5)
        
        header_contents = {}
        for file_path in header_samples:
            rel_path = str(file_path.relative_to(self.repo_path))
            content = self._read_file_content(file_path)
            if not content.startswith("Error") and not content.startswith("File too large"):
                header_contents[rel_path] = content

        impl_contents = {}
        for file_path in impl_samples:
            rel_path = str(file_path.relative_to(self.repo_path))
            content = self._read_file_content(file_path)
            if not content.startswith("Error") and not content.startswith("File too large"):
                impl_contents[rel_path] = content
        
        # Prepare prompt for Gemini
        prompt = f"""
        I need you to analyze the file structure of the V8 JavaScript engine codebase to help with a C++ to Go migration.
        
        Repository statistics:
        - Total C++ files: {cpp_files_count}
        - File extensions: {stats["extensions"]}
        - Number of directories: {stats["directory_count"]}
        
        Key directories (sample):
        {stats["directories"][:20]}
        
        Sample header files:
        """
        
        for file_name, content in header_contents.items():
            sample_content = content[:2000] + ("..." if len(content) > 2000 else "")
            prompt += f"\n--- {file_name} ---\n{sample_content}\n\n"
            
        prompt += "\nSample implementation files:\n"
        
        for file_name, content in impl_contents.items():
            sample_content = content[:2000] + ("..." if len(content) > 2000 else "")
            prompt += f"\n--- {file_name} ---\n{sample_content}\n\n"
            
        prompt += """
        Based on this information, please provide:
        
        1. An overview of the file organization pattern
        2. Identification of the main modules/components based on directory structure
        3. Analysis of the coding conventions and file organization
        4. Identification of potential challenges for Go migration related to code organization
        5. Recommendations for Go package structure based on the C++ organization
        
        Format your response as a structured analysis that will help plan the migration.
        """
        
        # Get analysis from Gemini
        file_structure_analysis = self._prompt_gemini(prompt)
        
        # Save results
        with open(self.output_dir / "file_structure_analysis.md", 'w') as f:
            f.write(file_structure_analysis)
            
        self.file_structure = {
            "stats": stats,
            "analysis": file_structure_analysis
        }
        
        print("File structure analysis complete and saved to", self.output_dir / "file_structure_analysis.md")
        return self.file_structure
    
    def analyze_module_structure(self, top_dirs=10):
        """Analyze the module structure by examining key directories"""
        print(f"Analyzing module structure for top {top_dirs} directories...")
        
        # Get top-level directories
        dirs = [d for d in self.repo_path.glob('*/') if d.is_dir()]
        dirs = sorted(dirs, key=lambda d: len(list(d.glob('**/*'))), reverse=True)
        dirs = dirs[:top_dirs]  # Focus on largest directories
        
        for dir_path in tqdm(dirs, desc="Analyzing directories"):
            rel_path = str(dir_path.relative_to(self.repo_path))
            
            # Get all C++ files in this directory
            cpp_files = list(dir_path.glob('**/*.h')) + list(dir_path.glob('**/*.cc'))
            
            if not cpp_files:
                continue
                
            # Sample files for analysis
            sample_files = cpp_files[:5]
            file_contents = {}
            
            for file_path in sample_files:
                rel_file_path = str(file_path.relative_to(self.repo_path))
                content = self._read_file_content(file_path)
                if not content.startswith("Error") and not content.startswith("File too large"):
                    file_contents[rel_file_path] = content
            
            if not file_contents:
                continue
                
            # Prepare prompt for Gemini
            prompt = f"""
            Analyze this module directory from the V8 JavaScript engine: {rel_path}

            Files in this directory (sample of {len(cpp_files)} total files):
            {[str(f.relative_to(self.repo_path)) for f in sample_files]}

            Sample file contents:
            """
            
            for file_name, content in file_contents.items():
                sample_content = content[:3000] + ("..." if len(content) > 3000 else "")
                prompt += f"\n--- {file_name} ---\n{sample_content}\n\n"
            
            prompt += """
            Based on these files, please:
            1. Describe the purpose of this module
            2. Identify the main classes and their responsibilities
            3. Explain how this module is likely to interface with other modules
            4. Note any C++-specific features that would need special handling in Go
            5. Suggest a Go package structure for this module
            
            Format your response as a structured analysis focused on future Go migration.
            """
            
            # Get analysis from Gemini
            module_analysis = self._prompt_gemini(prompt)
            
            # Save results
            safe_name = rel_path.replace('/', '_').replace('\\', '_')
            with open(self.output_dir / f"{safe_name}_analysis.md", 'w') as f:
                f.write(module_analysis)
                
            self.module_structure[rel_path] = {
                "files_count": len(cpp_files),
                "sample_files": [str(f.relative_to(self.repo_path)) for f in sample_files],
                "analysis": module_analysis
            }
        
        # Generate a summary of all modules
        modules_list = list(self.module_structure.keys())
        
        summary_prompt = f"""
        Based on the analysis of these modules from the V8 JavaScript engine:
        {modules_list}
        
        Please provide:
        1. A high-level overview of how the modules relate to each other
        2. The apparent architectural layers of the system
        3. Key interfaces between modules
        4. Recommendations for Go package organization based on this module structure
        
        Format your response as a structured summary that will help guide the migration from C++ to Go.
        """
        
        module_summary = self._prompt_gemini(summary_prompt)
        
        with open(self.output_dir / "module_structure_summary.md", 'w') as f:
            f.write(module_summary)
            
        with open(self.output_dir / "module_structure.json", 'w') as f:
            # Just save the keys and basic info, not all the content
            simple_structure = {k: {"files_count": v["files_count"]} for k, v in self.module_structure.items()}
            json.dump(simple_structure, f, indent=2)
            
        print("Module structure analysis complete. Results saved to", self.output_dir)
        return self.module_structure
            
    def analyze_structure(self):
        """Run complete structure analysis"""
        if not self.model:
            raise ValueError("API key not set. Use set_api_key() method.")
            
        self.analyze_file_structure()
        self.analyze_module_structure()
        
        print("Structure analysis complete!")
        return {
            "file_structure": self.file_structure,
            "module_structure": self.module_structure
        }

# Usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V8 Structure Analyzer for C++ to Go Migration")
    parser.add_argument("repo_path", help="Path to the V8 repository")
    parser.add_argument("--api-key", help="Google Gemini API key")
    parser.add_argument("--output-dir", default="structure_analysis", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    analyzer = V8StructureAnalyzer(args.repo_path, args.output_dir, args.api_key)
    analyzer.analyze_structure()