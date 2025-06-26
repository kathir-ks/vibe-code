import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class GridViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Grid Input-Output Viewer")
        self.root.geometry("1200x800")
        
        # Data storage
        self.data = None
        self.current_pair_index = 0
        self.current_problem_key = None
        self.problem_keys = []
        
        # Color mapping (adjust as needed based on your data)
        self.color_map = {
            0: 'white',
            1: 'red',
            2: 'blue', 
            3: 'green',
            4: 'yellow',
            5: 'purple',
            6: 'orange',
            7: 'pink',
            8: 'gray',
            9: 'brown'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top frame for file loading and navigation
        top_frame = tk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File loading
        tk.Button(top_frame, text="Load JSON File", command=self.load_file, 
                 bg='lightblue', font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        # Problem selector
        problem_frame = tk.Frame(top_frame)
        problem_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(problem_frame, text="Problem:", font=('Arial', 10)).pack(side=tk.LEFT)
        self.problem_var = tk.StringVar()
        self.problem_combo = ttk.Combobox(problem_frame, textvariable=self.problem_var, 
                                         state="readonly", width=15)
        self.problem_combo.pack(side=tk.LEFT, padx=5)
        self.problem_combo.bind('<<ComboboxSelected>>', self.on_problem_change)
        
        # Navigation frame
        nav_frame = tk.Frame(top_frame)
        nav_frame.pack(side=tk.LEFT)
        
        tk.Button(nav_frame, text="◀ Previous", command=self.prev_pair, 
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        self.pair_label = tk.Label(nav_frame, text="No data loaded", 
                                  font=('Arial', 10, 'bold'))
        self.pair_label.pack(side=tk.LEFT, padx=10)
        
        tk.Button(nav_frame, text="Next ▶", command=self.next_pair, 
                 font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        
        # Grids container
        grids_frame = tk.Frame(main_frame)
        grids_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input grid frame
        input_frame = tk.LabelFrame(grids_frame, text="INPUT", font=('Arial', 12, 'bold'))
        input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.input_grid_frame = tk.Frame(input_frame)
        self.input_grid_frame.pack(pady=10)
        
        # Output grid frame  
        output_frame = tk.LabelFrame(grids_frame, text="OUTPUT", font=('Arial', 12, 'bold'))
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        self.output_grid_frame = tk.Frame(output_frame)
        self.output_grid_frame.pack(pady=10)
        
        # Test section (if available)
        test_frame = tk.LabelFrame(main_frame, text="TEST INPUT", font=('Arial', 12, 'bold'))
        test_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.test_grid_frame = tk.Frame(test_frame)
        self.test_grid_frame.pack(pady=10)
        
    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.data = json.load(file)
                
                # Handle the new JSON structure where each key contains train/test data
                if isinstance(self.data, dict):
                    self.problem_keys = list(self.data.keys())
                    if self.problem_keys:
                        # Update problem selector
                        self.problem_combo['values'] = self.problem_keys
                        self.problem_combo.set(self.problem_keys[0])
                        
                        # Load first problem
                        self.current_problem_key = self.problem_keys[0]
                        self.load_current_problem()
                        
                        messagebox.showinfo("Success", f"Loaded {len(self.problem_keys)} problems")
                    else:
                        messagebox.showwarning("Warning", "No problems found in the file")
                else:
                    messagebox.showerror("Error", "Expected JSON object with problem keys")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def load_current_problem(self):
        if not self.current_problem_key or self.current_problem_key not in self.data:
            return
            
        problem_data = self.data[self.current_problem_key]
        self.train_data = problem_data.get('train', [])
        self.test_data = problem_data.get('test', [])
        
        self.current_pair_index = 0
        self.display_current_pair()
    
    def on_problem_change(self, event=None):
        selected_key = self.problem_var.get()
        if selected_key and selected_key in self.data:
            self.current_problem_key = selected_key
            self.load_current_problem()
    
    def create_grid(self, parent, grid_data, max_size=20):
        # Clear existing grid
        for widget in parent.winfo_children():
            widget.destroy()
            
        if not grid_data:
            tk.Label(parent, text="No data", font=('Arial', 10)).pack()
            return
            
        rows = len(grid_data)
        cols = len(grid_data[0]) if rows > 0 else 0
        
        # Calculate cell size based on grid size
        cell_size = max(15, min(30, max_size // max(rows, cols)))
        
        for r in range(rows):
            for c in range(cols):
                value = grid_data[r][c]
                color = self.color_map.get(value, 'lightgray')
                
                cell = tk.Frame(parent, bg=color, width=cell_size, height=cell_size, 
                               relief="solid", bd=1)
                cell.grid(row=r, column=c, padx=0, pady=0)
                cell.grid_propagate(False)
                
                # Add number label for clarity (optional)
                if cell_size >= 20:
                    label = tk.Label(cell, text=str(value), bg=color, 
                                   font=('Arial', max(6, cell_size//4)))
                    label.place(relx=0.5, rely=0.5, anchor='center')
    
    def display_current_pair(self):
        if not hasattr(self, 'train_data') or not self.train_data:
            return
            
        if self.current_pair_index >= len(self.train_data):
            self.current_pair_index = 0
            
        current_data = self.train_data[self.current_pair_index]
        
        # Update pair label
        self.pair_label.config(text=f"Example {self.current_pair_index + 1} of {len(self.train_data)}")
        
        # Display input grid
        input_data = current_data.get('input', [])
        self.create_grid(self.input_grid_frame, input_data)
        
        # Display output grid
        output_data = current_data.get('output', [])
        self.create_grid(self.output_grid_frame, output_data)
        
        # Display test data if available
        if self.test_data and len(self.test_data) > 0:
            # Show the first test case (or you could add navigation for multiple test cases)
            test_input = self.test_data[0].get('input', [])
            self.create_grid(self.test_grid_frame, test_input)
        else:
            # Clear test grid
            for widget in self.test_grid_frame.winfo_children():
                widget.destroy()
            tk.Label(self.test_grid_frame, text="No test data for this problem", 
                    font=('Arial', 10)).pack()
    
    def next_pair(self):
        if hasattr(self, 'train_data') and self.train_data and self.current_pair_index < len(self.train_data) - 1:
            self.current_pair_index += 1
            self.display_current_pair()
    
    def prev_pair(self):
        if hasattr(self, 'train_data') and self.train_data and self.current_pair_index > 0:
            self.current_pair_index -= 1
            self.display_current_pair()

# Create and run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = GridViewer(root)
    
    # Load the paste.txt file automatically if it exists
    try:
        with open('paste.txt', 'r') as file:
            content = file.read()
            # Try to parse as JSON
            data = json.loads(content)
            app.data = data
            
            # Handle the new JSON structure
            if isinstance(data, dict):
                app.problem_keys = list(data.keys())
                if app.problem_keys:
                    app.problem_combo['values'] = app.problem_keys
                    app.problem_combo.set(app.problem_keys[0])
                    app.current_problem_key = app.problem_keys[0]
                    app.load_current_problem()
                    print(f"Auto-loaded {len(app.problem_keys)} problems from paste.txt")
            
    except (FileNotFoundError, json.JSONDecodeError):
        print("Could not auto-load paste.txt - use Load JSON File button")
    
    root.mainloop()