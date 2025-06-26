import json
import tkinter as tk
import random

# load json file 
def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Create main window
root = tk.Tk()
root.title("Colored Grid")
root.geometry("500x400")

# Colors list
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'lightblue', 'lightgreen']

# Create frame for grid
grid_frame = tk.Frame(root)
grid_frame.pack(pady=20)


# Create grid of colored boxes showing the input
boxes = []
for row in range(8):
    box_row = []
    for col in range(10):
        color = random.choice(colors)
        # Create a frame with colored background (better than label for solid color)
        box = tk.Frame(grid_frame, bg=color, width=40, height=30, relief="raised", bd=2)
        box.grid(row=row, column=col, padx=1, pady=1)
        box.grid_propagate(False)  # Maintain fixed size
        box_row.append(box)
    boxes.append(box_row)

# create grid representing the output


# Run the application
root.mainloop()