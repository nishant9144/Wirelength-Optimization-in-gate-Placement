# Step 1: Define the input data (parsing the files)

# TC1
# # Gates positions and dimensions from the output file
# gate_positions = {
#     'g1': (0, 0),  # g1 placed at (0, 0)
#     'g2': (3, 0),  # g2 placed at (3, 0)
#     'g3': (0, 3),  # g3 placed at (0, 3)
#     'g4': (5, 0),  # g4 placed at (5, 0)
#     'g5': (3, 4)   # g5 placed at (3, 4)
# }

# # Gate dimensions and relative pin positions from the input file
# gate_pins = {
#     'g1': [(3, 2), (3, 0)],               # g1 pins
#     'g2': [(0, 1), (0, 3), (2, 4), (2, 2)],  # g2 pins
#     'g3': [(3, 4), (3, 3), (0, 2)],       # g3 pins
#     'g4': [(0, 1), (4, 1), (4, 0)],       # g4 pins
#     'g5': [(0, 4), (0, 1), (10, 1), (10, 4)]  # g5 pins
# }

# # Wire connections (pin pairs)
# wires = [
#     ('g1.p1', 'g2.p1'),  # Wire between g1.p1 and g2.p1
#     ('g1.p1', 'g2.p2'),  # Wire between g1.p1 and g2.p2
#     ('g1.p1', 'g3.p3'),  # Wire between g1.p1 and g3.p3
#     ('g2.p3', 'g4.p1'),  # Wire between g2.p3 and g4.p1
#     ('g2.p3', 'g3.p2'),  # Wire between g2.p3 and g3.p2
#     ('g3.p1', 'g4.p2'),  # Wire between g3.p1 and g4.p2
#     ('g3.p1', 'g5.p1'),  # Wire between g3.p1 and g5.p1
#     ('g2.p3', 'g5.p3'),  # Wire between g2.p3 and g5.p3
#     ('g4.p3', 'g5.p4')   # Wire between g4.p3 and g5.p4
# ]

# TC2
# gate_positions = {
#     'g1': (0, 0),
#     'g2': (3 ,2),
#     'g3': (1, 6),
#     'g4': (0, 3)
# }

# gate_pins = {
#     'g1': [(0, 0)],
#     'g2': [(0, 0)],
#     'g3': [(0, 0)],
#     'g4': [(0, 0)]
# }

# wires = [
#     ('g1.p1', 'g2.p1'),
#     ('g2.p1', 'g3.p1'),
#     ('g2.p1', 'g4.p1')
# ]

# taking the inputs from input.txt and output.txt files

def parse_input(input_file_path, output_file_path):
    gate_pins = {}
    wires = []
    gate_positions = {}
    bounding_box = None
    wire_length = None

    # Parse input.txt
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith('g'):
                parts = line.split()
                gate_name = parts[0]
                gate_pins[gate_name] = []
            elif line.startswith('pins'):
                parts = line.split()
                gate_name = parts[1]
                pins = [(int(parts[i]), int(parts[i+1])) for i in range(2, len(parts), 2)]
                gate_pins[gate_name] = pins
            elif line.startswith('wire'):
                parts = line.split()
                wires.append((parts[1], parts[2]))

    # Parse output.txt
    with open(output_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts[0] == 'bounding_box':
                bounding_box = (int(parts[1]), int(parts[2]))
            elif parts[0] == 'wire_length':
                wire_length = int(parts[1])
            elif parts[0].startswith('g'):
                gate_positions[parts[0]] = (int(parts[1]), int(parts[2]))

    return gate_positions, gate_pins, wires, bounding_box, wire_length

# Example usage
input_file_path = 'input.txt'  # Replace with your actual input file path
output_file_path = 'output.txt'  # Replace with your actual output file path
gate_positions, gate_pins, wires, bounding_box, wire_length = parse_input(input_file_path, output_file_path)


# The wires form connections across multiple gates, with several clusters of interconnected pins.


# Step 2: Calculate Absolute Pin Positions
pin_coordinates = {}  # Dictionary to store absolute positions of all pins

for gate, position in gate_positions.items():
    x_gate, y_gate = position
    pins = gate_pins[gate]
    for i, (x_rel, y_rel) in enumerate(pins):
        pin_name = f"{gate}.p{i+1}"  # pin naming follows format gX.pY
        pin_coordinates[pin_name] = (x_gate + x_rel, y_gate + y_rel)

# Step 3: Union-Find Algorithm to Cluster Pins
# Initialize Union-Find structure
parent = {}
rank = {}

# Initialize each pin to be its own parent (own set)
for pin in pin_coordinates.keys():
    parent[pin] = pin
    rank[pin] = 0

# Find function with path compression
def find(pin):
    if parent[pin] != pin:
        parent[pin] = find(parent[pin])
    return parent[pin]

# Union function with rank optimization
def union(pin1, pin2):
    root1 = find(pin1)
    root2 = find(pin2)
    
    if root1 != root2:
        if rank[root1] > rank[root2]:
            parent[root2] = root1
        elif rank[root1] < rank[root2]:
            parent[root1] = root2
        else:
            parent[root2] = root1
            rank[root1] += 1

# Process each wire connection and union connected pins
for wire in wires:
    pin1, pin2 = wire
    union(pin1, pin2)

# Step 4: Group Pins into Clusters
clusters = {}
for pin in pin_coordinates:
    root = find(pin)
    if root not in clusters:
        clusters[root] = []
    clusters[root].append(pin)

# Step 5: Calculate Bounding Boxes and Total Wire Length
total_wire_length = 0

for cluster_pins in clusters.values():
    min_x = min(pin_coordinates[pin][0] for pin in cluster_pins)
    max_x = max(pin_coordinates[pin][0] for pin in cluster_pins)
    min_y = min(pin_coordinates[pin][1] for pin in cluster_pins)
    max_y = max(pin_coordinates[pin][1] for pin in cluster_pins)
    
    # Bounding box width and height
    width = max_x - min_x
    height = max_y - min_y
    
    # Wire length is the perimeter of the bounding box
    wire_length = width + height
    total_wire_length += wire_length

# Step 6: Print the total wire length
print(f"Total wire length: {total_wire_length}")
