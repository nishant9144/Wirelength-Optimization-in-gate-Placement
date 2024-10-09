import random
import math
from tqdm import tqdm # used for progress bar (debugging)


class Pin:
    def __init__(self, pin_name, x, y):
        self.pin_name = pin_name
        self.x = x
        self.y = y

class Gate:
    def __init__(self, name, width, height, pins):
        self.name = name
        self.width = width
        self.height = height
        self.pins = pins  # dict of Pin objects
        self.connections = []  # Gates connected to this gate
        self.x = None  # x-coordinate of bottom-left corner
        self.y = None  # y-coordinate of bottom-left corner

    def add_connection(self, gate_name):
        self.connections.append(gate_name)

def parse_input(file_path):
    gates = {}
    wires = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("wire"):
            # Parsing a gate
            gate_info = line.split()
            gate_name = gate_info[0]
            width = int(gate_info[1])
            height = int(gate_info[2])
            
            i += 1
            # Now parsing pins line that starts with 'pins'
            pin_line = lines[i].strip()
            pin_parts = pin_line.split()[2:]  # Skip the 'pins' keyword
            pins = [Pin('p'+str(j//2+1) ,int(pin_parts[j]), int(pin_parts[j+1])) for j in range(0, len(pin_parts), 2)]
            pin_gate_dict = {pin.pin_name: pin for pin in pins}
            gates[gate_name] = Gate(gate_name, width, height, pin_gate_dict)
        else:
            # Parsing a wire
            wire_info = line.split()
            wire = (wire_info[1], wire_info[2])  # (source_pin, target_pin)
            wires.append(wire)
        
        i += 1
    
    # Create connections between gates based on wires
    for src, tgt in wires:
        src_gate = src.split('.')[0]
        tgt_gate = tgt.split('.')[0]
        if src_gate in gates and tgt_gate in gates:
            gates[src_gate].add_connection(tgt_gate)
            gates[tgt_gate].add_connection(src_gate)
    
    return gates, wires

def write_output(file_path, bounding_box, wire_length, gate_positions):
    with open(file_path, 'w') as f:
        # Write bounding box and wire length
        f.write(f"bounding_box {bounding_box[0]} {bounding_box[1]}\n")
        f.write(f"wire_length {wire_length}\n")
        
        # Write gate positions
        for gate_name, (x, y) in gate_positions.items():
            f.write(f"{gate_name} {x} {y}\n")

def is_overlapping(gate1, gate2):
    return not (gate1.x + gate1.width <= gate2.x or
                gate2.x + gate2.width <= gate1.x or
                gate1.y + gate1.height <= gate2.y or
                gate2.y + gate2.height <= gate1.y)

def check_overlap(new_gate, placed_gates):
    for placed_gate in placed_gates:
        if new_gate != placed_gate:
            if is_overlapping(new_gate, placed_gate):
                return True  # Overlap found
    return False  # No overlap

def nishant(wires, placed_gates):
    total_length = 0
    placed_gates_dict = {gate.name: gate for gate in placed_gates}
    for wire in wires:
        start = wire[0].split('.')
        end = wire[1].split('.')
        if start[0] not in placed_gates_dict or end[0] not in placed_gates_dict:
            continue
        start_gate = placed_gates_dict[start[0]]
        end_gate = placed_gates_dict[end[0]]
        start_x = start_gate.x + start_gate.pins[start[1]].x
        start_y = start_gate.y + start_gate.pins[start[1]].y
        end_x = end_gate.x + end_gate.pins[end[1]].x
        end_y = end_gate.y + end_gate.pins[end[1]].y
        total_length += abs(start_x - end_x) + abs(start_y - end_y)
    return total_length

def use_wire_estimator(placed_gates, wires):
    return nishant(wires, placed_gates)


def place_gate(gate, placed_gates, initial_position, wires):
    directions = [
        (-gate.width, 0),  # Left
        (gate.width, 0),   # Right
        (0, -gate.height), # Down
        (0, gate.height)    # Up
    ]
    
    best_position = None
    min_wire_length = float('inf')
    
    for dx, dy in directions:
        gate.x = initial_position[0] + dx
        gate.y = initial_position[1] + dy
        
        # Check for overlap
        if not check_overlap(gate, placed_gates):
            
            wire_length = use_wire_estimator(placed_gates, wires)
            if wire_length < min_wire_length:
                min_wire_length = wire_length
                best_position = (gate.x, gate.y)
    
    if best_position:
        gate.x, gate.y = best_position
        return True  # Successfully placed
    return False  # Could not place

def base_packing(gates, wires):
    # Calculate connections
    connection_counts = {gate_name: len(gate.connections) for gate_name, gate in gates.items()}
    
    # Sort gates by the number of connections
    sorted_gates = sorted(gates.values(), key=lambda g: connection_counts[g.name], reverse=True)

    placed_gates = []
    gate_positions = {}
    
    # Place the gate with the maximum connections at the center (arbitrarily placed at origin)
    center_gate = sorted_gates[0]
    center_gate.x, center_gate.y = 0, 0
    placed_gates.append(center_gate)
    gate_positions[center_gate.name] = (center_gate.x, center_gate.y)

    # Place connected gates around the center gate
    for gate in center_gate.connections:
        for candidate in sorted_gates:
            if candidate.name == gate and candidate not in placed_gates:
                if place_gate(candidate, placed_gates, (center_gate.x, center_gate.y), wires):
                    placed_gates.append(candidate)
                    gate_positions[candidate.name] = (candidate.x, candidate.y)

    # Place remaining gates
    with tqdm(total=len(sorted_gates), desc="Initial Packing") as pbar_initial:
        for gate in sorted_gates:
            pbar_initial.update(1)
            if gate not in placed_gates:
                position_found = False
                
                # Try placing the gate around all already placed gates
                for placed_gate in placed_gates:
                    if place_gate(gate, placed_gates, (placed_gate.x, placed_gate.y), wires):
                        placed_gates.append(gate)
                        gate_positions[gate.name] = (gate.x, gate.y)
                        position_found = True
                        break
                
                # If no position was found, try placing the gate around the origin as a last resort
                if not position_found:
                    if place_gate(gate, placed_gates, (0, 0)):
                        placed_gates.append(gate)
                        gate_positions[gate.name] = (gate.x, gate.y)
                    else:
                        raise RuntimeError(f"Could not place gate {gate.name} without overlap.")


    shift_x, shift_y = shift_to_first_quadrant(placed_gates)
    # Update gate positions to reflect shifts
    for gate in placed_gates:
        gate_positions[gate.name] = (gate.x, gate.y)

    
    return gate_positions,placed_gates

def get_bounding_box(placed_gates):
    # Find the minimum and maximum coordinates
    min_x = min(gate.x for gate in placed_gates)
    max_x = max(gate.x + gate.width for gate in placed_gates)
    min_y = min(gate.y for gate in placed_gates)
    max_y = max(gate.y + gate.height for gate in placed_gates)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    return (width, height)

def shift_to_first_quadrant(placed_gates):
    # Find the minimum x and y coordinates
    min_x = min(gate.x for gate in placed_gates)
    min_y = min(gate.y for gate in placed_gates)
    
    # Calculate the shift amounts
    shift_x = abs(min_x) if min_x < 0 else 0
    shift_y = abs(min_y) if min_y < 0 else 0

    # Shift all gates
    for gate in placed_gates:
        gate.x += shift_x
        gate.y += shift_y

    return shift_x, shift_y  # Return the shift values for reference

def generate_neighbor(placed_gates, wires):
    new_placed_gates = [gate for gate in placed_gates]
    gate_to_move = random.choice(new_placed_gates)
    
    possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    best_move = None
    best_cost = float('inf')
    
    for move_x, move_y in possible_moves:
        gate_to_move.x += move_x
        gate_to_move.y += move_y
        
        if not check_overlap(gate_to_move, new_placed_gates):
            new_cost = use_wire_estimator(new_placed_gates, wires)
            if new_cost < best_cost:
                best_cost = new_cost
                best_move = (move_x, move_y)
        
        gate_to_move.x -= move_x
        gate_to_move.y -= move_y
    
    if best_move:
        gate_to_move.x += best_move[0]
        gate_to_move.y += best_move[1]
    
    return new_placed_gates
def acceptance_probability(old_cost, new_cost, temperature):
    if new_cost < old_cost:
        return 1.0
    if temperature >= 3:
        return (math.exp((old_cost - new_cost) / temperature))**0.5
    return (math.exp((old_cost - new_cost) / temperature))
def cooling_schedule(temperature, alpha=0.99):
    return temperature * alpha

# bringing the local search algo implementation here
def local_search(placed_gates, wires):  
    # Example local search: try to move each gate slightly to reduce wire length
    for gate in placed_gates:
        best_position = (gate.x, gate.y)
        best_cost = use_wire_estimator(placed_gates, wires)
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                gate.x += dx
                gate.y += dy
                if not check_overlap(gate, placed_gates):
                    new_cost = use_wire_estimator(placed_gates, wires)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_position = (gate.x, gate.y)
                gate.x -= dx
                gate.y -= dy
        
        gate.x, gate.y = best_position

def simulated_annealing(placed_gates, wires, initial_temp, final_temp, alpha):
    best_placement = placed_gates.copy()
    current_cost = use_wire_estimator(placed_gates, wires)
    best_cost = current_cost
    temperature = initial_temp
    iteration = 0
    total_iterations = int(math.log(final_temp / initial_temp) / math.log(alpha))
    
    with tqdm(total=total_iterations, desc="Annealing") as pbar:
        while temperature > final_temp:
            new_placed_gates = generate_neighbor(placed_gates, wires)
            new_cost = use_wire_estimator(new_placed_gates, wires)
            if new_cost < best_cost:
                best_cost = new_cost
                best_placement = new_placed_gates.copy()
            if acceptance_probability(current_cost, new_cost, temperature) > random.random():
                placed_gates = new_placed_gates
                current_cost = new_cost
                local_search(placed_gates, wires)
                current_cost = use_wire_estimator(placed_gates, wires)
            
            temperature = cooling_schedule(temperature, alpha)
            iteration += 1
            pbar.update(1)
            #write output to output.txt
            gate_positions = {gate.name: (gate.x, gate.y) for gate in placed_gates}
            write_output('output.txt', [10,10], current_cost, gate_positions)  # Example bounding box
            vc = main('output.txt', 'input.txt')
            pbar.set_postfix({"Temp": f"{temperature:.2f}", "CC": current_cost, "VC": vc})
    return best_placement, best_cost

def adjust_coordinates(placed_gates):
    min_x = min(gate.x for gate in placed_gates)
    min_y = min(gate.y for gate in placed_gates)
    
    for gate in placed_gates:
        gate.x -= min_x
        gate.y -= min_y
    
    return placed_gates


offset = 0
# visualization length measurement
def parse_input1(data):
    gates = {}
    pins = {}
    wires = []

    for line in data:
        tokens = line.split()

        if tokens[0].startswith('g'):
            # Parsing gate dimensions
            gate_name = tokens[0]
            width, height = int(tokens[1])+offset, int(tokens[2])+offset
            gates[gate_name] = {"width": width, "height": height}
        
        elif tokens[0] == "pins":
            # Parsing pin coordinates
            gate_name = tokens[1]
            pin_coords = [(int(tokens[i]), int(tokens[i+1])) for i in range(2, len(tokens), 2)]
            pins[gate_name] = pin_coords

        elif tokens[0] == "wire":
            # Parsing wire connections
            wire_from = tokens[1].split('.')
            wire_to = tokens[2].split('.')
            wires.append((wire_from, wire_to))
    
    return gates, pins, wires

# Function to parse the gate positions from output.txt
def parse_gate_positions(data):
    gate_positions = {}
    
    for line in data:
        tokens = line.split()

        if tokens[0].startswith('g'):
            gate_name = tokens[0]
            x, y = int(tokens[1])+offset, int(tokens[2])+offset
            gate_positions[gate_name] = (x, y)
    
    return gate_positions

# Function to calculate pin coordinates based on gate placement
def calculate_pin_coordinates(gate_positions, gates, pins):
    pin_positions = {}

    for gate, position in gate_positions.items():
        gate_x, gate_y = position
        pin_positions[gate] = [(gate_x + px, gate_y + py) for (px, py) in pins[gate]]

    return pin_positions

# Function to calculate Manhattan distance between two points
def calculate_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Function to compute a 2D matrix of distances between all pairs of pins
def calculate_all_pin_distances(pin_positions):
    all_pins = []  # To store all pins and their coordinates
    pin_names = []  # To keep track of which pin belongs to which gate and index
    gate_names = []  # To track which gate each pin belongs to

    # Flattening the pin_positions dictionary into a list of pins with their coordinates and names
    for gate, pin_list in pin_positions.items():
        for i, pin_coords in enumerate(pin_list):
            pin_name = f"{gate}.p{i+1}"
            all_pins.append(pin_coords)
            pin_names.append(pin_name)
            gate_names.append(gate)  # Keep track of which gate each pin belongs to
    
    # Create a 2D matrix for distances
    distance_matrix = [[0] * len(all_pins) for _ in range(len(all_pins))]

    # Calculate distances between all pairs of pins
    for i in range(len(all_pins)):
        for j in range(len(all_pins)):
            if i != j:
                if gate_names[i] == gate_names[j]:
                    # Pins belong to the same gate, set distance to infinity
                    distance_matrix[i][j] = math.inf
                else:
                    # Calculate Manhattan distance for pins from different gates
                    distance_matrix[i][j] = calculate_distance(all_pins[i], all_pins[j])
    
    return distance_matrix, pin_names, gate_names

# Function to compute a 2D matrix of True/False for connected pins or same gate pins
def calculate_connection_matrix(pin_positions, pin_names, gate_names, wires):
    connection_matrix = [[False] * len(pin_names) for _ in range(len(pin_names))]

    # Explicitly mark False for pins belonging to the same gate
    for i in range(len(pin_names)):
        for j in range(len(pin_names)):
            if gate_names[i] == gate_names[j]:
                connection_matrix[i][j] = False  # Pins of the same gate must have False

    # Mark True for connected pins based on the wire connections
    for wire in wires:
        gate1, pin1 = wire[0]
        gate2, pin2 = wire[1]
        pin1_idx = int(pin1[1:]) - 1
        pin2_idx = int(pin2[1:]) - 1
        pin1_full = f"{gate1}.p{pin1_idx + 1}"
        pin2_full = f"{gate2}.p{pin2_idx + 1}"

        if pin1_full in pin_names and pin2_full in pin_names:
            idx1 = pin_names.index(pin1_full)
            idx2 = pin_names.index(pin2_full)
            connection_matrix[idx1][idx2] = True
            
    
    return connection_matrix

# New function to return a sequential list of pin coordinates
def get_pin_coordinates_in_order(pin_positions):
    ordered_pin_coordinates = []
    for gate, pin_list in pin_positions.items():
        ordered_pin_coordinates.extend(pin_list)  # Flatten the coordinates into a single list
    return ordered_pin_coordinates

def main(coordinates_file, dimensions_file):

    with open(dimensions_file, 'r') as f:
        input_data = f.readlines()

    # Reading gate positions from the output file
    with open(coordinates_file, 'r') as f:
        output_data = f.readlines()

    # Parse the input and output data
    gates, pins, wires = parse_input1(input_data)
    gate_positions = parse_gate_positions(output_data)

    # Calculate pin coordinates based on gate positions
    pin_positions = calculate_pin_coordinates(gate_positions, gates, pins)

    # Calculate distances between all pairs of pins
    distance_matrix, pin_names, gate_names = calculate_all_pin_distances(pin_positions)

    # Calculate connection matrix (True/False for connected pins or same gate pins)
    connection_matrix = calculate_connection_matrix(pin_positions, pin_names, gate_names, wires)

    # Get the list of pin coordinates in sequential order
    ordered_pin_coordinates = get_pin_coordinates_in_order(pin_positions)

    total_wire_length=0
    i=0
    j=0
    for i in range(0,len(connection_matrix)):
        temp=connection_matrix[i]

        if True in temp:
            bounding_x=list()
            bounding_y=list()
            bounding_x.append(ordered_pin_coordinates[i][0])
            bounding_y.append(ordered_pin_coordinates[i][1])

            for j in range(0,len(temp)):
                if connection_matrix[i][j]:
                    bounding_x.append(ordered_pin_coordinates[j][0])
                    bounding_y.append(ordered_pin_coordinates[j][1])
            
            xmin=min(bounding_x)
            xmax=max(bounding_x)
            ymin=min(bounding_y)
            ymax=max(bounding_y)

            total_wire_length=total_wire_length+xmax-xmin+ymax-ymin

    # print("total wire length is: ", total_wire_length)
    return total_wire_length


# adding skyline here
class Skyline:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.skyline = [(0, 0)]  # Start with a flat skyline

    def can_place_rectangle(self, width, height, position):
        # Check if the rectangle can be placed at the given position
        x, y = position
        if x + width > self.width:
            return False

        for i in range(len(self.skyline)):
            if self.skyline[i][0] >= x + width:
                break
            if self.skyline[i][1] > y:
                return False

        return True

    def find_position_for_rectangle(self, width, height):
        best_x = -1
        best_y = float('inf')
        best_index = -1

        for i in range(len(self.skyline)):
            x = self.skyline[i][0]
            if x + width > self.width:
                continue

            y = max([self.skyline[j][1] for j in range(i, len(self.skyline)) if self.skyline[j][0] < x + width])

            if y + height > self.height:
                continue

            if y < best_y:
                best_x = x
                best_y = y
                best_index = i

        return best_x, best_y, best_index

    def place_rectangle(self, rectangle):
        width = rectangle.width
        height = rectangle.height

        x, y, index = self.find_position_for_rectangle(width, height)
        if x == -1:
            return False

        rectangle.x = x
        rectangle.y = y
        rectangle.was_packed = True

        self.update_skyline(x, y, width, height)
        return True

    def update_skyline(self, x, y, width, height):
        new_skyline = []
        i = 0

        while i < len(self.skyline) and self.skyline[i][0] < x:
            new_skyline.append(self.skyline[i])
            i += 1

        new_skyline.append((x, y + height))
        while i < len(self.skyline) and self.skyline[i][0] < x + width:
            i += 1

        if i < len(self.skyline) and self.skyline[i][0] == x + width:
            new_skyline.append((x + width, self.skyline[i][1]))
            i += 1
        else:
            new_skyline.append((x + width, y))

        new_skyline.extend(self.skyline[i:])
        self.skyline = new_skyline

    def get_height(self):
        return max(y for _, y in self.skyline)

    def get_width(self):
        return max(x for x, _ in self.skyline)
    
class Pin_skyline:
    def __init__(self, gate_name, x_offset, y_offset):
        self.gate_name = gate_name
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.gate = None  # Will be set to the Gate object it belongs to

class Rectangle:
    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height
        self.x = 0 
        self.y = 0  
        self.pins = [] 
        self.was_packed = False  

class Net:
    def __init__(self, pins):
        self.pins = pins 

# Manhattan distance calculation between two points
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Total wire length using Manhattan distance
def total_wire_length(gates, nets):
    total_length = 0
    for net in nets:
        pin1 = net.pins[0]
        pin2 = net.pins[1]
        
        # Get absolute pin positions based on their gate's current position
        gate1 = pin1.gate
        gate2 = pin2.gate
        
        pin1_x = gate1.x + pin1.x_offset
        pin1_y = gate1.y + pin1.y_offset
        pin2_x = gate2.x + pin2.y_offset
        pin2_y = gate2.y + pin2.y_offset

        total_length += manhattan_distance(pin1_x, pin1_y, pin2_x, pin2_y)
    
    return total_length


def parse_input_file(filename):
    gates = []
    nets = []

    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            # Detect a gate line with gate name, width, and height
            if len(tokens) == 3 and all(token.isidentifier() or token.isdigit() for token in tokens):
                gate_name = tokens[0]
                width = int(tokens[1])
                height = int(tokens[2])
                gates.append(Rectangle(gate_name, width, height))
            # Detect a pin definition line
            elif tokens[0] == 'pins' and len(tokens) > 2:
                gate_name = tokens[1]
                for gate in gates:
                    if gate.name == gate_name:
                        for i in range(2, len(tokens), 2):
                            x_offset = int(tokens[i])
                            y_offset = int(tokens[i + 1])
                            pin = Pin_skyline(gate_name, x_offset, y_offset)
                            gate.pins.append(pin)
                            pin.gate = gate
            # Detect a wire connection between two gates' pins
            elif tokens[0] == 'wire' and len(tokens) == 3:
                try:
                    pin1_gate, pin1_pin = tokens[1].split('.')
                    pin2_gate, pin2_pin = tokens[2].split('.')
                    pin1 = next(gate.pins[int(pin1_pin[1:]) - 1] for gate in gates if gate.name == pin1_gate)
                    pin2 = next(gate.pins[int(pin2_pin[1:]) - 1] for gate in gates if gate.name == pin2_gate)
                    nets.append(Net([pin1, pin2]))
                except (ValueError, IndexError):
                    print(f"Error parsing wire connection: {tokens[1]} to {tokens[2]}. Ensure the pins exist.")
    return gates, nets

def calculate_rms(values):
    return math.sqrt(sum(x**2 for x in values) / len(values))

def write_output_file(filename, gates, total_wirelength, chip_width, chip_height):
    with open(filename, 'w') as file:
        file.write(f"bounding_box {chip_width} {chip_height}\n")
        for gate in gates: 
            file.write(f"{gate.name} {gate.x} {gate.y}\n")
        file.write(f"wire_length {total_wirelength}\n")

def skyline_call(bestwirelength):
    # Parse input
    gates, nets = parse_input_file('input.txt')  # Provide the path to your input file

    rectangles = gates
    rectangles.sort(key=lambda r: r.height, reverse=True)

    # Heuristic to determine initial bounding box size
    total_area = sum(rect.width * rect.height for rect in rectangles)
    initial_size = int((total_area ** 0.5) * 1.2)  # 20% larger than the square root of total area
    initial_width, initial_height = initial_size, initial_size 

    skyline = Skyline(initial_width, initial_height)

    while rectangles:
        # skyline = Skyline(initial_width, initial_height)
        unplaced_rectangles = []

        for rectangle in rectangles:
            if not skyline.place_rectangle(rectangle):
                unplaced_rectangles.append(rectangle)

        if not unplaced_rectangles:
            break  # All rectangles have been placed

        # Update the list of rectangles to only include unplaced ones
        rectangles = unplaced_rectangles
        # Double the bounding box dimensions
        initial_width *= 2
        initial_height *= 2
        skyline.width = initial_width
        skyline.height = initial_height

    total_wirelength = total_wire_length(gates, nets)
    if total_wirelength < bestwirelength:
        write_output_file('output.txt', gates, total_wirelength, skyline.get_width(), skyline.get_height())
    


# Example Usage
gates, wires = parse_input('input.txt')  # Provide the path to your input file
gate_positions, placed_gates = base_packing(gates,wires)


initial_temp = 10
final_temp = 0.1
if len(placed_gates) >= 100:
    initial_temp = 5
    alpha = 0.99
else:
    alpha = 0.9999

optimized_gates = adjust_coordinates(placed_gates)
bestplacement = optimized_gates.copy()
gate_positions = {gate.name: (gate.x, gate.y) for gate in bestplacement}
write_output('output.txt', [10,10], 10, gate_positions)  # Example bounding box
bestwirelength = main('output.txt', 'input.txt')
optimized_gates, wire_length = simulated_annealing(optimized_gates, wires, initial_temp, final_temp, alpha)
for i in range(1,6):
    optimized_gates, wire_length = simulated_annealing(optimized_gates, wires, initial_temp+i, i, alpha)
    if wire_length < bestwirelength:
        bestwirelength = wire_length
        bestplacement = optimized_gates.copy()

# Adjust coordinates to ensure all are positive
optimized_gates = adjust_coordinates(optimized_gates)
# Convert optimized_gates to gate_positions
gate_positions = {gate.name: (gate.x, gate.y) for gate in optimized_gates}
# Calculate bounding box
bounding_box = get_bounding_box(optimized_gates)
# Custom sorting function to sort gate names numerically
def numeric_sort_key(gate_name):
    return int(''.join(filter(str.isdigit, gate_name)))

sorted_gate_positions = dict(sorted(gate_positions.items(), key=lambda item: numeric_sort_key(item[0])))  # Sort gate positions by gate name numerically
write_output('output.txt', bounding_box, bestwirelength, sorted_gate_positions)  # Example bounding box
print("Base packing completed.")
skyline_call(bestwirelength)