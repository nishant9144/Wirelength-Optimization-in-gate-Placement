# Wirelength-Optimization-in-gate-Placement
## Problem
Given:
- a set of rectangular logic gates g1, g2...gn
- width and height of each gate gi
-the input and output pin locations (x and y co-ordinates) on the boundary of each gate gi
.p1, gi
.p2, ..., gi
.pm
(where gate gi has m pins)
- the pin-level connections between the gates
<br> 

Write a program to assign locations to all gates in a plane so that:
- no two gates are overlapping
- the sum of estimated wire lengths for all wires in the whole circuit is minimised.

## Constraint
- Assume that the gates cannot be re-oriented (rotated, etc.) in any way.
- Assuming that all wiring is horizontal and vertical
- Possible estimate for the wire length for a set of connected pins uses the semi-perimeter method: form
a rectangular bounding box of all the pin locations; the estimated wire length is half the perimeter of this
rectangle
<br>

Following are the input constraints:
- Corners of gate have integral coordinates
- Pins will have integral coordinates relative to the corresponding gate
- 0 < Number of gates ≤ 1000
- 0 < Width of gate ≤ 100
- 0 < Height of gate ≤ 100
- 0 < Number of pins on one side of a gate ≤ Height of Gate
- 0 < Total Number of pins ≤ 40000
- There is atleast 1 wire connecting a gate
## Solution
For our solution, we have combined two algorithms.<br>
First, we have used our algorithm used in software assignment 1 which is the Skyline
Algorithm.<br>
- We went for a heuristic method for packing rectangles onto a sheet using a
rectilinear skyline representation. This skyline is depicted as a sequence of
horizontal line segments that form the contour of vertical bars.<br>
The key properties of the skyline are:

- Consecutive segments have different y-coordinates.
- The x-coordinate of the right endpoint of each segment aligns with the xcoordinate of the left endpoint of the next.
- Initially, the skyline is a single segment representing the bottom of the sheet.

Rectangles are placed one by one, either with their bottom left corner touching a left
endpoint or their bottom right corner touching a right endpoint of a skyline segment.

The placement is determined by whether adjacent segments are higher or lower.
After placing a rectangle, the skyline is updated in two steps:
- A new segment is created for the top edge of the rectangle, and affected
segments are adjusted based on the rectangle's width.
- The algorithm identifies locally lowest segments—those lower than their
neighbors—and raises and merges them if no unplaced rectangles can fit.
This process continues until all segments are checked.

Reference: [https://www.researchgate.net/publication/221049934_A_Skyline-Based_Heuristic_for_the_2D_Rectangular_Strip_Packing_Problem](https://www.researchgate.net/publication/221049934_A_Skyline-Based_Heuristic_for_the_2D_Rectangular_Strip_Packing_Problem)

Second, we went for Simulated Annealing algorithm.<br>
Simulated Algorithm explores the solution space by randomly accepting worse
solutions with a probability based on temperature, which gradually decreases,
ensuring the system settles into an optimal or near-optimal solution.<br>
Steps of the Algorithm:
1. **Initial Solution**: Start with a random or heuristic-based layout of gates.
2. **Cost Function**: The cost function combines:
    - Wire length: The total Manhattan distance between connected gates.
    - Penalties: Applied for overlapping gates or boundary violations.
Cost = Wire Length + Penalty
3. **Neighbor Solution**: A new layout is generated by making small changes
(moving a gate) to the current solution.
4. **Acceptance Probability**: Even if the new solution is worse, it may still be
accepted based on a probability that decreases with temperature:<br>
<div style="text-align: center;">
P = exp(−ΔCost / T)
</div>

5. **Cooling Schedule**: The temperature decreases over time, allowing for more
exploration early on and more refinement later. The cooling schedule is
divided into phases:
    - Stochastic Search: Early acceptance of worse solutions to explore the
solution space.
    - Local Search: Focus on improving the current solution.
    - Uphill Search: Periodic temperature increases to escape local minima.
6. **Termination**: The process continues until the temperature reaches a
minimum value, resulting in an optimized gate layout.

Reference: [https://tilos-ai-institute.github.io/MacroPlacement/CodeElements/SimulatedAnnealing/](https://tilos-ai-institute.github.io/MacroPlacement/CodeElements/SimulatedAnnealing/)

## Time Complexity Anaylsis
( Refer to the report )