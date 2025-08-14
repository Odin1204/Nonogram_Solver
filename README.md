# Nonogram_Solver
A nonogram solver with pysat

## How to use
Currently when calling the program with `python3 nonogram_solver.py <size> <density>` it generates a random size x size field,
where the density is the percentage of coloured squares, and then solves this field.

If there is a specific field that you want to solve you have to change the program and call the gamefield constructor with 
`(width,height, [[hints_coloums]],[[hints_rows]])` where [] denotes a hint for a line with no coloured squares.

## How it works
The program translates a Nonogram field into a sat encoding. For every line it simulates a distinct finite automaton operating on a fixed input size.

The optimum performance is for fields with a density of 0.5-0.6 where it can solve fields of up to 100x100 in 1-2 minutes depending on the field.
Smaller fields are usually solved in under a second.

### Dependencies
- pysat

## TODO

- [] adding user interface

- [] adding error handeling
