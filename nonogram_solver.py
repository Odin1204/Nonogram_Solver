from pysat.solvers import Cadical153 as Solver;
from pysat.card import CardEnc;
import random

import sys

variable_count: int = 1

def get_variable():
    global variable_count
    temp = variable_count
    variable_count += 1
    return temp

def reset_count():
    global variable_count
    variable_count = 1

#def at_most_one(literal_set: list[int], solver):
#    for i in range(0, len(literal_set)):
#        for j in range(i+1, len(literal_set)):
#            if (literal_set[i] != literal_set[j]):
#                solver.add_clause([-literal_set[i], -literal_set[j]])

#def add_exactly_one(literal_set: list[int], solver):
#    solver.add_clause(literal_set)
#    at_most_one(literal_set, solver)

# improved exactly one with cardinality constraints
def add_exactly_one(literal_set: list[int], solver):
    global variable_count
    # Generate exactly-one encoding with auxiliary variables
    card = CardEnc.equals(lits=literal_set, bound=1, encoding=1, top_id=variable_count)
    
    # Add the generated clauses to the solver
    for clause in card.clauses:
        solver.add_clause(clause)
    
    # Update the variable counter to avoid overlapping IDs
    variable_count = max(variable_count, card.nv + 1)

class gamefield:

    @classmethod
    def generate_random_and_solve(cls, size: int, density: float = 0.5, seed: int = None):
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        grid = [
            [1 if random.random() < density else 0 for _ in range(size)]
            for _ in range(size)
        ]

        def extract_blocks(line):
            blocks = []
            count = 0
            for cell in line:
                if cell == 1:
                    count += 1
                elif count > 0:
                    blocks.append(count)
                    count = 0
            if count > 0:
                blocks.append(count)
            return blocks or []  # [] if the line is all empty

        row_blocks = [extract_blocks(row) for row in grid]
        col_blocks = [extract_blocks([grid[r][c] for r in range(size)]) for c in range(size)]

        print("Generated field:")
        for i in range(size):
            print("".join([("⬛" if (grid[i][j] > 0) else "⬜") for j in range(size)]))

        print("Solving field...")
        return cls(size,size, col_blocks, row_blocks)


    def __init__(self, field_width: int, field_height: int, column_blocks: list[list[int]], row_blocks: list[list[int]]):
        self.column_blocks = column_blocks
        self.row_blocks = row_blocks
        
        self.width = field_width
        self.height = field_height

        self.field: list[list[int]] = [
            [get_variable() for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self.solve()


    def get_row(self, row: int) -> list[int]:
        return self.field[row]

    def get_column(self, column: int) -> list[int]:
        return [row[column] for row in self.field]
    
    def solve(self):

        solver = Solver()

        for row in range(len(self.row_blocks)):
            automata = dfa(self.get_row(row), self.row_blocks[row])
            automata.add_automata_to_sat(solver)
        
        for col in range(len(self.column_blocks)):
            automata = dfa(self.get_column(col), self.column_blocks[col])
            automata.add_automata_to_sat(solver)

        reset_count()

        print(f"solving for {solver.nof_vars()} Variables and {solver.nof_clauses()} Clauses")

        
        if solver.solve():
            print("SATISFIABLE:")
            satisfying_assignment = solver.get_model()[:(self.height*self.width)]

            if self.check_solution(satisfying_assignment):
                print("valid")
                for i in range(self.height):
                    print("".join([("⬛" if (satisfying_assignment[i*self.width+j] > 0) else "⬜") for j in range(self.width)]))
            else:
                print("Error in solver")
        else:
            print("UNSAT")

    def check_solution(self, assignment: list[int]) -> bool:
        def extract_blocks(line_vars):
            # Convert variable IDs to booleans using the assignment
            line = [1 if var in assignment else 0 for var in line_vars]
            # Extract blocks of 1s
            blocks = []
            count = 0
            for val in line:
                if val == 1:
                    count += 1
                elif count > 0:
                    blocks.append(count)
                    count = 0
            if count > 0:
                blocks.append(count)
            return blocks

        # Check all rows
        for row_idx in range(self.height):
            expected = self.row_blocks[row_idx]
            actual = extract_blocks(self.get_row(row_idx))
            if actual != expected:
                print(f"Row {row_idx} mismatch: expected {expected}, got {actual}")
                return False

        # Check all columns
        for col_idx in range(self.width):
            expected = self.column_blocks[col_idx]
            actual = extract_blocks(self.get_column(col_idx))
            if actual != expected:
                print(f"Col {col_idx} mismatch: expected {expected}, got {actual}")
                return False

        return True

            


class dfa:
    #state 0 is failed state, 1 is starting state and num_states-1 is goal state
    num_states: int = 2

    goal_state: int

    transitions: dict[tuple[int, bool], int] = {}

    input: list[int]

    def __init__(self, input: list[int], block: list[int]):
        self.input = input
        
        #trivial transitions
        self.transitions[(0, True)] = 0
        self.transitions[(0, False)] = 0
        self.transitions[(1,False)] = 1

        #adding of intermediate states and first real transition
        #if a line should be empty it has to be denoted as [] not [0]
        if (len(block) != 0):
            self.num_states += len(block) - 1
            self.transitions[(1,True)] = 2
            self.add_all_transitions(block)
        else:
            self.transitions[(1,True)] = 0
            self.goal_state = 1

        return

    def add_all_transitions(self, block: list[int]):
        current_state = 2

        for i in range(len(block)):
            self.num_states += block[i]
            for _ in range(block[i]-1):
                self.transitions[(current_state, True)] = current_state+1 #advance
                self.transitions[(current_state, False)] = 0 #fail
                current_state += 1
            
            #transitions for goal state state
            if (i == len(block)-1):
                self.transitions[(current_state, True)] = 0
                self.transitions[(current_state, False)] = current_state
                self.goal_state = current_state
                break
            
            #go into intermediate state or fail
            self.transitions[(current_state, False)] = current_state + 1
            self.transitions[(current_state, True)] = 0
           
            current_state += 1
            
            #transitions for intermediate states
            self.transitions[(current_state, True)] = current_state+1
            self.transitions[(current_state, False)] = current_state
            current_state += 1
            
        return

    def __str__(self):
        lines = [
            f"DFA with {self.num_states} states",
            f"Goal state: {self.goal_state}",
            "Transitions:"
        ]
        # Sort transitions for consistent output
        for (state, bit) in sorted(self.transitions.keys()):
            next_state = self.transitions[(state, bit)]
            lines.append(f"  From state {state} on input {int(bit)} -> state {next_state}")
        return "\n".join(lines)

    def add_automata_to_sat(self, solver):
        
        if (self.goal_state == 1):
            for var in self.input:
                solver.add_clause([-var])
            return

        input_len = len(self.input)

        state_to_variable_map: dict[tuple[int, int], int] = {}
        

        #clauses for starting state
        false_transition = self.transitions[(1,False)]
        true_transition = self.transitions[(1,True)]

        temp_true = temp_false = 0

        possible_states: set[int] = {false_transition, true_transition}
        
        if (true_transition == false_transition):
            temp_false = temp_true = get_variable()

        else:
            temp_false = get_variable()
            temp_true = get_variable()
        

        state_to_variable_map[true_transition, 1] = temp_true
        state_to_variable_map[false_transition, 1] = temp_false
        

        solver.add_clause([self.input[0], temp_false]) #simulate i0 = F transition
        solver.add_clause([-self.input[0], temp_true]) #simulate i0 = T transition
        add_exactly_one([temp_true, temp_false], solver) # a transition must happen

        for i in range(1, input_len):
            temp_set: set[int] = set()

            for state in possible_states:
                state_var = state_to_variable_map[(state, i)]


                false_transition = self.transitions[(state, False)]
                true_transition = self.transitions[(state, True)]
                
                
                #transition leads to fail state
                if (false_transition == 0):
                    solver.add_clause([-state_var, self.input[i]])
                #goal state cannot be reached
                #this is only possible for this specific automaton construction since you can't skip states
                elif (self.goal_state - false_transition > input_len - i):
                    solver.add_clause([-state_var, self.input[i]])
                else:
                    temp_set.add(false_transition)
                    if ((false_transition, i+1) not in state_to_variable_map):
                        temp_false = get_variable()
                        state_to_variable_map[(false_transition, i+1)] = temp_false                
                    else:
                        temp_false = state_to_variable_map[(false_transition, i+1)]
                
                    solver.add_clause([-state_var, self.input[i],temp_false])
                
                if (true_transition == 0):
                    solver.add_clause([-state_var,-self.input[i]])
                #goal state cannot be reached
                #this is only possible for this specific automaton construction since you can't skip states
                elif (self.goal_state - true_transition > input_len - i):
                    solver.add_clause([-state_var, -self.input[i]])
                else:
                    temp_set.add(true_transition)
                    if ((true_transition, i+1) not in state_to_variable_map):
                        temp_true = get_variable()
                        state_to_variable_map[(true_transition, i+1)] = temp_true
                    else:
                        temp_true = state_to_variable_map[(true_transition, i+1)]


                    solver.add_clause([-state_var, -self.input[i], temp_true])
            
            add_exactly_one([state_to_variable_map[(s,i+1)] for s in temp_set], solver)
            possible_states = temp_set
        

        solver.add_clause([state_to_variable_map[(self.goal_state, input_len)]])





#print("Field ex 01")
#field_01 = gamefield(15, 15, 
#        [[1,3],[2,2,1],[1,5],[3,2],[1,4],[1,1,3,3],[1,1,1,1],[1,2,2],[1,1,1,3],[1,1,1,3,1],[3,2,2],[5,1],[1,7],[2,4,2,1],[1,3]], 
#         [[5],[1,5],[3,1],[2,2],[3,3],[5,4],[2,4,2],[1,1,2],[1,4,1],[4,7],[4,2,2],[2,2,2,1],[1,1,1,1],[1,1,1,1],[1,1,1]])


#print("Field ex 12")
#field_12 = gamefield(25, 25,
#[[4,2],[5,4,2],[1,1,1,2,2],[1,1,5,1,4],[1,1,7,1,3,1],[1,1,9,1],[1,1,1,4,4,2], [2,1,2,6,9], [2,3,6,3], [2,1,1,1,3],[2,2,1,1,2,3],
# [2,1,2,1,3,2],[2,1,1,6],[1,1,1,5],[2,2,2,1,1],[4,3,1],[2,2,2,1],[4,2,2,1],[1,3,3,2],[5,2,3,1],[1,4,3,1],[6,5],[2,3],[8,3],[2,2]],
#[[7],[2,5],[1,3],[2,2,3],[1,2],[2,2,2,2],[1,2,4],[1,2,1,2,1,2],[9,1,1,1,1],[1,2,2,2,2,1,4],[1,2,5,2,1,4],[1,2,3,2,1,2,1,1],
#[1,2,5,1,1,1,1],[1,2,2,1,2,2,1],[1,2,4,1,1,1,1],[5,1,1],[3,2,2,1],[3,2,1],[1,1,5,1],[2,2,2,4],[3,3,2,7],[2,4,2,7],[4,2,5,5],
#[8,4,1,2,1],[2,1,4]])

input_len = len(sys.argv)
if (input_len < 3 or input_len > 4):
    print(f"{sys.argv[0]} <size> <density> (seed)")
    exit()

size: int = int(sys.argv[1])
density: float = float(sys.argv[2])

if (input_len == 4):
    seed:int = int(sys.argv[3])
    gamefield.generate_random_and_solve(size,density, seed)
else:
    gamefield.generate_random_and_solve(size,density)
