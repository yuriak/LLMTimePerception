import json
import random
import time
import math
from collections import deque
import logging
from argparse import ArgumentParser
from llm import LLM  # Placeholder for actual LLM import
from utils import extract_json_from_text, validate_and_parse_json_output
import tqdm
import tiktoken
import pickle
from pydantic import BaseModel
from argparse import Namespace

logger = logging.getLogger(__name__)

SYS_PROMPT_STATIC_TREASURE = """
# Grid World Treasure-Hunt Task

You are an adventurer in a grid-world maze.  
Your mission: **locate and claim a hidden treasure chest** as efficiently as possible.

---

## 1. Environment and Coordinate System

- You are in a two-dimensional grid world with obstacles (walls).
- The grid uses a standard Cartesian coordinate system where [1,1] is the bottom-left corner.
- X-axis runs horizontally (left to right), Y-axis runs vertically (bottom to top).
- For example, in a X $\times$ Y grid:
  - Bottom-left corner is [1,1]
  - Bottom-right corner is [X,1]
  - Top-left corner is [1,Y]
  - Top-right corner is [X,Y]
- You will receive a map of the grid world represented as a 2D array.
- Each row in the array corresponds to a y-coordinate (bottom row = y:1, top row = y:Y).
- Each column corresponds to an x-coordinate (leftmost column = x:1, rightmost column = x:X).
- To convert between Cartesian coordinates [x,y] and the map array:
  - For example, position [7,6] refers to the 7th column from the left and the 6th row from the bottom

**Important Map Representation Notes:**

In the map representation:
- "X" represents your current position
- "#" represents walls (you cannot move through walls)
- "0" represents empty spaces where you can move
- The treasure is hidden somewhere in the grid (not shown on the map)

## 2. The Treasure Signal
- The treasure emits signals that help you locate it:
  - **Signal Direction**: you'll receive an angle pointing toward the treasure, consider you have a compass and the angle is the direction of the treasure:
    - 0 or 360 degrees = NORTH (the treasure is at north of you, consider going north)
    - 90 degrees = EAST (the treasure is at east of you, consider going east)
    - 180 degrees = SOUTH (the treasure is at south of you, consider going south)
    - 270 degrees = WEST (the treasure is at west of you, consider going west)
  - **Signal Distance**: You will also receive an estimated distance from your current position to the treasure.
    - This is an Euclidean distance, so you can use it to estimate how far the treasure is from you.
- Each time you **move** you automatically receive an updated "signal" pointing toward the treasure.

## 3. Available Actions
You can perform four different actions:
- "north": Move one cell north (y + 1)
- "south": Move one cell south (y – 1)
- "east": Move one cell east (x + 1)
- "west": Move one cell west (x – 1)
*Attempting to cross a wall leaves you in place and reports `blocked_by_wall`.*

## 4. Mission Termination
The episode ends when either:
1. You step onto the treasure's cell (**success**), or
2. You exceed the maximum allowed number of steps (**failure**).

## 5. State Format (input to you each turn)
For each step, you will receive the environment state as a JSON object:
```json
{
  "last_action": "start" or "moved" or "blocked_by_wall",
  "current_location": [x, y],
  "signal_direction": angle_in_degrees,
  "signal_distance": euclidean_distance_in_cells
}
```

## 6. Response Format
You must respond with a JSON object containing your action and reasoning without any additional text or explanations:
```json
{
  "action": "north" | "south" | "east" | "west",
  "reasoning": "Brief step-by-step thought process"
}
```

## 7.Strategy Guidelines

1. Always return a valid JSON with the action and reasoning fields.
2. Use signal direction and distance to guide movement toward the treasure.
3. Plan paths that avoid walls and minimise detours.
4. Track how the distance shrinks to verify you're heading the right way.
5. Keep reasoning short but clear, concise plans speed up play.


Happy hunting! may your steps lead straight to the prize!
"""


SYS_PROMPT_STATIC_BOMB = """
# Grid World Bomb Detection Task

You are a police in a grid world.
Your mission: **locate and defuse a hidden bomb before it explodes**.

---

## 1. Environment and Coordinate System

- You are in a two-dimensional grid world with obstacles (walls).
- The grid uses a standard Cartesian coordinate system where [1,1] is the bottom-left corner.
- X-axis runs horizontally (left to right), Y-axis runs vertically (bottom to top).
- For example, in a X $\times$ Y grid:
  - Bottom-left corner is [1,1]
  - Bottom-right corner is [X,1]
  - Top-left corner is [1,Y]
  - Top-right corner is [X,Y]
- You will receive a map of the grid world represented as a 2D array.
- Each row in the array corresponds to a y-coordinate (bottom row = y:1, top row = y:Y).
- Each column corresponds to an x-coordinate (leftmost column = x:1, rightmost column = x:X).
- To convert between Cartesian coordinates [x,y] and the map array:
  - For example, position [7,6] refers to the 7th column from the left and the 6th row from the bottom

**Important Map Representation Notes:**

In the map representation:
- "X" represents your current position
- "#" represents walls (you cannot move through walls)
- "0" represents empty spaces where you can move
- The bomb is hidden somewhere in the grid (not shown on the map)

## 2. The Bomb Signal
- The bomb emits signals that help you locate it:
  - **Signal Direction**: you'll receive an angle pointing toward the bomb, consider you have a compass and the angle is the direction of the bomb:
    - 0 or 360 degrees = NORTH (the bomb is at north of you, consider going north)
    - 90 degrees = EAST (the bomb is at east of you, consider going east)
    - 180 degrees = SOUTH (the bomb is at south of you, consider going south)
    - 270 degrees = WEST (the bomb is at west of you, consider going west)
  - **Signal Distance**: You will also receive an estimated distance from your current position to the bomb.
    - This is an Euclidean distance, so you can use it to estimate how far the bomb is from you.
- Each time you **move** you automatically receive an updated "signal" pointing toward the bomb.
- The bomb is a time bomb, and it will explode after a certain amount of time. You need to reach the bomb before it explodes.
  - The value of **remaining time** (in seconds) will be provided at each step.

## 3. Available Actions
You can perform four different actions:
- "north": Move one cell north (y + 1)
- "south": Move one cell south (y – 1)
- "east": Move one cell east (x + 1)
- "west": Move one cell west (x – 1)
*Attempting to cross a wall leaves you in place and reports `blocked_by_wall`.*

## 4. Time Management
**Time is Continuously Ticking — Use It Wisely**:
**In this task, time flows continuously, and every action—including your reasoning—consumes it.**
This includes:
- Movement, even if you are blocked by a wall, it also consumes time.
- Your thought process: the longer and more detailed your reasoning, the more time you lose, lengthy and detailed analysis will consume valuable time.
To succeed, you must:
- **Balance planning with speed** — avoid overthinking when time is limited.
- **Adapt to urgency** — simplify your decisions as the situation becomes more critical.
- **Stay aware of time already spent** — use past states and actions to estimate how much time remains before the bomb explodes.
Think smart. Act fast.

## 5. Mission Termination
The mission ends when either:
- You successfully locate the bomb (**success**)
- The bomb explodes before you reach it (failure)
- You exceed the maximum allowed number of steps (**failure**)

## 6. State Format (input to you each turn)
For each step, you will receive the environment state as a JSON object:
```json
{
  "last_action": "start" or "moved" or "blocked_by_wall",
  "current_location": [x, y],
  "signal_direction": angle_in_degrees,
  "signal_distance": euclidean_distance_in_cells,
  "remaining_time": time_in_seconds,
  "last_reasoning_time_consumed": time_in_seconds  // Time consumed in the last reasoning
}
```

## 7. Response Format
You must respond with a JSON object containing your action and reasoning without any additional text or explanations:
```json
{
  "action": "north" | "south" | "east" | "west",
  "reasoning": "Brief step-by-step thought process"
}
```

## 8. Strategy Guidelines

1. Always return a valid JSON with the action and reasoning fields.
2. Use signal direction and distance to guide movement toward the bomb.
3. Plan paths that avoid walls and minimise detours.
4. Track how the distance shrinks to verify you're heading the right way.
5. **Pay attention to the remaining time.**
6. Track the time passing based on previous states and actions.
7. Keep reasoning clear but concise, especially when time is limited.

Remember, lives are at stake! Your mission is to find the bomb as quickly as possible.
"""


SYS_PROMPT_MOVING_BOMB_DETECT = """
# Grid World Bomb Detection Task

You are a police in a grid world.
Your mission: **locate and defuse a hidden bomb before it explodes**.
This is a challenging task because the bomb is moving!

---

## 1. Environment and Coordinate System

- You are in a two-dimensional grid world with obstacles (walls).
- The grid uses a standard Cartesian coordinate system where [1,1] is the bottom-left corner.
- X-axis runs horizontally (left to right), Y-axis runs vertically (bottom to top).
- For example, in a X $\times$ Y grid:
  - Bottom-left corner is [1,1]
  - Bottom-right corner is [X,1]
  - Top-left corner is [1,Y]
  - Top-right corner is [X,Y]
- You will receive a map of the grid world represented as a 2D array.
- Each row in the array corresponds to a y-coordinate (bottom row = y:1, top row = y:Y).
- Each column corresponds to an x-coordinate (leftmost column = x:1, rightmost column = x:X).
- To convert between Cartesian coordinates [x,y] and the map array:
  - For example, position [7,6] refers to the 7th column from the left and the 6th row from the bottom

**Important Map Representation Notes:**

In the map representation:
- "X" represents your current position
- "#" represents walls (you cannot move through walls)
- "0" represents empty spaces where you can move
- The bomb is hidden somewhere in the grid (not shown on the map)

## 2. The Moving Bomb and Signal System

- The bomb is moving throughout the grid world and will never pass through walls.
- The bomb moves slower than you, but you need to track it with periodic detection ("detect" action).
- The bomb emits signals that help you locate it:
  - **Signal Direction**: When you perform a "detect" action, you'll receive an angle pointing toward the bomb, consider you have a compass and the angle is the direction of the bomb:
    - 0 or 360 degrees = NORTH (the bomb is at north of you, consider going north)
    - 90 degrees = EAST (the bomb is at east of you, consider going east)
    - 180 degrees = SOUTH (the bomb is at south of you, consider going south)
    - 270 degrees = WEST (the bomb is at west of you, consider going west)
  - **Signal Distance**: You will also receive an estimated distance from your current position to the bomb.
    - This is an Euclidean distance, so you can use it to estimate how far the bomb is from you.
- **Important**: You only receive updated directional and distance information when you perform a "detect" action. 
  - When you move, the directional and distance information is not automatically updated.
- The bomb is a time bomb, and it will explode after a certain amount of time. You need to reach the bomb before it explodes.
  - The value of **remaining time** (in seconds) will be provided at each step without the need to perform a "detect" action.

## 3. Available Actions
You can perform five different actions:
- "north": Move one cell north (y + 1)
- "south": Move one cell south (y – 1)
- "east": Move one cell east (x + 1)
- "west": Move one cell west (x – 1)
- "detect": Stay in place and receive updated information about the bomb's current location

## 4. Time Management
**Time is Continuously Ticking — Use It Wisely**:
**In this task, time flows continuously, and every action—including your reasoning—consumes it.**
This includes:
- Movement, even if you are blocked by a wall, it also consumes time.
- Detection, which consumes time but provides updated information about the bomb's location.
- Your thought process: the longer and more detailed your reasoning, the more time you lose, lengthy and detailed analysis will consume valuable time.
To succeed, you must:
- **Balance planning with speed** — avoid overthinking when time is limited.
- **Balance movement with detection** — you need updated info, but too many detections waste time.
- **Adapt to urgency** — simplify your decisions as the situation becomes more critical.
- **Stay aware of time already spent** — use past states and actions to estimate how much time remains before the bomb explodes.
Think smart. Act fast.


## 5. Mission Termination
The mission ends when either:
- You successfully locate the bomb (**success**)
- The bomb explodes before you reach it (failure)
- You exceed the maximum allowed number of steps (**failure**)

## 6. State Format (input to you each turn)
For each step, you will receive the environment state as a JSON object:
```json
{
  "last_action": "start" or "moved" or "blocked_by_wall" or "detected",
  "current_location": [x, y],
  "signal_direction": angle_in_degrees,  // Only present after a "detect" action or at the start
  "signal_distance": euclidean_distance_in_cells,  // Only present after a "detect" action or at the start
  "last_detected_signal_direction": angle_in_degrees,  // Present when you have previously detected but not performed a new detection
  "last_detected_signal_distance": euclidean_distance_in_cells,  // Present when you have previously detected but not performed a new detection
  "remaining_time": time_in_seconds,
  "last_reasoning_time_consumed": time_in_seconds  // Time consumed in the last reasoning
}
```

## 7. Response Format
You must respond with a JSON object containing your action and reasoning without any additional text or explanations:
```json
{
  "action": "north" | "south" | "east" | "west" | "detect",
  "reasoning": "Brief step-by-step thought process"
}
```

## 8. Strategy Guidelines

1. Always return a valid JSON with the action and reasoning fields.
2. Use signal direction and distance to guide movement toward the bomb.
3. **Balance movement with detection - you need updated info, but too many detections waste time.**
4. Plan paths that avoid walls and minimise detours.
5. Consider the bomb's movement when planning your path.
6. Track how the distance shrinks to verify you're heading the right way.
7. **Pay attention to the remaining time.**
8. Track the time passing based on previous states and actions.
9. Keep reasoning clear but concise, especially when time is limited.

Remember, lives are at stake! Your mission is to find the moving bomb as quickly as possible.
"""


USER_PROMPT_TEMPLATE = """
## Map

{map}

Map Size: {map_size}

## Wall Coordinates

{walls}

## Environment State

```json
{state}
```
"""


class Action(BaseModel):
    action: str
    reasoning: str


# Define directions and corresponding moves for Cartesian coordinates
# In Cartesian: up = increasing y, down = decreasing y
# left = decreasing x, right = increasing x
ACTION_SPACE = {
    "north": (0, 1),
    "south": (0, -1),
    "west": (-1, 0),
    "east": (1, 0),
    "detect": (0, 0),
    "invalid": (0, 0),
}


class GridWorldEnv:

    @classmethod
    def add_arguments(self, parser):
        """Add arguments to the argument parser"""

        # Game environment parameters
        parser.add_argument(
            "--grid_size",
            type=str,
            default="(10, 10)",
            help="Size of the grid world (width, height)",
        )
        parser.add_argument(
            "--wall_density",
            type=float,
            default=0.1,
            help="Density of walls in the grid world",
        )
        parser.add_argument(
            "--initial_remaining_time",
            type=int,
            default=300,
            help="Initial remaining time for the agent",
        )
        parser.add_argument(
            "--time_measurement_mode",
            type=str,
            choices=["wall", "token"],
            default="token",
            help="Mode for measuring time consumption",
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=50,
            help="Maximum number of steps for the simulation",
        )

        # Game rule parameters
        parser.add_argument(
            "--action_time_consume",
            type=str,
            default="{'get_state': 1, 'move': 1, 'think': 0.1, 'detect': 1, 'invalid': 1}",
            help="Time consumed for each action",
        )
        parser.add_argument(
            "--disable_time",
            action="store_true",
            help="Disable time count down",
        )
        parser.add_argument(
            "--disable_bomb_move",
            action="store_true",
            help="Disable bomb movement during the simulation",
        )
        parser.add_argument(
            "--bomb_move_ratio",
            type=int,
            default=3,
            help="Agent moves N steps, bomb moves 1 step (N:1 ratio)",
        )
        parser.add_argument(
            "--disable_detect",
            action="store_true",
            help="Disable detection action during the simulation",
        )

        # Program parameters
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode for detailed logging",
        )

        # LLM parameters
        parser.add_argument(
            "--disable_sys_prompt",
            action="store_true",
            help="Disable system prompt for LLM",
        )
        parser.add_argument(
            "--json_mode",
            action="store_true",
            help="Enable JSON mode for llm generation",
        )
        parser.add_argument(
            "--task_type",
            type=str,
            # choices=["treasure", "static_bomb", "static_bomb_hint", "static_bomb_hint_urge", "static_bomb_hint_urge2", "static_bomb_urge2_no_hint", "moving_bomb", "moving_bomb_detect"],
            choices=["treasure", "static_bomb", "moving_bomb_detect"],
            default="treasure",
            help="Type of task to be performed",
        )

    def __init__(self, args, llm: LLM = None, seed=1):

        self.args = args

        # Seed everything
        random.seed(seed)
        self.seed = seed

        # Game environment parameters
        self.size = eval(args.grid_size)
        self.wall_density = args.wall_density
        self.initial_remaining_time = args.initial_remaining_time
        self.time_measurement_mode = args.time_measurement_mode
        self.max_steps = args.max_steps

        # Game rule parameters
        self.action_time_consume = eval(args.action_time_consume)
        self.disable_time = args.disable_time
        self.disable_bomb_move = args.disable_bomb_move
        self.disable_detect = args.disable_detect
        self.bomb_move_ratio = args.bomb_move_ratio

        # Program parameters
        self.debug = args.debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")

        # LLM parameters
        self.task_type = args.task_type
        self.disable_sys_prompt = args.disable_sys_prompt
        self.json_mode = args.json_mode
        available_system_prompts = {
            "treasure": SYS_PROMPT_STATIC_TREASURE,
            "static_bomb": SYS_PROMPT_STATIC_BOMB,
            "moving_bomb_detect": SYS_PROMPT_MOVING_BOMB_DETECT,
        }
        self.system_prompt = available_system_prompts.get(
            self.task_type, SYS_PROMPT_STATIC_BOMB
        )
        self.user_prompt_template = {
            "static_bomb_hint": USER_PROMPT_TEMPLATE,
        }.get(self.task_type, USER_PROMPT_TEMPLATE)
        self.action_space = ACTION_SPACE

        # Simulation variables
        self.step = 0
        self.bomb_trajectory = []
        self.agent_trajectory = []
        self.record = []
        self.last_detected_signal_direction = None
        self.last_detected_signal_distance = None

        # LLM variables
        self.llm = llm
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        # self.reset()

    def reset(self):
        """Reset the grid world to initial state"""
        # Reinitialize the map with empty spaces
        # self.map = [["0" for _ in range(self.size[0])] for _ in range(self.size[1])]

        # Reset time
        self.remaining_time = self.initial_remaining_time

        # Reset walls
        self.walls = self.generate_walls()

        # Place agent and bomb at a random position (not the same as agent, and not blocked by walls)
        while True:
            agent_x = random.randint(0, self.size[0] - 1)
            agent_y = random.randint(0, self.size[1] - 1)
            # Check if agent is placed at wall position
            if (agent_x, agent_y) in self.walls:
                continue

            bomb_x = random.randint(0, self.size[0] - 1)
            bomb_y = random.randint(0, self.size[1] - 1)
            # Check if bomb is placed at wall position
            if (bomb_x, bomb_y) in self.walls:
                continue
            # Ensure bomb is not placed at agent position
            if (bomb_x, bomb_y) == (agent_x, agent_y):
                continue
            # Ensure there should be at least 1 valid path to the bomb, and should at least be 4 steps away
            valid_path = self.find_path((agent_x, agent_y), (bomb_x, bomb_y))
            if len(valid_path) < 4:
                continue
            # If all checks pass, break the loop
            self.agent_pos = (agent_x, agent_y)
            self.bomb_pos = (bomb_x, bomb_y)
            break

        # Reset counters and trajectories and record
        self.record = []
        self.step = 0
        self.last_detected_signal_direction = None
        self.last_detected_signal_distance = None
        self.agent_trajectory = [self.agent_pos]
        self.bomb_trajectory = [self.bomb_pos]

        if self.debug:
            logger.debug(f"Initial agent position: {self.agent_pos}")
            logger.debug(f"Initial bomb position: {self.bomb_pos}")
            logger.debug(f"Initial walls: {self.walls}")
            logger.debug(
                f"Initial map representation:\n{self.print_map(show_bomb=False, to_stdout=False)}"
            )
        self.llm.reset()

    def get_map_representation(self, show_bomb=False):
        """Get the current visual representation of the map"""
        # Update map to reflect current state
        # Create a new map with [y][x] indexing for Cartesian coordinates
        map_repr = [["0" for _ in range(self.size[0])] for _ in range(self.size[1])]

        # Add walls
        for wall_x, wall_y in self.walls:
            map_repr[wall_y][wall_x] = "#"

        # Add agent
        agent_x, agent_y = self.agent_pos
        map_repr[agent_y][agent_x] = "X"

        # Add bomb only if show_bomb is True
        if show_bomb:
            bomb_x, bomb_y = self.bomb_pos
            map_repr[bomb_y][bomb_x] = "B"

        return map_repr

    # def update_map_representation(self, show_bomb=False):
    #     """Update the visual representation of the map"""
    #     # Reset map to empty spaces
    #     self.map = self.get_map_representation(show_bomb=show_bomb)

    def print_map(self, show_bomb=True, to_stdout=False):
        """Print the grid map with walls, agent, and bomb"""
        # Update map first to ensure it reflects current state
        map_repr = self.get_map_representation(show_bomb=show_bomb)

        map_str = "\n".join(
            [
                str(row)
                .replace("'", "")
                .replace(",", "")
                .replace("[", "[ ")
                .replace("]", " ]")
                + f" => Y={self.size[1]-i}"
                for i, row in enumerate(reversed(map_repr))
            ]
        )
        map_str += (
            "\n"
            + str([str(i + 1) for i in range(self.size[0])])
            .replace("'", "")
            .replace(",", "")
            .replace("[", "[ ")
            .replace("]", " ]")
            + " => X-axis"
        )
        if to_stdout:
            for i, row in enumerate(reversed(map_repr)):
                print(" ".join(row) + f" => {self.size[1]-i}")
            print(" ".join([str(i + 1) for i in range(self.size[0])]))
            print()
        return map_str

    def get_wall_coordinates(self):
        """Get the current walls in the grid world"""
        return [
            (wall[0] + 1, wall[1] + 1)
            for wall in list(sorted(list(self.walls), key=lambda x: (x[0], x[1])))
        ]

    def print_trajectory(self, to_stdout=False):
        """Display the bomb and agent trajectories"""
        # Create a copy of the map
        traj_map = self.get_map_representation(show_bomb=False)

        # Mark the agent trajectory with 'a'
        for i, pos in enumerate(self.agent_trajectory):
            x, y = pos
            # if i > 0 and i < len(self.agent_trajectory):  # Skip first and last positions
            traj_map[y][x] = "a"

        # Mark the bomb trajectory with 'b'
        for i, pos in enumerate(self.bomb_trajectory):
            x, y = pos
            # if i > 0 and i < len(self.bomb_trajectory) - 1:  # Skip first and last positions
            # Don't overwrite agent position
            if traj_map[y][x] != "a":
                traj_map[y][x] = "b"

        # Mark current positions
        agent_x, agent_y = self.agent_pos
        traj_map[agent_y][agent_x] = "X"

        bomb_x, bomb_y = self.bomb_pos
        if bomb_x != agent_x or bomb_y != agent_y:  # Don't overwrite agent position
            traj_map[bomb_y][bomb_x] = "B"
        else:
            traj_map[bomb_y][bomb_x] = "F"

        map_str = "\n".join([str(row) for row in reversed(traj_map)])
        if to_stdout:
            print("Agent and Bomb Trajectories:")
            print("X: Current agent position")
            print("a: Previous agent positions")
            print("B: Current bomb position")
            print("b: Previous bomb positions")
            for row in reversed(traj_map):
                print(" ".join(row))
            print()
        return map_str

    def generate_walls(self):
        """Generate walls randomly based on wall density"""
        total_cells = self.size[0] * self.size[1]
        wall_count = int(total_cells * self.wall_density)
        walls = set()
        while len(walls) < wall_count:
            x, y = random.randint(0, self.size[0] - 1), random.randint(
                0, self.size[1] - 1
            )
            walls.add((x, y))
        return walls

    def find_path(self, agent_pos=None, bomb_pos=None):
        """Find the optimal path from agent to bomb using BFS"""
        if agent_pos is None:
            agent_pos = self.agent_pos
        if bomb_pos is None:
            bomb_pos = self.bomb_pos
        queue = deque([(agent_pos, [])])
        visited = set([agent_pos])
        valid_actions = list(filter(lambda x: sum(x) != 0, ACTION_SPACE.values()))
        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == bomb_pos:
                return path
            for dx, dy in valid_actions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size[0] and 0 <= ny < self.size[1]:
                    if (nx, ny) not in visited and (nx, ny) not in self.walls:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))
        return []

    def direction_angle(self):
        """
        Calculate direction angle from agent to bomb in degrees
        North is 0 degrees, East is 90 degrees, South is 180 degrees, West is 270 degrees
        """
        ax, ay = self.agent_pos
        bx, by = self.bomb_pos

        # Calculate the vector from agent to bomb
        dx = bx - ax
        dy = by - ay

        # In Cartesian coordinates, North is increasing y, East is increasing x
        angle_rad = math.atan2(dx, dy)

        # Convert to degrees and adjust to have North at 0 degrees
        angle_deg = (math.degrees(angle_rad) + 360) % 360

        return int(angle_deg)

    def get_euclidean_distance(self):
        """Calculate the Euclidean distance between agent and bomb"""
        ax, ay = self.agent_pos
        bx, by = self.bomb_pos
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def get_environment_state(
        self, last_action, has_detected=False, step_time_consumed=0
    ):
        """Get the current state of the environment"""
        state = {
            "last_action": last_action,
            "current_location": [self.agent_pos[0] + 1, self.agent_pos[1] + 1],
        }

        signal_direction = self.direction_angle()
        signal_distance = self.get_euclidean_distance()
        if self.disable_detect:
            state["signal_direction"] = signal_direction
            state["signal_distance"] = f"{signal_distance:.2f}"
        else:
            if has_detected or self.last_detected_signal_direction is None:
                self.last_detected_signal_direction = signal_direction
                self.last_detected_signal_distance = signal_distance
                state["signal_direction"] = signal_direction
                state["signal_distance"] = f"{signal_distance:.2f}"
            else:
                state["last_detected_signal_direction"] = (
                    self.last_detected_signal_direction
                )
                state["last_detected_signal_distance"] = (
                    f"{self.last_detected_signal_distance:.2f}"
                )

        if not self.disable_time:
            state["remaining_time"] = str(int(self.remaining_time)) + " seconds"
            state["last_reasoning_time_consumed"] = (
                str(int(step_time_consumed)) + " seconds"
            )
            if self.task_type in [ "static_bomb_hint", "static_bomb_hint_urge", "static_bomb_hint_urge2"]:
                state["tokens_used_for_reasoning"] = self.time_unit_to_len(step_time_consumed)

        return state, self.action_time_consume["get_state"]

    def move_bomb(self):
        """Move the bomb randomly, avoiding walls and staying in bounds"""
        # Possible directions for bomb to move (simplified to 4 directions)
        should_move_bomb = (
            self.step % self.bomb_move_ratio == 0
            if not self.disable_bomb_move
            else False
        )
        bomb_moved = False
        bomb_new_position = self.bomb_pos
        if self.disable_bomb_move:
            return should_move_bomb, bomb_moved, bomb_new_position

        if not should_move_bomb:
            return should_move_bomb, bomb_moved, bomb_new_position

        possible_moves = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self.bomb_pos[0] + dx, self.bomb_pos[1] + dy
            if (
                0 <= nx < self.size[0]
                and 0 <= ny < self.size[1]
                and (nx, ny) not in self.walls
            ):
                possible_moves.append((nx, ny))

        # If no valid moves, bomb stays in place
        if not possible_moves:
            return should_move_bomb, bomb_moved, bomb_new_position

        # Move bomb to a random valid position
        self.bomb_pos = random.choice(possible_moves)
        self.bomb_trajectory.append(self.bomb_pos)
        bomb_moved = True
        bomb_new_position = self.bomb_pos
        return should_move_bomb, bomb_moved, bomb_new_position

    def execute_action(self, action):
        """Move the agent in the specified direction or perform detection"""
        # Check if bomb should move (based on the move ratio)

        action_time_consume = self.action_time_consume.get(action, 0)
        action_feedback = "moved"
        action_is_detect = False
        if action == "detect":
            # Detection doesn't change position but consumes time
            action_is_detect = True
            action_feedback = "detected"
            return action_feedback, action_time_consume, action_is_detect

        if action == "invalid":
            # Invalid action, do nothing
            action_feedback = "invalid"
            return action_feedback, action_time_consume, action_is_detect

        dx, dy = ACTION_SPACE[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        if (
            not (0 <= nx < self.size[0] and 0 <= ny < self.size[1])
            or (nx, ny) in self.walls
        ):
            # If the move is blocked by a wall or out of bounds, stay in place
            action_feedback = "blocked_by_wall"

            return action_feedback, action_time_consume, action_is_detect

        self.agent_pos = (nx, ny)
        self.agent_trajectory.append(self.agent_pos)

        return action_feedback, action_time_consume, action_is_detect

    def run_simulation(self):
        """Run the simulation with LLM as the agent"""
        # Reset the environment
        self.reset()
        action_feedback = "start"
        has_detected = True
        logger.debug("Starting simulation...")
        step_time_consumed = 0
        for i in tqdm.tqdm(range(self.max_steps)):
            logger.debug("\n" + "---" * 20 + "\n")
            logger.debug(f"Step: {i + 1}/{self.max_steps}")

            should_move_bomb, bomb_moved, bomb_current_pos = self.move_bomb()
            if self.debug:
                self.print_map(show_bomb=True, to_stdout=True)
            logger.debug(f"Wall coordinates: {self.get_wall_coordinates()}")
            logger.debug(f"Should move bomb: {should_move_bomb}")
            logger.debug(f"Bomb at: {[self.bomb_pos[0] + 1, self.bomb_pos[1] + 1]}")
            logger.debug(f"Agent at: {[self.agent_pos[0] + 1, self.agent_pos[1] + 1]}")
            logger.debug(
                f"Current Location Info: {self.direction_angle()} {self.get_euclidean_distance()}"
            )
            logger.debug(f"At least {len(self.find_path())} steps away from bomb")
            logger.debug(f"Remaining time: {self.remaining_time}")
            # Get environment state
            env_state, time_consumed = self.get_environment_state(
                action_feedback, has_detected, step_time_consumed
            )
            step_time_consumed = time_consumed
            # self.remaining_time -= time_consumed
            logger.debug(f"Getting environment state: {env_state}")
            # Call LLM for decision
            action, llm_output, time_consumed = self.call_llm(env_state)
            step_time_consumed += time_consumed

            # Execute action
            action_feedback, action_time, has_detected = self.execute_action(action)
            step_time_consumed += action_time

            if not self.disable_time:
                self.remaining_time -= step_time_consumed
            logger.debug(f"Action: {action}")
            logger.debug(f"Action feedback: {action_feedback}")
            logger.debug(f"Remaining time: {self.remaining_time}")
            # Check outcomes
            if self.remaining_time <= 0:
                logger.debug("Time is up!")
                self.record.append(
                    (
                        env_state,
                        action,
                        "bomb_exploded",
                        self.remaining_time,
                        llm_output,
                    )
                )
                break
            if self.agent_pos == self.bomb_pos:
                logger.debug("Bomb found!")
                self.record.append(
                    (env_state, action, "found_bomb", self.remaining_time, llm_output)
                )
                break
            self.record.append(
                (env_state, action, action_feedback, self.remaining_time, llm_output)
            )
            self.step += 1
            logger.debug("\n" + "---" * 20 + "\n")
        return self.record

    def save_results(self, output_path="dynamic_simulation_result.pkl"):

        result = {
            "args": {**self.args.__dict__, "seed": self.seed},
            "record": self.record,
            "map_size": self.size,
            "walls": list(self.walls),
            "initial_agent_pos": self.agent_trajectory[0],
            "initial_bomb_pos": self.bomb_trajectory[0],
            "initial_remaining_time": self.initial_remaining_time,
            "agent_trajectory": self.agent_trajectory,
            "bomb_trajectory": self.bomb_trajectory,
            "actions": [h[1] for h in self.record if h[1]],
            "events": [h[2] for h in self.record],
            "remaining_times": [h[3] for h in self.record],
            "success": any(
                event == "found_bomb" for event in [h[2] for h in self.record]
            ),
        }

        # Advanced metrics
        result["steps_taken"] = len(self.record)
        result["detect_actions"] = len(
            [h for h in self.record if h[1] == "detect" and h[1] is not None]
        )
        result["move_actions"] = result["steps_taken"] - result["detect_actions"]
        result["bomb_moves"] = len(self.bomb_trajectory) - 1

        if result["success"]:
            result["time_efficiency"] = (
                result["remaining_times"][-1] / self.initial_remaining_time
            )
        else:
            result["time_efficiency"] = 0

        # Save LLM outputs for analysis
        result["llm_outputs"] = [h[4] for h in self.record if h[4] is not None]
        result["chat_history"] = self.llm.chat_history

        # Save into a pickle file
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        logger.debug(f"Simulation results saved to {output_path}")

    def call_llm(self, env_state):
        """Call the LLM for decision making"""
        start_time = time.time()
        message = []
        user_prompt = self.user_prompt_template.format(
            map=self.print_map(show_bomb=False, to_stdout=False),
            map_size=str(self.size),
            walls=str(self.get_wall_coordinates()),
            state=json.dumps(env_state, indent=4),
        )
        if self.step == 0:
            if not self.disable_sys_prompt:
                message.append({"role": "system", "content": self.system_prompt})
                message.append({"role": "user", "content": user_prompt})
            else:
                sys_and_user = (
                    self.system_prompt.strip()
                    + "\n\n---\n\nMission start, here is the initial state:\n\n"
                    + user_prompt.strip()
                )
                message.append({"role": "user", "content": sys_and_user})
        else:
            message.append({"role": "user", "content": user_prompt})

        response = self.llm.chat(
            message, no_tqdm=True, json_model=Action if self.json_mode else None
        )
        reasoning = response.get("reasoning", None)
        response = response.get("solution", "")

        if self.time_measurement_mode == "token":
            time_consumed = 0
            if reasoning:
                time_consumed = self.len_to_time_unit(reasoning)
            if reasoning != response:
                time_consumed += self.len_to_time_unit(response)
        else:
            time_consumed = (time.time() - start_time) * self.action_time_consume.get(
                "think", 1
            )
        try:
            action = validate_and_parse_json_output(response)[
                "action"
            ]
            if action not in ACTION_SPACE.keys():
                logger.warning(
                    f"Invalid action '{action}' from LLM. Defaulting to 'invalid'."
                )
                action = "invalid"
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            logger.debug(f"LLM output: {response}")
            action = "invalid"
        reasoning = "" if reasoning is None else reasoning
        llm_output = f"<think>{reasoning}</think><solution>{response}</solution>"
        return action, llm_output, time_consumed

    def len_to_time_unit(self, response):
        """Convert token length to time unit"""
        # Placeholder for actual token length calculation
        token_length = len(self.tokenizer.encode(response))
        time_consumed = token_length * self.action_time_consume.get(
            "think", 0.1
        )
        return time_consumed
    
    def time_unit_to_len(self, time_unit):
        """Convert time unit to token length"""
        # Placeholder for actual token length calculation
        token_length = time_unit / self.action_time_consume.get("think", 0.1)
        return int(token_length)
    
    @classmethod
    def resume_env(cls, simulation_result):
        """Resume the environment from a simulation result"""
        args = Namespace(**simulation_result["args"])
        env = cls(args=args)
        env.record = simulation_result["record"]
        env.agent_trajectory = simulation_result["agent_trajectory"]
        env.bomb_trajectory = simulation_result["bomb_trajectory"]
        env.agent_pos = simulation_result["initial_agent_pos"]
        env.bomb_pos = simulation_result["initial_bomb_pos"]
        env.walls = set(simulation_result["walls"])
        env.size = simulation_result["map_size"]
        env.remaining_time = simulation_result["initial_remaining_time"]
        env.step = len(env.record)
        env.bomb_pos = simulation_result["bomb_trajectory"][-1]
        env.agent_pos = simulation_result["agent_trajectory"][-1]
        return env

