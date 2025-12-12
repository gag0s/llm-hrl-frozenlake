import json

from src.llm.llm import BaseLLM


class PromptBox:
    """
    PromptBox = "glue" between the HRL agent and the LLM.

    Responsibilities:
    - build a natural-language prompt (env description + formatting rules + optional feedback)
    - call the LLM
    - parse the LLM response into a machine-readable subgoal list (list[int])

    Note:
    The prompt requests a Python-style list like [5, 48, 24, 63].
    This is compatible with JSON array syntax, so json.loads() works here.
    """
    def __init__(self, llm: BaseLLM, frozenlake_map: list[str]):
        self.llm = llm
        self.frozenlake_map = frozenlake_map

    def get_subgoals(self, subgoals, subgoal_data, avg_reward) -> list[int]:
        """Public entry point: returns a validated list[int] of subgoal indices."""

        prompt = self._get_prompt(subgoals, subgoal_data, avg_reward)
        response = self.llm.get_response(prompt)

        try:
            result = json.loads(response)
            # Ensure the LLM returned the exact contract: JSON/Python list of ints.
            if isinstance(result, list) and all(isinstance(x, int) for x in result):
                return result
            else:
                raise ValueError("LLM Response is not a list of integers.")

        except Exception as e:
            raise ValueError(f"Expected a JSON list of ints, but got unparsable output: {response}") from e

    def _get_prompt(self, subgoals, subgoal_data, avg_reward) -> str:
        """Build the full prompt string (base env info + optional feedback)."""

        prompt = f"I am currently training a hdqn algorithm with an llm. The game is " \
                     f"FrozenLakes from Gymnasium. " \
                     f"Here is the map: " + self._format_map() + \
                     f"S = Start; G = Goal; H = Hole; F = Normal Tile. " \
                     f"" \
                     f"If the agent reaches the Goal, the Meta Controller gets a reward of +1. " \
                     f"If the agent falls into a Hole, the Meta Controller gets a penalty of -1. " \
                     f"That means that the maximum reward the agent gets for solving the level is 1. " \
                     f"The lowest reward the agent can get is -1, when falling into a hole. " \
                     f"The agent is only allowed to step on F tiles and lastly the goal G. " \
                     f"Propose a set of subgoals (between 5 to 6 subgoals) the agent has to reach to solve the map. " \
                     f"A subgoal is a number from 0-63. 0 represents the top left corner. " \
                     f"1 represents the second tile in the top row. 7 represents the top right corner. " \
                     f"56 is bottom left tile, 63 is the goal tile. " \
                     f"Please return only a list of the subgoal indices. No explanation or questions allowed. " \
                     f"An example of a response would be: [5, 48, 24, 63]\n"

        # Only add feedback after the first iteration (when data exists).
        if subgoal_data is not None:
            feedback = self._get_feedback(subgoals, subgoal_data, avg_reward)
            prompt += feedback

        return prompt

    def _get_feedback(self, subgoals, subgoal_data, avg_reward) -> str:
        feedback = f"This is not your first try. The last time you proposed subgoals, your agent" \
                   f"got an average reward of {avg_reward}. The maximum reward is 1.0. If you got" \
                   f"the average reward of 1.0, DON'T change your subgoals. Otherwise DO change them" \
                   f"If there is still room for improvement, here is a list of the subgoals you" \
                   f"chose last time, how often the meta controller attempted them over 100 evaluation" \
                   f"runs, and the percentage of how often the controller reached the subgoals," \
                   f"when they were selected:\n"

        subgoal_data_str = self._format_subgoal_data(subgoal_data)
        feedback += subgoal_data_str

        if avg_reward < 0.99:
            feedback += "\nHere is also what each subgoal you picked represents in the map:\n" + self._describe_subgoals(
                subgoals) + "\n"

        if len(subgoal_data) >= 7 and avg_reward < 0.99:
            feedback += f"\nMaybe try picking less subgoals. {len(subgoal_data)} Subgoals are propably a " \
                        f"bit too much. Keeping it between 5 and 6 is subgoals is good.\n"

        if len(subgoal_data) <= 4 and avg_reward < 0.99:
            feedback += f"\nMaybe try picking more subgoals. {len(subgoal_data)} Subgoals are propably a bit " \
                        f"too little. Keeping it between 5 and 6 subgoals is good.\n"

        return feedback

    def _format_map(self) -> str:
        """
        Returns a formatted string version of the map (no indices).
        Each cell is spaced out for clarity.
        """
        formatted_rows = [" ".join(row) for row in self.frozenlake_map]
        formatted_map = "\n".join(formatted_rows)
        return f"Map:\n```\n{formatted_map}\n```"

    def _format_subgoal_data(self, subgoal_data) -> str:
        """
        Formats each entry as:
        'index: <idx>, attempts: <attempts>, success_percentage: <percentage>%'
        If a subgoal has 0 attempts, success_percentage is shown as '-'.
        """
        lines = []
        for row in subgoal_data:
            if row["attempts"] == 0:
                perc_str = "-"
            else:
                perc_str = f"{row['success_percentage']}%"
            lines.append(
                f"index: {row['index']}, attempts: {row['attempts']}, success_percentage: {perc_str}"
            )
        return "\n".join(lines)

    def _describe_subgoals(self, subgoals):
        # Flatten the map into a single string for easy index access
        flat_map = ''.join(self.frozenlake_map)

        tile_descriptions = {
            'S': 'the starting point',
            'F': 'a free tile',
            'H': 'a hole',
            'G': 'the goal'
        }

        descriptions = []
        for idx in subgoals:
            tile = flat_map[idx]
            desc = tile_descriptions.get(tile, 'an unknown tile')
            descriptions.append(f"Subgoal {idx} is {desc}.")
        return '\n'.join(descriptions)
