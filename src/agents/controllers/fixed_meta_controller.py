class FixedMetaController:
    def __init__(self, subgoals: list[int]):
        self.index = 0
        self.subgoals = subgoals

    def reset_counter(self):
        self.index = 0

    def subgoal_done(self):
        self.index += 1

    def select_goal(self, _, __) -> int:  # match MetaController.select_goal()
        # if invalid index, choose last subgoal
        if self.index >= len(self.subgoals) or self.index < 0:
            print("fixed meta index higher than len(subgoals)")
            return self.subgoals[-1]

        return self.subgoals[self.index]
