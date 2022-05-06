"""
A simple predator-prey multi-agent reinforcement learning environment in
python without dependencies, and an implementation of an example q-learning agent.
Python 3.9
https://gist.github.com/qpwo/a19f43368afd77288bf3b7db81fdc18b/
February, 2021 -- Public domain dedication
"""

from dataclasses import dataclass, field
import random

random.seed(1)

Action = int
Pair = tuple[int, int]


@dataclass
class Env:
    "A predator-prey environment"
    pred_locs: tuple[Pair, ...]
    prey_locs: tuple[Pair, ...]
    preds: tuple["Agent", ...]
    preys: tuple["Agent", ...]
    size: Pair

    def transition(self) -> None:
        "Get actions from agents and move them"
        h = self.hash()
        pred_rewards, prey_rewards = self.reward_of()
        pred_locs = list(self.pred_locs)
        for i, pd in enumerate(self.preds):
            action = pd.choose(h, pred_rewards[i])
            pred_locs[i] = move(pred_locs[i], action, self.size)
        prey_locs = list(self.prey_locs)
        for i, py in enumerate(self.preys):
            action = py.choose(h, prey_rewards[i])
            prey_locs[i] = move(prey_locs[i], action, self.size)
        self.pred_locs = tuple(pred_locs)
        self.prey_locs = tuple(prey_locs)

    def reward_of(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """
        Gives reward of current state for each agent.
        Predators get +1 reward for each 'captured' prey.
        Prey is considered captured if two or more predators are adjacent (corners included).
        Prey gets +1 reward if it is not captured.
        """
        adjacent: list[list[int]] = [[] for _ in range(len(self.prey_locs))]
        for i, pyl in enumerate(self.prey_locs):
            for j, pdl in enumerate(self.pred_locs):
                if is_adjacent(pyl, pdl):
                    adjacent[i].append(j)
        pred_rewards = [0] * len(self.preds)
        prey_rewards = [0] * len(self.preys)
        for i, ls in enumerate(adjacent):
            if len(ls) >= 2:  # two or more predators capture prey
                for j in ls:
                    pred_rewards[j] += 1
            else:
                prey_rewards[i] += 1
        return tuple(pred_rewards), tuple(prey_rewards)

    def hash(self) -> int:
        "Environment state can be hashed for use in agent's Q-table"
        return hash((self.pred_locs, self.prey_locs))

    def __repr__(self) -> str:
        "Display environment as grid"
        grid = [[" "] * self.size[1] for _ in range(self.size[0])]
        for pdl in self.pred_locs:
            grid[pdl[0]][pdl[1]] = "X"
        for prl in self.prey_locs:
            grid[prl[0]][prl[1]] = "O"

        return (
            ("_" * self.size[1] * 2 + "\n")
            + "\n".join("|" + " ".join(row) + "|" for row in grid)
            + ("\n" + " ̅" * self.size[1] * 2 + "\n")
        )


@dataclass
class Agent:
    "Simple Q-learning agent"
    Q: dict[tuple[int, int], float] = field(default_factory=lambda: {(-1, -1): 0})
    prev_action: Action = -1
    prev_state: int = -1
    alpha: float = 0.7
    gamma: float = 0.618
    epsilon: float = 0.05

    def choose(self, state: int, reward: float) -> Action:
        for a in ACTIONS:
            if (state, a) not in self.Q:
                self.Q[state, a] = random.random() * 0.00001

        best_reward = max(self.Q[state, ac] for ac in ACTIONS)
        self.Q[self.prev_state, self.prev_action] += self.alpha * (
            reward + self.gamma * best_reward - self.Q[self.prev_state, self.prev_action]
        )

        action = (
            max(ACTIONS, key=lambda ac: self.Q[state, ac])
            if random.random() > self.epsilon
            else random.choice(ACTIONS)
        )

        self.prev_action = action
        self.prev_state = state

        return action


ACTIONS = (0, 1, 2, 3)
ACTION_TO_STRING = ("up", "down", "left", "right")
ACTION_TO_PAIR = ((-1, 0), (1, 0), (0, -1), (0, 1))


def move(start: Pair, action: Action, size: Pair) -> Pair:
    "Move pair in direction of action. Possible addition: Forbid agents from occupying same cell"
    dir = ACTION_TO_PAIR[action]
    result = start[0] + dir[0], start[1] + dir[1]
    if not (0 <= result[0] < size[0] and 0 <= result[1] < size[1]):
        return start
    return result


def is_adjacent(p: Pair, q: Pair) -> bool:
    return -1 <= p[0] - q[0] <= 1 and -1 <= p[1] - q[1] <= 1


if __name__ == "__main__":
    preds = [Agent() for _ in range(3)]
    preys = [Agent() for _ in range(3)]
    rows = cols = 5

    def randpair() -> Pair:
        return (random.randrange(rows), random.randrange(cols))

    pred_locs = [randpair() for _ in range(len(preds))]
    prey_locs = [randpair() for _ in range(len(preys))]

    env = Env(tuple(pred_locs), tuple(prey_locs), tuple(preds), tuple(preys), (rows, cols))
    print("STARTING TRAINING")
    train_steps = 100_000
    for i in range(train_steps):
        if i % (train_steps // 20) == 0:
            print(f"Completed {i} / {train_steps} steps")
        env.transition()
    print("Now watch environment proceed step-by-step")
    while True:
        print(env)
        print(env.reward_of())
        env.transition()
        input("Press enter")