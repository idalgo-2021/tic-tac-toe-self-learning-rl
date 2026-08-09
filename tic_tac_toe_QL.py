import argparse
import random
import numpy as np

LINES = [
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
]


def check_winner(board):
    """Возвращает 1, если выиграл X, -1 если O, иначе None."""
    for a, b, c in LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_full(board):
    return all(c != 0 for c in board)


def print_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    rows = []
    for r in range(3):
        row = board[r * 3 : r * 3 + 3]
        rows.append(" ".join(symbols[c] for c in row))
    print("\n".join(rows))


def encode_state(board, player):
    """Доска с точки зрения текущего игрока: его фишки = +1, чужие = −1."""
    return [cell * player for cell in board]


def one_hot_action(move):
    v = np.zeros(9, dtype=float)
    v[move] = 1.0
    return v


# ---------------------------------------------------------------------------
# Нейросеть Q(s, a): 18 → hidden(36) → 1
#
# Выход:
#   Q(s, a) ∈ [−1; 1]  — оценка хода a в состоянии s
#   +1  — отличный ход для текущего игрока
#    0  — нейтральный / ничья
#   −1  — плохой ход
# ---------------------------------------------------------------------------


class TicTacToeQNet:
    def __init__(self, hidden=36, seed=None):
        rng = np.random.default_rng(seed)
        # вход: 9 (состояние) + 9 (one-hot действия)
        self.W1 = rng.normal(0, 0.4, (hidden, 18))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.4, hidden)
        self.b2 = 0.0

    def _features(self, state, move):
        """Вектор признаков [state | one_hot(move)]."""
        return np.concatenate([np.asarray(state, dtype=float), one_hot_action(move)])

    def forward(self, state, move):
        x = self._features(state, move)
        self._x = x
        self._z1 = self.W1.dot(x) + self.b1
        self._a1 = np.tanh(self._z1)
        z2 = self.W2.dot(self._a1) + self.b2
        self._out = np.tanh(z2)
        return self._out

    def train_step(self, state, move, target, lr):
        out = self.forward(state, move)
        d_out = (out - target) * (1.0 - out**2)  # dL/dz2
        dW2 = d_out * self._a1
        db2 = d_out
        da1 = d_out * self.W2
        dz1 = da1 * (1.0 - self._a1**2)
        dW1 = np.outer(dz1, self._x)
        db1 = dz1

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def q_values(self, state, moves):
        """Возвращает список (move, Q) для всех доступных ходов."""
        return [(m, float(self.forward(state, m))) for m in moves]

    def best_move(self, state, moves):
        """Жадный выбор: argmax_a Q(s, a)."""
        qs = self.q_values(state, moves)
        return max(qs, key=lambda t: t[1])[0]

    def max_q(self, state, moves):
        """max_a Q(s, a). Если ходов нет — 0."""
        if not moves:
            return 0.0
        return max(q for _, q in self.q_values(state, moves))

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], float(data["b2"])


def choose_move(net, board, player, moves, epsilon=0.0):
    """ε-greedy: с вероятностью ε — случайный ход, иначе argmax Q."""
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(moves)
    state = encode_state(board, player)
    return net.best_move(state, moves)


# ---------------------------------------------------------------------------
# Self-play + Q-learning
#
# После хода игрока:
#   если терминал → target = r  (+1 / −1 / 0)
#   иначе         → target = − max_{a'} Q(s', a')
#     (знак «−», потому что s' — позиция противника; его выгода = наш убыток)
#
#   Q(s, a) ← Q(s, a) + α · (target − Q(s, a))
# ---------------------------------------------------------------------------


def self_play_episode(net, epsilon, lr):
    board = [0] * 9
    player = 1  # X начинает

    while True:
        state = encode_state(board, player)
        moves = [i for i in range(9) if board[i] == 0]

        move = choose_move(net, board, player, moves, epsilon)
        board[move] = player

        winner = check_winner(board)

        if winner == player:
            target = 1.0
        elif winner is not None:
            # не должно случиться (только что ходили)
            target = -1.0
        elif is_full(board):
            target = 0.0
        else:
            # позиция после нашего хода - ход противника
            next_state = encode_state(board, -player)
            next_moves = [i for i in range(9) if board[i] == 0]
            # max Q для противника; наш target = − его max Q
            target = -net.max_q(next_state, next_moves)

        net.train_step(state, move, target, lr)

        if winner is not None or is_full(board):
            break

        player = -player


def random_move(board):
    moves = [i for i in range(9) if board[i] == 0]
    return random.choice(moves)


def evaluate_vs_random(net, games=200):
    """Сеть играет за X против случайных ходов O (ε = 0)."""
    wins = draws = losses = 0
    for _ in range(games):
        board = [0] * 9
        player = 1
        while True:
            if player == 1:
                moves = [i for i in range(9) if board[i] == 0]
                move = choose_move(net, board, 1, moves, epsilon=0.0)
            else:
                move = random_move(board)
            board[move] = player
            winner = check_winner(board)
            if winner is not None:
                wins += winner == 1
                losses += winner == -1
                break
            if is_full(board):
                draws += 1
                break
            player = -player
    return wins, draws, losses


def train(episodes, save_path, lr=0.05, hidden=36, seed=None):
    net = TicTacToeQNet(hidden=hidden, seed=seed)

    eps_start, eps_end = 0.5, 0.02

    for ep in range(1, episodes + 1):
        # линейно снижаем ε
        epsilon = eps_start + (eps_end - eps_start) * (ep / episodes)
        self_play_episode(net, epsilon, lr)

        if ep % max(1, episodes // 20) == 0:
            w, d, k = evaluate_vs_random(net, games=200)
            print(
                f"[{ep:>7}/{episodes}] eps={epsilon:.2f}  "
                f"vs random → побед {w}, ничьих {d}, поражений {k}"
            )

    net.save(save_path)
    print(f"\nМодель сохранена в: {save_path}")
    return net


def play(load_path):
    net = TicTacToeQNet()
    net.load(load_path)

    print("Клетки нумеруются так:")
    print("1 2 3\n4 5 6\n7 8 9\n")
    choice = (
        input("Хотите играть за X (ходит первым) или за O? [X/O]: ").strip().upper()
    )
    human = 1 if choice != "O" else -1
    ai = -human

    board = [0] * 9
    player = 1
    print_board(board)

    while True:
        if player == human:
            moves = [i for i in range(9) if board[i] == 0]
            while True:
                try:
                    raw = input("Ваш ход (1-9): ").strip()
                    idx = int(raw) - 1
                    if idx in moves:
                        break
                except ValueError:
                    pass
                print("Некорректный ввод, попробуйте снова.")
            board[idx] = player
        else:
            moves = [i for i in range(9) if board[i] == 0]
            idx = choose_move(net, board, ai, moves, epsilon=0.0)
            board[idx] = player
            print(f"\nХод нейросети: {idx + 1}")

        print()
        print_board(board)

        winner = check_winner(board)
        if winner is not None:
            if winner == human:
                print("\nВы выиграли!")
            else:
                print("\nПобедила нейросеть.")
            break
        if is_full(board):
            print("\nНичья.")
            break

        player = -player


def main():
    parser = argparse.ArgumentParser(
        description="Самообучающаяся нейросеть для крестиков-ноликов"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Обучить сеть через self-play + Q-learning")
    p_train.add_argument("--episodes", type=int, default=30000)
    p_train.add_argument("--save", type=str, default="model_ql.npz")
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--hidden", type=int, default=36)
    p_train.add_argument("--seed", type=int, default=None)

    p_play = sub.add_parser("play", help="Играть против обученной сети")
    p_play.add_argument("--load", type=str, default="model_ql.npz")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.episodes, args.save, lr=args.lr, hidden=args.hidden, seed=args.seed)
    elif args.mode == "play":
        play(args.load)


if __name__ == "__main__":
    main()
