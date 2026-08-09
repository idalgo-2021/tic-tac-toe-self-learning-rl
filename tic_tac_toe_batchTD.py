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
    """Возвращает 1, если выиграл X, -1 если O, иначе None"""
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


# ---------------------------------------------------------------------------
# Нейросеть (9 → hidden → 1)
# Выход V(s) ∈ [−1; 1]  — оценка позиции с точки зрения X
#   +1  выгодна X
#    0  ничья / равная
#   −1  выгодна O
# ---------------------------------------------------------------------------


class TicTacToeNet:
    def __init__(self, hidden=27, seed=None):
        rng = np.random.default_rng(seed)
        # W1, W2 - веса первого(9,27) и второго(27,1) слоёв
        # b1, b2 - смещения (bias)
        self.W1 = rng.normal(0, 0.5, (hidden, 9))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.5, hidden)
        self.b2 = 0.0

    def forward(self, x):
        self._x = x
        self._z1 = self.W1.dot(x) + self.b1  # (27,1): вычисляем W1 × x + b1
        self._a1 = np.tanh(self._z1)  # (27,1): каждое число чз активацию
        z2 = self.W2.dot(self._a1) + self.b2  # 27 весов(W2) х 27 значений(a1)
        self._out = np.tanh(z2)  # выход
        return self._out

    # Обратный ход:
    #   1. Вычисляем ошибку выхода
    #   2. Считаем градиенты второго слоя
    #   3. Распространяем ошибку на скрытый слой
    #   4. Считаем градиенты первого слоя
    #   5. Обновляем веса градиентным спуском
    def train_step(self, x, target, lr):
        out = self.forward(x)
        d_out = (out - target) * (1 - out**2)  # dL/dz2
        dW2 = d_out * self._a1
        db2 = d_out
        da1 = d_out * self.W2
        dz1 = da1 * (1 - self._a1**2)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], float(data["b2"])


def choose_move(net, board, player, moves, epsilon=0.0):
    """Выбирает ход: X максимизирует V(s'), O минимизирует V(s')."""
    if epsilon > 0 and random.random() < epsilon:
        return random.choice(moves)  # epsilon-greedy strategy

    # ОДНОШАГОВАЯ ОЦЕНОЧНАЯ ФУНКЦИЯ

    # Вычисляем оценку для каждой доступной позиции(предполагаемого хода)
    best_move, best_val = None, None
    for m in moves:
        b2 = list(board)
        b2[m] = player
        v = net.forward(np.array(b2, dtype=float))
        if (
            best_val is None
            or (player == 1 and v > best_val)
            or (player == -1 and v < best_val)
        ):
            best_val, best_move = v, m
    return best_move


# ---------------------------------------------------------------------------
# Self-play + Batch TD
#
# Партия целиком
#       |
#       v
# history [S1,S2,S3,S4,S5]
#       |
#       v
# обучение вперед S1 → S5


def self_play_episode(net, epsilon, lr):
    board = [0] * 9
    player = 1
    history = []

    while True:
        moves = [i for i in range(9) if board[i] == 0]
        move = choose_move(net, board, player, moves, epsilon)
        board[move] = player
        history.append(list(board))

        winner = check_winner(board)
        if winner is not None:
            result = float(winner)
            break
        if is_full(board):
            result = 0.0
            break
        player = -player

    # TD-обновление
    for i, state in enumerate(history):
        if i == len(history) - 1:
            target = result
        else:
            target = net.forward(np.array(history[i + 1], dtype=float))
        net.train_step(np.array(state, dtype=float), target, lr)


def random_move(board):
    moves = [i for i in range(9) if board[i] == 0]
    return random.choice(moves)


def evaluate_vs_random(net, games=200):
    """Сеть за X против случайного O."""
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


def train(episodes, save_path, lr=0.05, hidden=27, seed=None):
    net = TicTacToeNet(hidden=hidden, seed=seed)

    eps_start, eps_end = 0.5, 0.02  # epsilon-greedy strategy

    for ep in range(1, episodes + 1):
        epsilon = eps_start + (eps_end - eps_start) * (
            ep / episodes
        )
        self_play_episode(net, epsilon, lr)

        if ep % max(1, episodes // 20) == 0:
            w, d, k = evaluate_vs_random(net, games=200)
            print(
                f"[{ep:>7}/{episodes}] eps={epsilon:.2f}  "
                f"vs random -> побед {w}, ничьих {d}, поражений {k}"
            )

    net.save(save_path)
    print(f"\nМодель сохранена в: {save_path}")
    return net


# ---------------------------------------------------------------------------
# РЕЖИМ ИГРЫ ПОЛЬЗОВАТЕЛЯ С ПРОГРАММОЙ(СЕТЬЮ)


def play(load_path):
    net = TicTacToeNet()
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


# ---------------------------------------------------------------------------
#


def main():
    parser = argparse.ArgumentParser(
        description="Самообучающаяся нейросеть для крестиков-ноликов"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_train = sub.add_parser("train", help="Обучить сеть через self-play")
    p_train.add_argument("--episodes", type=int, default=30000)
    p_train.add_argument("--save", type=str, default="model_batch.npz")
    p_train.add_argument("--lr", type=float, default=0.05)
    p_train.add_argument("--hidden", type=int, default=27)
    p_train.add_argument("--seed", type=int, default=None)

    p_play = sub.add_parser("play", help="Играть против обученной сети")
    p_play.add_argument("--load", type=str, default="model_batch.npz")

    args = parser.parse_args()

    if args.mode == "train":
        train(args.episodes, args.save, lr=args.lr, hidden=args.hidden, seed=args.seed)
    elif args.mode == "play":
        play(args.load)


if __name__ == "__main__":
    main()
