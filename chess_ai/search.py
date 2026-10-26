from typing import Any
import math
import chess
import torch
import torch.nn.functional as F
from chess_ai.encoder import encode_board
from chess_ai.model import ChessNet

class MCTSNode:
    __slots__ = ('parent', 'prior', 'visit_count', 'value_sum', 'children', 'is_expanded', 'board')

    def __init__(self, board: chess.Board | None=None, parent=None, prior: float=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:

    def __init__(self, model: Any, device: str='cpu', c_puct: float=1.5):
        self.model = model
        self.device = device
        self.c_puct = c_puct
        p0 = next(model.parameters(), None)
        self._dtype = p0.dtype if p0 is not None else torch.float32

    def evaluate_board(self, board: chess.Board, history: list[chess.Board] | None=None) -> tuple[dict[chess.Move, float], float, list[float]]:
        moves = list(board.legal_moves)
        if not moves:
            val = -1.0 if board.is_check() else 0.0
            return ({}, val, [0.0, 1.0, 0.0])
        t = encode_board(board, history).unsqueeze(0).to(self.device, dtype=self._dtype, non_blocking=True)
        with torch.inference_mode():
            p_logits, v_logits = self.model(t)
            move_indices = torch.tensor([m.from_square << 6 | m.to_square for m in moves], dtype=torch.long, device=self.device)
            move_logits = p_logits[0].view(4096)[move_indices]
            move_probs_t = F.softmax(move_logits.float(), dim=0)
            wdl_t = F.softmax(v_logits[0].float(), dim=0)
            probs = move_probs_t.cpu().tolist()
            wdl = wdl_t.cpu().tolist()
        val_white = wdl[0] - wdl[2]
        turn_val = val_white if board.turn == chess.WHITE else -val_white
        move_priors = {}
        total_p = 0.0
        for m, p in zip(moves, probs):
            p_clamped = max(p, 1e-06)
            move_priors[m] = p_clamped
            total_p += p_clamped
        if total_p > 0:
            inv_total = 1.0 / total_p
            for m in move_priors:
                move_priors[m] *= inv_total
        return (move_priors, turn_val, wdl)

    def search(self, root_board: chess.Board, history: list[chess.Board] | None=None, num_simulations: int=100, add_dirichlet_noise: bool=False, temperature: float=0.0) -> tuple[chess.Move | None, dict[chess.Move, float], list[float]]:
        if not list(root_board.legal_moves):
            return (None, {}, [0.33, 0.33, 0.33])
        root = MCTSNode(board=root_board)
        priors, root_val, wdl = self.evaluate_board(root_board, history)
        if add_dirichlet_noise and priors:
            noise = torch.distributions.Dirichlet(torch.full((len(priors),), 0.3)).sample().tolist()
            for idx, m in enumerate(priors):
                priors[m] = 0.75 * priors[m] + 0.25 * noise[idx]
        for move, prior in priors.items():
            root.children[move] = MCTSNode(parent=root, prior=prior)
        root.is_expanded = True
        working_board = root_board.copy()
        history_boards: list[chess.Board] = list(history or [])
        for _ in range(num_simulations):
            node = root
            path_moves: list[chess.Move] = []
            while node.is_expanded and node.children:
                best_score = -float('inf')
                best_move = None
                best_child = None
                sqrt_total = math.sqrt(max(1, node.visit_count))
                for move, child in node.children.items():
                    u = self.c_puct * child.prior * (sqrt_total / (1 + child.visit_count))
                    q = -child.value
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_move = move
                        best_child = child
                if best_move is None:
                    break
                history_boards.append(working_board.copy())
                working_board.push(best_move)
                path_moves.append(best_move)
                node = best_child
            moves = list(working_board.legal_moves)
            if not moves:
                leaf_value = -1.0 if working_board.is_check() else 0.0
            elif working_board.is_insufficient_material() or working_board.is_repetition(3) or working_board.can_claim_fifty_moves():
                leaf_value = 0.0
            else:
                leaf_priors, leaf_value, _ = self.evaluate_board(working_board, history=history_boards)
                for move, prior in leaf_priors.items():
                    node.children[move] = MCTSNode(parent=node, prior=prior)
                node.is_expanded = True
            for _ in range(len(path_moves)):
                working_board.pop()
                history_boards.pop()
            curr = node
            curr_val = leaf_value
            while curr is not None:
                curr.visit_count += 1
                curr.value_sum += curr_val
                curr_val = -curr_val
                curr = curr.parent
        visit_counts = {m: child.visit_count for m, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        if total_visits == 0 or not visit_counts:
            best_move = max(priors, key=priors.get) if priors else None
            return (best_move, priors, wdl)
        policy = {m: count / total_visits for m, count in visit_counts.items()}
        if temperature <= 0.01:
            best_move = max(visit_counts, key=visit_counts.get)
        else:
            weights = [cnt ** (1.0 / temperature) for cnt in visit_counts.values()]
            tot = sum(weights)
            probs = [w / tot for w in weights]
            moves = list(visit_counts.keys())
            idx = torch.multinomial(torch.tensor(probs), 1).item()
            best_move = moves[idx]
        return (best_move, policy, wdl)
