"""
Loss functions for 3DPPE.
Consolidated and modernized from legacy loss implementations.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SetCriterion(nn.Module):
    """
    Set-based criterion for DETR-like object detection.
    Computes classification and regression losses.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: Optional[nn.Module] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes"]
    ):
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict or {
            "loss_ce": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        }
        self.eos_coef = eos_coef
        self.losses = losses

        # Classification loss
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]], num_boxes: int) -> Dict[str, Tensor]:
        """Classification loss."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]], num_boxes: int) -> Dict[str, Tensor]:
        """Bounding box regression loss."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(src_boxes),
            self.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Get source permutation indices."""
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        """Get target permutation indices."""
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def box_cxcywh_to_xyxy(x: Tensor) -> Tensor:
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """
        Generalized IoU from https://giou.stanford.edu/
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

        iou, union = SetCriterion.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

    @staticmethod
    def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute IoU between two sets of boxes."""
        area1 = SetCriterion.box_area(boxes1)
        area2 = SetCriterion.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    @staticmethod
    def box_area(boxes: Tensor) -> Tensor:
        """Compute area of boxes."""
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Forward pass computing all losses."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = getattr(self, f'loss_{loss}')(aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class DPPESetCriterion(SetCriterion):
    """
    3DPPE-specific set criterion with depth-aware losses.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: Optional[nn.Module] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes", "depth"]
    ):
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses)

    def loss_depth(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices: List[Tuple[Tensor, Tensor]], num_boxes: int) -> Dict[str, Tensor]:
        """Depth estimation loss."""
        if 'pred_depth' not in outputs:
            return {}

        idx = self._get_src_permutation_idx(indices)
        src_depth = outputs['pred_depth'][idx]

        # Get target depth from targets (if available)
        target_depth = torch.cat([t.get('depth', torch.zeros_like(src_depth[:len(t['labels'])])) for t, (_, i) in zip(targets, indices)], dim=0)

        if len(target_depth) == 0:
            return {'loss_depth': torch.tensor(0.0, device=src_depth.device)}

        loss_depth = F.l1_loss(src_depth, target_depth, reduction='none')
        losses = {'loss_depth': loss_depth.sum() / num_boxes}

        return losses


class ScaleInvariantLoss(nn.Module):
    """
    Scale-invariant loss for depth estimation.
    From "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """

    def __init__(self, lambda_: float = 0.5):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            pred: Predicted depth map (B, 1, H, W)
            target: Target depth map (B, 1, H, W)
            mask: Valid pixel mask (B, 1, H, W)
        """
        if mask is None:
            mask = torch.ones_like(target)

        # Convert to log space
        log_pred = torch.log(torch.clamp(pred, min=1e-8))
        log_target = torch.log(torch.clamp(target, min=1e-8))

        diff = log_pred - log_target
        diff = diff * mask

        # Scale-invariant loss
        loss = torch.mean(diff ** 2) - self.lambda_ * (torch.mean(diff) ** 2)

        return loss


class BalancedL1Loss(nn.Module):
    """
    Balanced L1 loss for depth estimation.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 1.5, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: Predicted values
            target: Target values
        """
        diff = torch.abs(pred - target)

        # Balanced L1 loss
        loss = torch.where(
            diff < self.beta,
            self.alpha * (diff ** self.gamma) / self.beta ** (self.gamma - 1),
            diff - self.alpha * self.beta ** self.gamma / (self.gamma - 1) + self.alpha * self.beta
        )

        return loss.mean()


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DETR-style object detection.
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """Perform the matching."""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concatenate the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -self.generalized_box_iou(
            self.box_cxcywh_to_xyxy(out_bbox),
            self.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [self.linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def box_cxcywh_to_xyxy(self, x: Tensor) -> Tensor:
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    def generalized_box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tensor:
        """
        Generalized IoU from https://giou.stanford.edu/
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

        iou, union = self.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

    def box_iou(self, boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
        area1 = self.box_area(boxes1)
        area2 = self.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    def box_area(self, boxes: Tensor) -> Tensor:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def linear_sum_assignment(self, cost_matrix: Tensor) -> Tuple[List[int], List[int]]:
        """Solve linear sum assignment problem."""
        try:
            import scipy.optimize
            indices = scipy.optimize.linear_sum_assignment(cost_matrix)
        except ImportError:
            # Fallback to greedy assignment
            indices = self.greedy_assignment(cost_matrix)

        return indices

    def greedy_assignment(self, cost_matrix: Tensor) -> Tuple[List[int], List[int]]:
        """Greedy assignment as fallback."""
        # Simple greedy assignment - assign each query to the best available target
        num_queries, num_targets = cost_matrix.shape
        assigned = set()
        indices = []

        for i in range(min(num_queries, num_targets)):
            # Find the minimum cost assignment
            flat_idx = torch.argmin(cost_matrix)
            query_idx = flat_idx // num_targets
            target_idx = flat_idx % num_targets

            # Skip if target already assigned
            if target_idx in assigned:
                cost_matrix[query_idx, target_idx] = float('inf')
                continue

            indices.append((query_idx.item(), target_idx.item()))
            assigned.add(target_idx)

        return tuple(zip(*indices)) if indices else ([], [])
