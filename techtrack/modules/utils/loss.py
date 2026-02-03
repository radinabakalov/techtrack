import itertools
import numpy as np

class Loss:
    """
    *Modified* YOLO Loss for Hard Negative Mining.

    Attributes:
        num_classes (int): Number of classes.
        iou_threshold (float): Intersection over Union (IoU) threshold.
        lambda_coord (float): Weighting factor for localization loss.
        lambda_noobj (float): Weighting factor for no object confidence loss.
    """

    def __init__(self, iou_threshold=0.5, lambda_coord=0.5, lambda_obj=0.5, lambda_noobj=0.5, lambda_cls=0.5, num_classes=20):
        """
        Initialize the Loss object with the given parameters.

        Internal Process:
        1. Stores the provided hyperparameters as instance attributes.
        2. Defines the column names for loss components to track them in results.

        Args:
            num_classes (int): Number of classes.
            lambda_coord (float): Weighting factor for localization loss.
            lambda_obj (float): Weighting factor for objectness loss.
            lambda_noobj (float): Weighting factor for no object confidence loss.
            lambda_cls (float): Weighting factor for classification loss.
        """
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.columns = [
            'total_loss', 
            'loc_loss', 
            'conf_loss_obj', 
            'conf_loss_noobj', 
            'class_loss'
        ]
        self.iou_threshold = iou_threshold
    
    def get_predictions(self, predictions):
        """
        Extracts bounding box coordinates, objectness scores, and class scores from predictions.

        Internal Process:
        1. Iterates over predictions to extract bounding box coordinates.
        2. Extracts objectness scores.
        3. Extracts class scores.

        Args:
            predictions (list): List of predicted bounding boxes and associated scores.
        
        Returns:
            tuple: (bounding boxes, objectness scores, class scores)
        """
        if predictions is None:
            return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0, self.num_classes), dtype=float)

        # If already passed in as (boxes, obj, cls)
        if isinstance(predictions, tuple) and len(predictions) == 3:
            pred_box, pred_obj, pred_cls = predictions
            pred_box = np.array(pred_box, dtype=float).reshape(-1, 4)
            pred_obj = np.array(pred_obj, dtype=float).reshape(-1)
            pred_cls = np.array(pred_cls, dtype=float)
            return pred_box, pred_obj, pred_cls

        pred_boxes = []
        pred_obj = []
        pred_cls = []

        for p in predictions:
            # Dictionary style
            if isinstance(p, dict):
                box = p.get("bbox") or p.get("box") or p.get("bbox_xywh")
                obj = p.get("score") or p.get("objectness") or p.get("conf")
                cls = p.get("class_scores") or p.get("cls_scores") or p.get("probs")

                if box is None:
                    continue

                pred_boxes.append(list(box))
                pred_obj.append(float(obj) if obj is not None else 0.0)

                if cls is None:
                    pred_cls.append(np.zeros(self.num_classes, dtype=float))
                else:
                    cls_arr = np.array(cls, dtype=float).flatten()
                    if cls_arr.size == self.num_classes:
                        pred_cls.append(cls_arr)
                    else:
                        # Adjust to correct size
                        vec = np.zeros(self.num_classes, dtype=float)
                        take = min(self.num_classes, cls_arr.size)
                        vec[:take] = cls_arr[:take]
                        pred_cls.append(vec)
                continue

            # List/tuple row style: [x, y, w, h, obj, class_scores...]
            row = np.array(p, dtype=float).flatten()
            if row.size < 5:
                continue

            pred_boxes.append(row[:4].tolist())
            pred_obj.append(float(row[4]))

            cls_part = row[5:]
            if cls_part.size == self.num_classes:
                pred_cls.append(cls_part.astype(float))
            elif cls_part.size == 0:
                pred_cls.append(np.zeros(self.num_classes, dtype=float))
            else:
                vec = np.zeros(self.num_classes, dtype=float)
                take = min(self.num_classes, cls_part.size)
                vec[:take] = cls_part[:take]
                pred_cls.append(vec)

        pred_box = np.array(pred_boxes, dtype=float).reshape(-1, 4)
        pred_obj = np.array(pred_obj, dtype=float).reshape(-1)
        pred_cls = np.array(pred_cls, dtype=float)

        return pred_box, pred_obj, pred_cls 
    
    def get_annotations(self, annotations):
        """
        Extract ground truth bounding boxes and class IDs from annotations.
        
        Internal Process:
        1. Iterates over annotations to extract bounding box coordinates.
        2. Extracts the corresponding class labels.
        
        Args:
            annotations (list): List of ground truth annotations.
        
        Returns:
            tuple: (ground truth bounding boxes, class labels)
        """
        if annotations is None:
            return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)

        # If already passed in as (boxes, classes)
        if isinstance(annotations, tuple) and len(annotations) == 2:
            gt_box, gt_cls = annotations
            gt_box = np.array(gt_box, dtype=float).reshape(-1, 4)
            gt_cls = np.array(gt_cls, dtype=int).reshape(-1)
            return gt_box, gt_cls

        gt_boxes = []
        gt_classes = []

        for a in annotations:
            # Dictionary style
            if isinstance(a, dict):
                box = a.get("bbox") or a.get("box") or a.get("bbox_xywh")
                cls = a.get("class_id") or a.get("class") or a.get("label")

                if box is None or cls is None:
                    continue

                gt_boxes.append(list(box))
                gt_classes.append(int(cls))
                continue

            # Row style: [x, y, w, h, class_id]
            row = np.array(a, dtype=float).flatten()
            if row.size < 5:
                continue

            gt_boxes.append(row[:4].tolist())
            gt_classes.append(int(row[4]))

        gt_box = np.array(gt_boxes, dtype=float).reshape(-1, 4)
        gt_cls = np.array(gt_classes, dtype=int).reshape(-1)

        return gt_box, gt_cls

    def compute(self, predictions, annotations):
        """
        Compute the YOLO loss components.

        Internal Process:
        1. Extracts predictions and annotations of a single image/frame.
        2. Iterates through annotations to compute localization, confidence, and class loss.
        3. Computes total loss using predefined weighting factors.

        Args:
            predictions (list): List of predictions of a single image.
            annotations (list): List of ground truth annotations of a single image.

        Returns:
            dict: Dictionary containing the computed loss components.
        """
        loc_loss = 0 # localization loss
        class_loss = 0 # classification loss
        conf_loss_obj = 0 # with object (or confidence) loss
        conf_loss_noobj = 0 # no object (or confidence) loss
        total_loss = 0 # aggregate loss including loc_loss, class_loss, conf_loss_obj, etc.

        pred_boxes, pred_obj_scores, pred_cls_scores = self.get_predictions(predictions)
        gt_boxes, gt_class_ids = self.get_annotations(annotations)

        # Small helper as things get too long without it
        def iou_xywh(box_a, box_b):
            ax, ay, aw, ah = box_a
            bx, by, bw, bh = box_b

            ax2, ay2 = ax + aw, ay + ah
            bx2, by2 = bx + bw, by + bh

            inter_x1 = max(ax, bx)
            inter_y1 = max(ay, by)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            area_a = max(0, aw) * max(0, ah)
            area_b = max(0, bw) * max(0, bh)
            union_area = area_a + area_b - inter_area

            if union_area == 0:
                return 0.0
            return inter_area / union_area

        matched_pred_indices = set()

        # Match each ground truth box to best prediction
        for gt_idx, gt_box in enumerate(gt_boxes):
            gt_class_id = int(gt_class_ids[gt_idx])

            best_iou = 0.0
            best_pred_idx = -1

            for pred_idx, pred_box in enumerate(pred_boxes):
                if pred_idx in matched_pred_indices:
                    continue

                iou = iou_xywh(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_pred_idx != -1 and best_iou >= self.iou_threshold:
                matched_pred_indices.add(best_pred_idx)

                pred_box = pred_boxes[best_pred_idx]
                pred_obj = float(pred_obj_scores[best_pred_idx])
                pred_scores_vec = pred_cls_scores[best_pred_idx]

                # Localization loss (bounding box coordinates)
                box_diff = np.array(pred_box, dtype=float) - np.array(gt_box, dtype=float)
                loc_loss += float(np.sum(box_diff ** 2))

                # Objectness loss for matched predictions
                conf_loss_obj += (1.0 - pred_obj) ** 2

                # Classification loss
                gt_score = 0.0
                if pred_scores_vec is not None and len(pred_scores_vec) > gt_class_id >= 0:
                    gt_score = float(pred_scores_vec[gt_class_id])

                class_loss += (1.0 - gt_score) ** 2

            else:
                # No matching prediction found
                conf_loss_obj += 1.0
                class_loss += 1.0

        # Unmatched predictions (false positives)
        for pred_idx, pred_obj in enumerate(pred_obj_scores):
            if pred_idx in matched_pred_indices:
                continue
            conf_loss_noobj += float(pred_obj) ** 2

        total_loss = (
            self.lambda_coord * loc_loss
            + self.lambda_obj * conf_loss_obj
            + self.lambda_noobj * conf_loss_noobj
            + self.lambda_cls * class_loss)

        return {
            "total_loss": total_loss, 
            "loc_loss": loc_loss, 
            "conf_loss_obj": conf_loss_obj, 
            "conf_loss_noobj": conf_loss_noobj, 
            "class_loss": class_loss
        }
