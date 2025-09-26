import unittest
from unittest import mock
import glob
import os
import tempfile
import pandas as pd
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from techtrack.modules.inference.model import Detector
from techtrack.modules.inference.nms import NMS
from techtrack.modules.utils.loss import Loss  
from techtrack.modules.rectification.hard_negative_mining import HardNegativeMiner
from techtrack.modules.inference.preprocessing import Preprocessing

###############################################################################
# Tests for Loss (-5pts for each failed test)
###############################################################################
class TestLoss(unittest.TestCase):
    def setUp(self):
        # Use a smaller number of classes for easier testing.
        self.loss = Loss(iou_threshold=0.5, lambda_coord=0.5, lambda_noobj=0.5, num_classes=3)

    def test_get_predictions(self):
        """
        Test get_predictions() with dummy prediction data.
        Expected:
          - Two predictions (one per image).
          - Each prediction list: [x1, y1, x2, y2, objectness, class_score_0, class_score_1, class_score_2].
          - Example:
              Image 1: [10, 10, 20, 20, 0.9, 0.1, 0.9, 0.0]
              Image 2: [30, 30, 40, 40, 0.8, 0.7, 0.1, 0.2]
        """
        # Get the test description including code snippet (if any) from the docstring.
        code_used = TestLoss.test_get_predictions.__doc__
        predictions = [
            [[10, 10, 20, 20, 0.9, 0.1, 0.9, 0.0]],
            [[30, 30, 40, 40, 0.8, 0.7, 0.1, 0.2]]
        ]
        pred_box, objectness_score, class_scores = self.loss.get_predictions(predictions)
        
        self.assertEqual(
            pred_box.shape, (2, 4),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected predicted boxes shape to be (2,4) for input predictions = {}\n"
            "Got shape: {} with boxes: {}\nPlease check the box extraction logic.\n"
            "Code used:\n{}\n******\n".format(predictions, pred_box.shape, pred_box, code_used)
        )
        self.assertEqual(
            objectness_score.shape, (2,),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected objectness scores shape to be (2,) for input predictions = {}\n"
            "Got shape: {} with scores: {}\nPlease review the objectness extraction logic.\n"
            "Code used:\n{}\n******\n".format(predictions, objectness_score.shape, objectness_score, code_used)
        )
        self.assertEqual(
            class_scores.shape, (2, 3),
            "\n******\nTest: test_get_predictions\nFunction: get_predictions()\n"
            "Error: Expected class scores shape to be (2,3) for input predictions = {}\n"
            "Got shape: {} with class scores: {}\nPlease verify the class score extraction.\n"
            "Code used:\n{}\n******\n".format(predictions, class_scores.shape, class_scores, code_used)
        )

    def test_get_annotations(self):
        """
        Test get_annotations() with dummy annotation data.
        Expected:
          - Two annotations in the form: [class_id, x1, y1, x2, y2].
          - Example:
              Annotation 1: [1, 10, 10, 20, 20]
              Annotation 2: [0, 30, 30, 40, 40]
        """
        code_used = TestLoss.test_get_annotations.__doc__
        annotations = [
            [1, 10, 10, 20, 20],
            [0, 30, 30, 40, 40]
        ]
        gt_box, gt_class_id = self.loss.get_annotations(annotations)
        
        self.assertEqual(
            gt_box.shape, (2, 4),
            "\n******\nTest: test_get_annotations\nFunction: get_annotations()\n"
            "Error: Expected ground truth boxes shape to be (2,4) for input annotations = {}\n"
            "Got shape: {} with boxes: {}\nPlease check annotation parsing for boxes.\n"
            "Code used:\n{}\n******\n".format(annotations, gt_box.shape, gt_box, code_used)
        )
        self.assertEqual(
            gt_class_id.shape, (2,),
            "\n******\nTest: test_get_annotations\nFunction: get_annotations()\n"
            "Error: Expected ground truth class IDs shape to be (2,) for input annotations = {}\n"
            "Got shape: {} with class IDs: {}\nPlease verify the extraction of class IDs from annotations.\n"
            "Code used:\n{}\n******\n".format(annotations, gt_class_id.shape, gt_class_id, code_used)
        )

    def test_compute_loss(self):
        """
        Test compute() with a simple matching prediction and annotation.
        Expected:
          - Prediction: [10, 10, 20, 20, 0.9, 0.7, 0.3, 0.0]
          - Annotation: [0, 10, 10, 20, 20] (class 0, matching the box)
          - All loss components (total_loss, loc_loss, conf_loss_obj, conf_loss_noobj, class_loss) must be non-negative.
        """
        code_used = TestLoss.test_compute_loss.__doc__
        predictions = [
            [[10, 10, 20, 20, 0.9, 0.7, 0.3, 0.0]]
        ]
        annotations = [
            [0, 10, 10, 20, 20]
        ]
        losses = self.loss.compute(predictions, annotations)
        
        for key in losses:
            self.assertGreaterEqual(
                losses[key], 0,
                "\n******\nTest: test_compute_loss\nFunction: compute()\n"
                "Error: Expected loss component '{}' to be non-negative for inputs:\n"
                "predictions = {}\nannotations = {}\nGot {}.\nPlease inspect the loss computation for {}.\n"
                "Code used:\n{}\n******\n".format(key, predictions, annotations, losses[key], key, code_used)
            )
        self.assertIn(
            "total_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'total_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nPlease verify the loss aggregation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "loc_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'loc_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nCheck localization loss computation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "conf_loss_obj", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'conf_loss_obj' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nReview objectness loss for true detections.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "conf_loss_noobj", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'conf_loss_noobj' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nReview no-object loss computation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )
        self.assertIn(
            "class_loss", losses,
            "\n******\nTest: test_compute_loss\nFunction: compute()\n"
            "Error: Expected key 'class_loss' in losses dictionary for inputs:\n"
            "predictions = {}\nannotations = {}\nGot keys: {}.\nVerify class loss calculation.\n"
            "Code used:\n{}\n******\n".format(predictions, annotations, list(losses.keys()), code_used)
        )

    def test_compute_loss_increases_with_lower_class_prediction(self):
        """
        Test that compute() yields a higher total loss when the predicted probability for the correct class decreases.
        Example:
          - Scenario A (High confidence): [10,10,20,20,0.9,0.8,0.1,0.1]
          - Scenario B (Low confidence): [10,10,20,20,0.9,0.3,0.35,0.35]
        Expected:
          - Total loss and class loss in Scenario B must be greater than in Scenario A.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_A = [[[10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]]]
          predictions_B = [[[10, 10, 20, 20, 0.9, 0.3, 0.35, 0.35]]]
        """
        code_used = TestLoss.test_compute_loss_increases_with_lower_class_prediction.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_A = [[[10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]]]
        predictions_B = [[[10, 10, 20, 20, 0.9, 0.3, 0.35, 0.35]]]
        loss_A = self.loss.compute(predictions_A, annotations)
        loss_B = self.loss.compute(predictions_B, annotations)
        
        self.assertGreater(
            loss_B["total_loss"], loss_A["total_loss"],
            "\n******\nTest: test_compute_loss_increases_with_lower_class_prediction\nFunction: compute()\n"
            "Error: Expected total_loss in Scenario B ({}) to be greater than in Scenario A ({}).\n"
            "Inputs:\nannotations = {}\npredictions_A = {}\npredictions_B = {}\nReview class probability impact on loss.\n"
            "Code used:\n{}\n******\n".format(loss_B["total_loss"], loss_A["total_loss"], annotations, predictions_A, predictions_B, code_used)
        )
        self.assertGreater(
            loss_B["class_loss"], loss_A["class_loss"],
            "\n******\nTest: test_compute_loss_increases_with_lower_class_prediction\nFunction: compute()\n"
            "Error: Expected class_loss in Scenario B ({}) to exceed that in Scenario A ({}).\n"
            "Inputs:\nannotations = {}\npredictions_A = {}\npredictions_B = {}\nCheck class prediction handling.\n"
            "Code used:\n{}\n******\n".format(loss_B["class_loss"], loss_A["class_loss"], annotations, predictions_A, predictions_B, code_used)
        )

    def test_compute_loss_increases_when_objectness_decreases(self):
        """
        Test that compute() yields a higher loss when the objectness score decreases for a true detection.
        Example:
          - High objectness: [10,10,20,20,0.9,0.9,0.05,0.05]
          - Low objectness:  [10,10,20,20,0.3,0.9,0.05,0.05]
        Expected:
          - Lower objectness should yield a higher objectness loss and higher total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_high_obj = [[[10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]]]
          predictions_low_obj = [[[10, 10, 20, 20, 0.3, 0.9, 0.05, 0.05]]]
        """
        code_used = TestLoss.test_compute_loss_increases_when_objectness_decreases.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_high_obj = [[[10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]]]
        predictions_low_obj = [[[10, 10, 20, 20, 0.3, 0.9, 0.05, 0.05]]]
        loss_high = self.loss.compute(predictions_high_obj, annotations)
        loss_low = self.loss.compute(predictions_low_obj, annotations)
        
        self.assertGreater(
            loss_low["total_loss"], loss_high["total_loss"],
            "\n******\nTest: test_compute_loss_increases_when_objectness_decreases\nFunction: compute()\n"
            "Error: Expected total_loss with low objectness ({}) to be greater than with high objectness ({}).\n"
            "Inputs:\nannotations = {}\npredictions_high_obj = {}\npredictions_low_obj = {}\nCheck impact of objectness on loss.\n"
            "Code used:\n{}\n******\n".format(loss_low["total_loss"], loss_high["total_loss"], annotations, predictions_high_obj, predictions_low_obj, code_used)
        )

    def test_all_highly_overlapping_boxes_are_computed(self):
        """
        Test that compute() accumulates loss from all predicted boxes overlapping a ground truth.
        Example:
          - Two identical predictions overlapping [10,10,20,20].
          - One prediction overlapping [10,10,20,20].
        Expected:
          - Total loss for two predictions must be greater than for one.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          predictions_two = [[
              [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1],
              [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]
          ]]
          predictions_one = [[[10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]]]
        """
        code_used = TestLoss.test_all_highly_overlapping_boxes_are_computed.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        predictions_two = [[
            [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1],
            [10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]
        ]]
        predictions_one = [[[10, 10, 20, 20, 0.1, 0.8, 0.1, 0.1]]]
        loss_two = self.loss.compute(predictions_two, annotations)
        loss_one = self.loss.compute(predictions_one, annotations)
        
        self.assertGreaterEqual(
            loss_two["total_loss"], loss_one["total_loss"],
            "\n******\nTest: test_all_highly_overlapping_boxes_are_computed\nFunction: compute()\n"
            "Error: Expected total_loss for two overlapping predictions ({}) to exceed that for one prediction ({}).\n"
            "Inputs:\nannotations = {}\npredictions_two = {}\npredictions_one = {}\nEnsure all overlaps contribute to the loss.\n"
            "Code used:\n{}\n******\n".format(loss_two["total_loss"], loss_one["total_loss"], annotations, predictions_two, predictions_one, code_used)
        )

    def test_non_overlapping_boxes_do_not_contribute(self):
        """
        Test that compute() ignores predicted boxes not overlapping the ground truth.
        Example:
          - Overlapping prediction: [10,10,20,20]
          - Non-overlapping prediction: [100,100,110,110]
        Expected:
          - The objectness loss for true detections (conf_loss_obj) remains unchanged when a non-overlapping prediction is added.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          overlapping_pred = [10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]
          non_overlapping_pred = [100, 100, 110, 110, 0.9, 0.8, 0.1, 0.1]
        """
        code_used = TestLoss.test_non_overlapping_boxes_do_not_contribute.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        overlapping_pred = [10, 10, 20, 20, 0.9, 0.8, 0.1, 0.1]
        non_overlapping_pred = [100, 100, 110, 110, 0.9, 0.8, 0.1, 0.1]
        predictions_overlap_only = [[overlapping_pred]]
        predictions_combined = [[overlapping_pred, non_overlapping_pred]]
        loss_overlap = self.loss.compute(predictions_overlap_only, annotations)
        loss_combined = self.loss.compute(predictions_combined, annotations)
        
        self.assertAlmostEqual(
            loss_overlap["conf_loss_obj"], loss_combined["conf_loss_obj"], places=6,
            msg="\n******\nTest: test_non_overlapping_boxes_do_not_contribute\nFunction: compute()\n"
            "Error: Expected conf_loss_obj to remain unchanged when adding non-overlapping predictions.\n"
            "Inputs:\nannotations = {}\nwith overlapping_pred = {} and non_overlapping_pred = {}\n"
            "Got conf_loss_obj: {} (overlap only) vs. {} (combined).\nPlease check overlap handling in loss computation.\n"
            "Code used:\n{}\n******\n".format(annotations, overlapping_pred, non_overlapping_pred, loss_overlap["conf_loss_obj"], loss_combined["conf_loss_obj"], code_used)
        )

    def test_loss_increases_with_bbox_shift(self):
        """
        Test that compute() yields a higher loss when the predicted bounding box shifts from the ground truth.
        Example:
          - Perfect match: [10,10,20,20]
          - Shifted prediction: [12,10,22,20] (shifted 2 pixels horizontally)
        Expected:
          - A shifted box yields higher localization loss and higher total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          prediction_perfect = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
          prediction_shifted = [12, 10, 22, 20, 0.9, 0.9, 0.05, 0.05]
        """
        code_used = TestLoss.test_loss_increases_with_bbox_shift.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        prediction_perfect = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
        prediction_shifted = [12, 10, 22, 20, 0.9, 0.9, 0.05, 0.05]
        predictions_perfect = [[prediction_perfect]]
        predictions_shifted = [[prediction_shifted]]
        loss_perfect = self.loss.compute(predictions_perfect, annotations)
        loss_shifted = self.loss.compute(predictions_shifted, annotations)
        
        self.assertGreater(
            loss_shifted["total_loss"], loss_perfect["total_loss"],
            "\n******\nTest: test_loss_increases_with_bbox_shift\nFunction: compute()\n"
            "Error: Expected total_loss for shifted prediction ({}) to exceed that for perfect prediction ({}).\n"
            "Inputs:\nannotations = {}\nprediction_perfect = {}\nprediction_shifted = {}\nReview localization loss calculation.\n"
            "Code used:\n{}\n******\n".format(loss_shifted["total_loss"], loss_perfect["total_loss"], annotations, prediction_perfect, prediction_shifted, code_used)
        )
        self.assertGreater(
            loss_shifted["loc_loss"], loss_perfect["loc_loss"],
            "\n******\nTest: test_loss_increases_with_bbox_shift\nFunction: compute()\n"
            "Error: Expected loc_loss for shifted prediction ({}) to be higher than for perfect prediction ({}).\n"
            "Inputs:\nannotations = {}\nprediction_perfect = {}\nprediction_shifted = {}\nCheck box regression impact.\n"
            "Code used:\n{}\n******\n".format(loss_shifted["loc_loss"], loss_perfect["loc_loss"], annotations, prediction_perfect, prediction_shifted, code_used)
        )

    def test_loss_increases_when_true_object_objectness_is_low(self):
        """
        Test that compute() yields a higher loss for a true object when its predicted objectness is low.
        Example:
          - High objectness: [10,10,20,20,0.9,0.9,0.05,0.05]
          - Low objectness:  [10,10,20,20,0.0,0.9,0.05,0.05]
        Expected:
          - Lower objectness increases the objectness loss (conf_loss_obj) and total loss.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          prediction_high_obj = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
          prediction_low_obj = [10, 10, 20, 20, 0.0, 0.9, 0.05, 0.05]
        """
        code_used = TestLoss.test_loss_increases_when_true_object_objectness_is_low.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        prediction_high_obj = [10, 10, 20, 20, 0.9, 0.9, 0.05, 0.05]
        prediction_low_obj = [10, 10, 20, 20, 0.0, 0.9, 0.05, 0.05]
        predictions_high = [[prediction_high_obj]]
        predictions_low = [[prediction_low_obj]]
        loss_high = self.loss.compute(predictions_high, annotations)
        loss_low = self.loss.compute(predictions_low, annotations)
        
        self.assertGreater(
            loss_low["total_loss"], loss_high["total_loss"],
            "\n******\nTest: test_loss_increases_when_true_object_objectness_is_low\nFunction: compute()\n"
            "Error: Expected total_loss for low objectness ({}) to be greater than for high objectness ({}).\n"
            "Inputs:\nannotations = {}\nprediction_high_obj = {}\nprediction_low_obj = {}\nExamine objectness contribution to total loss.\n"
            "Code used:\n{}\n******\n".format(loss_low["total_loss"], loss_high["total_loss"], annotations, prediction_high_obj, prediction_low_obj, code_used)
        )
        self.assertGreater(
            loss_low["conf_loss_obj"], loss_high["conf_loss_obj"],
            "\n******\nTest: test_loss_increases_when_true_object_objectness_is_low\nFunction: compute()\n"
            "Error: Expected conf_loss_obj for low objectness ({}) to exceed that for high objectness ({}).\n"
            "Inputs:\nannotations = {}\nprediction_high_obj = {}\nprediction_low_obj = {}\nReview objectness loss calculation.\n"
            "Code used:\n{}\n******\n".format(loss_low["conf_loss_obj"], loss_high["conf_loss_obj"], annotations, prediction_high_obj, prediction_low_obj, code_used)
        )

    def test_false_positive_objectness_loss(self):
        """
        Test that compute() yields a significantly higher no-object loss (conf_loss_noobj) when a false positive has high objectness.
        Example:
          - False positive with near-zero objectness: [30,30,40,40,0.0,0.1,0.1,0.8]
          - False positive with high objectness: [30,30,40,40,0.9,0.1,0.1,0.8]
        Expected:
          - The no-object loss for the high objectness false positive should be significantly higher.
        Code used:
          annotations = [[0, 10, 10, 20, 20]]
          false_pred_low = [30,30,40,40,0.0,0.1,0.1,0.8]
          false_pred_high = [30,30,40,40,0.9,0.1,0.1,0.8]
        """
        code_used = TestLoss.test_false_positive_objectness_loss.__doc__
        annotations = [[0, 10, 10, 20, 20]]
        false_pred_low = [30, 30, 40, 40, 0.0, 0.1, 0.1, 0.8]
        false_pred_high = [30, 30, 40, 40, 0.9, 0.1, 0.1, 0.8]
        predictions_low = [[false_pred_low]]
        predictions_high = [[false_pred_high]]
        loss_low = self.loss.compute(predictions_low, annotations)
        loss_high = self.loss.compute(predictions_high, annotations)
        
        self.assertGreater(
            loss_high["conf_loss_noobj"], loss_low["conf_loss_noobj"],
            "\n******\nTest: test_false_positive_objectness_loss\nFunction: compute()\n"
            "Error: The computed no-object loss (conf_loss_noobj) for a false positive with high objectness "
            "is not greater than that for a false positive with near-zero objectness.\n"
            "Inputs used:\n  annotations: {}\n  false_pred_low: {}\n  false_pred_high: {}\n"
            "Computed losses:\n  conf_loss_noobj (low objectness): {}\n  conf_loss_noobj (high objectness): {}\n"
            "Details: A false positive with a higher objectness score should incur a larger penalty in the no-object loss term.\n"
            "Please review the handling of objectness in the loss computation.\n"
            "Code used:\n{}\n******\n".format(annotations, false_pred_low, false_pred_high, loss_low["conf_loss_noobj"], loss_high["conf_loss_noobj"], code_used)
        )

###############################################################################
# Tests for HardNegativeMining (-5pts for each failed test)
###############################################################################
class TestHardNegativeMinerWithFullLoss(unittest.TestCase):

    def setUp(self):
        self.mock_model = mock.MagicMock()
        self.mock_nms = mock.MagicMock()
        self.mock_measure = mock.MagicMock()
        self.mock_measure.columns = [
            'total_loss', 'loc_loss', 'conf_loss_obj', 'conf_loss_noobj', 'class_loss'
        ]

        self.miner = HardNegativeMiner(
            model=self.mock_model,
            measure=self.mock_measure,
            dataset_dir="mock_dataset"
        )

        # Disable actual table construction
        self.miner._HardNegativeMiner__construct_table = mock.MagicMock()

    def set_mock_table(self, rows):
        self.miner.table = pd.DataFrame(rows)

    def test_sample_by_total_loss(self):
        self.set_mock_table([
            {'image_file': 'a.jpg', 'annotation_file': 'a.txt',
             'total_loss': 0.3, 'loc_loss': 0.1, 'conf_loss_obj': 0.05,
             'conf_loss_noobj': 0.05, 'class_loss': 0.1},
            {'image_file': 'b.jpg', 'annotation_file': 'b.txt',
             'total_loss': 0.9, 'loc_loss': 0.4, 'conf_loss_obj': 0.2,
             'conf_loss_noobj': 0.1, 'class_loss': 0.2}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        self.assertEqual(df.iloc[0]['total_loss'], 0.9)
        self.assertEqual(df.iloc[0]['loc_loss'], 0.4)

    def test_all_loss_components_exist(self):
        self.set_mock_table([
            {'image_file': 'x.jpg', 'annotation_file': 'x.txt',
             'total_loss': 0.5, 'loc_loss': 0.2, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.1}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        for col in self.mock_measure.columns:
            self.assertIn(col, df.columns)

    def test_sorting_by_class_loss(self):
        self.set_mock_table([
            {'image_file': 'a.jpg', 'annotation_file': 'a.txt',
             'total_loss': 0.5, 'loc_loss': 0.1, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.05},
            {'image_file': 'b.jpg', 'annotation_file': 'b.txt',
             'total_loss': 0.4, 'loc_loss': 0.1, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.2}
        ])
        df = self.miner.sample_hard_negatives(1, 'class_loss')
        self.assertEqual(df.iloc[0]['class_loss'], 0.2)

    def test_equal_total_loss(self):
        self.set_mock_table([
            {'image_file': f'img{i}.jpg', 'annotation_file': f'img{i}.txt',
             'total_loss': 0.5, 'loc_loss': 0.1, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.2}
            for i in range(3)
        ])
        df = self.miner.sample_hard_negatives(2, 'total_loss')
        self.assertEqual(len(df), 2)
        self.assertTrue((df['total_loss'] == 0.5).all())

    def test_negative_loss_values(self):
        self.set_mock_table([
            {'image_file': 'img.jpg', 'annotation_file': 'img.txt',
             'total_loss': -0.3, 'loc_loss': -0.1,
             'conf_loss_obj': 0.0, 'conf_loss_noobj': -0.05, 'class_loss': -0.15}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        self.assertEqual(df.iloc[0]['total_loss'], -0.3)
        self.assertLess(df.iloc[0]['loc_loss'], 0)

    def test_zero_loss_values(self):
        self.set_mock_table([
            {'image_file': 'img.jpg', 'annotation_file': 'img.txt',
             'total_loss': 0.0, 'loc_loss': 0.0,
             'conf_loss_obj': 0.0, 'conf_loss_noobj': 0.0, 'class_loss': 0.0}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        self.assertEqual(df.iloc[0]['total_loss'], 0.0)
        self.assertTrue((df.iloc[0][self.mock_measure.columns] == 0.0).all())

    def test_different_criteria_change_order(self):
        self.set_mock_table([
            {'image_file': 'img1.jpg', 'annotation_file': 'img1.txt',
             'total_loss': 0.6, 'loc_loss': 0.4, 'conf_loss_obj': 0.05,
             'conf_loss_noobj': 0.05, 'class_loss': 0.1},
            {'image_file': 'img2.jpg', 'annotation_file': 'img2.txt',
             'total_loss': 0.5, 'loc_loss': 0.9, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.1}
        ])
        df_total = self.miner.sample_hard_negatives(1, 'total_loss')
        df_loc = self.miner.sample_hard_negatives(1, 'loc_loss')
        self.assertNotEqual(df_total.iloc[0]['image_file'], df_loc.iloc[0]['image_file'])

    def test_fewer_than_requested(self):
        self.set_mock_table([
            {'image_file': 'x.jpg', 'annotation_file': 'x.txt',
             'total_loss': 0.4, 'loc_loss': 0.1, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.05, 'class_loss': 0.15}
        ])
        df = self.miner.sample_hard_negatives(3, 'total_loss')
        self.assertEqual(df.shape[0], 1)

    def test_invalid_column(self):
        self.set_mock_table([
            {'image_file': 'x.jpg', 'annotation_file': 'x.txt', 'total_loss': 0.4}
        ])
        with self.assertRaises(KeyError):
            self.miner.sample_hard_negatives(1, 'nonexistent_metric')

    def test_preserves_metadata_columns(self):
        self.set_mock_table([
            {'image_file': 'img.jpg', 'annotation_file': 'ann.txt', 'tag': 'hard',
             'total_loss': 0.9, 'loc_loss': 0.2, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.5}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        self.assertIn('tag', df.columns)
        self.assertEqual(df.iloc[0]['tag'], 'hard')

    def test_data_types_preserved(self):
        self.set_mock_table([
            {'image_file': 'img.jpg', 'annotation_file': 'ann.txt',
             'total_loss': 0.9, 'loc_loss': 0.1, 'conf_loss_obj': 0.1,
             'conf_loss_noobj': 0.1, 'class_loss': 0.6}
        ])
        df = self.miner.sample_hard_negatives(1, 'total_loss')
        self.assertIsInstance(df.iloc[0]['total_loss'], float)

###############################################################################
# DummyVideoCapture for Preprocessing Tests
###############################################################################
class DummyVideoCapture:
    """
    A dummy video capture class to simulate cv2.VideoCapture behavior.
    
    If provided with a list of frames, it uses them directly. If provided with a string
    and the string is a path to a directory, it loads all image files from that directory 
    (sorted alphabetically) as frames.
    """
    def __init__(self, source, open_success=True):
        self.index = 0
        self.open_success = open_success

        if isinstance(source, list):
            self.frames = source
        elif isinstance(source, str) and os.path.isdir(source):
            files = sorted(glob.glob(os.path.join(source, "*")))
            self.frames = []
            for f in files:
                img = cv2.imread(f)
                if img is not None:
                    self.frames.append(img)
        else:
            self.frames = []

    def isOpened(self):
        return self.open_success

    def read(self):
        if self.index < len(self.frames):
            frame = self.frames[self.index]
            self.index += 1
            return True, frame
        return False, None

    def release(self):
        pass

###############################################################################
# Integration Tests for Detector and NMS using Real Images (-5pts for each failed test)
###############################################################################
class TestDetectorIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary class file with sample class labels.
        self.temp_class_file = tempfile.NamedTemporaryFile(delete=False, mode="w+t")
        self.classes = ["barcode",
                        "car",
                        "cardboard box",
                        "fire",
                        "forklift",
                        "freight container",
                        "gloves",
                        "helmet",
                        "ladder",
                        "license plate",
                        "person",
                        "qr code",
                        "road sign",
                        "safety vest",
                        "smoke",
                        "traffic cone",
                        "traffic light",
                        "truck",
                        "van",
                        "wood pallet"]
        self.temp_class_file.write("\n".join(self.classes))
        self.temp_class_file.flush()
        self.temp_class_file.close()

        # Define a dummy network output that is predictable.
        self.dummy_output = [np.array([[0.5, 0.5, 0.2, 0.2, 0.0, 0.6, 0.4, 0.0]])]
        self.dummy_net = MagicMock()
        self.dummy_net.getLayerNames.return_value = ["layer1", "layer2"]
        self.dummy_net.forward.return_value = self.dummy_output

    def tearDown(self):
        os.unlink(self.temp_class_file.name)

    @patch("cv2.dnn.readNet")
    def test_detector_integration_with_storage_images(self, mock_readNet):
        """
        For each JPEG image found in 'storage/test_images', run detector.predict() and
        detector.post_process() using a dummy network output, then apply NMS.filter().
        
        Also, manually call cv2.dnn.NMSBoxes on the post-processed bounding boxes and compare
        the indices with those returned by NMS.filter(). This test confirms that the calculations
        (conversion from center-based to top-left coordinates, scaling based on image dimensions,
        and NMS filtering) are accurately implemented.
        """
        code_used = TestDetectorIntegration.test_detector_integration_with_storage_images.__doc__
        mock_readNet.return_value = self.dummy_net

        images_pattern = os.path.join("storage", "test_images", "*.jpg")
        image_files = sorted(glob.glob(images_pattern))
        if not image_files:
            self.skipTest(f"No test images found in {images_pattern}")

        for image_file in image_files:
            img = cv2.imread(image_file)
            self.assertIsNotNone(img, f"\nTest: test_detector_integration_with_storage_images\nFunction: Integration\nError: Failed to load image: {image_file}\nCode used:\n{code_used}\n")
            detector = Detector(
                "storage/yolo_models/yolov4-tiny-logistics_size_416_1.weights",
                "storage/yolo_models/yolov4-tiny-logistics_size_416_1.cfg",
                self.temp_class_file.name,
                score_threshold=0.5
            )
            outputs = detector.predict(img)
            H, W = img.shape[:2]
            expected_bbox = [int(0.4 * W), int(0.4 * H), int(0.2 * W), int(0.2 * H)]
            bboxes, class_ids, confidence_scores, class_scores = detector.post_process(outputs)

            self.assertGreaterEqual(len(bboxes), 1,
                                    f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Expected at least one detection for {image_file}.\nCode used:\n{code_used}\n")
            self.assertEqual(bboxes[0], expected_bbox,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Incorrect bounding box conversion for image {image_file}. Expected {expected_bbox}, got {bboxes[0]}.\nCode used:\n{code_used}\n")
            self.assertGreaterEqual(confidence_scores[0], 0.5,
                                      f"\nTest: test_detector_integration_with_storage_images\nFunction: post_process()\nError: Confidence score too low for image {image_file}.\nCode used:\n{code_used}\n")
            nms_instance = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
            filtered = nms_instance.filter(bboxes, class_ids, confidence_scores, class_scores)
            indices = cv2.dnn.NMSBoxes(bboxes, confidence_scores, 0.5, 0.4)
            if len(indices) > 0:
                indices = indices.flatten().tolist()
            else:
                indices = []
            expected_filtered_bboxes = [bboxes[i] for i in indices]
            expected_filtered_class_ids = [class_ids[i] for i in indices]
            self.assertEqual(filtered[0], expected_filtered_bboxes,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: NMS.filter()\nError: Detector NMS filtered bounding boxes for {image_file} do not match manual NMS results.\nCode used:\n{code_used}\n")
            self.assertEqual(filtered[1], expected_filtered_class_ids,
                             f"\nTest: test_detector_integration_with_storage_images\nFunction: NMS.filter()\nError: Detector NMS filtered class IDs for {image_file} do not match manual NMS results.\nCode used:\n{code_used}\n")


###############################################################################
# Tests for Preprocessing (using DummyVideoCapture) (-5pts for each failed test)
###############################################################################
class TestPreprocessing(unittest.TestCase):
    @patch('cv2.VideoCapture')
    def test_capture_video_yields_every_nth_frame_from_list(self, mock_VideoCapture):
        """
        Verify that Preprocessing.capture_video yields every nth frame from a list.
        
        Code used:
            dummy_frames = [np.full((100, 100, 3), fill_value=i, dtype=np.uint8) for i in range(15)]
            drop_rate = 3
            mock_capture = DummyVideoCapture(dummy_frames)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing("dummy_path.mp4", drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())
        Example:
            With 15 dummy frames and drop_rate=3, expected frames are at indices 0, 3, 6, 9, 12.
        """
        code_used = TestPreprocessing.test_capture_video_yields_every_nth_frame_from_list.__doc__
        dummy_frames = [np.full((100, 100, 3), fill_value=i, dtype=np.uint8) for i in range(15)]
        drop_rate = 3

        mock_capture = DummyVideoCapture(dummy_frames)
        mock_VideoCapture.return_value = mock_capture

        preprocessing = Preprocessing("dummy_path.mp4", drop_rate=drop_rate)
        captured_frames = list(preprocessing.capture_video())

        expected_frames = [dummy_frames[i] for i in range(0, len(dummy_frames), drop_rate)]
        self.assertEqual(len(captured_frames), len(expected_frames),
                         f"\nTest: test_capture_video_yields_every_nth_frame_from_list\nFunction: capture_video()\nError: The number of captured frames ({len(captured_frames)}) does not match the expected count ({len(expected_frames)}).\nCode used:\n{code_used}\n")
        for cap_frame, exp_frame in zip(captured_frames, expected_frames):
            np.testing.assert_array_equal(cap_frame, exp_frame,
                                          err_msg=f"\nTest: test_capture_video_yields_every_nth_frame_from_list\nFunction: capture_video()\nError: The content of a captured frame does not match the expected frame.\nCode used:\n{code_used}\n")

    @patch('cv2.VideoCapture')
    def test_capture_video_from_directory_with_50_frames(self, mock_VideoCapture):
        """
        Verify that Preprocessing.capture_video correctly loads images from a directory
        and yields every nth frame.
        
        Code used:
            Create 50 dummy images in a temporary directory.
            drop_rate = 5
            mock_capture = DummyVideoCapture(tmpdirname)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing(tmpdirname, drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())
        Example:
            With a directory of 50 images and drop_rate=5, expected frames are at indices 0, 5, 10, ..., 45.
        """
        code_used = TestPreprocessing.test_capture_video_from_directory_with_50_frames.__doc__
        with tempfile.TemporaryDirectory() as tmpdirname:
            num_frames = 50
            frame_shape = (50, 50, 3)
            for i in range(num_frames):
                dummy_img = np.full(frame_shape, fill_value=i, dtype=np.uint8)
                filename = os.path.join(tmpdirname, f"frame_{i:03d}.png")
                cv2.imwrite(filename, dummy_img)

            drop_rate = 5

            mock_capture = DummyVideoCapture(tmpdirname)
            mock_VideoCapture.return_value = mock_capture

            preprocessing = Preprocessing(tmpdirname, drop_rate=drop_rate)
            captured_frames = list(preprocessing.capture_video())

            expected_frames = []
            for i in range(0, num_frames, drop_rate):
                filename = os.path.join(tmpdirname, f"frame_{i:03d}.png")
                img = cv2.imread(filename)
                if img is not None:
                    expected_frames.append(img)

            self.assertEqual(len(captured_frames), len(expected_frames),
                             f"\nTest: test_capture_video_from_directory_with_50_frames\nFunction: capture_video()\nError: The number of frames captured ({len(captured_frames)}) does not match the expected count ({len(expected_frames)}) based on the drop rate.\nCode used:\n{code_used}\n")
            for cap_frame, exp_frame in zip(captured_frames, expected_frames):
                np.testing.assert_array_equal(cap_frame, exp_frame,
                                              err_msg=f"\nTest: test_capture_video_from_directory_with_50_frames\nFunction: capture_video()\nError: The content of the captured frame does not match the expected image content.\nCode used:\n{code_used}\n")

    @patch('cv2.VideoCapture')
    def test_video_file_not_opened(self, mock_VideoCapture):
        """
        Verify that if cv2.VideoCapture cannot open a video file, capture_video raises a ValueError.
        
        Code used:
            mock_capture = DummyVideoCapture([], open_success=False)
            mock_VideoCapture.return_value = mock_capture
            preprocessing = Preprocessing("nonexistent.mp4", drop_rate=5)
            list(preprocessing.capture_video())
        Example:
            For a non-existent file, the error message should mention 'Unable to open video file'.
        """
        code_used = TestPreprocessing.test_video_file_not_opened.__doc__
        mock_capture = DummyVideoCapture([], open_success=False)
        mock_VideoCapture.return_value = mock_capture

        preprocessing = Preprocessing("nonexistent.mp4", drop_rate=5)
        with self.assertRaises(ValueError, msg=f"\nTest: test_video_file_not_opened\nFunction: capture_video()\nError: If the video file cannot be opened, a ValueError must be raised.\nCode used:\n{code_used}\n"):
            list(preprocessing.capture_video())


if __name__ == '__main__':
    unittest.main()
