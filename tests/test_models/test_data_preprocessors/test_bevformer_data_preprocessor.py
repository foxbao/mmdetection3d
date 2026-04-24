# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet3d.structures import Det3DDataSample
from projects.BEVFormer.bevformer import BEVFormerDataPreprocessor


class TestBEVFormerDataPreprocessor(TestCase):

    def test_forward_keeps_history_points(self):
        processor = BEVFormerDataPreprocessor()
        points = torch.randn((10, 4))
        history_points = [torch.randn((5, 4)), torch.randn((6, 4))]

        data = dict(
            inputs=dict(points=[points], history_points=[history_points]),
            data_samples=[Det3DDataSample()])
        out = processor(data)

        batch_inputs = out['inputs']
        self.assertIn('history_points', batch_inputs)
        self.assertEqual(len(batch_inputs['history_points']), 1)
        self.assertEqual(len(batch_inputs['history_points'][0]), 2)
        self.assertTrue(
            torch.equal(batch_inputs['history_points'][0][0], history_points[0]))
        self.assertTrue(
            torch.equal(batch_inputs['history_points'][0][1], history_points[1]))
