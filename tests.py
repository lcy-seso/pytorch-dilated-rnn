#!/usr/bin/env python3

import unittest
import drnn
import torch


class TestForward(unittest.TestCase):
    def test(self):
        model = drnn.DRNN(
            n_input=10,
            n_hidden=10,
            n_layers=4,
            dropout=0,
            cell_type='GRU',
            batch_first=False).to("cuda:0")

        x = torch.randn(23, 3, 10, device="cuda:0")
        out = model(x)

        # self.assertTrue(out.size(0) == 23)
        # self.assertTrue(out.size(1) == 3)
        # self.assertTrue(out.size(2) == 10)


# class TestReshape(unittest.TestCase):
#     def test(self):
#         model = drnn.DRNN(10, 10, 4, 0, 'GRU')
#
#         x = torch.randn(24, 3, 10)
#
#         split_x = model._prepare_inputs(x, 2)
#
#         second_block = x[1::2]
#         check = split_x[:, x.size(1):, :]
#
#         self.assertTrue((second_block == check).all())
#
#         unsplit_x = model._split_outputs(split_x, 2)
#
#         self.assertTrue((x == unsplit_x).all())
#
#
# class TestHidden(unittest.TestCase):
#     def test(self):
#         model = drnn.DRNN(10, 10, 4, 0, 'GRU')
#
#         x = torch.randn(23, 3, 10)
#
#         hidden = model(x)[1]
#
#         self.assertEqual(len(hidden), 4)
#
#         for hid in hidden:
#             print(hid.size())
#
#
# class TestPassHidden(unittest.TestCase):
#     def test(self):
#         model = drnn.DRNN(10, 10, 4, 0, 'GRU')
#
#         hidden = []
#         for i in range(4):
#             hidden.append(torch.randn(2 ** i, 3, 10))
#
#         x = torch.randn(24, 3, 10)
#         hidden = model(x, hidden)
#

if __name__ == '__main__':
    unittest.main()
