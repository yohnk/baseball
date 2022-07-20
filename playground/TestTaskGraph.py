import asyncio
import unittest

import numpy as np

import TaskGraph as tg


class TestTaskGraph(unittest.TestCase):

    def test_seed_int(self):
        sn = tg.SeedNode(5)
        collect = tg.CollectNode()
        sn.add_child(collect)
        asyncio.run(sn.start())
        result = collect.result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 5)

    def test_seed_list(self):
        sn = tg.SeedNode(["Hello World"])
        collect = tg.CollectNode()
        sn.add_child(collect)
        asyncio.run(sn.start())
        result = collect.result()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ["Hello World"])

    @staticmethod
    async def async_square(num):
        return np.square(await num)

    @staticmethod
    def square(num):
        return np.square(num)

    @staticmethod
    def average(one, two, three):
        return np.mean([one, two, three])

    def test_chain(self):
        one = tg.SeedNode(2)
        two = one.add_child(tg.AsyncNode(self.async_square))
        three = two.add_child(tg.AsyncNode(self.async_square))
        result = three.add_child(tg.CollectNode())
        asyncio.run(one.start())
        self.assertEqual(16, result.result()[0])

    def test_main_task(self):
        one = tg.SeedNode(2)
        two = one.add_child(tg.AsyncNode(self.async_square))
        three = two.add_child(tg.MainNode(self.square))
        result = three.add_child(tg.CollectNode())
        asyncio.run(one.start())
        self.assertEqual(16, three.result())
        self.assertEqual(16, result.result()[0])

    def test_multi_input(self):
        root = tg.SeedNode(2)

        c1 = root.add_child(tg.AsyncNode(self.async_square))
        c2 = c1.add_child(tg.AsyncNode(self.async_square))
        c3 = c2.add_child(tg.AsyncNode(self.async_square))

        m = tg.MainNode(self.average)
        c1.add_child(m)
        c2.add_child(m)
        c3.add_child(m)

        asyncio.run(root.start())

        self.assertEqual(92.0, m.result())


if __name__ == '__main__':
    unittest.main()
