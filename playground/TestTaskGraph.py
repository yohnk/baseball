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
    async def async_divide(num):
        return (await num)/2

    @staticmethod
    async def async_multiply(num):
        return (await num) * 2

    @staticmethod
    def square(num):
        return np.square(num)

    @staticmethod
    def average_three(one, two, three):
        return np.mean([one, two, three])

    @staticmethod
    def async_average(*argv):
        return np.mean(argv)

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

        m = tg.MainNode(self.average_three)
        c1.add_child(m)
        c2.add_child(m)
        c3.add_child(m)

        asyncio.run(root.start())

        self.assertEqual(92.0, m.result())

    def test_split(self):
        root = tg.SeedNode(2)
        output = root.split(
            tg.AsyncNode(self.async_divide).main_task(self.square),
            tg.AsyncNode(self.async_multiply).async_task(self.async_square)
        )

        asyncio.run(root.start())

        self.assertEqual(2, len(output.result()))
        self.assertIn(1, output.result())
        self.assertIn(16, output.result())

    def test_roots_leafs(self):
        root1 = tg.AsyncNode()
        child11 = root1.add_child(tg.AsyncNode())
        root2 = tg.AsyncNode()
        child21 = root2.add_child(tg.MainNode())
        collector = tg.CollectNode()
        child11.add_child(collector)
        child21.add_child(collector)

        roots = collector.roots()

        self.assertEqual(2, len(roots))
        self.assertIn(root1, roots)
        self.assertIn(root2, roots)

        leafs = root1.leafs()
        self.assertEqual(1, len(leafs))
        self.assertIn(collector, leafs)

        leafs = root2.leafs()
        self.assertEqual(1, len(leafs))
        self.assertIn(collector, leafs)

    def test_all_children(self):
        root1 = tg.AsyncNode()
        child11 = root1.add_child(tg.AsyncNode())
        root2 = tg.AsyncNode()
        child21 = root2.add_child(tg.MainNode())
        collector = tg.CollectNode()
        child11.add_child(collector)
        child21.add_child(collector)

        r1c = root1.tree()
        self.assertEqual(3, len(r1c))
        self.assertIn(root1, r1c)
        self.assertIn(child11, r1c)
        self.assertIn(collector, r1c)

        r2c = root2.tree()
        self.assertEqual(3, len(r2c))
        self.assertIn(root2, r2c)
        self.assertIn(child21, r2c)
        self.assertIn(collector, r2c)

    def test_all_nodes(self):
        root1 = tg.AsyncNode()
        child11 = root1.add_child(tg.AsyncNode())
        root2 = tg.AsyncNode()
        child21 = root2.add_child(tg.MainNode())
        collector = tg.CollectNode()
        child11.add_child(collector)
        child21.add_child(collector)

        all_nodes = set([root1, child11, root2, child21, collector])

        for n in all_nodes:
            test = n.all_nodes()
            self.assertEqual(test, all_nodes)

    def test_generations(self):
        root1 = tg.AsyncNode()
        child11 = root1.add_child(tg.AsyncNode())
        child12 = child11.add_child(tg.AsyncNode())
        root2 = tg.AsyncNode()
        child21 = root2.add_child(tg.MainNode())
        collector = tg.CollectNode()
        child12.add_child(collector)
        child21.add_child(collector)

        self.assertEqual(0, root1.generation)
        self.assertEqual(0, root2.generation)
        self.assertEqual(1, child11.generation)
        self.assertEqual(1, child21.generation)
        self.assertEqual(2, child12.generation)
        self.assertEqual(3, collector.generation)




if __name__ == '__main__':
    unittest.main()
