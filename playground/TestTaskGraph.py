import asyncio
import unittest
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import TaskGraph as tg


class TestTaskGraph(unittest.TestCase):

    @staticmethod
    async def async_square(num):
        anum = await num
        o = np.square(anum)
        return o

    @staticmethod
    async def async_divide(num):
        anum = await num
        o = anum / 2
        return o

    @staticmethod
    def divide(num):
        o = num / 2
        return o

    @staticmethod
    async def async_multiply(num):
        anum = await num
        o = anum * 2
        return o

    @staticmethod
    def square(num):
        o = np.square(num)
        return o

    @staticmethod
    def average_three(one, two, three):
        o = np.mean([one, two, three])
        return o

    @staticmethod
    def async_average(*argv):
        o = np.mean(argv)
        return o

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

        self.assertIsNone(output.result())

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

        all_nodes = {root1, child11, root2, child21, collector}

        for n in all_nodes:
            test = n.all_nodes()
            self.assertEqual(test, all_nodes)

    def test_generations(self):
        root1 = tg.AsyncNode()
        child11 = tg.AsyncNode()
        child12 = tg.AsyncNode()

        root2 = tg.AsyncNode()
        child21 = tg.MainNode()

        collector = tg.CollectNode()
        multi_gen = tg.AsyncNode()

        root1.split(
            child11.add_child(child12).add_child(collector), multi_gen
        )
        root2.add_child(child21).add_child(collector)

        self.assertEqual(0, root1.generation)
        self.assertEqual(0, root2.generation)
        self.assertEqual(1, child11.generation)
        self.assertEqual(1, child21.generation)
        self.assertEqual(1, multi_gen.generation)
        self.assertEqual(2, child12.generation)
        self.assertEqual(3, collector.generation)

        collector.add_child(multi_gen)
        self.assertEqual(4, multi_gen.generation)
        self.assertEqual(3, collector.generation)

        g = root1._generations()

        self.assertIn(root1, g[0])
        self.assertIn(root2, g[0])
        self.assertEqual(2, len(g[0]))
        self.assertIn(child11, g[1])
        self.assertIn(child21, g[1])
        self.assertEqual(2, len(g[1]))
        self.assertIn(child12, g[2])
        self.assertEqual(1, len(g[2]))
        self.assertIn(collector, g[3])
        self.assertEqual(1, len(g[3]))
        self.assertIn(multi_gen, g[4])
        self.assertEqual(1, len(g[4]))

    def test_executor(self):
        executor = ProcessPoolExecutor()
        c = tg.SeedNode(3, executor=executor).async_task(self.async_square).process_task(self.square).async_task(
            self.async_multiply).process_task(self.divide).collect()
        asyncio.run(c.start())
        self.assertEqual(1, len(c.result()))
        self.assertEqual(81, c.result()[0])
        executor.shutdown(wait=True, cancel_futures=False)
        self.assertEqual(0, len(executor._pending_work_items))
        self.assertEqual(2, executor._queue_count)


if __name__ == '__main__':
    unittest.main()
