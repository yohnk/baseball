from __future__ import annotations
import asyncio
from abc import ABC
import enum
from asyncio import Future
from typing import Set, List, Dict
from itertools import chain


class ET(enum.Enum):
    ASYNC = 0,
    PROCESS = 1,
    THREAD = 2,
    MAIN = 3


async def async_noop(*args):
    return args


def noop(*args):
    return args


def is_async(x):
    return x is not None and (asyncio.iscoroutine(x) or asyncio.iscoroutinefunction(x) or asyncio.isfuture(x))


class Node(ABC):

    def __init__(self, work=async_noop, executor=None):
        self.parents: Set[Node] = set()
        self.children: Set[Node] = set()
        self.work = work
        self.task = None
        self.generation = 0
        self.executor = executor

    async def start(self):
        generations = self._generations()
        for gen in sorted(generations.keys()):
            for node in generations[gen]:
                node.task = node.create_task(await node.parent_results())

    async def parent_results(self) -> List:
        o = [await p.result() for p in self.parents if is_async(p.result)]
        o.extend([p.result() for p in self.parents if not is_async(p.result)])
        return o

    def add_child(self, c: AsyncNode) -> Node:
        self.children.add(c)
        c.parents.add(self)
        c._set_generation()

        if self.executor is not None and c.executor is None:
            c.executor = self.executor

        return c

    def _set_generation(self):
        for p in self.parents:
            if self.generation <= p.generation:
                self.generation = p.generation + 1
        for c in self.children:
            c._set_generation()

    def _generations(self) -> Dict[int, List[Node]]:
        o = {}
        for n in self.all_nodes():
            if n.generation not in o:
                o[n.generation] = set()
            o[n.generation].add(n)
        return o

    def all_nodes(self) -> Set[Node]:
        if self.is_leaf():
            return set(chain(*[r.tree() for r in self.roots()]))
        else:
            return set(chain(*[l.all_nodes() for l in self.leafs()]))

    def tree(self) -> Set[Node]:
        s = set(chain(*[c.tree() for c in self.children]))
        s.add(self)
        return s

    def is_root(self) -> bool:
        return len(self.parents) == 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def roots(self) -> Set[Node]:
        if self.is_root():
            return {self}
        else:
            return set(chain(*[p.roots() for p in self.parents]))

    def leafs(self) -> Set[Node]:
        if self.is_leaf():
            return {self}
        else:
            return set(chain(*[c.leafs() for c in self.children]))

    async def result(self):
        return self.task

    def split(self, *nodes: Node) -> Node:
        split_node = CollectNode()
        for n in nodes:
            for r in n.roots():
                self.add_child(r)
            for l in n.leafs():
                l.add_child(split_node)
        return split_node

    def main_task(self, work=noop) -> Node:
        return self.add_child(MainNode(work=work))

    def async_task(self, work=async_noop) -> Node:
        return self.add_child(AsyncNode(work=work))

    def process_task(self, work=noop):
        return self.add_child(ProcessNode(work=work))

    def collect(self) -> Node:
        return self.add_child(CollectNode())


class AsyncNode(Node):

    def create_task(self, results):
        return asyncio.create_task(self.work(*results))

    async def parent_results(self) -> List:
        o = [await p.result() for p in self.parents if is_async(p.result)]
        for p in self.parents:
            if not is_async(p.result):
                f = Future()
                f.set_result(p.result())
                o.append(f)
        return o


class CollectNode(AsyncNode):

    def __init__(self, work=async_noop):
        self.results = None
        self.exceptions = []
        super().__init__(work=work)

    async def parent_results(self):
        results = [await x for x in [await p.result() for p in self.parents if is_async(p.result)]]
        results.extend([p.result() for p in self.parents if not is_async(p.result)])
        self.results = results
        return results

    def result(self):
        return self.results


class MainNode(CollectNode):

    def __init__(self, work=noop):
        super().__init__(work=work)

    def create_task(self, results):
        self.results = self.work(*results)
        return None


class ProcessNode(CollectNode):

    def create_task(self, results):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self.executor, self.work, *results)

    async def result(self):
        return self.task


class SeedNode(AsyncNode):

    def __init__(self, value, executor=None):
        super().__init__(work=self.get_work(value), executor=executor)

    @staticmethod
    def get_work(value):
        async def anon_work(*args):
            return value
        return anon_work
