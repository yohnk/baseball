from __future__ import annotations

import asyncio
import time
import typing
from abc import ABC, abstractmethod
import enum
from concurrent.futures import ProcessPoolExecutor
from typing import List
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


class ParentNotStartedException(Exception):
    pass


class Node(ABC):

    def __init__(self, work=async_noop, executor=None):
        self.parents: List[Node] = set()
        self.children: List[Node] = set()
        self.work = work
        self.task = None
        self.generation = None
        self.executor = executor
        self.started = False

    async def start(self):
        try:
            self.task = self.create_task(await self.parent_results())
            self.started = True
            await self._start_children()
        except ParentNotStartedException:
            pass

    async def _start_children(self):
        [await c.start() for c in self.children if is_async(c.start)]
        [c.start() for c in self.children if not is_async(c.start)]

    async def parent_results(self):
        o = [await p.result() for p in self.parents if is_async(p.result)]
        o.extend([p.result() for p in self.parents if not is_async(p.result)])
        return o

    def add_child(self, c: AsyncNode):
        self.children.add(c)
        c.parents.add(self)

        if self.generation is not None and (c.generation is None or c.generation <= self.generation):
            c.generation = self.generation + 1

        if self.generation is None and c.generation is not None:
            self.generation = c.generation - 1

        return c

    def all_nodes(self):
        if self.is_leaf():
            return set(chain(*[r.tree() for r in self.roots()]))
        else:
            return set(chain(*[l.all_nodes() for l in self.leafs()]))

    def tree(self):
        s = set(*[c.tree() for c in self.children])
        s.add(self)
        return s

    def is_root(self):
        return len(self.parents) == 0

    def is_leaf(self):
        return len(self.children) == 0

    def roots(self):
        if self.is_root():
            return {self}
        else:
            return set(chain(*[p.roots() for p in self.parents]))

    def leafs(self):
        if self.is_leaf():
            return [self]
        else:
            return set(chain(*[c.leafs() for c in self.children]))

    async def result(self):
        return self.task

    def split(self, *nodes: Node):
        split_node = CollectNode()
        for n in nodes:
            for r in n.roots():
                self.add_child(r)
            for l in n.leafs():
                l.add_child(split_node)
        return split_node

    def main_task(self, work=noop):
        return self.add_child(MainNode(work=work))

    def async_task(self, work=async_noop):
        return self.add_child(AsyncNode(work=work))

    def collect(self):
        return self.add_child(CollectNode())


class AsyncNode(Node):

    def create_task(self, results):
        return asyncio.create_task(self.work(*results))


class CollectNode(AsyncNode):

    def __init__(self, work=async_noop):
        self.results = None
        self.exceptions = []
        super().__init__(work=work)

    async def parent_results(self):
        if not all([p.started for p in self.parents]):
            raise ParentNotStartedException()

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




# class ProcessNode(AsyncNode):
#
#     def __init__(self, work):
#         super().__init__(exec_type=ET.PROCESS, work=work)
#
#     def start(self, executor: ProcessPoolExecutor):
#         loop = asyncio.get_event_loop()
#         results = [x.result(wait=True) for x in self.parents]
#         self.task = loop.run_in_executor(executor, self.work(*results))
#         super().start()
#
#     def result(self):
#         return self.task


class SeedNode(AsyncNode):

    def __init__(self, value):
        super().__init__(work=self.get_work(value))

    @staticmethod
    def get_work(value):
        async def anon_work(*args):
            return value
        return anon_work
