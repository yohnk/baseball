from __future__ import annotations

import asyncio
import time
import typing
from abc import ABC, abstractmethod
import enum
from concurrent.futures import ProcessPoolExecutor
from typing import List


class ET(enum.Enum):
    ASYNC = 0,
    PROCESS = 1,
    THREAD = 2,
    MAIN = 3


async def async_noop(*args):
    pass


def noop(*args):
    pass


def is_async(x):
    return x is not None and (asyncio.iscoroutine(x) or asyncio.iscoroutinefunction(x) or asyncio.isfuture(x))


class ParentNotStartedException(Exception):
    pass


class Node(ABC):

    def __init__(self, work=async_noop):
        self.parents: List[Node] = []
        self.children: List[Node] = []
        self.work = work
        self.task = None
        self.started = False

    async def start(self, executor: ProcessPoolExecutor):
        try:
            self.task = self.create_task(await self.parent_results())
            self.started = True
            await self._start_children(executor)
        except ParentNotStartedException:
            pass

    async def _start_children(self, executor):
        [await c.start(executor) for c in self.children if is_async(c.start)]
        [c.start(executor) for c in self.children if not is_async(c.start)]

    async def parent_results(self):
        o = [await p.result() for p in self.parents if is_async(p.result)]
        o.extend([p.result() for p in self.parents if not is_async(p.result)])
        return o

    def add_child(self, c: AsyncNode):
        self.children.append(c)
        c.parents.append(self)
        return c

    @staticmethod
    def result(self):
        pass


class AsyncNode(Node):

    def create_task(self, results):
        return asyncio.create_task(self.work(*results))

    async def result(self):
        return self.task


class CollectNode(AsyncNode):

    def __init__(self):
        self.results = None
        self.exceptions = []
        super().__init__(work=async_noop)

    async def parent_results(self):
        if not all([p.started for p in self.parents]):
            raise ParentNotStartedException()

        results = [await x for x in [await p.result() for p in self.parents if is_async(p.result)]]
        results.extend([p.result() for p in self.parents if not is_async(p.result)])
        self.results = results
        return results

    def result(self):
        return self.results


class MainNode(AsyncNode):

    def __init__(self, work=noop):
        super(self).__init__()
        self.main_work = work

    async def start(self, executor: ProcessPoolExecutor):
        results = [await x.result() for x in self.parents]
        self.task = asyncio.create_task(self.work(*results))

        for c in self.children:
            await c.start(executor)




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
