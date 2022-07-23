from __future__ import annotations
import asyncio
import enum
from abc import ABC, abstractmethod
from asyncio import Future
from itertools import chain
from typing import Set, List, Dict


class ET(enum.Enum):
    ASYNC = 0,
    PROCESS = 1,
    THREAD = 2,
    MAIN = 3


async def async_noop(*args):
    try:
        if len(args) == 1:
            if is_async(args[0]):
                return await args[0]
            else:
                return args[0]
    except Exception as e:
        pass
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
        self.exceptions = []

    async def start(self, raise_exception=False) -> List:
        generations = self._generations()
        for gen in sorted(generations.keys()):
            for node in generations[gen]:
                if len(node.exceptions) == 0:
                    try:
                        node.task = node.create_task(await node.parent_results())
                    except Exception as e:
                        node._add_exception(e)

        if raise_exception:
            [n._raise() for n in set(chain(*generations.values()))]

        out = [n.result() for n in set(chain(*generations.values())) if n.is_leaf() and n.is_collector()]
        return out

    async def parent_results(self) -> List:
        out = []
        for p in self.parents:
            try:
                if is_async(p.result):
                    out.append(await p.result())
                else:
                    out.append(p.result())
            except Exception as e:
                p._add_exception(e)
        return out

    def _raise(self):
        if len(self.exceptions) > 0:
            raise self.exceptions[0]

    def _add_exception(self, e):
        self.exceptions.append(e)
        for c in self.children:
            c._add_exception(e)

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

    @staticmethod
    def is_collector():
        return False

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

    def seed_task(self, value) -> Node:
        return self.add_child(SeedNode(value=value))

    def process_task(self, work=noop, executor=None):
        return self.add_child(ProcessNode(work=work, executor=executor))

    def collect(self) -> Node:
        return self.add_child(CollectNode())

    @abstractmethod
    def create_task(self, results):
        pass


class AsyncNode(Node):

    def create_task(self, results):
        return asyncio.create_task(self.work(*results))

    async def parent_results(self) -> List:
        out = []
        for p in self.parents:
            try:
                if is_async(p.result):
                    out.append(await p.result())
                else:
                    f = Future()
                    f.set_result(p.result())
                    out.append(f)
            except Exception as e:
                p._add_exception(e)
        return out


class CollectNode(AsyncNode):

    def __init__(self, work=async_noop, executor=None):
        self.results = None
        super().__init__(work=work, executor=executor)

    async def parent_results(self):
        out = []
        for p in self.parents:
            try:
                if is_async(p.result):
                    out.append(await (await p.result()))
                else:
                    out.append(p.result())
            except Exception as e:
                p._add_exception(e)
        self.results = out
        return out

    def result(self):
        try:
            if len(self.results) == 1:
                return self.results[0]
        except Exception as e:
            pass
        return self.results

    @staticmethod
    def is_collector():
        return True


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
