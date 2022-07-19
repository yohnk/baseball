from __future__ import annotations
from inspect import getfullargspec
import asyncio
import enum
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from threading import Thread


def noop():
    pass


class ET(enum.Enum):
    ASYNC = 0,
    PROCESS = 1,
    THREAD = 2,
    MAIN = 3


class TNode:

    def __init__(self, exec_t: ET = ET.PROCESS, work=noop, collector=False):
        self.children = []
        self.parents = []
        self.exec_t = exec_t
        self.work = work
        self.results = []
        self.future = None
        self.complete = False
        self.collector = collector

    def __repr__(self):
        return self.work.__qualname__ + "_" + str(self.exec_t)

    def add_child(self, child: TNode):
        self.children.append(child)
        child.parents.append(self)
        return child

    def is_complete(self):
        return self.complete

    def is_execed(self):
        return self.future is not None and self.future.done()

    def is_ready(self):
        return self.future is None and len(self.results) == len(self.parents)

    def descendants(self):
        o = {self}
        for child in self.children:
            o.update(child.descendants())
        return o

    def async_task(self, m):
        return self._add_task(ET.ASYNC, m)

    def thread_task(self, m):
        return self._add_task(ET.THREAD, m)

    def main_task(self, m):
        return self._add_task(ET.MAIN, m)

    def process_task(self, m):
        return self._add_task(ET.PROCESS, m)

    def _add_task(self, exec_t, work):
        return self.add_child(TNode(exec_t=exec_t, work=work))

    def split(self, *children):
        tn = TNode(exec_t=ET.MAIN, collector=False)
        tn.work = tn._get_result
        [child.add_child(tn) for child in [self.add_child(child) for child in children]]
        return tn

    def supply(self, val):
        tn = TNode(exec_t=ET.MAIN)
        f = Future()
        f.set_result(val)
        tn.future = f
        return self.add_child(tn)

    def collect(self):
        tn = TNode(exec_t=ET.MAIN, collector=True)
        tn.work = tn._get_result
        return self.add_child(tn)

    def _get_result(self):
        return self.results

    def add_result(self, *r):
        for x in r:
            self.results.append(x)


class TaskGraph:

    def __init__(self):
        self.r = TNode(exec_t=ET.MAIN)
        self.p_executor = ProcessPoolExecutor(max_workers=10)
        self.t_executor = ThreadPoolExecutor(max_workers=2)
        self.loop = asyncio.get_event_loop()
        self.loop_thread = Thread(target=self.loop.run_forever)
        self.loop_thread.start()

    def root(self) -> TNode:
        return self.r.add_child(TNode(exec_t=ET.MAIN))

    def run(self):
        family_tree = self.r.descendants()
        while not all([d.is_complete() for d in family_tree]):

            for node in [n for n in family_tree if n.is_execed() and not n.is_complete()]:
                for child in node.children:
                    child.add_result(node.future.result())
                node.complete = True

            for node in [n for n in family_tree if n.is_ready() and not n.is_execed()]:
                self._start(node, node.results)

        return [n.results for n in family_tree if n.collector]

    def close(self):
        self.p_executor.shutdown(wait=True, cancel_futures=True)
        self.t_executor.shutdown(wait=True, cancel_futures=True)
        self.loop.call_soon_threadsafe(self.loop.stop)
        while self.loop.is_running():
            pass

    def _start(self, node: TNode, input_param=()):
        args = [arg for arg in getfullargspec(node.work).args if arg != "self"]
        if len(args) == 0:
            input_param = ()
        f = Future()
        if node.exec_t == ET.PROCESS:
            f = self.p_executor.submit(node.work, *input_param)
        if node.exec_t == ET.THREAD:
            f = self.t_executor.submit(node.work, *input_param)
        if node.exec_t == ET.ASYNC:
            t = self.wrap(node.work(*input_param), f)
            self.loop.call_soon_threadsafe(asyncio.ensure_future, t)
        if node.exec_t == ET.MAIN:
            f.set_result(node.work(*input_param))
        node.future = f

    @staticmethod
    async def wrap(m, f: Future):
        f.set_result(m)

    @staticmethod
    def async_task(m):
        return TNode(exec_t=ET.ASYNC, work=m)

    @staticmethod
    def process_task(m):
        return TNode(exec_t=ET.PROCESS, work=m)

    @staticmethod
    def main_task(m):
        return TNode(exec_t=ET.MAIN, work=m)

    @staticmethod
    def thread_task(m):
        return TNode(exec_t=ET.THREAD, work=m)
