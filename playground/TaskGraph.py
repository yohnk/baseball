# from __future__ import annotations
#
# import asyncio
# import enum
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
#
#
# def noop():
#     pass
#
#
# class ET(enum.Enum):
#     ASYNC = 0,
#     PROCESS = 1,
#     THREAD = 2,
#     MAIN = 3
#
#
# class TNode:
#
#     def __init__(self, exec_t: ET = ET.PROCESS, work=noop):
#         self.children = []
#         self.exec_t = exec_t
#         self.work = work
#
#     def add_child(self, child: TNode):
#         self.children.append(child)
#         return child
#
#     def async_task(self, m):
#         return self._add_task(ET.ASYNC, m)
#
#     def thread_task(self, m):
#         return self._add_task(ET.THREAD, m)
#
#     def main_task(self, m):
#         return self._add_task(ET.MAIN, m)
#
#     def process_task(self, m):
#         return self._add_task(ET.PROCESS, m)
#
#     def _add_task(self, exec_t, work):
#         return self.add_child(TNode(exec_t=exec_t, work=work))
#
#     def split(self, *children):
#         collector_node = self._collector()
#         [child.add_child(collector_node) for child in [self.add_child(child) for child in children]]
#         return collector_node
#
#     @staticmethod
#     def _collector():
#         return TNode(exec_t=ET.MAIN)
#
#     def collect(self):
#         return self.add_child(self._collector())
#
#
# class TaskGraph:
#
#     def __init__(self):
#         self.roots = []
#         self.p_executor = ProcessPoolExecutor(max_workers=2)
#         self.t_executor = ThreadPoolExecutor(max_workers=2)
#         self.loop = asyncio.get_event_loop()
#
#     def root(self):
#         self.roots.append(TNode(exec_t=ET.PROCESS))
#         return self.roots[-1]
#
#     def run(self):
#         running = []
#         for r in self.roots:
#
#
#     def _start(self, node: TNode, iparam=None):
#         if node.exec_t == ET.PROCESS:
#             return self.p_executor.submit(node.work, iparam)
#         if node.exec_t == ET.THREAD:
#             return self.t_executor.submit(node.work, iparam)
#         if node.exec_t == ET.ASYNC:
#             return self.loop.create_future(node.work)
#         if node.exec_t == ET.MAIN:
#
#
#     @staticmethod
#     def async_task(m):
#         return TNode(exec_t=ET.ASYNC, work=m)
#
#     @staticmethod
#     def process_task(m):
#         return TNode(exec_t=ET.PROCESS, work=m)
#
#     @staticmethod
#     def main_task(m):
#         return TNode(exec_t=ET.MAIN, work=m)
#
#     @staticmethod
#     def thread_task(m):
#         return TNode(exec_t=ET.THREAD, work=m)
