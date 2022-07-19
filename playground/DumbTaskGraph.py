import asyncio
import random
import math
import time
import traceback
import numpy as np

from playground.TaskGraph import TaskGraph


async def generate():
    await asyncio.sleep(1)
    out = random.random() * 100
    print("Generate", out)
    return out


def sqrt(num):
    out = math.sqrt(num)
    print("sqrt", out)
    return out


def square(num):
    out = num**2
    print("square", out)
    return out


def avg(nums):
    m = np.mean(nums)
    print("mean", m)
    return m


if __name__ == "__main__":
    tg = TaskGraph()
    try:
        tg.root().supply(9).split(
            tg.process_task(sqrt, name="sqrt"),
            tg.thread_task(square, name="square")
        ).main_task(avg, name=avg).collect()

        print(tg.run())
    except Exception as e:
        traceback.print_exc()
    finally:
        tg.close()