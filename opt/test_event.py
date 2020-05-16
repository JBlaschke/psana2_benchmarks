#!/usr/bin/env python
# -*- coding: utf-8 -*-


from time import sleep

from benchmarking import Event, EventLogger



def log_timer(e1):
    logger = EventLogger()
    logger.add(e1)



def log_inplace():
    logger = EventLogger()
    logger.add(Event())



def test_timers(n):

    e1 = Event()
    sleep(n)  # sleep for n seconds
    e2 = Event()

    return e1, e2



def test_logger(n):

    e1 = Event()
    sleep(n)  # sleep for n seconds
    e2 = Event()

    log_timer(e1)
    log_timer(e2)


if __name__ == "__main__":

    e1, e2 = test_timers(1)
    print("Testing event timers:")
    print(f"e1 = {e1.timestamp}\ne2 = {e2.timestamp}")

    test_logger(1)
    print("Testing event timers:")
    logger = EventLogger()
    for i, ts in enumerate(logger.timestamps):
        print(f"e{i+1} = {ts}")
