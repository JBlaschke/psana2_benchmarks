#!/usr/bin/env python
# -*- coding: utf-8 -*-


from time import sleep

from benchmarking import Event, EventLogger



def log_timer(e1):
    EventLogger().add(e1)



def log_inplace():
    EventLogger().add(Event())



def test_timers(n):

    e1 = Event()
    sleep(n)  # sleep for n seconds
    e2 = Event()

    return e1, e2



def test_logger(n):
    EventLogger().clear()

    e1 = Event()
    sleep(n)  # sleep for n seconds
    e2 = Event()

    log_timer(e1)
    log_timer(e2)



def test_inplace_logger(n):
    EventLogger().clear()

    log_inplace()
    sleep(n)  # sleep for n seconds
    log_inplace()



if __name__ == "__main__":

    e1, e2 = test_timers(1)
    print("Testing event timers:")
    print(f"e1 = {e1.timestamp}\ne2 = {e2.timestamp}")

    test_logger(1)
    print("Testing event logger:")
    logger = EventLogger()
    for i, ts in enumerate(logger.timestamps):
        print(f"e{i+1} = {ts}")

    test_inplace_logger(1)
    print("Testing inplace event logger:")
    logger = EventLogger()
    for i, ts in enumerate(logger.timestamps):
        print(f"e{i+1} = {ts}")
