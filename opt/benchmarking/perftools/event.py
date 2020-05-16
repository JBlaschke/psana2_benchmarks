#!/usr/bin/env python
# -*- coding: utf-8 -*-


import math
import time



class Event(object):


    def __init__(self):
        self._t_evt = self._time()


    @staticmethod
    def _time():

        t = time.time()
        s = int(math.floor(t))
        return (s, int(round((t - s) * 1000)))


    @property
    def timestamp(self):
        
        s, t = self._t_evt
        return time.strftime("%Y-%m-%dT%H:%MZ%S", time.gmtime(t)) + (".%03d" % s)
