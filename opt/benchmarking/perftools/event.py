#!/usr/bin/env python
# -*- coding: utf-8 -*-


from math import floor
from time import time, strftime, gmtime


from ..util import Singleton


class Event(object):


    def __init__(self):
        self._t_evt = self._time()


    @staticmethod
    def _time():

        t = time()
        s = int(floor(t))
        return (s, int(round((t - s) * 1000)))


    @property
    def timestamp(self):
        
        s, t = self._t_evt
        return strftime("%Y-%m-%dT%H:%MZ%S", gmtime(t)) + (".%03d" % s)


