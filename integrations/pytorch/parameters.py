"""Describe and allocate the APC regions for PyTorch model state."""

from itertools import zip_longest
import re

import torch

import SuperNovaBind as sn