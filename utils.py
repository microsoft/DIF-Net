# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)