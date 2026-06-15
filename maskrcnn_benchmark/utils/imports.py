# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified for PyTorch 2.x compatibility (torch._six removed)
import sys
import importlib
import importlib.util


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
def import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module
