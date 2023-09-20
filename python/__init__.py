import functools
import hashlib
import os
import re
import sysconfig
import tempfile
from pathlib import Path
import traceback

def printc(obj, color="cyan"):
    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }
    colored_text = f"\033[{color_code[color]}m{obj}\033[0m" if color in color_code else obj
    print(colored_text)
import setuptools
import torch


#_________________________________________________________________________________________________

import triton._C.libvk_backend_for_triton.triton as _triton
from triton.common.backend import BaseBackend, register_backend
from triton.compiler.make_launcher import make_so_cache_key
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import DriverBase
from triton.runtime.jit import version_key


def _add_external_libs(mod, libs):
    for name, path in libs.items():
        if len(name) == 0 or len(path) == 0:
            return
    _triton.add_external_libs(mod, list(libs.keys()), list(libs.values()))


# SPIRV translation

def ttgir_to_spirv(mod, extern_libs, arch):
    if extern_libs:
        _add_external_libs(mod, extern_libs)
    spirv_code, share_memory_size = _triton.translate_triton_gpu_to_spirv(str(mod), arch)
    mod.share_memory_size = share_memory_size
    return spirv_code


def spirv_to_spvbin(spirv: str, compute_capability: int):
    # return _triton.compile_spirv_to_spvbin(spirv, compute_capability)
    return _triton.compile_spirv_to_spvbin(spirv, 80)


def spirv_get_kernel_name(spirv: str) -> str:
    '''
    Get kernel name from SPIRV code.
    This Kernel name is required when launching the kernel.
    '''
    assert spirv
    decl_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if line.startswith('OpName'):
            decl_ops += [line.split()[-1]]
    def_ops = []
    for line in spirv.split('\n'):
        line = line.strip()
        if re.compile(r'\bOpEntryPoint\b').search(line):
            def_op = line.split()[2][1:]
            if '"{}"'.format(def_op) in decl_ops:
                def_ops += [def_op]
    assert len(def_ops) == 1, "expect only one kernel per spriv"
    return def_ops[0]


def ty_to_cpp(ty):
    if ty[0] == '*':
        return "void*"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


def generate_launcher(constants, signature):
    arg_decls = ', '.join(f"{ty_to_cpp(ty)} arg{i}" for i, ty in signature.items())

    def _extracted_type_pybind11(ty):
        if ty[0] == '*':
            return "py::object"
        return {
            'i1': 'int32_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            'fp32': 'float',
            'f32': 'float',
            'fp64': 'double',
        }[ty]

    #TODO interface with pipeline.cpp    
    


def _build_xpu_ext(name, src, srcdir):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))

    # fallback on setuptools
    extra_compile_args = ['-fPIC', '-w']
    # library_dirs = [cuda_lib_dir]
    # include_dirs = [srcdir, cu_include_dir]
    # library_dirs = []
    # include_dirs = [srcdir]
    libraries = ['ze_loader']
    # extra arguments
    # extra_link_args = []
    # create extension module
    # build extension module

    # create extension module
    ext = DPCPPExtension(name,
                         [src],
                         extra_compile_args=extra_compile_args,
                         libraries=libraries)

    args = ['build_ext']
    args.append('--build-temp=' + srcdir)
    args.append('--build-lib=' + srcdir)
    args.append('-q')
    args = dict(
        name=name,
        ext_modules=[ext],
        cmdclass={
            'build_ext': DpcppBuildExtension},
        script_args=args,
    )
    # with quiet():
    setuptools.setup(**args)
    return so




class VKBackend(BaseBackend):
    stub_so_path = ""
    # breakpoint()

    def __init__(self, device_type: str) -> None:
        super(VKBackend, self).__init__(device_type)
        printc("sdsds",'blue')
        # self.driver = SYCLDriver()
        printc("gen driver class",'blue')

    def add_stages(self, arch, extern_libs, stages):
        filter_in_stages = ["ast", "ttir", "ttgir"]
        filter_out_stages = []
        for key, _ in stages.items():
            if key not in filter_in_stages:
                filter_out_stages.append(key)
        for filter_out_key in filter_out_stages:
            stages.pop(filter_out_key)

        stages["spirv"] = (lambda path: Path(path).read_text(),
                           lambda src: ttgir_to_spirv(src, extern_libs, arch))
        stages["spvbin"] = (lambda path: Path(path).read_bytes(),
                            lambda src: spirv_to_spvbin(src, arch))

    def add_meta_info(self, ir, module, next_module, metadata, asm):
        if ir == "spirv":
            metadata["name"] = spirv_get_kernel_name(next_module)
            if "shared" not in metadata:
                metadata["shared"] = module.share_memory_size

        if ir == "spvbin":
            asm[ir] = next_module

    def get_driver(self):
        return self.driver

    def get_stream(self, idx=None):
        # if idx is None:
        #     idx = self.get_current_device()
        # return torch.xpu.current_stream(idx).sycl_queue
        raise NotImplementedError("get_stream not implemented yet")


    @functools.lru_cache(None)
    def get_device_properties(self, device):
        # return self.driver.utils.get_device_properties(torch.xpu.device(device).to_sycl_dev())
        raise NotImplementedError("")

    def get_current_device(self):
        # return torch.xpu.current_device()
        raise NotImplementedError("")


    def set_current_device(self, device):
        # torch.xpu.set_device(device)
        raise NotImplementedError("")

    def get_load_binary_fn(self):

        # def _load_binary_fn(kernel_name, binary, shared_size, device):
        #     return self.driver.utils.load_binary(kernel_name, binary, shared_size, torch.xpu.device(device).to_sycl_dev())

        # return _load_binary_fn
        raise NotImplementedError("")

    def get_kernel_bin(self):
        return "spvbin"

    def get_architecture_descriptor(self, **kwargs):
        # dev_props = torch.xpu.get_device_properties(torch.xpu.current_device())
        # max_work_group_size = 256
        # max_num_subgroup = 128
        # subgroup_sizes = 64
        # # TODO: chose a reasonable subgroup size
        # threads_per_warp = 32
        # num_warps = max_work_group_size // threads_per_warp
        # capability = {"num_warps": num_warps, "threads_per_warp": threads_per_warp}
        # return capability
        raise NotImplementedError("")

    def make_launcher_stub(self, name, signature, constants):
        # name of files that are cached
        so_cache_key = make_so_cache_key(version_key(), signature, constants)
        so_cache_manager = get_cache_manager(so_cache_key)
        so_name = f"{name}.so"
        # retrieve stub from cache if it exists
        cache_path = so_cache_manager.get_file(so_name)
        if cache_path is None:
            with tempfile.TemporaryDirectory() as tmpdir:
                src = generate_launcher(constants, signature)
                src_path = os.path.join(tmpdir, "main.c")
                with open(src_path, "w") as f:
                    f.write(src)
                so = _build_xpu_ext(name, src_path, tmpdir)
                with open(so, "rb") as f:
                    return so_cache_manager.put(f.read(), so_name, binary=True)
        else:
            return cache_path


register_backend("cpu", VKBackend)
