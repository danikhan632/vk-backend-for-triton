import triton._C.libvk_backend_for_triton.triton as _triton
mod=r'''
module attributes {"triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_0d1d2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    %2 = tt.load %1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
    %3 = tt.addptr %arg1, %0 : !tt.ptr<f32>, i32
    %4 = tt.load %3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
    %5 = arith.addf %2, %4 : f32
    %6 = tt.addptr %arg2, %0 : !tt.ptr<f32>, i32
    tt.store %6, %5 {cache = 1 : i32, evict = 1 : i32} : f32
    tt.return
  }
}

'''
arch={'num_warps': 8, 'threads_per_warp': 32} #just example nubers



spirv_code, share_memory_size = _triton.translate_triton_gpu_to_spirv(str(mod), arch)



open('kernel.spirv','w').write(spirv_code)
print("shared memory: ", share_memory_size)