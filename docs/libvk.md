
This is where work is being made to ensure that the SPIRV geenerated from ttgir is Vulkan-compatible
Below is sample for the libvk being used, and what the output looks like



```python

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

```





above is code that passes in pre-generated TTGIR using python 

Current points to address in SPIRV that present issues
```
  spirv.EntryPoint "GLCompute" @main, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spirv.ExecutionMode @main "SubgroupSize", 32
  ...
  OpEntryPoint GLCompute %main "main" %__builtin_var_WorkgroupId__ %__builtin_var_LocalInvocationId__
```
these will have to be global variables



```spvasm
Starting translation from TritonGPU to SPIRV...
Initializing dialect registry...
Parsing MLIR module...
Operation:  tt.get_program_id
Operation:  tt.addptr
Operation:  tt.load
Operation:  tt.addptr
Operation:  tt.load
Operation:  arith.addf
Operation:  tt.addptr
Operation:  tt.store
Operation:  tt.return
Operation:  tt.func
Operation:  builtin.module
Translating to SPIRV IR...
MLIRContext Address: 0x7ffc03b3d750
Number of Registered Dialects: 13

numWarps 32
threadsPerWarp 32
spirv.module Logical Vulkan requires #spirv.vce<v1.4, [Shader, Float16Buffer, Int64, Int16, Int8, Vector16, VariablePointersStorageBuffer, Float64, VulkanMemoryModel, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, GroupNonUniformShuffle], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_KHR_vulkan_memory_model]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16Buffer, Int64, Int16, Int8, Vector16, VariablePointersStorageBuffer, VulkanMemoryModel, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, GroupNonUniformShuffle], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_KHR_vulkan_memory_model]>, api=Vulkan, #spirv.resource_limits<>>, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  spirv.GlobalVariable @__builtin_var_LocalInvocationId__ built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.func @kernel_0d1d2d(%arg0: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}) "None" attributes {noinline = false, spirv.entry_point_abi = #spirv.entry_point_abi<>, sym_visibility = "public"} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
    %2 = spirv.UConvert %1 : i32 to i64
    %3 = spirv.SConvert %2 : i64 to i32
    %4 = spirv.PtrAccessChain %arg0[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    %true = spirv.Constant true
    %5 = spirv.Undef : i32
    spirv.BranchConditional %true, ^bb1, ^bb2(%5 : i32)
  ^bb1:  // pred: ^bb0
    %6 = spirv.Bitcast %4 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    %7 = spirv.Load "StorageBuffer" %6 : i32
    spirv.Branch ^bb2(%7 : i32)
  ^bb2(%8: i32):  // 2 preds: ^bb0, ^bb1
    %9 = spirv.Bitcast %8 : i32 to f32
    %10 = spirv.PtrAccessChain %arg1[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    spirv.BranchConditional %true, ^bb3, ^bb4(%5 : i32)
  ^bb3:  // pred: ^bb2
    %11 = spirv.Bitcast %10 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    %12 = spirv.Load "StorageBuffer" %11 : i32
    spirv.Branch ^bb4(%12 : i32)
  ^bb4(%13: i32):  // 2 preds: ^bb2, ^bb3
    %14 = spirv.Bitcast %13 : i32 to f32
    %15 = spirv.FAdd %9, %14 : f32
    %16 = spirv.PtrAccessChain %arg2[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
    %17 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi32>
    %18 = spirv.CompositeExtract %17[0 : i32] : vector<3xi32>
    %19 = spirv.UConvert %18 : i32 to i64
    %20 = spirv.SConvert %19 : i64 to i32
    %cst1_i32 = spirv.Constant 1 : i32
    %21 = spirv.ULessThan %20, %cst1_i32 : i32
    %22 = spirv.LogicalAnd %true, %21 : i1
    spirv.BranchConditional %22, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %23 = spirv.Bitcast %15 : f32 to i32
    %24 = spirv.Bitcast %16 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    spirv.Store "StorageBuffer" %24, %23 : i32
    spirv.Branch ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    spirv.Return
  }
}
#spirv.entry_point_abi<>
Entered getInterfaceVariables
Descriptor counter value: 1
Descriptor counter value: 2
Function being processed:
Total interface variables found: 2
Added to interfaceVars: __builtin_var_WorkgroupId__
Added to interfaceVars: __builtin_var_LocalInvocationId__
Contents of interfaceVars: 
__builtin_var_WorkgroupId__
__builtin_var_LocalInvocationId__
spirv.module Logical Vulkan requires #spirv.vce<v1.4, [Shader, Float16Buffer, Int64, Int16, Int8, Vector16, VariablePointersStorageBuffer, Float64, VulkanMemoryModel, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, GroupNonUniformShuffle], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_KHR_vulkan_memory_model]> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Shader, Float16Buffer, Int64, Int16, Int8, Vector16, VariablePointersStorageBuffer, VulkanMemoryModel, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, GroupNonUniformShuffle], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_KHR_vulkan_memory_model]>, api=Vulkan, #spirv.resource_limits<>>, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  spirv.GlobalVariable @__builtin_var_LocalInvocationId__ bind(1, 0) built_in("LocalInvocationId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.GlobalVariable @__builtin_var_WorkgroupId__ bind(0, 0) built_in("WorkgroupId") : !spirv.ptr<vector<3xi32>, Input>
  spirv.func @main(%arg0: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f32, StorageBuffer> {tt.divisibility = 16 : i32}) "None" attributes {noinline = false} {
    %__builtin_var_WorkgroupId___addr = spirv.mlir.addressof @__builtin_var_WorkgroupId__ : !spirv.ptr<vector<3xi32>, Input>
    %0 = spirv.Load "Input" %__builtin_var_WorkgroupId___addr : vector<3xi32>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi32>
    %2 = spirv.UConvert %1 : i32 to i64
    %3 = spirv.SConvert %2 : i64 to i32
    %4 = spirv.PtrAccessChain %arg0[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    %true = spirv.Constant true
    %5 = spirv.Undef : i32
    spirv.BranchConditional %true, ^bb1, ^bb2(%5 : i32)
  ^bb1:  // pred: ^bb0
    %6 = spirv.Bitcast %4 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    %7 = spirv.Load "StorageBuffer" %6 : i32
    spirv.Branch ^bb2(%7 : i32)
  ^bb2(%8: i32):  // 2 preds: ^bb0, ^bb1
    %9 = spirv.Bitcast %8 : i32 to f32
    %10 = spirv.PtrAccessChain %arg1[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    spirv.BranchConditional %true, ^bb3, ^bb4(%5 : i32)
  ^bb3:  // pred: ^bb2
    %11 = spirv.Bitcast %10 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    %12 = spirv.Load "StorageBuffer" %11 : i32
    spirv.Branch ^bb4(%12 : i32)
  ^bb4(%13: i32):  // 2 preds: ^bb2, ^bb3
    %14 = spirv.Bitcast %13 : i32 to f32
    %15 = spirv.FAdd %9, %14 : f32
    %16 = spirv.PtrAccessChain %arg2[%3] : !spirv.ptr<f32, StorageBuffer>, i32
    %__builtin_var_LocalInvocationId___addr = spirv.mlir.addressof @__builtin_var_LocalInvocationId__ : !spirv.ptr<vector<3xi32>, Input>
    %17 = spirv.Load "Input" %__builtin_var_LocalInvocationId___addr : vector<3xi32>
    %18 = spirv.CompositeExtract %17[0 : i32] : vector<3xi32>
    %19 = spirv.UConvert %18 : i32 to i64
    %20 = spirv.SConvert %19 : i64 to i32
    %cst1_i32 = spirv.Constant 1 : i32
    %21 = spirv.ULessThan %20, %cst1_i32 : i32
    %22 = spirv.LogicalAnd %true, %21 : i1
    spirv.BranchConditional %22, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %23 = spirv.Bitcast %15 : f32 to i32
    %24 = spirv.Bitcast %16 : !spirv.ptr<f32, StorageBuffer> to !spirv.ptr<i32, StorageBuffer>
    spirv.Store "StorageBuffer" %24, %23 : i32
    spirv.Branch ^bb6
  ^bb6:  // 2 preds: ^bb4, ^bb5
    spirv.Return
  }
  spirv.EntryPoint "GLCompute" @main, @__builtin_var_WorkgroupId__, @__builtin_var_LocalInvocationId__
  spirv.ExecutionMode @main "SubgroupSize", 32
}
Starting first optimization stage...
Registering passes for optimization...
Executing optimization...
Binary remains unchanged after optimization.
Starting second optimization stage...
Registering passes for optimization...
Executing optimization...
Binary remains unchanged after optimization.
Optimization complete.
OpCapability Shader
OpCapability Float16Buffer
OpCapability Int64
OpCapability Int16
OpCapability Int8
OpCapability Vector16
OpCapability VariablePointersStorageBuffer
OpCapability Float64
OpCapability VulkanMemoryModel
OpCapability AtomicFloat32AddEXT
OpCapability ExpectAssumeKHR
OpCapability SubgroupDispatch
OpCapability GroupNonUniformShuffle
OpExtension "SPV_EXT_shader_atomic_float_add"
OpExtension "SPV_KHR_expect_assume"
OpExtension "SPV_KHR_vulkan_memory_model"
OpExtension "VK_KHR_16bit_storage"
OpMemoryModel Logical Vulkan
OpEntryPoint GLCompute %main "main" %__builtin_var_WorkgroupId__ %__builtin_var_LocalInvocationId__
OpExecutionMode %main SubgroupSize 32
OpName %__builtin_var_LocalInvocationId__ "__builtin_var_LocalInvocationId__"
OpName %__builtin_var_WorkgroupId__ "__builtin_var_WorkgroupId__"
OpName %main "main"
OpDecorate %__builtin_var_LocalInvocationId__ Binding 0
OpDecorate %__builtin_var_LocalInvocationId__ BuiltIn LocalInvocationId
OpDecorate %__builtin_var_LocalInvocationId__ DescriptorSet 1
OpDecorate %__builtin_var_WorkgroupId__ Binding 0
OpDecorate %__builtin_var_WorkgroupId__ BuiltIn WorkgroupId
OpDecorate %__builtin_var_WorkgroupId__ DescriptorSet 0
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%__builtin_var_LocalInvocationId__ = OpVariable %_ptr_Input_v3uint Input
%__builtin_var_WorkgroupId__ = OpVariable %_ptr_Input_v3uint Input
%void = OpTypeVoid
%float = OpTypeFloat 32
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
%6 = OpTypeFunction %void %_ptr_StorageBuffer_float %_ptr_StorageBuffer_float %_ptr_StorageBuffer_float
%ulong = OpTypeInt 64 0
%bool = OpTypeBool
%true = OpConstantTrue %bool
%23 = OpUndef %uint
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
%uint_1 = OpConstant %uint 1
%main = OpFunction %void None %6
%11 = OpFunctionParameter %_ptr_StorageBuffer_float
%12 = OpFunctionParameter %_ptr_StorageBuffer_float
%13 = OpFunctionParameter %_ptr_StorageBuffer_float
%14 = OpLabel
%15 = OpLoad %v3uint %__builtin_var_WorkgroupId__
%16 = OpCompositeExtract %uint %15 0
%18 = OpUConvert %ulong %16
%19 = OpSConvert %uint %18
%20 = OpPtrAccessChain %_ptr_StorageBuffer_float %11 %19
OpBranchConditional %true %24 %25
%24 = OpLabel
%27 = OpBitcast %_ptr_StorageBuffer_uint %20
%28 = OpLoad %uint %27
OpBranch %25
%25 = OpLabel
%29 = OpPhi %uint %28 %24 %23 %14
%30 = OpBitcast %float %29
%31 = OpPtrAccessChain %_ptr_StorageBuffer_float %12 %19
OpBranchConditional %true %32 %33
%32 = OpLabel
%34 = OpBitcast %_ptr_StorageBuffer_uint %31
%35 = OpLoad %uint %34
OpBranch %33
%33 = OpLabel
%36 = OpPhi %uint %35 %32 %23 %25
%37 = OpBitcast %float %36
%38 = OpFAdd %float %30 %37
%39 = OpPtrAccessChain %_ptr_StorageBuffer_float %13 %19
%40 = OpLoad %v3uint %__builtin_var_LocalInvocationId__
%41 = OpCompositeExtract %uint %40 0
%42 = OpUConvert %ulong %41
%43 = OpSConvert %uint %42
%45 = OpULessThan %bool %43 %uint_1
%46 = OpLogicalAnd %bool %true %45
OpBranchConditional %46 %47 %48
%47 = OpLabel
%49 = OpBitcast %uint %38
%50 = OpBitcast %_ptr_StorageBuffer_uint %39
OpStore %50 %49
OpBranch %48
%48 = OpLabel
OpReturn
OpFunctionEnd

Translation complete. Returning SPIRV module and shared attribute...
shared memory:  0
```


