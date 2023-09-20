## Current docs and status


## Proper MLIR to SPIRV Conversion


The SPIRV standard, on paper, promises interoperability with both OpenCL and Vulkan. However, practical implementation has revealed compatibility challenges. Intel's approach to SPIRV translation is predominantly OpenCL-centric, leading to discrepancies when adapting for Vulkan. This document outlines some of the significant issues encountered and potential solutions.

## 1. MLIR to SPIR-V Conversion

The theoretical interoperability of SPIR-V with both OpenCL and Vulkan has posed certain challenges during practical implementation. Intel's SPIR-V translation, which is predominantly rooted in OpenCL, necessitates modifications to accommodate storage buffers and computing capabilities for Vulkan. 

Several pressing concerns are being addressed:

- **Entry Point Concern**: Vulkan's SPIR-V mandates a main function devoid of parameters, contrasting with OpenCL's version of SPIR-V.
  
- **Descriptor Sets**: Vulkan requires descriptor sets for its compute pipeline stages. Initial support for this is in place, but its efficacy remains unconfirmed due to existing issues with the generated SPIR-V.
  
- **Usage of "get_ptr" vs "tl.load"**: The "get_ptr" operation is specific to Intel's SPIR-V extension. Attempts to employ 'tl.load' have resulted in this operation's appearance. Its role in a Vulkan context is ambiguous. For GPUs with architectures like Hopper/Ampere, 'tl.load' implies loading operands into the L2 Cache (this understanding should be approached with caution). The exact storage location for these operands in the context of Vulkan is yet to be ascertained.

Efforts are ongoing to address these challenges and ensure a smooth translation from MLIR to SPIR-V suitable for Vulkan.

## 2. Compute Kernel/Pipeline Launching in Vulkan

Launching compute kernels in Vulkan introduces challenges distinct from its SyCL counterpart. While SyCL offers a relatively straightforward implementation, Vulkan demands the establishment of an entire compute pipeline. The task is to create a flexible pipeline that can accommodate a myriad of configurations.



1. **Complexity of Vulkan Compute Pipelines**:
    - **Problem**: Vulkan’s compute pipeline setup is inherently more intricate than the SyCL's streamlined approach. The design needs to account for a vast array of potential configurations.
    - **Solution**: Develop a modular and scalable architecture for the Vulkan compute pipeline. This design should allow for easy adjustments and tweaks to cater to different configurations.

2. **Reference Implementations**:
    - **Problem**: While there are existing examples of Vulkan Compute, they might not be equipped to handle dynamic and diverse configurations.
    - **Solution**: Study and analyze the best practices from existing Vulkan Compute examples. Combine the insights from these studies with custom implementations to cater to the specific needs of the project.

3. **JIT Compilation and Code Injection**:
    - **Problem**: The Intel extension approach of treating C++ code as a string and JIT entering values seems promising. However, replicating this for Vulkan comes with the added complexity of managing the compute pipelines.
    - **Solution**: Design a robust JIT compilation system tailored for Vulkan. Ensure that the JIT system can dynamically adapt and interact seamlessly with the Vulkan compute pipeline, allowing for efficient code injection and execution.


Transitioning compute kernel launching to Vulkan necessitates a paradigm shift from the SyCL methodology. Addressing the highlighted challenges will pave the way for a robust and adaptable Vulkan-based compute pipeline. Collaborative efforts and continuous refinement will be essential to achieve this goal.


## 3. Dynamic C++ Compilation for Vulkan with Clang/g++

### Overview

Transitioning from Intel's DPP, OneAPI, and LevelZero-based dynamic C++ compilation requires a shift towards Clang/g++ and Vulkan integration.

### Key Steps:

1. **Replace DPP/OneAPI/LevelZero**: 
    - Develop a Clang/g++ backend for dynamic C++ compilation, ensuring system compatibility.
  
2. **Vulkan Integration**: 
    - Design an interface to link Clang/g++ compiled output with Vulkan, addressing translations and runtime considerations.

3. **Transition Strategy**: 
    - Emphasize modular design to simplify debugging, testing, and scalability.

### Conclusion

Moving from Intel's tools to Clang/g++ and Vulkan might pose challenges, but with a modular approach and robust integration, it can be streamlined and efficient.
However it does present challenges with without oneAPI


## 4. Integration with PyTorch and Addressing Memory Model Challenges


As the project progresses towards leveraging Vulkan for compute, a critical challenge arises: how to integrate with PyTorch's memory model and ensure compatibility, especially considering potential extensions to systems like AMD without ROCm support.

### Key Considerations:

1. **Memory Location & Management**:
    - **Problem**: When using Vulkan for compute, determining where memory resides becomes complex, especially considering potential future integrations.
    - **Potential Solutions**: Assess whether using `torch.device("cpu")` or introducing a new `torch.device("vk")` would be more appropriate. This decision depends on the specifics of memory management and interoperability requirements.

2. **PyTorch Extension Development**:
    - **Problem**: Developing a dedicated extension for PyTorch, similar to Intel's approach, may provide a seamless integration path. However, this could be resource-intensive.
    - **Potential Solutions**: Evaluate the benefits of a dedicated extension against its development time and complexity. Consider alternative integration methods that might be more efficient.

3. **Performance Implications**:
    - **Problem**: Integration changes can impact the overall performance of the system, especially in memory-intensive operations.
    - **Potential Solutions**: Implement performance benchmarking and monitoring to identify bottlenecks and areas for optimization.


### Conclusion

The integration of Vulkan with PyTorch introduces complexities, especially concerning memory management. A thorough understanding of both systems, combined with iterative testing and benchmarking, will be crucial to navigate these challenges successfully.


## 5. Preparation for macOS ARM64 Binaries

Building for the macOS ARM64 architecture is anticipated to be straightforward, given the existing work on LLVM for Triton. The transition will employ MoltenVK in place of Vulkan. MoltenVK is expected to support a comparable range of compute capabilities, ensuring consistent functionality.



## 5B. Evaluating SPIRV-Cross for macOS Integration

### Overview

The compatibility of Vulkan-compatible SPIRV with SPIRV-Cross has emerged as a concern. While preliminary tests show inconsistencies, further refinements in SPIRV conversion may yield better results with SPIRV-Cross.

### Key Considerations:

1. **SPIRV Compatibility**:
    - **Issue**: Vulkan-optimized SPIRV doesn't always align with SPIRV-Cross expectations.
    - **Potential Solution**: Adjust SPIRV conversion processes to enhance compatibility with SPIRV-Cross.

2. **Re-evaluation of Metal Compute Pipeline**:
    - **Issue**: If SPIRV-Cross integration proves successful, there's an opportunity to reconsider using Metal's compute pipeline for macOS, instead of Vulkan.
    - **Trade-off**: This decision involves balancing between code reuse efficiency and optimal performance.

### Conclusion

The integration decision between Vulkan and Metal for macOS hinges on the success of SPIRV-Cross compatibility. A careful evaluation of performance implications and development efficiency will guide the optimal path forward.

