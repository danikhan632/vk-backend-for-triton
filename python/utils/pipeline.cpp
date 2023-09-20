#include <iostream>
#include <fstream>
#include <chrono>
// Comment this to disable VMA support
#define WITH_VMA

#include <vulkan/vulkan.hpp>

#ifdef WITH_VMA
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#endif

int main()
{
    // Instance Variables
    const float QueuePriority = 1.0f;
    const uint32_t NumElements = 10000000;
    const std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBinding = {
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
    };

    vk::ApplicationInfo AppInfo;
    std::vector<const char*> Layers;
    vk::InstanceCreateInfo InstanceCreateInfo;
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::PhysicalDeviceProperties DeviceProps;
    uint32_t ApiVersion;
    vk::PhysicalDeviceLimits DeviceLimits;
    std::vector<vk::QueueFamilyProperties> QueueFamilyProps;
    uint32_t ComputeQueueFamilyIndex;
    vk::DeviceQueueCreateInfo DeviceQueueCreateInfo;
    vk::DeviceCreateInfo DeviceCreateInfo;
    vk::Device Device;
    uint32_t BufferSize;
    vk::BufferCreateInfo BufferCreateInfo;
    VmaAllocatorCreateInfo AllocatorInfo;
    VmaAllocator Allocator;
    VkBuffer InBufferRaw;
    VkBuffer OutBufferRaw;
    VmaAllocationCreateInfo AllocationInfo;
    vk::Buffer InBuffer;
    vk::Buffer OutBuffer;
    std::vector<char> ShaderContents;
    vk::ShaderModuleCreateInfo ShaderModuleCreateInfo;
    vk::ShaderModule ShaderModule;
    vk::DescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo;
    vk::DescriptorSetLayout DescriptorSetLayout;
    vk::PipelineLayoutCreateInfo PipelineLayoutCreateInfo;
    vk::PipelineLayout PipelineLayout;
    vk::PipelineCache PipelineCache;
    vk::PipelineShaderStageCreateInfo PipelineShaderCreateInfo;
    vk::ComputePipelineCreateInfo ComputePipelineCreateInfo;
    vk::Pipeline ComputePipeline;
    vk::DescriptorPoolSize DescriptorPoolSize;
    vk::DescriptorPoolCreateInfo DescriptorPoolCreateInfo;
    vk::DescriptorPool DescriptorPool;
    vk::DescriptorSetAllocateInfo DescriptorSetAllocInfo;
    std::vector<vk::DescriptorSet> DescriptorSets;
    vk::DescriptorSet DescriptorSet;
    vk::DescriptorBufferInfo InBufferInfo;
    vk::DescriptorBufferInfo OutBufferInfo;
    std::vector<vk::WriteDescriptorSet> WriteDescriptorSets;
    vk::CommandPoolCreateInfo CommandPoolCreateInfo;
    vk::CommandPool CommandPool;
    vk::CommandBufferAllocateInfo CommandBufferAllocInfo;
    std::vector<vk::CommandBuffer> CmdBuffers;
    vk::CommandBuffer CmdBuffer;
    vk::CommandBufferBeginInfo CmdBufferBeginInfo;
    vk::Queue Queue;
    vk::Fence Fence;
    vk::SubmitInfo SubmitInfo;
    int32_t* InBufferPtr;
    int32_t* OutBufferPtr;
    // Create configuration for VulkanComputePipeline
    json config = {{Python_input_config}};
    // json config = {
    //     {"inputType", "float"},
    //     {"outputType", "float"},
    //     {"numWarps", 32},
    //     {"groupCountX", {}},
    //     {"groupCountY", 1},
    //     {"groupCountZ", 1},
    //     {"bufferSize", 1024},
    //     {"shaderFile", "kernel.spvbin"},
    //     {"descriptors", {
    //         {
    //             {"binding", 0},
    //             {"descriptorType", "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"},
    //             {"descriptorCount", 1},
    //             {"poolSize", 1},
    //             {"stageFlags", "VK_SHADER_STAGE_COMPUTE_BIT"},
    //             {"data", arrayA}
    //         },
    //         {
    //             {"binding", 1},
    //             {"descriptorType", "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"},
    //             {"descriptorCount", 1},
    //             {"poolSize", 1},
    //             {"stageFlags", "VK_SHADER_STAGE_COMPUTE_BIT"},
    //             {"data", arrayB}
    //         },
    //         {
    //             {"binding", 2},
    //             {"descriptorType", "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER"},
    //             {"descriptorCount", 1},
    //             {"poolSize", 1},
    //             {"stageFlags", "VK_SHADER_STAGE_COMPUTE_BIT"},
    //             {"isOutput", true}
    //         }
    //     }},
    //     {"maxSets", 1}
    // };
        const std::string inputType = config["inputType"];
        const std::string outputType = config["outputType"];
        const uint32_t numWarps = config["numWarps"];
        const uint32_t groupCountX = config["groupCountX"];
        const uint32_t groupCountY = config["groupCountY"];
        const uint32_t groupCountZ = config["groupCountZ"];
        const uint32_t bufferSize = config["bufferSize"];
        const std::string shaderFile = config["shaderFile"];
        const uint32_t maxSets = config["maxSets"];

        // Descriptors
        std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutBindings;
        std::vector<vk::DescriptorPoolSize> DescriptorPoolSizes;
        for (const auto &descriptor : config["descriptors"])
        {
            uint32_t binding = descriptor["binding"];
            vk::DescriptorType descriptorType = toVulkanDescriptorType(descriptor["descriptorType"]); // todo: handle
            uint32_t descriptorCount = descriptor["descriptorCount"];
            uint32_t poolSize = descriptor["poolSize"];
            vk::ShaderStageFlagBits stageFlags = toVulkanShaderStage(descriptor["stageFlags"]); // todo : handle
            
            DescriptorSetLayoutBindings.push_back(
                {binding, descriptorType, descriptorCount, stageFlags}
            );

            DescriptorPoolSizes.push_back(
                {descriptorType, descriptorCount * poolSize}
            );
        }

    return 0;
}
