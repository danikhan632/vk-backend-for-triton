# Vulkan Backend for Triton

This is designed as a semi-hardware abstract backend enabling Vulkan Compatible Devices to utilize Triton.
Currently the first real hardware backend planned is for Apple Metal devices using MoltenVK* however progress is currently being made more towards ensuring that it produces Valid SPIRV ASM with developement for Vulkan Compute pipelines and Pytorch Memory Management currently under construction

# Credits
This does use a lot of the code from Intel's Extension for Triton and I'd like to give a huge thanks to Intel and Eikan Wang. Also thanks to the triton team @ openai, none of this 
would be possible without their help.
[OpenAI Triton](https://github.com/openai/triton)
[Intel® Extension for Triton ](https://github.com/intel/intel-xpu-backend-for-triton)

## Prerequisites

As of now please follow the instructions from [Intel® Extension for Triton ](https://github.com/intel/intel-xpu-backend-for-triton)
Once down with that follow instructions below
## Build VK Backend

```Bash
# Clone OpenAI/Triton
git clone https://github.com/openai/triton.git
cd triton
git checkout 5df904233c11a65bd131ead7268f84cca7804275
#note that this is an Older version of Triton hopefully will be rebased soon.

cd third_party
git clone https://github.com/danikhan632/vk-backend-for-triton.git
mv ./vk-backend-for-triton ./vk_backend

```

Now Build Triton with VK backend enabled:

```Bash
cd {triton-root-dir}
cd python
TRITON_CODEGEN_VK_BACKEND=1 python setup.py develop
```
# Usage Guide

At present, the functionality of this software is limited, with only `libvk` being operational. We're actively working on it to generate Vulkan-compatible SPIRV. For a comprehensive understanding and usage details, please refer to our documentation.

# Contributing

We welcome and encourage contributions from the community! Whether you're a seasoned developer or someone new to the field, your ideas, bug fixes, and enhancements can make a difference.

**How to Contribute:**
1. Fork the repository.
2. Make your changes or additions.
3. Create a pull request with a detailed description of your updates.

Also feel free to reach out on [discord](https://discord.gg/Mg5zYBwt)
# License

This project is licensed under the MIT License. This means you're free to use, modify, distribute, and sublicense the software, provided you include the original copyright and permission notices. For complete terms and conditions, please refer to the `LICENSE` file in the repository.

