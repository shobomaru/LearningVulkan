// based on SimpleTexture.cpp

#define _WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string_view>
#include <exception>
#include <functional>
#include <fstream>
#define VK_USE_PLATFORM_WIN32_KHR
#include <Vulkan/vulkan.hpp>
#include <wrl/client.h>
#include <dxc/dxcapi.h>
#include <DirectXMath.h>

#pragma comment(lib, "vulkan-1.lib")
#pragma comment(lib, "dxcompiler.lib")

using namespace std;
using Microsoft::WRL::ComPtr;

namespace
{
	const int WINDOW_WIDTH = 640;
	const int WINDOW_HEIGHT = 360;
	const int BUFFER_COUNT = 3;
};

struct destructor_callback
{
	function<void()> mFun;
	~destructor_callback() { if (mFun) mFun(); }
};

class VLK
{
	void CHK(vk::Result r)
	{
		if (r != vk::Result::eSuccess)
			throw runtime_error("VkResult is failed value.");
	}
	void CHK(VkResult r)
	{
		if (r != VK_SUCCESS)
			throw runtime_error("VkResult is failed value.");
	}
	void CHK(HRESULT hr)
	{
		if (FAILED(hr)) {
			throw runtime_error("HRESULT is failed value.");
		}
	}
	void ASSERT(bool r, const char* msg)
	{
		if (!r)
			throw runtime_error(msg);
	}
	size_t ALIGN(size_t s, size_t align)
	{
		return (s + align - 1) & ~(align - 1);
	}

	vk::UniqueInstance mInstance;
	VkDebugUtilsMessengerEXT mDebugUtilsMessenger = {};
	destructor_callback mDebugUtilsDeleter;
	vk::UniqueSurfaceKHR mSurface;
	vk::UniqueDevice mDevice;
	uint32_t mQueueFamilyGfxIdx = 0;
	uint32_t mQueueFamilyDmaIdx = 0;
	vk::Queue mQueue = {};
	vk::Queue mQueuePresent = {};
	vk::Queue mQueueDma = {};
	vk::UniqueSwapchainKHR mSwapchain;
	std::vector<vk::Image> mBackBuffers;
	std::vector<vk::UniqueImageView> mBackBuffersView;
	vk::UniqueSemaphore mSwapchainSema;
	vk::UniqueSemaphore mDrawingSema;

	uint32_t mBackBufferIdx = 0;
	uint64_t mFrameCount = 0;
	vk::UniqueCommandPool mCmdPool;
	std::vector<vk::UniqueCommandBuffer> mCmdBuf;
	vk::UniqueFence mCmdFence[2];
	vk::UniqueDescriptorPool mDescPools[2];

	vk::UniqueRenderPass mRenderPass;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutBuf;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutTex;
	vk::UniquePipelineLayout mPipelineLayout;
	vk::UniquePipeline mPSO;

	vk::Extent2D mSceneExtent;
	vk::UniqueImage mSceneColor;
	vk::UniqueDeviceMemory mSceneColorMemory;
	vk::UniqueImageView mSceneColorView;
	vk::UniqueImage mSceneDepth;
	vk::UniqueDeviceMemory mSceneDepthMemory;
	vk::UniqueImageView mSceneDepthView;
	vk::UniqueFramebuffer mSceneFramebuffer;

	struct VertexElement
	{
		float position[3];
		float normal[3];
		float texcoord[2];
	};
	static constexpr int SphereSlices = 12;
	static constexpr int SphereStacks = 12;

	vk::UniqueBuffer mSphereVB;
	vk::UniqueBuffer mSphereIB;
	vk::UniqueDeviceMemory mSphereMemory;

	static constexpr int UniformBufferSize = 1 * 1000 * 1000;
	vk::UniqueBuffer mUniformBuffers[2];
	vk::UniqueDeviceMemory mUniformMemory;
	vk::DeviceSize mUniformMemoryOffsets[2];

	vk::UniqueCommandPool mDmaCmdPool;
	vk::UniqueCommandBuffer mDmaCmdBuf;
	vk::UniqueSemaphore mDmaSema;
	vk::UniqueBuffer mImageUploadBuffer;
	vk::UniqueDeviceMemory mImageUploadMemory;

	vk::UniqueImage mImgList[6];
	vk::UniqueImageView mImgListView[6];
	vk::UniqueDeviceMemory mImagesMemory;

	vk::DescriptorSet mSphereDescSetBuf[2];
	vk::DescriptorSet mSphereDescSetTex[2];

	// Define single color lists for image creation
	uint32_t mImageDataList[6] = {
		0xFF0000FF, 0xFF00FFFF, 0xFF00FF00, 0xFFFFFF00, 0xFFFF0000, 0xFFFF00FF,
	};

public:
	~VLK()
	{
		mQueue.waitIdle();
		mQueuePresent.waitIdle();
	}

	VLK(int width, int height, HWND hwnd)
	{
		static_assert(_countof(mImageDataList) == _countof(mImgList));

#ifdef _DEBUG
		const char* const layers[] = {
			"VK_LAYER_KHRONOS_validation",
		};
		const uint32_t layerCount = _countof(layers);
#else
		const char** const layers{};
		const uint32_t layerCount = 0;
#endif

		const char* const extensions[] = {
			VK_KHR_SURFACE_EXTENSION_NAME,
			VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
#ifdef _DEBUG
			VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
#endif
		};

		const char* const deviceExtensions[] = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME,
			VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
			VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
		};

		// Create a Vulkan instance
		const auto appInfo = vk::ApplicationInfo(
			"MyApp", VK_MAKE_API_VERSION(0, 0, 0, 0),
			"LearningVulkan", VK_MAKE_API_VERSION(0, 0, 0, 0), VK_API_VERSION_1_2);
		const auto createInfo = vk::InstanceCreateInfo(
			vk::InstanceCreateFlags(), &appInfo,
			layerCount, layers,
			_countof(extensions), extensions);
		mInstance = vk::createInstanceUnique(createInfo);

		// Check extension capabilities
#if _DEBUG
		if (layerCount > 0)
		{
			const auto supportedLayers = vk::enumerateInstanceLayerProperties();
			for (auto& layer : layers)
			{
				ASSERT(find_if(begin(supportedLayers), end(supportedLayers),
						[&](const vk::LayerProperties& s) { return string_view(s.layerName) == layer; }
					) != end(supportedLayers), "Layer not available");
			}
		}
#endif
		for (auto& ext : extensions)
		{
			const auto supportedExts = vk::enumerateInstanceExtensionProperties();
			ASSERT(find_if(begin(supportedExts), end(supportedExts),
					[&](const vk::ExtensionProperties& s) { return string_view(s.extensionName) == ext; }
				) != end(supportedExts), "Extension not available");
		}

		// Set debug util
#ifdef _DEBUG
		{
			auto info = vk::DebugUtilsMessengerCreateInfoEXT(
				vk::DebugUtilsMessengerCreateFlagsEXT(),
				vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
				vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
				[](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
					cout << pCallbackData->pMessage << endl;
					OutputDebugStringA(pCallbackData->pMessage);
					OutputDebugStringW(L"\n");
					if (messageSeverity == VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
						throw runtime_error("Validation error");
					return VK_FALSE;
				},
				nullptr);
			auto f = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(mInstance.get(), "vkCreateDebugUtilsMessengerEXT");
			auto r = f(*mInstance, reinterpret_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&info), nullptr, &mDebugUtilsMessenger);
			CHK(r);
			mDebugUtilsDeleter.mFun = [&]() {
				auto f = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(mInstance.get(), "vkDestroyDebugUtilsMessengerEXT");
				f(*mInstance, mDebugUtilsMessenger, nullptr);
				mDebugUtilsMessenger = {};
			};
		}
#endif

		// Create a surface
		const auto surfaceInfo = vk::Win32SurfaceCreateInfoKHR({}, (HINSTANCE)GetModuleHandle(0), hwnd);
		mSurface = mInstance->createWin32SurfaceKHRUnique(surfaceInfo);

		// Get device queue
		const vk::PhysicalDevice physDevice = mInstance->enumeratePhysicalDevices()[0];
		const auto queueFamilyProps = physDevice.getQueueFamilyProperties();
		const uint32_t queueGfxIdx = [&]() {
			for (uint32_t i = 0; i < queueFamilyProps.size(); ++i)
			{
				if (queueFamilyProps[i].queueFlags & vk::QueueFlagBits::eGraphics)
				{
					ASSERT(physDevice.getSurfaceSupportKHR(i, *mSurface) == VK_TRUE, "Separate queue currently not supported");
					return i;
				}
			}
			throw runtime_error("No grapgics queue found");
			return 0u;
		}();
		mQueueFamilyGfxIdx = queueGfxIdx;
		const uint32_t queueDmaIdx = [&]() {
			const vk::QueueFlags mask =
				vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer;
			uint32_t f = ~0u;
			for (uint32_t i = 0; i < queueFamilyProps.size(); ++i)
			{
				auto queueFlags = queueFamilyProps[i].queueFlags & mask;
				if (queueFlags == vk::QueueFlagBits::eTransfer)
				{
					return i;
				}
				if (queueFlags & vk::QueueFlagBits::eTransfer)
				{
					f = i;
				}
			}
			ASSERT(f != ~0u, "No transfer queue found");
			return f;
		}();
		mQueueFamilyDmaIdx = queueDmaIdx;

		// Check device extension capabilities
		for (auto& ext : deviceExtensions)
		{
			const auto supportedExts = physDevice.enumerateDeviceExtensionProperties();
			ASSERT(find_if(begin(supportedExts), end(supportedExts),
					[&](const vk::ExtensionProperties& s) { return string_view(s.extensionName) == ext; }
				) != end(supportedExts), "Extension not available");
		}

		// Check device features
		const auto features = physDevice.getFeatures();
		const auto props = physDevice.getProperties();
		cout << "Device Name: " << props.deviceName << endl;
		ASSERT(features.samplerAnisotropy, "Anisotropic sampling is not supported");
		ASSERT(props.limits.maxBoundDescriptorSets >= 4, "maxBoundDescriptorSets is insufficient");
		ASSERT(props.limits.maxSamplerAnisotropy >= 4, "maxSamplerAnisotropy is insufficient");

		// Create a device
		float queueDefaultPriority = 1.0f;
		vector<vk::DeviceQueueCreateInfo> deviceQueueInfos = {
			{{}, queueGfxIdx, 1, &queueDefaultPriority}
		};
		if (queueGfxIdx != queueDmaIdx) {
			deviceQueueInfos.push_back({ {}, queueDmaIdx, 1, &queueDefaultPriority });
		}
		const auto enableFeatures12 = vk::PhysicalDeviceVulkan12Features()
			// descriptor indexing
			.setDescriptorIndexing(1)
			.setRuntimeDescriptorArray(1) // SPIR-V capability
			.setDescriptorBindingPartiallyBound(1) // (option)
			.setDescriptorBindingVariableDescriptorCount(1)  // (option)
			// non uniform resource index
			.setShaderSampledImageArrayNonUniformIndexing(1);
		const auto deviceCreateInfo = vk::DeviceCreateInfo(
			{}, (uint32_t)deviceQueueInfos.size(), deviceQueueInfos.data(),
			0, nullptr, _countof(deviceExtensions), deviceExtensions
		).setPNext(&enableFeatures12);
		mDevice = physDevice.createDeviceUnique(deviceCreateInfo);

		// Get device queues
		mQueue = mDevice->getQueue(queueGfxIdx, 0);
		mQueuePresent = mDevice->getQueue(queueGfxIdx, 0);
		mQueueDma = mDevice->getQueue(queueDmaIdx, 0);

		// Check surface format
		const auto surfaceFormat = vk::Format::eB8G8R8A8Unorm;
		const auto surfaceColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
		const auto supportedSurfaceFormats = physDevice.getSurfaceFormatsKHR(*mSurface);
		ASSERT(find_if(supportedSurfaceFormats.begin(),
				supportedSurfaceFormats.end(),
				[&](vk::SurfaceFormatKHR s) {
					return s.format == surfaceFormat && s.colorSpace == surfaceColorSpace;
				}
			) != supportedSurfaceFormats.end() , "Surface format mismatch");

		// Create a swapchain
		const auto surfaceCaps = physDevice.getSurfaceCapabilitiesKHR(*mSurface);
		ASSERT(surfaceCaps.currentExtent.width == width, "Invalid swapchain width");
		ASSERT(surfaceCaps.currentExtent.height == height, "Invalid swapchain height");
		const auto presentMode = vk::PresentModeKHR::eFifo;
		const auto surfacePresentModes = physDevice.getSurfacePresentModesKHR(*mSurface);
		ASSERT(find(surfacePresentModes.begin(), surfacePresentModes.end(), presentMode)
				!= surfacePresentModes.end(), "Unsupported presentation mode");
		const auto swapChainCreateInfo = vk::SwapchainCreateInfoKHR(
			{}, *mSurface, BUFFER_COUNT, surfaceFormat, surfaceColorSpace,
			vk::Extent2D(width, height), 1,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eColorAttachment,
			vk::SharingMode::eExclusive, 1, &queueGfxIdx)
			.setPresentMode(presentMode);
		mSwapchain = mDevice->createSwapchainKHRUnique(swapChainCreateInfo);
		mBackBuffers = mDevice->getSwapchainImagesKHR(*mSwapchain);
		ASSERT(mBackBuffers.size() == BUFFER_COUNT, "Invalid back buffer count");

		for (int i = 0; i < BUFFER_COUNT; ++i)
		{
			const auto viewInfo = vk::ImageViewCreateInfo(
				{}, mBackBuffers[i], vk::ImageViewType::e2D, surfaceFormat)
				.setSubresourceRange(
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
			mBackBuffersView.push_back(mDevice->createImageViewUnique(viewInfo));
		}

		mSwapchainSema = mDevice->createSemaphoreUnique({});
		mDrawingSema = mDevice->createSemaphoreUnique({});

		// Create commands
		mCmdPool = mDevice->createCommandPoolUnique(
			vk::CommandPoolCreateInfo(
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				queueGfxIdx));
		mCmdBuf = mDevice->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo(*mCmdPool, vk::CommandBufferLevel::ePrimary, 2));
		for (int i = 0; i < _countof(mCmdFence); ++i)
		{
			mCmdFence[i] = mDevice->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
		}

		// Create descriptor pools
		const auto descPoolSizes = {
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 100),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 100),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 10),
		};
		for (auto& p : mDescPools)
		{
			p = mDevice->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(
				{}, //vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind, // descriptor indexing (option)
				10,
				descPoolSizes
			));
		}

		// Create a render pass
		const auto colorAttachmentDesc = vk::AttachmentDescription(
			{}, surfaceFormat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal
		);
		const auto depthAttachmentDesc = vk::AttachmentDescription(
			{}, vk::Format::eD32Sfloat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal
		);
		const auto attachmentsDesc = { colorAttachmentDesc , depthAttachmentDesc };
		const auto colorAttachmentRef = vk::AttachmentReference(
			0, vk::ImageLayout::eColorAttachmentOptimal
		);
		const auto depthAttachmentRef = vk::AttachmentReference(
			1, vk::ImageLayout::eDepthStencilAttachmentOptimal
		);
		const auto subpassDesc = vk::SubpassDescription(
			{}, vk::PipelineBindPoint::eGraphics, {}, colorAttachmentRef, {}, &depthAttachmentRef
		);
		mRenderPass = mDevice->createRenderPassUnique(
			vk::RenderPassCreateInfo({}, attachmentsDesc, subpassDesc)
		);

		// Create a descriptor set
		// Set 0
		{
			const auto descriptorBinding = vk::DescriptorSetLayoutBinding(
				0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex
			);
			const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo(
				{}, descriptorBinding
			);
			mDescriptorSetLayoutBuf = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfo);
		}
		// Set 1
		{
			const auto descriptorBinding = {
				vk::DescriptorSetLayoutBinding(
					0, vk::DescriptorType::eSampledImage, 50/*variable descriptor count*/, vk::ShaderStageFlagBits::eFragment
				),
			};
			// descriptor indexing
			const array<vk::DescriptorBindingFlags, 1> bindingFlags = {
				vk::DescriptorBindingFlagBits::ePartiallyBound | vk::DescriptorBindingFlagBits::eVariableDescriptorCount
			};
			const auto descriptorBindingFlagsInfo = vk::DescriptorSetLayoutBindingFlagsCreateInfo(bindingFlags);

			const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo(
				{}, //vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool, // descriptor indexing (option)
				descriptorBinding
			).setPNext(&descriptorBindingFlagsInfo);
			mDescriptorSetLayoutTex = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfo);
		}

		// Create a pipeline layout
		const auto pipelineDescSets = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
		const auto pipelinePushConsts = vk::PushConstantRange(vk::ShaderStageFlagBits::eFragment, 0, 4);
		mPipelineLayout = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, pipelineDescSets, pipelinePushConsts)
		);

		// Create modules
		static const char shaderCodeSceneVS[] = R"#(
[[vk::binding(0, 0)]] cbuffer CScene {
	float4x4 ViewProj;
};
struct Output {
	float4 position : SV_Position;
	float3 world : WorldPosition;
	float3 normal : Normal;
	float2 texcoord : Texcoord;
};
Output main(float3 position : Position, float3 normal : Normal, float2 texcoord : Texcoord) {
	Output output;
	output.position = mul(float4(position, 1), ViewProj);
	output.world = position;
	output.normal = normalize(normal);
	output.texcoord = texcoord;
	return output;
}
)#";

		static const char shaderCodeScenePS[] = R"#(
#define MAX_BINDLESS_TEXTURE_NUM (6)

struct PushConst { int32_t tex_index; };
[[vk::push_constant]] PushConst pushConst;
[[vk::binding(0, 1)]] Texture2D Tex[]; // bindless
struct Input {
	float4 position : SV_Position;
	float3 world : WorldPosition;
	float3 normal : Normal;
	float2 texcoord : Texcoord;
};
float4 main(Input input) : SV_Target {
	float4 color;
	if (pushConst.tex_index >= MAX_BINDLESS_TEXTURE_NUM) {
		uint idx = (uint)input.position.y % MAX_BINDLESS_TEXTURE_NUM;
		color = Tex[NonUniformResourceIndex(idx)].Load(int3(0, 0, 0));
	}
	else {
		color = Tex[pushConst.tex_index].Load(int3(0, 0, 0));
	}
	return color;
}
)#";

		SetDllDirectory(L"../dll/");

		ComPtr<IDxcCompiler> dxc;
		CHK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc)));
		ComPtr<IDxcLibrary> dxcLib;
		CHK(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&dxcLib)));

		ComPtr<IDxcBlobEncoding> dxcTxtSceneVS, dxcTxtScenePS;
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeSceneVS, _countof(shaderCodeSceneVS) - 1, CP_UTF8, &dxcTxtSceneVS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeScenePS, _countof(shaderCodeScenePS) - 1, CP_UTF8, &dxcTxtScenePS));

		ComPtr<IDxcBlob> dxcBlobShadowVS, dxcBlobSceneVS, dxcBlobScenePS;
		ComPtr<IDxcBlobEncoding> dxcError;
		ComPtr<IDxcOperationResult> dxcRes;
		const wchar_t* shaderArgsVS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv", L"-fvk-invert-y"
		};
		const wchar_t* shaderArgsPS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv", L"-fvk-use-dx-position-w"
		};

		dxc->Compile(dxcTxtSceneVS.Get(), nullptr, L"main", L"vs_6_0", shaderArgsVS, _countof(shaderArgsVS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobSceneVS);
		dxc->Compile(dxcTxtScenePS.Get(), nullptr, L"main", L"ps_6_0", shaderArgsPS, _countof(shaderArgsPS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobScenePS);

		const auto vsCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobSceneVS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobSceneVS->GetBufferPointer()));
		auto vsModule =  mDevice->createShaderModuleUnique(vsCreateInfo);
		const auto psCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobScenePS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobScenePS->GetBufferPointer()));
		auto fsModule = mDevice->createShaderModuleUnique(psCreateInfo);

		// Create a PSO
		const auto pipelineShadersInfo = {
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vsModule, "main"),
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fsModule, "main"),
		};
		const auto vertexInputBindingDesc
			= vk::VertexInputBindingDescription(0, 32, vk::VertexInputRate::eVertex);
		const auto vertexInputAttrsDesc = {
			vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, 0),
			vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, 12),
			vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, 24),
		};
		const auto pipelineVertexInputsInfo = vk::PipelineVertexInputStateCreateInfo(
			{}, vertexInputBindingDesc, vertexInputAttrsDesc
		);
		const auto pipelineInputAssemblyStateInfo = vk::PipelineInputAssemblyStateCreateInfo(
			{}, vk::PrimitiveTopology::eTriangleList
		);
		const auto viewportStateInfo = vk::PipelineViewportStateCreateInfo()
			.setViewportCount(1)
			.setScissorCount(1);
		const auto pipelineRSInfo = vk::PipelineRasterizationStateCreateInfo()
			.setCullMode(vk::CullModeFlagBits::eBack)
			.setFrontFace(vk::FrontFace::eClockwise)
			.setLineWidth(1.0f);
		const auto pipelineMSAAInfo = vk::PipelineMultisampleStateCreateInfo();
		const auto pipelineDSSInfo = vk::PipelineDepthStencilStateCreateInfo()
			.setDepthTestEnable(VK_TRUE).setDepthWriteEnable(VK_TRUE)
			.setDepthCompareOp(vk::CompareOp::eGreaterOrEqual);
		const auto blendAttachmentState = vk::PipelineColorBlendAttachmentState()
			.setColorWriteMask(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
				vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
		const auto pipelineBSInfo = vk::PipelineColorBlendStateCreateInfo()
			.setAttachments(blendAttachmentState);
		const auto dynamicStates = {
			vk::DynamicState::eViewport, vk::DynamicState::eScissor
		};
		const auto pipelineDynamicStatesInfo = vk::PipelineDynamicStateCreateInfo({}, dynamicStates);
		const auto pipelineInfo = vk::GraphicsPipelineCreateInfo(
			{}, pipelineShadersInfo, &pipelineVertexInputsInfo, &pipelineInputAssemblyStateInfo,
			nullptr, &viewportStateInfo, &pipelineRSInfo, &pipelineMSAAInfo, &pipelineDSSInfo,
			&pipelineBSInfo, &pipelineDynamicStatesInfo, *mPipelineLayout, *mRenderPass
		);
		auto vkres = mDevice->createGraphicsPipelineUnique(nullptr, pipelineInfo);
		CHK(vkres.result);
		mPSO = std::move(vkres.value);

		// Get memory props
		const auto memoryProps = physDevice.getMemoryProperties();
		auto GetMemTypeIndex = [&](const vk::MemoryRequirements& memReq, bool hostVisible) {
			const vk::MemoryPropertyFlags preferFlags = vk::MemoryPropertyFlagBits::eDeviceLocal;
			uint32_t fallbackType = ~0u;
			for (uint32_t i = 0; i < memoryProps.memoryTypeCount; ++i)
			{
				if (memReq.memoryTypeBits & (1u << i))
				{
					auto props = memoryProps.memoryTypes[i].propertyFlags;
					if (hostVisible && !(props & vk::MemoryPropertyFlagBits::eHostVisible))
						continue;
					if (props & preferFlags)
						return i;
					if (fallbackType == ~0u)
						fallbackType = i;
				}
			}
			ASSERT(fallbackType != ~0u, "Memory type mismatched");
			return fallbackType;
		};

		// Create a color buffer
		mSceneColor = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, surfaceFormat, vk::Extent3D(width, height, 1),
			1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc
		).setInitialLayout(vk::ImageLayout::eUndefined)
		);
		const auto colorMemReq = mDevice->getImageMemoryRequirements(*mSceneColor);
		mSceneColorMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			colorMemReq.size, GetMemTypeIndex(colorMemReq, false)
		));
		mDevice->bindImageMemory(*mSceneColor, *mSceneColorMemory, 0);
		mSceneColorView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mSceneColor, vk::ImageViewType::e2D, surfaceFormat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		));

		// Create a depth buffer
		mSceneDepth = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, vk::Format::eD32Sfloat, vk::Extent3D(width, height, 1),
			1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment
		).setInitialLayout(vk::ImageLayout::eUndefined)
		);
		const auto depthMemReq = mDevice->getImageMemoryRequirements(*mSceneDepth);
		mSceneDepthMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			depthMemReq.size, GetMemTypeIndex(depthMemReq, false)
		));
		mDevice->bindImageMemory(*mSceneDepth, *mSceneDepthMemory, 0);
		mSceneDepthView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mSceneDepth, vk::ImageViewType::e2D, vk::Format::eD32Sfloat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)
		));

		// Create a frame buffer
		const auto framebufferAttachments = { *mSceneColorView, *mSceneDepthView };
		mSceneFramebuffer = mDevice->createFramebufferUnique(vk::FramebufferCreateInfo(
			{}, *mRenderPass, framebufferAttachments, width, height, 1));
		mSceneExtent = vk::Extent2D(width, height);

		// Generate a sphere
		struct IndexList
		{
			short a[6];
		};
		vector<VertexElement> vertices;
		vector<IndexList> indices;
		vertices.reserve((SphereStacks + 1)* (SphereSlices + 1));
		for (int y = 0; y < SphereStacks + 1; ++y)
		{
			for (int x = 0; x < SphereSlices + 1; ++x)
			{
				// Generate a evenly tesselated plane
				float v[3] = { (float)x / (float)(SphereSlices), (float)y / (float)(SphereStacks), 0.0f };
				// Convert to spherical coordinate system
				float theta = 2 * 3.14159265f * v[0];
				float phi = 2 * 3.14159265f * v[1] / 2.0f;
				VertexElement ve = { sinf(phi) * sinf(theta), cosf(phi), sinf(phi) * cosf(theta) };
				// Setup normal
				float r = 1.0f;
				ve.normal[0] = ve.position[0] / r;
				ve.normal[1] = ve.position[1] / r;
				ve.normal[2] = ve.position[2] / r;
				// Setup uv
				ve.texcoord[0] = (float)x / SphereSlices;
				ve.texcoord[1] = (float)y / SphereStacks;
				vertices.push_back(ve);
			}
		}
		indices.reserve(SphereStacks * SphereSlices);
		for (int y = 0; y < SphereStacks; ++y)
		{
			for (int x = 0; x < SphereSlices; ++x)
			{
				short b = static_cast<short>(y * (SphereSlices + 1) + x);
				short s = SphereSlices + 1;
				IndexList il = { b, b + s, b + 1, b + s, b + s + 1, b + 1 };
				indices.push_back(il);
			}
		}

		// Create sphere buffers
		mSphereVB = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer
		));
		const auto sphereVBMemReq = mDevice->getBufferMemoryRequirements(*mSphereVB);
		mSphereIB = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer
		));
		const auto sphereIBMemReq = mDevice->getBufferMemoryRequirements(*mSphereIB);
		mSphereMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(sphereVBMemReq.size, sphereIBMemReq.alignment) + sphereIBMemReq.size,
			GetMemTypeIndex(sphereVBMemReq, true)
		));
		mDevice->bindBufferMemory(*mSphereVB, *mSphereMemory, 0);
		mDevice->bindBufferMemory(*mSphereIB, *mSphereMemory, (uint64_t)ALIGN(sphereVBMemReq.size, sphereIBMemReq.alignment));

		// Upload sphere data
		uint8_t *pData = reinterpret_cast<uint8_t*>(mDevice->mapMemory(*mSphereMemory, 0, VK_WHOLE_SIZE));
		memcpy(pData, vertices.data(), sizeof(vertices[0])* vertices.size());
		pData += ALIGN(sphereVBMemReq.size, sphereIBMemReq.alignment);
		memcpy(pData, indices.data(), sizeof(indices[0])* indices.size());
		mDevice->unmapMemory(*mSphereMemory);

		// Generate a plane
		vertices.clear();
		indices.clear();
		vertices.push_back({ -1, -1, +1,  0, +1,  0, 0, 0 });
		vertices.push_back({ +1, -1, +1,  0, +1,  0, 1, 0 });
		vertices.push_back({ -1, -1, -1,  0, +1,  0, 0, 1 });
		vertices.push_back({ +1, -1, -1,  0, +1,  0, 1, 1 });
		for (auto& v : vertices) { v.position[0] *= 3; v.position[1] *= 3; v.position[2] *= 3; }
		indices.push_back({ 0, 1, 2, 2, 1, 3 });

		// Create uniform buffers
		for (auto& ub : mUniformBuffers)
		{
			ub = mDevice->createBufferUnique(vk::BufferCreateInfo(
				{}, UniformBufferSize, vk::BufferUsageFlagBits::eUniformBuffer
			));
		}
		const auto uniformBufMemReq = mDevice->getBufferMemoryRequirements(*mUniformBuffers[0]);
		mUniformMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			_countof(mUniformBuffers) * ALIGN(uniformBufMemReq.size, uniformBufMemReq.alignment),
			GetMemTypeIndex(uniformBufMemReq, true)
		));
		for (int i = 0; i < _countof(mUniformBuffers); ++i)
		{
			auto ofs = i * ALIGN(uniformBufMemReq.size, uniformBufMemReq.alignment);
			mDevice->bindBufferMemory(*mUniformBuffers[i], *mUniformMemory, ofs);
			mUniformMemoryOffsets[i] = ofs;
		}

		// Prepare a transfer command and a buffer
		mDmaCmdPool = mDevice->createCommandPoolUnique(
			vk::CommandPoolCreateInfo(
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				queueDmaIdx));
		mDmaCmdBuf = move(mDevice->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(
			*mDmaCmdPool, vk::CommandBufferLevel::ePrimary, 1))[0]);
		mDmaSema = mDevice->createSemaphoreUnique({});
		mImageUploadBuffer = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, 8 * 1024 * 1024, vk::BufferUsageFlagBits::eTransferSrc
		));
		const auto imageUploadMemReq = mDevice->getBufferMemoryRequirements(*mImageUploadBuffer);
		mImageUploadMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			imageUploadMemReq.size, GetMemTypeIndex(imageUploadMemReq, true)
		));
		mDevice->bindBufferMemory(*mImageUploadBuffer, *mImageUploadMemory, 0);

		// Create images
		for (auto& img : mImgList)
		{
			img = mDevice->createImageUnique(vk::ImageCreateInfo(
				{}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, vk::Extent3D(1, 1, 1),
				1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
				vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
			));
		}
		const auto imgMemReq = mDevice->getImageMemoryRequirements(*mImgList[0]);
		mImagesMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(imgMemReq.size, imgMemReq.alignment) * _countof(mImgList),
			GetMemTypeIndex(imgMemReq, false)
		));
		for (auto& view : mImgListView)
		{
			const auto i = distance(begin(mImgListView), &view);
			mDevice->bindImageMemory(*mImgList[i], *mImagesMemory, (uint64_t)ALIGN(imgMemReq.size, imgMemReq.alignment) * i);
			view = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
				{}, *mImgList[i], vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
			));
		}

		// Upload image data
		pData = reinterpret_cast<uint8_t*>(mDevice->mapMemory(*mImageUploadMemory, 0, VK_WHOLE_SIZE));
		memcpy(pData, mImageDataList, sizeof(mImageDataList));
		mDevice->unmapMemory(*mImageUploadMemory);

		// Setup an image transfer command
		mDmaCmdBuf->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		array<vk::ImageMemoryBarrier, _countof(mImgList)> barriers;
		for (int i = 0; i < _countof(mImgList); ++i)
		{
			barriers[i] = vk::ImageMemoryBarrier(
				{}, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyDmaIdx, *mImgList[i],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
			);
		};
		mDmaCmdBuf->pipelineBarrier(
			vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTransfer, {},
			{}, {}, barriers);
		for (int i = 0; i < _countof(mImgList); ++i)
		{
			mDmaCmdBuf->copyBufferToImage(
				*mImageUploadBuffer, *mImgList[i], vk::ImageLayout::eTransferDstOptimal,
				vk::BufferImageCopy(
					sizeof(mImageDataList[0]) * i, 0, 0,
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1), {}, vk::Extent3D(1, 1, 1)));
		}
		// Release exclusive ownership if (mQueueFamilyDmaIdx != mQueueFamilyGfxIdx)
		for (int i = 0; i < _countof(mImgList); ++i)
		{
			barriers[i] = vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite, {},
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mImgList[i],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
			);
		};
		mDmaCmdBuf->pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {},
			{}, {}, barriers);
		mDmaCmdBuf->end();

		// Execute a transfer command
		const auto submitInfo = vk::SubmitInfo({}, {}, *mDmaCmdBuf, *mDmaSema);
		mQueueDma.submit(submitInfo, VK_NULL_HANDLE);

		// Wait a grapihcs queue
		const vk::PipelineStageFlags submitPipelineStage = vk::PipelineStageFlagBits::eBottomOfPipe;
		const auto submitInfoGfx = vk::SubmitInfo(*mDmaSema, submitPipelineStage, {}, {});
		mQueue.submit(submitInfoGfx, VK_NULL_HANDLE);
	}

	void Draw()
	{
		const auto backBufferIdx = mDevice->acquireNextImageKHR(*mSwapchain, 100000000000, *mSwapchainSema);
		CHK(backBufferIdx.result);
		mBackBufferIdx = backBufferIdx.value;

		const auto fence = *mCmdFence[mFrameCount % 2];
		auto r = mDevice->waitForFences(fence, VK_TRUE, 100000000000);
		CHK(r);
		mDevice->resetFences(fence);

		mFrameCount++;
		
		// Make uniform buffer

		auto fov = DirectX::XMConvertToRadians(45.0f);
		auto aspect = 1.0f * mSceneExtent.width / mSceneExtent.height;
		auto nearClip = 0.01f;
		auto farClip = 100.0f;

		auto worldMat = DirectX::XMMatrixIdentity();
		auto viewMat = DirectX::XMMatrixLookAtLH(mCameraPos, mCameraTarget, mCameraUp);
		auto projMat = DirectX::XMMatrixPerspectiveFovLH(fov, aspect, farClip, nearClip); // Reversed depth

		auto wvpMat = DirectX::XMMatrixTranspose(worldMat * viewMat * projMat);

		void* pUB = mDevice->mapMemory(*mUniformMemory, mUniformMemoryOffsets[mFrameCount % 2], UniformBufferSize);
		*reinterpret_cast<decltype(&wvpMat)>(pUB) = wvpMat;
		mDevice->unmapMemory(*mUniformMemory);

		// Start drawing

		auto cmdBuf = *mCmdBuf[mFrameCount % 2];
		cmdBuf.reset();
		cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		// Acquire exclusive ownership
		if (mFrameCount == 1 && mQueueFamilyDmaIdx != mQueueFamilyGfxIdx)
		{
			array<vk::ImageMemoryBarrier, _countof(mImgList)> barriers;
			for (int i = 0; i < _countof(mImgList); ++i)
			{
				barriers[i] = vk::ImageMemoryBarrier(
					{}/*ignored*/, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mImgList[i],
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
				);
			};
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {},
				{}, {}, barriers);
		}

		const std::array<vk::ClearValue, 2> sceneClearValue = {
			vk::ClearColorValue(std::array<float, 4>({0.1f,0.2f,0.4f,1.0f})),
			vk::ClearDepthStencilValue(0.0f)
		};
		const auto renderPassInfo = vk::RenderPassBeginInfo(
			*mRenderPass, *mSceneFramebuffer, vk::Rect2D({}, mSceneExtent), sceneClearValue
		);

		// The initial layout of the render pass is "Undefined", so any layout can be accepted
		cmdBuf.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

		// Draw a sphere
		if (!(mSphereDescSetBuf[mFrameCount % 2]))
		{
			const auto descSetLayouts = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
			// variable descriptor count
			const array<uint32_t, 2> descCounts = { 1, _countof(mImgList) };
			const auto variableDescriptorCountInfo = vk::DescriptorSetVariableDescriptorCountAllocateInfo(descCounts);
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], descSetLayouts
			).setPNext(&variableDescriptorCountInfo)
			);
			mSphereDescSetBuf[mFrameCount % 2] = descSets[0];
			mSphereDescSetTex[mFrameCount % 2] = descSets[1];
		}
		vk::WriteDescriptorSet wdescSets[10];
		auto descBufInfo = vk::DescriptorBufferInfo(*mUniformBuffers[mFrameCount % 2], 0, 2048);
		wdescSets[0] = vk::WriteDescriptorSet(
			mSphereDescSetBuf[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eUniformBuffer
			).setBufferInfo(descBufInfo);
		// Debug layer says we need to set all variable descriptors in one VkWriteDescriptorSet that are belonged to the descriptor set
		array<vk::DescriptorImageInfo, _countof(mImgList)> descImageInfos;
		for (int i = 0; i < _countof(mImgList); ++i)
		{
			descImageInfos[i] = vk::DescriptorImageInfo({}, *mImgListView[i], vk::ImageLayout::eShaderReadOnlyOptimal);
		}
		wdescSets[1] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descImageInfos);
		mDevice->updateDescriptorSets(2, wdescSets, 0, nullptr);
		cmdBuf.setViewport(0, vk::Viewport(0, 0, (float)mSceneExtent.width, (float)mSceneExtent.height, 0, 1));
		vk::Rect2D scissor({}, mSceneExtent);
		cmdBuf.setScissor(0, scissor);
		cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *mPSO);
		cmdBuf.bindVertexBuffers(0, *mSphereVB, { 0 });
		cmdBuf.bindIndexBuffer(*mSphereIB, 0, vk::IndexType::eUint16);
		const auto sphereDescSets = {
			mSphereDescSetBuf[mFrameCount % 2], mSphereDescSetTex[mFrameCount % 2]
		};
		cmdBuf.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, *mPipelineLayout, 0, sphereDescSets, {}
		);
		cmdBuf.pushConstants(*mPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, 4, &mTexIndex);
		cmdBuf.drawIndexed(6 * SphereStacks * SphereSlices, 1, 0, 0, 0);

		cmdBuf.endRenderPass();

		// Blit to back buffer
		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
			vk::ImageMemoryBarrier(
				{ /*no access*/ }, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				mBackBuffers[mBackBufferIdx],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		const auto imageColorSubres0 = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
		cmdBuf.copyImage(*mSceneColor, vk::ImageLayout::eTransferSrcOptimal,
			mBackBuffers[mBackBufferIdx], vk::ImageLayout::eTransferDstOptimal,
			vk::ImageCopy(imageColorSubres0, {}, imageColorSubres0, {}, vk::Extent3D(mSceneExtent, 1))
		);

		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite,
				vk::AccessFlagBits::eTransferRead,
				vk::ImageLayout::eTransferDstOptimal,
				vk::ImageLayout::ePresentSrcKHR,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				mBackBuffers[mBackBufferIdx],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		cmdBuf.end();

		// Execute a command

		const vk::PipelineStageFlags submitPipelineStage = vk::PipelineStageFlagBits::eBottomOfPipe;
		const auto submitInfo = vk::SubmitInfo(*mSwapchainSema, submitPipelineStage, cmdBuf, *mDrawingSema);
		mQueue.submit(submitInfo, fence);
	}

	void Present()
	{
		const auto presentInfo = vk::PresentInfoKHR(*mDrawingSema, *mSwapchain, mBackBufferIdx);
		const auto r = mQueuePresent.presentKHR(presentInfo);
	}

	void IncTexIndex()
	{
		mTexIndex++;
	}
	void DecTexIndex()
	{
		mTexIndex--;
		//ASSERT(mTexIndex >= 0, "Oops, descriptor index is out of bounds. This may cause GPU page fault.");
	}

private:
	DirectX::XMVECTOR mCameraPos = DirectX::XMVectorSet(0.0f, 4.0f, -4.0f, 0);
	DirectX::XMVECTOR mCameraTarget = DirectX::XMVectorSet(0, 0, 0, 0);
	DirectX::XMVECTOR mCameraUp = DirectX::XMVectorSet(0, 1, 0, 0);

	int32_t mTexIndex = 0;
};

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message) {
	case WM_KEYDOWN:
		if (wParam == VK_ESCAPE) {
			PostMessage(hWnd, WM_DESTROY, 0, 0);
			return 0;
		}
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hWnd, message, wParam, lParam);
	}
	return 0;
}

static HWND setupWindow(int width, int height)
{
	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WndProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = (HINSTANCE)GetModuleHandle(0);
	wcex.hIcon = nullptr;
	wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
	wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
	wcex.lpszMenuName = nullptr;
	wcex.lpszClassName = L"WindowClass";
	wcex.hIconSm = nullptr;
	if (!RegisterClassEx(&wcex)) {
		throw runtime_error("RegisterClassEx()");
	}

	RECT rect = { 0, 0, width, height };
	AdjustWindowRect(&rect, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, FALSE);
	const int windowWidth = (rect.right - rect.left);
	const int windowHeight = (rect.bottom - rect.top);

	HWND hWnd = CreateWindow(L"WindowClass", L"Window",
		WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU, CW_USEDEFAULT, 0, windowWidth, windowHeight,
		nullptr, nullptr, nullptr, nullptr);
	if (!hWnd) {
		throw runtime_error("CreateWindow()");
	}

	return hWnd;
}

int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	MSG msg;
	ZeroMemory(&msg, sizeof msg);

	HWND hwnd = 0;
#ifdef NDEBUG
	try
#endif
	{
		hwnd = setupWindow(WINDOW_WIDTH, WINDOW_HEIGHT);
		ShowWindow(hwnd, SW_SHOW);
		UpdateWindow(hwnd);

		VLK vlk(WINDOW_WIDTH, WINDOW_HEIGHT, hwnd);

		while (msg.message != WM_QUIT) {
			BOOL r = PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE);
			if (r == 0) {
				vlk.Draw();
				vlk.Present();
			}
			else {
				DispatchMessage(&msg);
				if (msg.message == WM_KEYDOWN)
				{
					if (msg.wParam == VK_RIGHT) vlk.IncTexIndex();
					else if (msg.wParam == VK_LEFT) vlk.DecTexIndex();
				}
			}
		}
	}
#ifdef NDEBUG
	catch (std::exception& e) {
		MessageBoxA(hwnd, e.what(), "Exception occuured.", MB_ICONSTOP);
	}
#endif

	return static_cast<int>(msg.wParam);
}

