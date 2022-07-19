// based on SimpleTexture.cpp

// Inspired fron Rainbow Six Siege
// Standard MSAA x2 sample patterin is (0.75, 0.75), (0.25, 0.25)
// so I add 1px border on checkerboard scene, and offset (0.0, 0.0), (0.5, 0.0) each frame
// First time I have implemented using VK_EXT_sample_locations, but did not work properly on Radeon and Intel
// and I found it's too cumbersome to modify varying PS parameters such as texcoord :(

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

	// Checkerboard resolve resources
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutResolveDummy;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutResolveTex;
	vk::UniquePipelineLayout mPipelineLayoutResolve;
	vk::UniquePipeline mPSOResolve;
	vk::Extent2D mResolveExtent;
	vk::UniqueImage mResolveColor[2]; // ping pong buffering
	vk::UniqueDeviceMemory mResolveColorMemory;
	vk::UniqueImageView mResolveColorView[2];

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
	vk::UniqueBuffer mPlaneVB;
	vk::UniqueBuffer mPlaneIB;
	vk::UniqueDeviceMemory mPlaneMemory;

	static constexpr int UniformBufferSize = 1 * 1000 * 1000;
	vk::UniqueBuffer mUniformBuffers[2];
	vk::UniqueDeviceMemory mUniformMemory;
	vk::DeviceSize mUniformMemoryOffsets[2];

	vk::UniqueCommandPool mDmaCmdPool;
	vk::UniqueCommandBuffer mDmaCmdBuf;
	vk::UniqueSemaphore mDmaSema;
	vk::UniqueBuffer mImageUploadBuffer;
	vk::UniqueDeviceMemory mImageUploadMemory;

	vk::UniqueImage mSailboatImg;
	vk::UniqueImageView mSailboatView;
	uint32_t mSailboatMipLevels;
	vk::UniqueImage mLennaImg;
	vk::UniqueImageView mLennaView;
	uint32_t mLennaMipLevels;
	vk::UniqueDeviceMemory mImagesMemory;

	vk::UniqueSampler mSampler;

	vk::DescriptorSet mSphereDescSetBuf[2];
	vk::DescriptorSet mSphereDescSetTex[2];
	vk::DescriptorSet mPlaneDescSetBuf[2];
	vk::DescriptorSet mPlaneDescSetTex[2];
	vk::DescriptorSet mResolveDescSetDummy[2][2];
	vk::DescriptorSet mResolveDescSetTex[2][2];

public:
	~VLK()
	{
		mQueue.waitIdle();
		mQueuePresent.waitIdle();
	}

	VLK(int width, int height, HWND hwnd)
	{
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
			"LearningVulkan", VK_MAKE_API_VERSION(0, 0, 0, 0), VK_API_VERSION_1_0);
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
		const auto enableFeatures = vk::PhysicalDeviceFeatures()
			.setSamplerAnisotropy(1)
			.setSampleRateShading(1);
		const auto deviceCreateInfo = vk::DeviceCreateInfo(
			{}, (uint32_t)deviceQueueInfos.size(), deviceQueueInfos.data(),
			0, nullptr, _countof(deviceExtensions), deviceExtensions, &enableFeatures
		);
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
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 100),
		};
		for (auto& p : mDescPools)
		{
			p = mDevice->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo({}, 10, descPoolSizes));
		}

		// Create a render pass
		const auto colorAttachmentDesc = vk::AttachmentDescription(
			{}, surfaceFormat, vk::SampleCountFlagBits::e2,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal
		);
		const auto depthAttachmentDesc = vk::AttachmentDescription(
			{}, vk::Format::eD32Sfloat, vk::SampleCountFlagBits::e2,
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

		// Create a scene descriptor set
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
					0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					1, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment
				),
			};
			const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo(
				{}, descriptorBinding
			);
			mDescriptorSetLayoutTex = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfo);
		}

		// Create a resolve descriptor set
		// Set 0
		{
			mDescriptorSetLayoutResolveDummy = mDevice->createDescriptorSetLayoutUnique({});
		}
		// Set 1
		{
			const auto descriptorBinding = {
				vk::DescriptorSetLayoutBinding(
					0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute
				),
				vk::DescriptorSetLayoutBinding(
					1, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute
				),
				vk::DescriptorSetLayoutBinding(
					2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute
				),
			};
			const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo(
				{}, descriptorBinding
			);
			mDescriptorSetLayoutResolveTex = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfo);
		}

		// Create a scene pipeline layout
		const auto pipelineDescSets = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
		mPipelineLayout = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, pipelineDescSets)
		);

		// Create a resolve pipeline layout
		const array<vk::DescriptorSetLayout, 2> resolvePipelineDescSets = { *mDescriptorSetLayoutResolveDummy, *mDescriptorSetLayoutResolveTex };
		const auto pipelinePushConsts = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, 4);
		mPipelineLayoutResolve = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, resolvePipelineDescSets, pipelinePushConsts)
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
[[vk::binding(0, 1)]] Texture2D Tex;
[[vk::binding(1, 1)]] SamplerState SS;
struct Input {
	float4 position : SV_Position;
	float3 world : WorldPosition;
	float3 normal : Normal;
	float2 texcoord : Texcoord;
};
float4 main(Input input) : SV_Target {
	// We needs to modify ddx/ddy in order to match derivatives in full resolution
	float4 color = Tex.SampleBias(SS, input.texcoord, -1.0);
	return color;
}
)#";

		static const char shaderCodeResolveCS[] = R"#(
struct PushConst { uint OddFrameFlag; };
[[vk::push_constant]] PushConst pushConst;
[[vk::binding(0, 1)]] Texture2DMS<float4> Input;
[[vk::binding(1, 1)]] Texture2D Previous;
[[vk::binding(2, 1)]] RWTexture2D<float4> Output;

[numthreads(8, 8, 1)]
void main(uint2 dtid : SV_DispatchThreadID) {
	uint2 intpos = dtid;
	uint2 halfpos = dtid / 2;
	float4 color;
	if (pushConst.OddFrameFlag) {
		if ((intpos.x ^ intpos.y) & 1) {
			color = Previous.Load(int3(intpos, 0));
		}
		else {
			color = (intpos.y & 1) ? Input.Load(halfpos, 0) : Input.Load(halfpos, 1);
		}
	}
	else {
		if ((intpos.x ^ intpos.y) & 1) {
			color = (intpos.y & 1) ? Input.Load(halfpos, 0) : Input.Load(halfpos + uint2(1, 0), 1);
		}
		else {
			color = Previous.Load(int3(intpos, 0));
		}
	}
	Output[intpos] = color;
}
)#";

		SetDllDirectory(L"../dll/");

		ComPtr<IDxcCompiler> dxc;
		CHK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc)));
		ComPtr<IDxcLibrary> dxcLib;
		CHK(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&dxcLib)));

		ComPtr<IDxcBlobEncoding> dxcTxtSceneVS, dxcTxtScenePS, dxcTxtResolveCS;
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeSceneVS, _countof(shaderCodeSceneVS) - 1, CP_UTF8, &dxcTxtSceneVS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeScenePS, _countof(shaderCodeScenePS) - 1, CP_UTF8, &dxcTxtScenePS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeResolveCS, _countof(shaderCodeResolveCS) - 1, CP_UTF8, &dxcTxtResolveCS));

		ComPtr<IDxcBlob> dxcBlobSceneVS, dxcBlobScenePS, dxcBlobResolveCS;
		ComPtr<IDxcBlobEncoding> dxcError;
		ComPtr<IDxcOperationResult> dxcRes;
		const wchar_t* shaderArgsVS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv", L"-fvk-invert-y"
		};
		const wchar_t* shaderArgsPS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv", L"-fvk-use-dx-position-w"
		};
		const wchar_t* shaderArgsCS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv"
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
		dxc->Compile(dxcTxtResolveCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobResolveCS);

		const auto vsCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobSceneVS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobSceneVS->GetBufferPointer()));
		auto vsModule =  mDevice->createShaderModuleUnique(vsCreateInfo);
		const auto psCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobScenePS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobScenePS->GetBufferPointer()));
		auto fsModule = mDevice->createShaderModuleUnique(psCreateInfo);
		const auto csCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobResolveCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobResolveCS->GetBufferPointer()));
		auto csModule = mDevice->createShaderModuleUnique(csCreateInfo);

		// Create a scene PSO
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
		const auto pipelineMSAAInfo = vk::PipelineMultisampleStateCreateInfo()
			.setRasterizationSamples(vk::SampleCountFlagBits::e2)
			.setSampleShadingEnable(VK_TRUE);
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

		// Create a resolve PSO
		const auto resolvePipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csModule, "main");
		const auto resolvePipelineInfo = vk::ComputePipelineCreateInfo({}, resolvePipelineShadersInfo, *mPipelineLayoutResolve);
		vkres = mDevice->createComputePipelineUnique(nullptr, resolvePipelineInfo);
		CHK(vkres.result);
		mPSOResolve = std::move(vkres.value);

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
			{}, vk::ImageType::e2D, surfaceFormat, vk::Extent3D(width / 2 + 1, height / 2, 1),
			1, 1, vk::SampleCountFlagBits::e2, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled
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
			{}, vk::ImageType::e2D, vk::Format::eD32Sfloat, vk::Extent3D(width / 2 + 1, height / 2, 1),
			1, 1, vk::SampleCountFlagBits::e2, vk::ImageTiling::eOptimal,
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

		// Create resolve buffers
		for (auto& img : mResolveColor)
		{
			img = mDevice->createImageUnique(vk::ImageCreateInfo(
				{}, vk::ImageType::e2D, surfaceFormat, vk::Extent3D(width, height, 1),
				1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
				vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
				| vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst
			).setInitialLayout(vk::ImageLayout::eUndefined)
			);
		}
		const auto resolveColorMemReq = mDevice->getImageMemoryRequirements(*mResolveColor[0]);
		mResolveColorMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(resolveColorMemReq.size, resolveColorMemReq.alignment) + resolveColorMemReq.size,
			GetMemTypeIndex(resolveColorMemReq, false)
		));
		for (int i = 0; i < _countof(mResolveColorView); ++i)
		{
			mDevice->bindImageMemory(
				*mResolveColor[i], *mResolveColorMemory,
				i * ALIGN(resolveColorMemReq.size, resolveColorMemReq.alignment));
			mResolveColorView[i] = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
				{}, *mResolveColor[i], vk::ImageViewType::e2D, surfaceFormat, {},
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
			));
		}

		// Create a frame buffer
		const auto framebufferAttachments = { *mSceneColorView, *mSceneDepthView };
		mSceneFramebuffer = mDevice->createFramebufferUnique(vk::FramebufferCreateInfo(
			{}, *mRenderPass, framebufferAttachments, width / 2 + 1, height / 2, 1));
		mSceneExtent = vk::Extent2D(width / 2 + 1, height / 2);

		mResolveExtent = vk::Extent2D(width, height);

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

		// Create plane buffers
		mPlaneVB = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer
		));
		const auto planeVBMemReq = mDevice->getBufferMemoryRequirements(*mPlaneVB);
		mPlaneIB = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer
		));
		const auto planeIBMemReq = mDevice->getBufferMemoryRequirements(*mPlaneIB);
		mPlaneMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(planeVBMemReq.size, planeIBMemReq.alignment) + planeIBMemReq.size,
			GetMemTypeIndex(planeVBMemReq, true)
		));
		mDevice->bindBufferMemory(*mPlaneVB, *mPlaneMemory, 0);
		mDevice->bindBufferMemory(*mPlaneIB, *mPlaneMemory, (uint64_t)ALIGN(planeVBMemReq.size, planeIBMemReq.alignment));

		// Upload plane data
		pData = reinterpret_cast<uint8_t*>(mDevice->mapMemory(*mPlaneMemory, 0, VK_WHOLE_SIZE));
		memcpy(pData, vertices.data(), sizeof(vertices[0])* vertices.size());
		pData += ALIGN(planeVBMemReq.size, planeIBMemReq.alignment);
		memcpy(pData, indices.data(), sizeof(indices[0])* indices.size());
		mDevice->unmapMemory(*mPlaneMemory);

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

		// Create a sampler
		mSampler = mDevice->createSamplerUnique(vk::SamplerCreateInfo(
			{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
			0.0f, VK_TRUE, 4.0f, VK_FALSE, {}, 0.0f, VK_LOD_CLAMP_NONE
		));

		// Load images and generate mipmaps
		struct ImageData
		{
			vk::Extent3D extent;
			uint32_t size;
			unique_ptr<uint8_t[]> data;
		};
		auto loadBitmap = [&](const char* path)
		{
			ifstream ifs(path, ios::binary);
			BITMAPFILEHEADER fileHeader = {};
			ifs.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
			ASSERT(fileHeader.bfType == 0x4D42, "Invalid BMP format");
			BITMAPINFOHEADER header;
			ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
			auto oneline = make_unique<uint8_t[]>(header.biWidth * 3); // rgb
			auto data = make_unique<uint8_t[]>(header.biWidth * header.biHeight * 4); // rgba
			for (int y = 0; y < header.biHeight; y++)
			{
				ifs.read(reinterpret_cast<char*>(oneline.get()), header.biWidth * 3);
				for (int x = 0; x < header.biWidth; ++x)
				{
					const int p = (header.biHeight - 1 - y) * header.biWidth + x;
					data[p * 4] = oneline[x * 3 + 2];
					data[p * 4 + 1] = oneline[x * 3 + 1];
					data[p * 4 + 2] = oneline[x * 3];
					data[p * 4 + 3] = '\xFF';
				}
			}
			return ImageData{
				vk::Extent3D(header.biWidth, header.biHeight, 1),
				uint32_t(4 * header.biWidth * header.biHeight),
				move(data) };
		};
		auto generateMipmap = [](ImageData&& mip0)
		{
			auto downsample = [](const ImageData& high)
			{
				const auto ext = vk::Extent3D(max(1u, high.extent.width / 2), max(1u, high.extent.height / 2), 1);
				auto data = make_unique<uint8_t[]>(ext.width * ext.height * 4);
				for (uint32_t y = 0; y < ext.height; ++y)
				{
					for (uint32_t x = 0; x < ext.width; ++x)
					{
						const auto pd = y * ext.width + x;
						const auto ps = 2 * y * 2 * ext.width + 2 * x;
						for (int c = 0; c < 4; ++c)
						{

							uint32_t d = high.data[ps * 4 + c];
							d += high.data[ps * 4 + 1 * 4 + c];
							d += high.data[ps * 4 + high.extent.width * 4 + c];
							d += high.data[ps * 4 + high.extent.width * 4 + 1 * 4 + c];
							data[pd * 4 + c] = static_cast<uint8_t>((d + 2) / 4);
						}
					}
				}
				return ImageData{ ext, uint32_t(ext.width * ext.height * 4), move(data) };
			};
			vector<ImageData> v;
			v.push_back(ImageData{ mip0.extent, mip0.size, move(mip0.data) });
			while (v.rbegin()->extent.width != 1 || v.rbegin()->extent.height != 1)
			{
				v.push_back(downsample(*v.rbegin()));
			}
			return v;
		};
		auto sailboatData = generateMipmap(loadBitmap("../res/Sailboat.bmp"));
		auto lennaData = generateMipmap(loadBitmap("../res/Lenna.bmp"));
		mSailboatMipLevels = (uint32_t)sailboatData.size();
		mLennaMipLevels = (uint32_t)lennaData.size();

		// Create images
		mSailboatImg = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, sailboatData[0].extent,
			mSailboatMipLevels, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
		));
		const auto sailboatMemReq = mDevice->getImageMemoryRequirements(*mSailboatImg);
		mLennaImg = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, vk::Format::eR8G8B8A8Unorm, lennaData[0].extent,
			mLennaMipLevels, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
			vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
		));
		const auto lennaMemReq = mDevice->getImageMemoryRequirements(*mLennaImg);
		mImagesMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(sailboatMemReq.size, lennaMemReq.alignment) + lennaMemReq.size,
			GetMemTypeIndex(sailboatMemReq, false)
		));
		mDevice->bindImageMemory(*mSailboatImg, *mImagesMemory, 0);
		mDevice->bindImageMemory(*mLennaImg, *mImagesMemory, (uint64_t)ALIGN(sailboatMemReq.size, lennaMemReq.alignment));
		mSailboatView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mSailboatImg, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mSailboatMipLevels, 0, 1)
		));
		mLennaView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mLennaImg, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mLennaMipLevels, 0, 1)
		));

		// Upload image data
		pData = reinterpret_cast<uint8_t*>(mDevice->mapMemory(*mImageUploadMemory, 0, VK_WHOLE_SIZE));
		for (int i = 0; i < sailboatData.size(); ++i)
		{
			memcpy(pData, sailboatData[i].data.get(), sailboatData[i].size);
			pData += sailboatData[i].size;
		}
		for (int i = 0; i < lennaData.size(); ++i)
		{
			memcpy(pData, lennaData[i].data.get(), lennaData[i].size);
			pData += lennaData[i].size;
		}
		mDevice->unmapMemory(*mImageUploadMemory);

		// Setup an image transfer command
		mDmaCmdBuf->begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
		const auto barriers = {
			vk::ImageMemoryBarrier(
				{}, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyDmaIdx, *mSailboatImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mSailboatMipLevels, 0, 1)
			),
			vk::ImageMemoryBarrier(
				{}, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyDmaIdx, *mLennaImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mLennaMipLevels, 0, 1)
			),
		};
		mDmaCmdBuf->pipelineBarrier(
			vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTransfer, {},
			{}, {}, barriers);
		size_t bufferOffset = 0;
		for (int i = 0; i < sailboatData.size(); ++i)
		{
			mDmaCmdBuf->copyBufferToImage(
				*mImageUploadBuffer, *mSailboatImg, vk::ImageLayout::eTransferDstOptimal,
				vk::BufferImageCopy(
					bufferOffset, 0, 0,
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1), {}, sailboatData[i].extent));
			bufferOffset += sailboatData[i].size;
		}
		for (int i = 0; i < lennaData.size(); ++i)
		{
			mDmaCmdBuf->copyBufferToImage(
				*mImageUploadBuffer, *mLennaImg, vk::ImageLayout::eTransferDstOptimal,
				vk::BufferImageCopy(
					bufferOffset, 0, 0,
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, i, 0, 1), {}, lennaData[i].extent));
			bufferOffset += lennaData[i].size;
		}
		// Release exclusive ownership if (mQueueFamilyDmaIdx != mQueueFamilyGfxIdx)
		const auto barriers2 = {
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite, {},
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mSailboatImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mSailboatMipLevels, 0, 1)
			),
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite, {},
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mLennaImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mLennaMipLevels, 0, 1)
			),
		};
		mDmaCmdBuf->pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe, {},
			{}, {}, barriers2);
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
		auto aspect = 1.0f * mResolveExtent.width / mResolveExtent.height;
		auto nearClip = 0.01f;
		auto farClip = 100.0f;

		auto worldMat = DirectX::XMMatrixIdentity();
		auto viewMat = DirectX::XMMatrixLookAtLH(mCameraPos, mCameraTarget, mCameraUp);
		auto projMat = DirectX::XMMatrixPerspectiveFovLH(fov, aspect, farClip, nearClip); // Reversed depth

		// Offset clip space position to make checkerboard pattern
		if (mFrameCount & 1)
		{
			auto halfPx = 0.5f / mSceneExtent.width;
			auto ofs = DirectX::XMVectorSet(-2.0f * halfPx, 0, 0, 0);
			projMat.r[2] = DirectX::XMVectorAdd(projMat.r[2], ofs);
		}

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
			const auto barriers = {
				vk::ImageMemoryBarrier(
					{}/*ignored*/, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mSailboatImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mSailboatMipLevels, 0, 1)
				),
				vk::ImageMemoryBarrier(
					{}/*ignored*/, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mLennaImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mLennaMipLevels, 0, 1)
				),
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

		cmdBuf.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

		// Draw a sphere
		if (!(mSphereDescSetBuf[mFrameCount % 2]))
		{
			const auto descSetLayouts = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], descSetLayouts
			));
			mSphereDescSetBuf[mFrameCount % 2] = descSets[0];
			mSphereDescSetTex[mFrameCount % 2] = descSets[1];
		}
		vk::WriteDescriptorSet wdescSets[10];
		auto descBufInfo = vk::DescriptorBufferInfo(*mUniformBuffers[mFrameCount % 2], 0, 2048);
		auto descTexInfo = vk::DescriptorImageInfo({}, *mSailboatView, vk::ImageLayout::eShaderReadOnlyOptimal);
		auto descSamplerInfo = vk::DescriptorImageInfo(*mSampler, {}, {});
		wdescSets[0] = vk::WriteDescriptorSet(
			mSphereDescSetBuf[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eUniformBuffer
			).setBufferInfo(descBufInfo);
		wdescSets[1] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descTexInfo);
		wdescSets[2] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 1, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descSamplerInfo);
		mDevice->updateDescriptorSets(3, wdescSets, 0, nullptr);
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
		cmdBuf.drawIndexed(6 * SphereStacks * SphereSlices, 1, 0, 0, 0);

		// Draw a plane
		if (!(mPlaneDescSetBuf[mFrameCount % 2]))
		{
			const auto descSetLayouts = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], descSetLayouts
			));
			mPlaneDescSetBuf[mFrameCount % 2] = descSets[0];
			mPlaneDescSetTex[mFrameCount % 2] = descSets[1];
		}
		descBufInfo = vk::DescriptorBufferInfo(*mUniformBuffers[mFrameCount % 2], 0, 2048);
		descTexInfo = vk::DescriptorImageInfo({}, *mLennaView, vk::ImageLayout::eShaderReadOnlyOptimal);
		descSamplerInfo = vk::DescriptorImageInfo(*mSampler, {}, {});
		wdescSets[0] = vk::WriteDescriptorSet(
			mPlaneDescSetBuf[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eUniformBuffer
		).setBufferInfo(descBufInfo);
		wdescSets[1] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descTexInfo);
		wdescSets[2] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 1, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descSamplerInfo);
		mDevice->updateDescriptorSets(3, wdescSets, 0, nullptr);
		cmdBuf.bindVertexBuffers(0, *mPlaneVB, { 0 });
		cmdBuf.bindIndexBuffer(*mPlaneIB, 0, vk::IndexType::eUint16);
		const auto planeDescSets = {
			mPlaneDescSetBuf[mFrameCount % 2], mPlaneDescSetTex[mFrameCount % 2]
		};
		cmdBuf.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, *mPipelineLayout, 0, planeDescSets, {}
		);
		cmdBuf.drawIndexed(6, 1, 0, 0, 0);

		cmdBuf.endRenderPass();

		// Clear the reprojection image for the first time
		if (mFrameCount == 1 || mFrameCount == 2)
		{
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
				vk::ImageMemoryBarrier(
					{/*no access*/}, vk::AccessFlagBits::eTransferWrite,
					vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mResolveColor[(mFrameCount + 1) & 1],
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
			cmdBuf.clearColorImage(
				*mResolveColor[(mFrameCount + 1) & 1],
				vk::ImageLayout::eTransferDstOptimal,
				vk::ClearColorValue(),
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
			);
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
				vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mResolveColor[(mFrameCount + 1) & 1],
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
		}
		// ... or change layout to shader readable
		else {
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
				vk::ImageMemoryBarrier(
					{/*no access*/}, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mResolveColor[(mFrameCount + 1) & 1],
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
		}
		// Change a resolve image layout to shader writable
		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eBottomOfPipe, vk::PipelineStageFlagBits::eComputeShader, {}, {}, {},
			vk::ImageMemoryBarrier(
				{/*no access*/}, vk::AccessFlagBits::eShaderWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				*mResolveColor[mFrameCount & 1],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		// Resolve checkerboard
		if (!(mResolveDescSetDummy[mFrameCount % 2][mFrameCount & 1]))
		{
			const auto descSetLayouts = { *mDescriptorSetLayoutResolveDummy, *mDescriptorSetLayoutResolveTex };
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], descSetLayouts
			));
			mResolveDescSetDummy[mFrameCount % 2][mFrameCount & 1] = descSets[0];
			mResolveDescSetTex[mFrameCount % 2][mFrameCount & 1] = descSets[1];
		}
		const auto resolveInputTexInfo = vk::DescriptorImageInfo({}, *mSceneColorView, vk::ImageLayout::eShaderReadOnlyOptimal);
		const auto resolvePrevTexInfo = vk::DescriptorImageInfo(
			{}, *mResolveColorView[(mFrameCount + 1) & 1], vk::ImageLayout::eShaderReadOnlyOptimal
		);
		const auto resolveOutputTexInfo = vk::DescriptorImageInfo(
			{}, *mResolveColorView[mFrameCount & 1], vk::ImageLayout::eGeneral
		);
		wdescSets[0] = vk::WriteDescriptorSet(
			mResolveDescSetTex[mFrameCount % 2][mFrameCount & 1], 0, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(resolveInputTexInfo);
		wdescSets[1] = vk::WriteDescriptorSet(
			mResolveDescSetTex[mFrameCount % 2][mFrameCount & 1], 1, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(resolvePrevTexInfo);
		wdescSets[2] = vk::WriteDescriptorSet(
			mResolveDescSetTex[mFrameCount % 2][mFrameCount & 1], 2, 0, 1, vk::DescriptorType::eStorageImage
		).setImageInfo(resolveOutputTexInfo);
		mDevice->updateDescriptorSets(3, wdescSets, 0, nullptr);
		const auto resolveDescSets = {
			mResolveDescSetDummy[mFrameCount % 2][mFrameCount & 1], mResolveDescSetTex[mFrameCount % 2][mFrameCount & 1]
		};
		cmdBuf.bindDescriptorSets(
			vk::PipelineBindPoint::eCompute, *mPipelineLayoutResolve, 0, resolveDescSets, {}
		);
		cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOResolve);
		const uint32_t oddFrame = mFrameCount & 1;
		cmdBuf.pushConstants(*mPipelineLayoutResolve, vk::ShaderStageFlagBits::eCompute, 0, 4, &oddFrame);
		cmdBuf.dispatch((mResolveExtent.width + 7) / 8, (mResolveExtent.height + 7) / 8, 1);

		// Blit to back buffer
		const array<vk::ImageMemoryBarrier, 2> imageBarriersCopy = {
			vk::ImageMemoryBarrier(
				{ /*no access*/ }, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				mBackBuffers[mBackBufferIdx],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)),
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				*mResolveColor[mFrameCount & 1],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)),
		};
		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
			imageBarriersCopy
		);

		const auto imageColorSubres0 = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
		cmdBuf.copyImage(*mResolveColor[mFrameCount & 1], vk::ImageLayout::eTransferSrcOptimal,
			mBackBuffers[mBackBufferIdx], vk::ImageLayout::eTransferDstOptimal,
			vk::ImageCopy(imageColorSubres0, {}, imageColorSubres0, {}, vk::Extent3D(mResolveExtent, 1))
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

	void MoveUp()
	{
		auto d = DirectX::XMMatrixRotationX(DirectX::XMConvertToRadians(-3.f));
		auto p = DirectX::XMVector3TransformCoord(DirectX::XMVectorSubtract(mCameraTarget, mCameraPos), d);
		mCameraTarget = DirectX::XMVectorAdd(p, mCameraPos);
		mCameraUp = DirectX::XMVector3TransformCoord(mCameraUp, d);
	}
	void MoveDown()
	{
		auto d = DirectX::XMMatrixRotationX(DirectX::XMConvertToRadians(3.f));
		auto p = DirectX::XMVector3TransformCoord(DirectX::XMVectorSubtract(mCameraTarget, mCameraPos), d);
		mCameraTarget = DirectX::XMVectorAdd(p, mCameraPos);
		mCameraUp = DirectX::XMVector3TransformCoord(mCameraUp, d);
	}
	void MoveFwd()
	{
		auto d = DirectX::XMVectorScale(DirectX::XMVectorSubtract(mCameraPos, mCameraTarget), -0.08f);
		mCameraPos = DirectX::XMVectorAdd(d, mCameraPos);
		mCameraTarget = DirectX::XMVectorAdd(d, mCameraTarget);
	}
	void MoveBack()
	{
		auto d = DirectX::XMVectorScale(DirectX::XMVectorSubtract(mCameraPos, mCameraTarget), 0.08f);
		mCameraPos = DirectX::XMVectorAdd(d, mCameraPos);
		mCameraTarget = DirectX::XMVectorAdd(d, mCameraTarget);
	}
	void MoveRight()
	{
		auto d = DirectX::XMMatrixRotationY(DirectX::XMConvertToRadians(-3.f));
		auto p = DirectX::XMVector3TransformCoord(DirectX::XMVectorSubtract(mCameraTarget, mCameraPos), d);
		mCameraTarget = DirectX::XMVectorAdd(p, mCameraPos);
	}
	void MoveLeft()
	{
		auto d = DirectX::XMMatrixRotationY(DirectX::XMConvertToRadians(3.f));
		auto p = DirectX::XMVector3TransformCoord(DirectX::XMVectorSubtract(mCameraTarget, mCameraPos), d);
		mCameraTarget = DirectX::XMVectorAdd(p, mCameraPos);
	}

private:
	DirectX::XMVECTOR mCameraPos = DirectX::XMVectorSet(0.0f, 4.0f, -4.0f, 0);
	DirectX::XMVECTOR mCameraTarget = DirectX::XMVectorSet(0, 0, 0, 0);
	DirectX::XMVECTOR mCameraUp = DirectX::XMVectorSet(0, 1, 0, 0);
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
					if (msg.wParam == 'E') vlk.MoveUp();
					else if (msg.wParam == 'Q') vlk.MoveDown();
					else if (msg.wParam == 'W') vlk.MoveFwd();
					else if (msg.wParam == 'S') vlk.MoveBack();
					else if (msg.wParam == 'D') vlk.MoveRight();
					else if (msg.wParam == 'A') vlk.MoveLeft();
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

