// based on PBR.cpp

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

	vk::UniqueSemaphore mSwapchainSema[BUFFER_COUNT];
	vk::UniqueSemaphore mDrawingSema[BUFFER_COUNT];

	uint32_t mBackBufferIdx = 0;
	uint64_t mFrameCount = 0;
	vk::UniqueCommandPool mCmdPool;
	std::vector<vk::UniqueCommandBuffer> mCmdBuf;
	vk::UniqueFence mCmdFence[2];
	vk::UniqueDescriptorPool mDescPools[2];

	vk::UniqueRenderPass mRenderPass;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutBuf;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutTex;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutPost;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutABRDF;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutEnvFilter;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutProjSH;
	vk::UniqueDescriptorSetLayout mDescriptorSetLayoutConvSH;
	vk::UniquePipelineLayout mPipelineLayout;
	vk::UniquePipelineLayout mPipelineLayoutPost;
	vk::UniquePipelineLayout mPipelineLayoutABRDF;
	vk::UniquePipelineLayout mPipelineLayoutEnvFilter;
	vk::UniquePipelineLayout mPipelineLayoutProjSH;
	vk::UniquePipelineLayout mPipelineLayoutConvSH;
	vk::UniquePipeline mPSO;
	vk::UniquePipeline mPSOPost;
	vk::UniquePipeline mPSOABRDF;
	vk::UniquePipeline mPSOEnvFilter;
	vk::UniquePipeline mPSOEnvDiffuse;
	vk::UniquePipeline mPSOProjSH;
	vk::UniquePipeline mPSOConvSH;

	vk::Extent2D mSceneExtent;
	vk::UniqueImage mSceneColor;
	vk::UniqueDeviceMemory mSceneColorMemory;
	vk::UniqueImageView mSceneColorView;
	vk::UniqueImage mSceneDepth;
	vk::UniqueDeviceMemory mSceneDepthMemory;
	vk::UniqueImageView mSceneDepthView;
	vk::UniqueFramebuffer mSceneFramebuffer[3];

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
	vk::UniqueImage mEnvMapImg;
	vk::UniqueImageView mEnvMapView;
	vector<vk::UniqueImageView> mEnvMapMipView;
	vector<vk::UniqueImageView> mEnvMapMipCubeView;
	uint32_t mEnvMapMipLevels;
	vk::UniqueImage mEnvDiffuseImg; // no mip
	vk::UniqueImageView mEnvDiffuseView, mEnvDiffuseArrayView;
	vk::Extent3D mEnvDiffuseExtent;
	vk::UniqueDeviceMemory mImagesMemory;

	vk::UniqueImage mAmbientBrdfImg;
	vk::UniqueImageView mAmbientBrdfView;
	vk::UniqueDeviceMemory mAmbientBrdfMemory;

	vk::UniqueBuffer mSHBuf;
	vk::UniqueDeviceMemory mSHMemory;

	vk::UniqueSampler mSampler;
	vk::UniqueSampler mEnvMapSampler;
	vk::UniqueSampler mAmbientBrdfSampler;

	vk::DescriptorSet mSphereDescSetBuf[2];
	vk::DescriptorSet mSphereDescSetTex[2];
	vk::DescriptorSet mPlaneDescSetBuf[2];
	vk::DescriptorSet mPlaneDescSetTex[2];
	vk::DescriptorSet mPostDescSet[2];

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
			VK_KHR_FORMAT_FEATURE_FLAGS_2_EXTENSION_NAME, // UAV image
		};

		// Create a Vulkan instance
		const auto appInfo = vk::ApplicationInfo(
			"MyApp", VK_MAKE_API_VERSION(0, 0, 0, 0),
			"LearningVulkan", VK_MAKE_API_VERSION(0, 0, 0, 0), VK_API_VERSION_1_1);
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
				[](vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity, vk::DebugUtilsMessageTypeFlagsEXT messageType, const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
					cout << pCallbackData->pMessage << endl;
					OutputDebugStringA(pCallbackData->pMessage);
					OutputDebugStringW(L"\n");
					if (messageSeverity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
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
		// VK_KHR_shader_draw_parameters
		auto enableShaderDrawParam = vk::PhysicalDeviceShaderDrawParametersFeatures(1);
		const auto enableFeatures = vk::PhysicalDeviceFeatures()
			.setSamplerAnisotropy(1);
		auto enableFeatures2 = vk::PhysicalDeviceFeatures2(enableFeatures)
			.setPNext(&enableShaderDrawParam);
		const auto deviceCreateInfo = vk::DeviceCreateInfo(
			{}, (uint32_t)deviceQueueInfos.size(), deviceQueueInfos.data(),
			0, nullptr, _countof(deviceExtensions), deviceExtensions
		).setPNext(&enableFeatures2);
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
			mSwapchainSema[i] = mDevice->createSemaphoreUnique({});
			mDrawingSema[i] = mDevice->createSemaphoreUnique({});
		}

		// Create commands
		mCmdPool = mDevice->createCommandPoolUnique(
			vk::CommandPoolCreateInfo(
				vk::CommandPoolCreateFlagBits::eTransient | vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
				queueGfxIdx));
		mCmdBuf = mDevice->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo(*mCmdPool, vk::CommandBufferLevel::ePrimary, 2));
		for (int i = 0; i < 2; ++i)
		{
			mCmdFence[i] = mDevice->createFenceUnique(vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
		}

		// Create descriptor pools
		const auto descPoolSizes = {
			vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 500),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage, 500),
			vk::DescriptorPoolSize(vk::DescriptorType::eSampler, 50),
			vk::DescriptorPoolSize(vk::DescriptorType::eStorageImage, 20),
		};
		for (auto& p : mDescPools)
		{
			p = mDevice->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo({}, 100, descPoolSizes));
		}

		// Create a render pass
		const auto sceneFormat = vk::Format::eB10G11R11UfloatPack32;
		const auto colorAttachmentDesc = vk::AttachmentDescription(
			{}, sceneFormat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal
		);
		const auto depthAttachmentDesc = vk::AttachmentDescription(
			{}, vk::Format::eD32Sfloat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal
		);
		const auto postAttachmentDesc = vk::AttachmentDescription(
			{}, surfaceFormat, vk::SampleCountFlagBits::e1,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore,
			vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
			vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR
		);
		const auto attachmentsDesc = {
			colorAttachmentDesc, depthAttachmentDesc, postAttachmentDesc
		};
		const auto colorAttachmentRef = vk::AttachmentReference(
			0, vk::ImageLayout::eColorAttachmentOptimal
		);
		const auto depthAttachmentRef = vk::AttachmentReference(
			1, vk::ImageLayout::eDepthStencilAttachmentOptimal
		);
		const auto colorReadAttachmentRef = vk::AttachmentReference(
			0, vk::ImageLayout::eShaderReadOnlyOptimal
		);
		const auto postAttachmentRef = vk::AttachmentReference(
			2, vk::ImageLayout::eColorAttachmentOptimal
		);
		const auto subpassDescs = {
			// Pass1: Forward lighting
			vk::SubpassDescription(
				{}, vk::PipelineBindPoint::eGraphics, {}, colorAttachmentRef, {}, &depthAttachmentRef
			),
			// Pass2: Post process (Tone mapping, sRGB conversion)
			vk::SubpassDescription(
				{}, vk::PipelineBindPoint::eGraphics, colorReadAttachmentRef, postAttachmentRef, {}
			),
		};
		const auto subpassDeps = vk::SubpassDependency(
			0, 1,
			vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader,
			vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
			vk::DependencyFlagBits::eByRegion
		);
		mRenderPass = mDevice->createRenderPassUnique(
			vk::RenderPassCreateInfo({}, attachmentsDesc, subpassDescs, subpassDeps)
		);

		// Create descriptor set layouts
		// Set 0
		{
			const auto descriptorBinding = {
				vk::DescriptorSetLayoutBinding(
					0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex
				),
				vk::DescriptorSetLayoutBinding(
					10, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					11, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment
				),
			};
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
				vk::DescriptorSetLayoutBinding(
					2, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					3, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					4, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					5, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eFragment
				),
				vk::DescriptorSetLayoutBinding(
					6, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eFragment
				),
			};
			const auto descriptorSetLayoutInfo = vk::DescriptorSetLayoutCreateInfo(
				{}, descriptorBinding
			);
			mDescriptorSetLayoutTex = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfo);
		}

		const auto descriptorBindingPost = vk::DescriptorSetLayoutBinding(
			0, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment
		);
		const auto descriptorSetLayoutInfoPost = vk::DescriptorSetLayoutCreateInfo(
			{}, descriptorBindingPost
		);
		mDescriptorSetLayoutPost = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfoPost);

		const auto descriptorBindingABRDF = {
			vk::DescriptorSetLayoutBinding(
				0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute
			),
		};
		const auto descriptorSetLayoutInfoABRDF = vk::DescriptorSetLayoutCreateInfo(
			{}, descriptorBindingABRDF
		);
		mDescriptorSetLayoutABRDF = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfoABRDF);

		const auto descriptorBindingEnvFilter = {
			vk::DescriptorSetLayoutBinding(
				0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute
			),
			vk::DescriptorSetLayoutBinding(
				1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute
			),
			vk::DescriptorSetLayoutBinding(
				2, vk::DescriptorType::eSampler, 1, vk::ShaderStageFlagBits::eCompute
			),
		};
		const auto descriptorSetLayoutInfoEnvFilter = vk::DescriptorSetLayoutCreateInfo(
			{}, descriptorBindingEnvFilter
		);
		mDescriptorSetLayoutEnvFilter = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfoEnvFilter);

		const auto descriptorBindingProjSH = {
			vk::DescriptorSetLayoutBinding(
				0, vk::DescriptorType::eSampledImage, 1, vk::ShaderStageFlagBits::eCompute
			),
			vk::DescriptorSetLayoutBinding(
				1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute
			),
		};
		const auto descriptorSetLayoutInfoProjSH = vk::DescriptorSetLayoutCreateInfo(
			{}, descriptorBindingProjSH
		);
		mDescriptorSetLayoutProjSH = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfoProjSH);

		const auto descriptorBindingConvSH = {
			vk::DescriptorSetLayoutBinding(
				0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute
			),
		};
		const auto descriptorSetLayoutInfoConvSH = vk::DescriptorSetLayoutCreateInfo(
			{}, descriptorBindingConvSH
		);
		mDescriptorSetLayoutConvSH = mDevice->createDescriptorSetLayoutUnique(descriptorSetLayoutInfoConvSH);

		// Create pipeline layouts
		const auto pipelineDescSets = { *mDescriptorSetLayoutBuf, *mDescriptorSetLayoutTex };
		mPipelineLayout = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, pipelineDescSets)
		);
		mPipelineLayoutPost = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, *mDescriptorSetLayoutPost)
		);
		mPipelineLayoutABRDF = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, *mDescriptorSetLayoutABRDF)
		);
		mPipelineLayoutEnvFilter = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, *mDescriptorSetLayoutEnvFilter)
		);
		mPipelineLayoutProjSH = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, *mDescriptorSetLayoutProjSH)
		);
		mPipelineLayoutConvSH = mDevice->createPipelineLayoutUnique(
			vk::PipelineLayoutCreateInfo({}, *mDescriptorSetLayoutConvSH)
		);

		// Create modules
		static const char shaderCodeSceneVS[] = R"#(
[[vk::binding(0, 0)]] cbuffer CScene {
	float4x4 ViewProj;
	float4x4 Model[1 + 18];
	float2 Metallic;
	float2 Roughness;
};
struct Output {
	float4 position : SV_Position;
	float3 world : WorldPosition;
	float3 normal : Normal;
	float2 texcoord : Texcoord;
	float metallic : Metallic;
	float roughness : Roughness;
	float2 clearCoat : ClearCoat;
};
Output main(uint instanceID : SV_InstanceID, //[[vk::builtin("BaseInstance")]] uint baseInstanceID : BaseInstanceID,
			 float3 position : Position, float3 normal : Normal, float2 texcoord : Texcoord) {
	// Workaround: -fvk-support-nonzero-base-instance is disabled because of dxc internal bug
	const uint baseInstanceID = 0;
	float4 wpos = mul(float4(position, 1), Model[instanceID + baseInstanceID]);
	Output output;
	output.position = mul(wpos, ViewProj);
	output.world = wpos.xyz / wpos.w;
	output.normal = normalize(mul(normal, (float3x3)Model[instanceID + baseInstanceID]));
	output.texcoord = texcoord;
	output.metallic = lerp(Metallic[0], Metallic[1], saturate((float)(instanceID / 6)));
	output.roughness = lerp(Roughness[0], Roughness[1], (float)(instanceID % 6) / 5);
	output.clearCoat = float2((instanceID >= 12 && instanceID < 18) ? 1 : 0, 0.1/*roughness*/);
	return output;
}
)#";

		static const char shaderCodeScenePS[] = R"#(
static const float MaxEnvMapMipLevel = 7.0;
[[vk::binding(10, 0)]] cbuffer CLight {
	float3 CameraPosition;
	float3 SunLightIntensity;
	float3 SunLightDirection;
};
[[vk::binding(11, 0)]] StructuredBuffer<float> SHFactorBuf;
[[vk::binding(0, 1)]] Texture2D BaseColor;
[[vk::binding(1, 1)]] SamplerState SS;
[[vk::binding(2, 1)]] TextureCube<float3> EnvMap;
//[[vk::binding(3, 1)]] TextureCube<float3> EnvDiffuseMap; // baked on SH
[[vk::binding(4, 1)]] SamplerState EnvMapSS;
[[vk::binding(5, 1)]] Texture2D<float2> AmbientBrdf;
[[vk::binding(6, 1)]] SamplerState AmbientBrdfSS;
struct Input {
	float4 position : SV_Position;
	float3 world : WorldPosition;
	float3 normal : Normal;
	float2 texcoord : Texcoord;
	float metallic : Metallic;
	float roughness : Roughness;
	float2 clearCoat : ClearCoat;
};
// https://google.github.io/filament/Filament.html
// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
#define PI (3.14159265f)
#define F0 (0.04f)
float D_GGX(float NoH, float a) {
	float a2 = a * a;
	float f = (NoH * a2 - NoH) * NoH + 1.0;
	return a2 / (PI * f * f);
}
float V_SmithGGXCorrelated(float NoV, float NoL, float roughness) {
	float a2 = roughness * roughness;
	float GGXV = NoL * sqrt(NoV * NoV * (1.0 - a2) + a2);
	float GGXL = NoV * sqrt(NoL * NoL * (1.0 - a2) + a2);
	return 0.5 / (GGXV + GGXL);
}
float3 F_Schlick(float u, float3 f0, float f90 = 1.0) {
	return f0 + (1.0 - f0) * pow(1.0 - u, 5.0);
}
float Fd_Lambert() {
	return 1 / PI;
}
float Fd_Burley(float NoV, float NoL, float LoH, float roughness) {
	float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
	float lightScatter = F_Schlick(NoL, 1.0, f90).x;
	float viewScatter = F_Schlick(NoV, 1.0, f90).x;
	return lightScatter * viewScatter * (1.0 / PI);
}
float V_Kelemen(float LoH) {
	return 0.25 / (LoH * LoH);
}
float computeLODFromRoughness(float perceptualRoughness) {
	return (perceptualRoughness * MaxEnvMapMipLevel);
}
float3 ApproximateSpecularIBL(float3 SpecularColor, float Roughness, float3 N, float3 V, out float2 EnvBRDF) {
	float NoV = saturate(dot(N, V));
	float3 R = 2 * dot(V, N) * N - V;
	float lod = computeLODFromRoughness(Roughness);
	float3 PrefilteredColor = EnvMap.SampleLevel(EnvMapSS, R, lod);
	EnvBRDF = AmbientBrdf.SampleLevel(AmbientBrdfSS, float2(NoV, Roughness * Roughness), 0);
	return PrefilteredColor * (SpecularColor * EnvBRDF.x + EnvBRDF.y);
}
float3 irradianceSH(float3 n, float3 sh[9]) {
	float3 c =  sh[0]
		+ sh[1] * n.y
		+ sh[2] * n.z
		+ sh[3] * n.x
#if 1
		+ sh[4] * (n.y * n.x)
		+ sh[5] * (n.y * n.z)
		+ sh[6] * (3 * n.z * n.z - 1)
		+ sh[7] * (n.z * n.x)
		+ sh[8] * (n.x * n.x - n.y * n.y)
#endif
		;
	return max(0, c) / PI;
}
void readSH(out float3 sh[9]) {
	sh[0] = float3(SHFactorBuf[0], SHFactorBuf[9], SHFactorBuf[18]);
	sh[1] = float3(SHFactorBuf[1], SHFactorBuf[19], SHFactorBuf[19]);
	sh[2] = float3(SHFactorBuf[2], SHFactorBuf[11], SHFactorBuf[20]);
	sh[3] = float3(SHFactorBuf[3], SHFactorBuf[12], SHFactorBuf[21]);
	sh[4] = float3(SHFactorBuf[4], SHFactorBuf[13], SHFactorBuf[22]);
	sh[5] = float3(SHFactorBuf[5], SHFactorBuf[14], SHFactorBuf[23]);
	sh[6] = float3(SHFactorBuf[6], SHFactorBuf[15], SHFactorBuf[24]);
	sh[7] = float3(SHFactorBuf[7], SHFactorBuf[16], SHFactorBuf[25]);
	sh[8] = float3(SHFactorBuf[8], SHFactorBuf[17], SHFactorBuf[26]);
}
float4 main(Input input) : SV_Target {
	//float4 baseColor = BaseColor.Sample(SS, input.texcoord);
	float4 baseColor = 1; // White
	float3 diffColor = (input.metallic > 0.0) ? 0.0 : baseColor.rgb;
	float3 specColor = (input.metallic > 0.0) ? baseColor.rgb : F0.xxx;
	input.normal = normalize(input.normal);

	float3 viewDir = normalize(CameraPosition - input.world);
	float3 halfVector = normalize(viewDir + SunLightDirection);
	float dotNV = abs(dot(input.normal, viewDir)) + 1e-5;
	float dotNL = saturate(dot(input.normal, SunLightDirection));
	float dotNH = saturate(dot(input.normal, halfVector));
	float dotLH = saturate(dot(SunLightDirection, halfVector));
	// https://ubm-twvideo01.s3.amazonaws.com/o1/vault/gdc2017/Presentations/Hammon_Earl_PBR_Diffuse_Lighting.pdf
	float lenSq_LV = 2 + 2 * dot(SunLightDirection, viewDir);
	float rcpLen_LV = rsqrt(lenSq_LV);
	dotNH = (dotNL + dotNV) * rcpLen_LV;
	dotLH = rcpLen_LV + rcpLen_LV * dot(SunLightDirection, viewDir);
	float roughness = input.roughness * input.roughness;
	float termD = D_GGX(dotNH, roughness);
	float termV = V_SmithGGXCorrelated(dotNV, dotNL, roughness);
	float3 termF = F_Schlick(dotLH, specColor);
	float3 Fr = termD * termV * termF;
	float Fd = Fd_Burley(dotNV, dotNL, dotLH, roughness);

	float clearCoatStrength = input.clearCoat.x;
	float clearCoatPerceptualRoughness = input.clearCoat.y;
	float clearCoatRoughness = clearCoatPerceptualRoughness * clearCoatPerceptualRoughness;
	float termDc = D_GGX(dotNH, clearCoatRoughness);
	float termVc = V_Kelemen(dotLH);
	float termFc = F_Schlick(dotLH, 0.04).r * clearCoatStrength;
	float Frc = termDc * termVc * termFc;

	//float3 F = Fr + Fd * diffColor; // Diffuse + Specular
	float3 F = (Fr + Fd * diffColor) * (1 - termFc) + Frc.rrr;

	float3 lit = 0; //SunLightIntensity * F * dotNL;
	float Fc = F_Schlick(dotNH, 0.04).r * clearCoatStrength;
	float2 envBrdf, envBrdfClearCoat;
	lit += ApproximateSpecularIBL(specColor, input.roughness, input.normal, viewDir, envBrdf)
			* (1 - Fc) * (1 - Fc)
		+ ApproximateSpecularIBL((float3)1, clearCoatRoughness, input.normal, viewDir, envBrdfClearCoat)
			* Fc;
	float3 sh[9];
	readSH(sh);
	float3 specReflectance = specColor;
	lit += irradianceSH(input.normal, sh) * diffColor * (1 - specReflectance) * Fd_Lambert() * (1 - Fc);
	return float4(lit, 1.0);
}
)#";

		static const char shaderCodePostVS[] = R"#(
float4 main(uint vid : SV_VertexID) : SV_Position {
	return float4((vid & 1) ? 3 : -1, (vid & 2) ? -3 : 1, 0, 1);
}
)#";

		static const char shaderCodePostPS[] = R"#(
// http://filmicworlds.com/blog/filmic-tonemapping-operators/
[[vk::input_attachment_index(0)]] SubpassInput Input;
float luminance(float3 rgb) {
	return dot(rgb, float3(0.2, 0.7, 0.1));
}
float3 tonemapping(float3 z) {
	float a = 0.15, b = 0.50, c = 0.10, d = 0.20, e = 0.02, f = 0.30;
	return ((z * (a * z + c * b) + d * e) / (z * (a * z + b) + d * f)) - e / f;
}
float3 linearToSrgb(float3 lin) {
	lin = saturate(lin);
	float3 s1 = 1.055 * pow(lin, 1 / 2.4) - 0.055;
	float3 s2 = lin * 12.92;
	return step(lin, 0.0031308) * s2 + select(step(lin, 0.0031308), 0, 1) * s1;
}
float4 main() : SV_Target {
	float3 color = Input.SubpassLoad().rgb;
	float exposure = exp2(2.0); // You can change the EV
	float lum = luminance(color);
	float3 sat = color / max(lum, 0.00001);
	color = sat * tonemapping(exposure * lum);
	color = linearToSrgb(color);
	return float4(color, 1);
}
)#";
		
		static const char shaderCodeABRDFCS[] = R"#(
#define PI (3.14159265f)
static const uint sampleCount = 1024;
float3 importanceSampleGGX(float2 Xi, float Roughness, float3 N)
{
	float a = Roughness * Roughness;
	float Phi = 2 * PI * Xi.x;
	float CosTheta = sqrt( (1 - Xi.y) / ( 1 + (a*a - 1) * Xi.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );
	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	float3 UpVector = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
	float3 TangentX = normalize( cross( UpVector, N ) );
	float3 TangentY = cross( N, TangentX );
	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}
float2 hammersley(uint i, float numSamples) {
	return float2(i / numSamples, reversebits(i) / 4294967296.0);
}
float GDFG(float NoV, float NoL, float a) {
	float a2 = a * a;
	float GGXL = NoV * sqrt((-NoL * a2 + NoL) * NoL + a2);
	float GGXV = NoL * sqrt((-NoV * a2 + NoV) * NoV + a2);
	return (2 * NoL) / (GGXV + GGXL);
}
static const float3 N = float3(0, 0, 1);
float2 DFG(float NoV, float a) {
	float3 V;
	V.x = sqrt(1.0f - NoV*NoV);
	V.y = 0.0f;
	V.z = NoV;

	float2 r = 0.0f;
	for (uint i = 0; i < sampleCount; i++) {
		float2 Xi = hammersley(i, sampleCount);
		float3 H = importanceSampleGGX(Xi, a, N);
		float3 L = 2.0f * dot(V, H) * H - V;

		float VoH = saturate(dot(V, H));
		float NoL = saturate(L.z);
		float NoH = saturate(H.z);

		if (NoL > 0.0f) {
			float G = GDFG(NoV, NoL, a);
			float Gv = G * VoH / NoH;
			float Fc = pow(1 - VoH, 5.0f);
			r.x += Gv * (1 - Fc);
			r.y += Gv * Fc;
		}
	}
	return r * (1.0f / sampleCount);
}
RWTexture2D<float2> Output;
[numthreads(8, 8, 1)]
void main(uint2 dtid : SV_DispatchThreadID) {
	float width, height;
	Output.GetDimensions(width, height);
	float percepturalRoughness = (0.5 + dtid.x) / width;
	float roughness = percepturalRoughness * percepturalRoughness;
	float dotNV = (0.5 + dtid.y) / height;
	Output[dtid] = DFG(roughness, dotNV);
}
)#";

		static const char shaderCodeEnvFilterCS[] = R"#(
#define PI (3.14159265f)
static const uint sampleCount = 2048;
static const float clampLuminance = 500;
float3 importanceSampleGGX(float2 Xi, float Roughness, float3 N)
{
	float a = Roughness * Roughness;
	float Phi = 2 * PI * Xi.x;
	float CosTheta = sqrt( (1 - Xi.y) / ( 1 + (a*a - 1) * Xi.y ) );
	float SinTheta = sqrt( 1 - CosTheta * CosTheta );
	float3 H;
	H.x = SinTheta * cos( Phi );
	H.y = SinTheta * sin( Phi );
	H.z = CosTheta;
	float3 UpVector = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
	float3 TangentX = normalize( cross( UpVector, N ) );
	float3 TangentY = cross( N, TangentX );
	// Tangent to world space
	return TangentX * H.x + TangentY * H.y + N * H.z;
}
float2 hammersley(uint i, float numSamples) {
	return float2(i / numSamples, reversebits(i) / 4294967296.0);
}
float3 directionFrom3D(float x, float y, uint z) {
	switch (z) {
	case 0: return float3(+1, -y, -x);
	case 1: return float3(-1, -y, +x);
	case 2: return float3(+x, +1, +y);
	case 3: return float3(+x, -1, -y);
	case 4: return float3(+x, -y, +1);
	case 5: return float3(-x, -y, -1);
	}
	return (float3)0;
}
[[vk::binding(0, 0)]] TextureCube<float3> Input;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> Output; // float3 cannot accept for RGBA16 imageView in vk1.0-1.2
[[vk::binding(2, 0)]] SamplerState SS;
float3 PrefilterEnvMap(float Roughness, float3 R) {
	float3 N = R;
	float3 V = R;
	float3 PrefilteredColor = 0;
	float TotalWeight = 0;
	for (uint i = 0; i < sampleCount; i++) {
		float2 Xi = hammersley(i, sampleCount);
		float3 H = importanceSampleGGX(Xi, Roughness, N);
		float3 L = 2.0f * dot(V, H) * H - V;
		float NoL = saturate(dot(N, L));
		if (NoL > 0) {
			PrefilteredColor += min(clampLuminance, Input.SampleLevel(SS, L, 0).rgb) * NoL;
			TotalWeight += NoL;
		}
	}
	return PrefilteredColor / TotalWeight;
}
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
	float width, height, arrayLayer;
	Output.GetDimensions(width, height, arrayLayer);
	float roughness = 1.0 - ((float)firstbitlow(width) / firstbitlow(128));
	float2 uv = ((float2)dtid.xy + 0.5) / float2(width, height);
	float3 dir = directionFrom3D(uv.x * 2 - 1, uv.y * 2 - 1, dtid.z);
	Output[dtid] = float4(PrefilterEnvMap(roughness, normalize(dir)), 1);
}
)#";

		static const char shaderCodeEnvDiffuseCS[] = R"#(
// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
#define PI (3.14159265f)
static const uint sampleCount = 32768;
static const float clampLuminance = 8000;
void importanceSampleCosDir(float2 u, float3 N, out float3 L, out float NdotL, out float pdf)
{
	// Local referencial
	float3 upVector = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
	float3 tangentX = normalize(cross(upVector, N));
	float3 tangentY = cross(N, tangentX);
	float u1 = u.x;
	float u2 = u.y;
	float r = sqrt(u1);
	float phi = u2 * PI * 2;
	float3 L0 = float3(r * cos(phi), r * sin(phi), sqrt(max(0, 1 - u1)));
	L = normalize(tangentX * L0.x + tangentY * L0.y + N * L0.z);
	NdotL = dot(L, N);
	pdf = NdotL / PI;
}
float2 hammersley(uint i, float numSamples) {
	return float2(i / numSamples, reversebits(i) / 4294967296.0);
}
float3 directionFrom3D(float x, float y, uint z) {
	switch (z) {
	case 0: return float3(+1, -y, -x);
	case 1: return float3(-1, -y, +x);
	case 2: return float3(+x, +1, +y);
	case 3: return float3(+x, -1, -y);
	case 4: return float3(+x, -y, +1);
	case 5: return float3(-x, -y, -1);
	}
	return (float3)0;
}
[[vk::binding(0, 0)]] TextureCube<float3> Input;
[[vk::binding(1, 0)]] RWTexture2DArray<float4> Output; // float3 cannot accept for RGBA16 imageView in vk1.0-1.2
[[vk::binding(2, 0)]] SamplerState SS;
float3 integrateDiffuseCube(float3 N) {
	float3 accBrdf = 0;
	for (uint i = 0; i < sampleCount; i++) {
		float2 eta = hammersley(i, sampleCount);
		float3 L;
		float NdotL, pdf;
		importanceSampleCosDir(eta, N, L, NdotL, pdf);
		if (NdotL > 0) {
			accBrdf += min(clampLuminance, Input.SampleLevel(SS, L, 0).rgb);
		}
	}
	return (accBrdf / sampleCount);
}
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
	float width, height, arrayLayer;
	Output.GetDimensions(width, height, arrayLayer);
	float2 uv = ((float2)dtid.xy + 0.5) / float2(width, height);
	float3 dir = directionFrom3D(uv.x * 2 - 1, uv.y * 2 - 1, dtid.z);
	Output[dtid] = float4(integrateDiffuseCube(normalize(dir)), 1);
}
)#";

		static const char shaderCodeProjSHCS[] = R"#(
// https://andrew-pham.blog/2019/08/26/spherical-harmonics/
// https://www.ppsloan.org/publications/StupidSH36.pdf
// https://www.ppsloan.org/publications/SHJCGT.pdf
#define PI (3.14159265f)
float3 directionFrom3D(float x, float y, uint z) {
	switch (z) {
	case 0: return float3(+1, -y, -x);
	case 1: return float3(-1, -y, +x);
	case 2: return float3(+x, +1, +y);
	case 3: return float3(+x, -1, -y);
	case 4: return float3(+x, -y, +1);
	case 5: return float3(-x, -y, -1);
	}
	return (float3)0;
}
void SHNewEval3(float fX, float fY, float fZ, out float pSH[9]) {
	float fC0, fC1, fS0, fS1, fTmpA, fTmpB, fTmpC;
	float fZ2 = fZ * fZ;
	pSH[0] = 0.2820947917738781;
	pSH[2] = 0.4886025119029199 * fZ;
	pSH[6] = 0.9461746957575601 * fZ2 - 0.3153915652525201;
	fC0 = fX;
	fS0 = fY;
	fTmpA = -0.48860251190292;
	pSH[3] = fTmpA * fC0;
	pSH[1] = fTmpA * fS0;
	fTmpB = -1.092548430592079 * fZ;
	pSH[7] = fTmpB * fC0;
	pSH[5] = fTmpB * fS0;
	fC1 = fX * fC0 - fY * fS0;
	fS1 = fX * fS0 + fY * fC0;
	fTmpC = 0.5462742152960395;
	pSH[8] = fTmpC * fC1;
	pSH[4] = fTmpC * fS1;
}
void SHScale(float input[9], float scale, out float output[9]) {
	for (int i = 0; i < 9; ++i) {
		output[i] = input[i] * scale;
	}
}
void SHAdd(float inputA[9], float inputB[9], out float output[9]) {
	for (int i = 0; i < 9; ++i) {
		output[i] = inputA[i] + inputB[i];
	}
}
int toFixedPointWt(float input) {
	return int(input * 16384);
}
void toFixedPoint(float input[9], out int output[9]) {
	for (int i = 0; i < 9; ++i) {
		output[i] = int(input[i] * 65536);
	}
}
[[vk::binding(0, 0)]] Texture2DArray<float3> Input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> Output;
void writeSH(int r[9], int idx) {
	for (int i = 0; i < 9; ++i) {
		InterlockedAdd(Output[9 * idx + i], r[i]);
	}
}
void processProjectSH(float3 dir, float3 color, float2 uv) {
	float u = uv.x * 2 - 1;
	float v = uv.y * 2 - 1;
	float temp = 1.0 + u * u + v * v;
	float weight = 4 / (sqrt(temp) * temp);
	float basis[9];
	SHNewEval3(-dir.x, -dir.y, -dir.z, basis);
	float tempR[9], tempG[9], tempB[9];
	SHScale(basis, color.r * weight, tempR);
	SHScale(basis, color.g * weight, tempG);
	SHScale(basis, color.b * weight, tempB);
	int iTempR[9], iTempG[9], iTempB[9];
	toFixedPoint(tempR, iTempR);
	toFixedPoint(tempG, iTempG);
	toFixedPoint(tempB, iTempB);
	int iWeight = toFixedPointWt(weight);
	writeSH(iTempR, 0);
	writeSH(iTempG, 1);
	writeSH(iTempB, 2);
	InterlockedAdd(Output[27], iWeight);
}
[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
	float width, height, arrayLayer;
	Input.GetDimensions(width, height, arrayLayer);
	float2 uv = ((float2)dtid.xy + 0.5) / float2(width, height);
	float3 dir = directionFrom3D(uv.x * 2 - 1, uv.y * 2 - 1, dtid.z);
	float3 color = Input.Load(int4(dtid, 0));
	processProjectSH(normalize(dir), color, uv);
}
)#";

		static const char shaderCodeConvSHCS[] = R"#(
#define PI (3.14159265f)
float fromFixedPointWt(int input) {
	return float(input) / 16384;
}
void fromFixedPoint(int input, out float output) {
	output = float(input) / 65536;
}
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> InOut;
groupshared float sNorm;
[numthreads(27, 1, 1)]
void main(uint dtid : SV_DispatchThreadID) {
	if (dtid == 0) {
		float fWtSum = fromFixedPointWt(InOut[27]);
		sNorm = 4 * PI / fWtSum; // area of sphere
		InOut[27] = asuint(sNorm);
	}
	GroupMemoryBarrierWithGroupSync();
	uint vFixed = InOut[dtid];
	float v;
	fromFixedPoint(vFixed, v);
	float normProj = sNorm;
	v *= normProj;
	InOut[dtid] = asuint(v);
}
)#";

		SetDllDirectory(L"../dll/");

		ComPtr<IDxcCompiler> dxc;
		CHK(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxc)));
		ComPtr<IDxcLibrary> dxcLib;
		CHK(DxcCreateInstance(CLSID_DxcLibrary, IID_PPV_ARGS(&dxcLib)));

		ComPtr<IDxcBlobEncoding> dxcTxtSceneVS, dxcTxtScenePS, dxcTxtPostVS, dxcTxtPostPS,
			dxcTxtABRDFCS, dxcTxtEnvFilterCS, dxcTxtEnvDiffuseCS, dxcTxtProjSHCS, dxcTxtConvSHCS;
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeSceneVS, _countof(shaderCodeSceneVS) - 1, CP_UTF8, &dxcTxtSceneVS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeScenePS, _countof(shaderCodeScenePS) - 1, CP_UTF8, &dxcTxtScenePS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodePostVS, _countof(shaderCodePostVS) - 1, CP_UTF8, &dxcTxtPostVS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodePostPS, _countof(shaderCodePostPS) - 1, CP_UTF8, &dxcTxtPostPS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeABRDFCS, _countof(shaderCodeABRDFCS) - 1, CP_UTF8, &dxcTxtABRDFCS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeEnvFilterCS, _countof(shaderCodeEnvFilterCS) - 1, CP_UTF8, &dxcTxtEnvFilterCS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeEnvDiffuseCS, _countof(shaderCodeEnvDiffuseCS) - 1, CP_UTF8, &dxcTxtEnvDiffuseCS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeProjSHCS, _countof(shaderCodeProjSHCS) - 1, CP_UTF8, &dxcTxtProjSHCS));
		CHK(dxcLib->CreateBlobWithEncodingFromPinned(shaderCodeConvSHCS, _countof(shaderCodeConvSHCS) - 1, CP_UTF8, &dxcTxtConvSHCS));

		ComPtr<IDxcBlob> dxcBlobSceneVS, dxcBlobScenePS, dxcBlobPostVS, dxcBlobPostPS,
			dxcBlobABRDFCS, dxcBlobEnvFilterCS, dxcBlobEnvDiffuseCS, dxcBlobProjSHCS, dxcBlobConvSHCS;
		ComPtr<IDxcBlobEncoding> dxcError;
		ComPtr<IDxcOperationResult> dxcRes;
		const wchar_t* shaderArgsVS[] = {
			L"-Zi", L"-all_resources_bound", L"-Qembed_debug", L"-spirv", L"-fvk-invert-y",// L"-fvk-support-nonzero-base-instance",
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
		dxc->Compile(dxcTxtPostVS.Get(), nullptr, L"main", L"vs_6_0", shaderArgsVS, _countof(shaderArgsVS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobPostVS);
		dxc->Compile(dxcTxtPostPS.Get(), nullptr, L"main", L"ps_6_0", shaderArgsPS, _countof(shaderArgsPS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobPostPS);
		dxc->Compile(dxcTxtABRDFCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobABRDFCS);
		dxc->Compile(dxcTxtEnvFilterCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobEnvFilterCS);
		dxc->Compile(dxcTxtEnvDiffuseCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobEnvDiffuseCS);
		dxc->Compile(dxcTxtProjSHCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobProjSHCS);
		dxc->Compile(dxcTxtConvSHCS.Get(), nullptr, L"main", L"cs_6_0", shaderArgsCS, _countof(shaderArgsCS), nullptr, 0, nullptr, &dxcRes);
		dxcRes->GetErrorBuffer(&dxcError);
		if (dxcError->GetBufferSize()) {
			OutputDebugStringA(reinterpret_cast<char*>(dxcError->GetBufferPointer()));
			throw runtime_error("Shader compile error.");
		}
		dxcRes->GetResult(&dxcBlobConvSHCS);

		const auto vsCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobSceneVS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobSceneVS->GetBufferPointer()));
		auto vsModule =  mDevice->createShaderModuleUnique(vsCreateInfo);
		const auto psCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobScenePS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobScenePS->GetBufferPointer()));
		auto fsModule = mDevice->createShaderModuleUnique(psCreateInfo);
		const auto vsPostCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobPostVS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobPostVS->GetBufferPointer()));
		auto vsPostModule = mDevice->createShaderModuleUnique(vsPostCreateInfo);
		const auto psPostCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobPostPS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobPostPS->GetBufferPointer()));
		auto fsPostModule = mDevice->createShaderModuleUnique(psPostCreateInfo);
		const auto csABRDFCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobABRDFCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobABRDFCS->GetBufferPointer()));
		auto csABRDFModule = mDevice->createShaderModuleUnique(csABRDFCreateInfo);
		const auto csEnvFilterCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobEnvFilterCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobEnvFilterCS->GetBufferPointer()));
		auto csEnvFilterModule = mDevice->createShaderModuleUnique(csEnvFilterCreateInfo);
		const auto csEnvDiffuseCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobEnvDiffuseCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobEnvDiffuseCS->GetBufferPointer()));
		auto csEnvDiffuseModule = mDevice->createShaderModuleUnique(csEnvDiffuseCreateInfo);
		const auto csProjSHCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobProjSHCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobProjSHCS->GetBufferPointer()));
		auto csProjSHModule = mDevice->createShaderModuleUnique(csProjSHCreateInfo);
		const auto csConvSHCreateInfo = vk::ShaderModuleCreateInfo(
			{}, dxcBlobConvSHCS->GetBufferSize(), reinterpret_cast<uint32_t*>(dxcBlobConvSHCS->GetBufferPointer()));
		auto csConvSHModule = mDevice->createShaderModuleUnique(csConvSHCreateInfo);

		// Create PSOs
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

		const auto pipelinePostShadersInfo = {
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex, *vsPostModule, "main"),
			vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment, *fsPostModule, "main"),
		};
		const auto pipelinePostDSSInfo = vk::PipelineDepthStencilStateCreateInfo();
		const auto pipelinePostInfo = vk::GraphicsPipelineCreateInfo(
			{}, pipelinePostShadersInfo, &pipelineVertexInputsInfo, &pipelineInputAssemblyStateInfo,
			nullptr, &viewportStateInfo, &pipelineRSInfo, &pipelineMSAAInfo, &pipelinePostDSSInfo,
			&pipelineBSInfo, &pipelineDynamicStatesInfo, *mPipelineLayoutPost, *mRenderPass, 1
		);
		vkres = mDevice->createGraphicsPipelineUnique(nullptr, pipelinePostInfo);
		CHK(vkres.result);
		mPSOPost = std::move(vkres.value);

		const auto abrdfPipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csABRDFModule, "main");
		const auto abrdfPipelineInfo = vk::ComputePipelineCreateInfo({}, abrdfPipelineShadersInfo, *mPipelineLayoutABRDF);
		vkres = mDevice->createComputePipelineUnique(nullptr, abrdfPipelineInfo);
		CHK(vkres.result);
		mPSOABRDF = std::move(vkres.value);

		const auto envFilterPipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csEnvFilterModule, "main");
		const auto envFilterPipelineInfo = vk::ComputePipelineCreateInfo({}, envFilterPipelineShadersInfo, *mPipelineLayoutEnvFilter);
		vkres = mDevice->createComputePipelineUnique(nullptr, envFilterPipelineInfo);
		CHK(vkres.result);
		mPSOEnvFilter = std::move(vkres.value);

		const auto envDiffusePipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csEnvDiffuseModule, "main");
		// Reuse the pipeline layout
		const auto envDiffusePipelineInfo = vk::ComputePipelineCreateInfo({}, envDiffusePipelineShadersInfo, *mPipelineLayoutEnvFilter);
		vkres = mDevice->createComputePipelineUnique(nullptr, envDiffusePipelineInfo);
		CHK(vkres.result);
		mPSOEnvDiffuse = std::move(vkres.value);

		const auto projSHPipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csProjSHModule, "main");
		const auto projSHPipelineInfo = vk::ComputePipelineCreateInfo({}, projSHPipelineShadersInfo, *mPipelineLayoutProjSH);
		vkres = mDevice->createComputePipelineUnique(nullptr, projSHPipelineInfo);
		CHK(vkres.result);
		mPSOProjSH = std::move(vkres.value);

		const auto convSHPipelineShadersInfo = vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eCompute, *csConvSHModule, "main");
		const auto convSHPipelineInfo = vk::ComputePipelineCreateInfo({}, convSHPipelineShadersInfo, *mPipelineLayoutConvSH);
		vkres = mDevice->createComputePipelineUnique(nullptr, convSHPipelineInfo);
		CHK(vkres.result);
		mPSOConvSH = std::move(vkres.value);

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
			{}, vk::ImageType::e2D, sceneFormat, vk::Extent3D(width, height, 1),
			1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eColorAttachment
			| vk::ImageUsageFlagBits::eInputAttachment // Subpass load
			| vk::ImageUsageFlagBits::eTransientAttachment // Store temporary data
		).setInitialLayout(vk::ImageLayout::eUndefined)
		);
		const auto colorMemReq = mDevice->getImageMemoryRequirements(*mSceneColor);
		mSceneColorMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			colorMemReq.size, GetMemTypeIndex(colorMemReq, false)
			// TODO: Consider vk::MemoryPropertyFlagBits::eLazilyAllocated
			//       for intermediate color/depth buffers
		));
		mDevice->bindImageMemory(*mSceneColor, *mSceneColorMemory, 0);
		mSceneColorView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mSceneColor, vk::ImageViewType::e2D, sceneFormat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
		));

		// Create a depth buffer
		mSceneDepth = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, vk::Format::eD32Sfloat, vk::Extent3D(width, height, 1),
			1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment
			| vk::ImageUsageFlagBits::eTransientAttachment // Store temporary data
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
		for (int i = 0; i < BUFFER_COUNT; ++i)
		{
			const auto framebufferAttachments = { *mSceneColorView, *mSceneDepthView, *mBackBuffersView[i] };
			mSceneFramebuffer[i] = mDevice->createFramebufferUnique(vk::FramebufferCreateInfo(
				{}, *mRenderPass, framebufferAttachments, width, height, 1));
		}
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

		// Create samplers
		mSampler = mDevice->createSamplerUnique(vk::SamplerCreateInfo(
			{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
			0.0f, VK_TRUE, 4.0f, VK_FALSE, {}, 0.0f, VK_LOD_CLAMP_NONE
		));
		mEnvMapSampler = mDevice->createSamplerUnique(vk::SamplerCreateInfo(
			{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
			0.0f, VK_FALSE, 1.0f, VK_FALSE, {}, 0.0f, VK_LOD_CLAMP_NONE
		));

		// Create an ambient BRDF texture
		mAmbientBrdfImg = mDevice->createImageUnique(vk::ImageCreateInfo(
			{}, vk::ImageType::e2D, vk::Format::eR16G16Sfloat, vk::Extent3D(256, 256, 1),
			1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
			vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
		));
		const auto ambinetBrdfMemReq = mDevice->getImageMemoryRequirements(*mAmbientBrdfImg);
		mAmbientBrdfMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ambinetBrdfMemReq.size, GetMemTypeIndex(ambinetBrdfMemReq, false)
		));
		mDevice->bindImageMemory(*mAmbientBrdfImg, *mAmbientBrdfMemory, 0);
		mAmbientBrdfView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mAmbientBrdfImg, vk::ImageViewType::e2D, vk::Format::eR16G16Sfloat)
			.setSubresourceRange(
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1))
		);
		// Create a sampler for ambient BRDF texture
		mAmbientBrdfSampler = mDevice->createSamplerUnique(vk::SamplerCreateInfo(
			{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
			vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
			0.0f, VK_FALSE, 1.0f, VK_FALSE, {}, 0.0f, VK_LOD_CLAMP_NONE
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

		// Load IBL maps
		auto loadHdrBinary = [&](const char* path)
		{
			const int w = 128;
			const int h = 128;
			ifstream ifs(path, ios::binary);
			auto oneline = make_unique<float[]>(w * sizeof(float) * 4); // float4
			auto data = make_unique<uint8_t[]>(w * h * sizeof(short) * 4); // half4
			for (int y = 0; y < h; y++)
			{
				ifs.read(reinterpret_cast<char*>(oneline.get()), w * sizeof(float) * 4);
				for (int x = 0; x < w; ++x)
				{
					auto s = reinterpret_cast<short*>(data.get() + (y * w + x) * sizeof(short) * 4);
					short h[4];
					_mm_storel_epi64((__m128i*)h, _mm_cvtps_ph(_mm_loadu_ps(&oneline[4 * x]), 0));
					s[0] = h[0];
					s[1] = h[1];
					s[2] = h[2];
					s[3] = h[3];
				}
			}
			return ImageData{
				vk::Extent3D(w, h, 1),
				uint32_t(w * h * sizeof(short) * 4),
				move(data) };
		};
		array<ImageData, 6> envMapData = {
			loadHdrBinary("../res/syferfontein_1d_clear/posx.bin"),
			loadHdrBinary("../res/syferfontein_1d_clear/negx.bin"),
			loadHdrBinary("../res/syferfontein_1d_clear/posy.bin"),
			loadHdrBinary("../res/syferfontein_1d_clear/negy.bin"),
			loadHdrBinary("../res/syferfontein_1d_clear/posz.bin"),
			loadHdrBinary("../res/syferfontein_1d_clear/negz.bin"),
		};
		ASSERT(envMapData[0].extent.width == 128 && envMapData[0].extent.height == 128, "Unsupported size");
		mEnvMapMipLevels = 8; // log2(128)+1==8

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
		mEnvMapImg = mDevice->createImageUnique(vk::ImageCreateInfo(
			vk::ImageCreateFlagBits::eCubeCompatible,
			vk::ImageType::e2D, vk::Format::eR16G16B16A16Sfloat, envMapData[0].extent,
			mEnvMapMipLevels, 6, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
		));
		const auto envMapMemReq = mDevice->getImageMemoryRequirements(*mEnvMapImg);
		mEnvDiffuseExtent = vk::Extent3D(envMapData[0].extent.width / 4, envMapData[0].extent.height / 4, 1);
		mEnvDiffuseImg = mDevice->createImageUnique(vk::ImageCreateInfo(
			vk::ImageCreateFlagBits::eCubeCompatible,
			vk::ImageType::e2D, vk::Format::eR16G16B16A16Sfloat, mEnvDiffuseExtent,
			1, 6, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage,
			vk::SharingMode::eExclusive, {}, vk::ImageLayout::eUndefined
		));
		const auto envDiffuseMemReq = mDevice->getImageMemoryRequirements(*mEnvDiffuseImg);
		mImagesMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			ALIGN(ALIGN(ALIGN(sailboatMemReq.size, lennaMemReq.alignment) + lennaMemReq.size
				, envMapMemReq.alignment) + envMapMemReq.size
				, envDiffuseMemReq.alignment) + envDiffuseMemReq.size,
			GetMemTypeIndex(sailboatMemReq, false)
		));
		mDevice->bindImageMemory(*mSailboatImg, *mImagesMemory, 0);
		size_t imgMemOffset = ALIGN(sailboatMemReq.size, lennaMemReq.alignment);
		mDevice->bindImageMemory(*mLennaImg, *mImagesMemory, imgMemOffset);
		imgMemOffset = ALIGN(imgMemOffset + lennaMemReq.size, envMapMemReq.alignment);
		mDevice->bindImageMemory(*mEnvMapImg, *mImagesMemory, imgMemOffset);
		imgMemOffset = ALIGN(imgMemOffset + envMapMemReq.size, envDiffuseMemReq.alignment);
		mDevice->bindImageMemory(*mEnvDiffuseImg, *mImagesMemory, imgMemOffset);
		mSailboatView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mSailboatImg, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mSailboatMipLevels, 0, 1)
		));
		mLennaView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mLennaImg, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mLennaMipLevels, 0, 1)
		));
		mEnvMapView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mEnvMapImg, vk::ImageViewType::eCube, vk::Format::eR16G16B16A16Sfloat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mEnvMapMipLevels, 0, 6)
		));
		mEnvMapMipView.resize(mEnvMapMipLevels);
		mEnvMapMipCubeView.resize(mEnvMapMipLevels);
		for (int i = 0; i < (int)mEnvMapMipLevels; ++i)
		{
			mEnvMapMipView[i] = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
				{}, *mEnvMapImg, vk::ImageViewType::e2DArray, vk::Format::eR16G16B16A16Sfloat, {},
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, i, 1, 0, VK_REMAINING_ARRAY_LAYERS)
			));
			mEnvMapMipCubeView[i] = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
				{}, *mEnvMapImg, vk::ImageViewType::eCube, vk::Format::eR16G16B16A16Sfloat, {},
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, i, 1, 0, VK_REMAINING_ARRAY_LAYERS)
			));
		}
		mEnvDiffuseView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mEnvDiffuseImg, vk::ImageViewType::eCube, vk::Format::eR16G16B16A16Sfloat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
		));
		mEnvDiffuseArrayView = mDevice->createImageViewUnique(vk::ImageViewCreateInfo(
			{}, *mEnvDiffuseImg, vk::ImageViewType::e2DArray, vk::Format::eR16G16B16A16Sfloat, {},
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
		));

		// Create a SH buffer
		mSHBuf = mDevice->createBufferUnique(vk::BufferCreateInfo(
			{}, sizeof(float) * (9 * 3 + 1),
			vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst
		));
		const auto shMemReq = mDevice->getBufferMemoryRequirements(*mSHBuf);
		mSHMemory = mDevice->allocateMemoryUnique(vk::MemoryAllocateInfo(
			shMemReq.size, GetMemTypeIndex(shMemReq, false)
		));
		mDevice->bindBufferMemory(*mSHBuf, *mSHMemory, 0);

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
		for (int f = 0; f < envMapData.size(); ++f)
		{
			memcpy(pData, envMapData[f].data.get(), envMapData[f].size);
			pData += envMapData[f].size;
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
			vk::ImageMemoryBarrier(
				{}, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyDmaIdx, *mEnvMapImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
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
		for (int f = 0; f < envMapData.size(); ++f)
		{
			mDmaCmdBuf->copyBufferToImage(
				*mImageUploadBuffer, *mEnvMapImg, vk::ImageLayout::eTransferDstOptimal,
				vk::BufferImageCopy(
					bufferOffset, 0, 0,
					vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, f, 1), {}, envMapData[f].extent));
			bufferOffset += envMapData[f].size;
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
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite, {},
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
				mQueueFamilyDmaIdx, mQueueFamilyGfxIdx, *mEnvMapImg,
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
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
		mFrameCount++;

		const auto fence = *mCmdFence[mFrameCount % 2];
		auto r = mDevice->waitForFences(fence, VK_TRUE, 100000000000);
		CHK(r);
		mDevice->resetFences(fence);

		auto& drawingSema = mDrawingSema[mFrameCount % BUFFER_COUNT];
		const auto backBufferIdx = mDevice->acquireNextImageKHR(*mSwapchain, 100000000000, *drawingSema);
		CHK(backBufferIdx.result);
		mBackBufferIdx = backBufferIdx.value;
		auto& swapchainSema = mSwapchainSema[mBackBufferIdx];
		
		// Make uniform buffer

		auto fov = DirectX::XMConvertToRadians(45.0f);
		auto aspect = 1.0f * mSceneExtent.width / mSceneExtent.height;
		auto nearClip = 0.01f;
		auto farClip = 100.0f;

		auto viewMat = DirectX::XMMatrixLookAtLH(mCameraPos, mCameraTarget, mCameraUp);
		auto projMat = DirectX::XMMatrixPerspectiveFovLH(fov, aspect, farClip, nearClip); // Reversed depth

		auto vpMat = DirectX::XMMatrixTranspose(viewMat * projMat);

		struct CBuffer
		{
			DirectX::XMMATRIX ViewProj;
			DirectX::XMMATRIX Model[1 + 18];
			DirectX::XMVECTOR Metallic_Roughness;
		} cbufBuf;
		cbufBuf.ViewProj = vpMat;
		cbufBuf.Model[0] = DirectX::XMMatrixIdentity();
		for (int i = 0; i < 18; ++i)
		{
			float x = (float)(i % 6) * 1.1f - 3.3f + 0.65f;
			float y = (float)(i / 6) * 1.1f + 1.f;
			float s = 0.5f;
			cbufBuf.Model[1 + i] = DirectX::XMMatrixTranspose(
				DirectX::XMMatrixMultiply(DirectX::XMMatrixScaling(s, s, s) ,DirectX::XMMatrixTranslation(x, y, 0))
			);
		};
		cbufBuf.Metallic_Roughness = DirectX::XMVectorSet(0, 1, 0.05f, 0.95f);

		struct CLight
		{
			DirectX::XMVECTOR CameraPosition;
			DirectX::XMVECTOR SunLightIntensity;
			DirectX::XMVECTOR SunLightDirection;
		} cbufLight;
		cbufLight.CameraPosition = mCameraPos;
		cbufLight.SunLightIntensity = DirectX::XMVectorSet(3.0f, 3.0f, 3.0f, 1.0f);
		cbufLight.SunLightDirection = DirectX::XMVector3Normalize(DirectX::XMVectorSet(-0.1f, 0.1f, -1.0f, 1.0f));

		auto* pUB = (uint8_t*)mDevice->mapMemory(*mUniformMemory, mUniformMemoryOffsets[mFrameCount % 2], UniformBufferSize);
		memcpy(pUB, &cbufBuf, sizeof cbufBuf);
		memcpy(pUB + 2048, &cbufLight, sizeof cbufLight);
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
				vk::ImageMemoryBarrier(
					{}/*ignored*/, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyDmaIdx, mQueueFamilyGfxIdx,* mEnvMapImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
				),
			};
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader, {},
				{}, {}, barriers);
		}

		// Precompute IBL
		if (mFrameCount == 1)
		{
			// Generate ambinet BRDF
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader, {},
				{}, {}, vk::ImageMemoryBarrier(
					{}/*ignored*/, vk::AccessFlagBits::eShaderWrite,
					vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mAmbientBrdfImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
				));

			cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOABRDF);
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutABRDF
			));
			auto abrdfImageInfo = vk::DescriptorImageInfo({}, *mAmbientBrdfView, vk::ImageLayout::eGeneral);
			auto wdesc = {
				vk::WriteDescriptorSet(
					descSets[0], 0, 0, 1, vk::DescriptorType::eStorageImage
				).setImageInfo(abrdfImageInfo)
			};
			mDevice->updateDescriptorSets(wdesc, {});
			cmdBuf.bindDescriptorSets(
				vk::PipelineBindPoint::eCompute, *mPipelineLayoutABRDF, 0, descSets[0], {}
			);
			cmdBuf.dispatch(256 / 8, 256 / 8, 1);

			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eFragmentShader | vk::PipelineStageFlagBits::eComputeShader, {},
				{}, {}, vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mAmbientBrdfImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)
				));

			// Generate GGX prefiltered cubemaps
			for (int i = 1; i < (int)mEnvMapMipLevels; ++i)
			{
				cmdBuf.pipelineBarrier(
					(i == 1) ? vk::PipelineStageFlagBits::eTopOfPipe : vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader, {},
					{}, {}, vk::ImageMemoryBarrier(
						{/*no access*/ }, vk::AccessFlagBits::eShaderWrite,
						vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
						mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mEnvMapImg,
						vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, i, 1, 0, 6)
					));

				uint32_t threadNum = 1 << (mEnvMapMipLevels - 1 - i);
				cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOEnvFilter);
				auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
					*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutEnvFilter
				));
				auto mipUpImageInfo = vk::DescriptorImageInfo({}, *mEnvMapMipCubeView[i - 1], vk::ImageLayout::eShaderReadOnlyOptimal);
				auto mipDownImageInfo = vk::DescriptorImageInfo({}, *mEnvMapMipView[i], vk::ImageLayout::eGeneral);
				auto sampler = vk::DescriptorImageInfo(*mEnvMapSampler, {}, {});
				auto wdesc = {
					vk::WriteDescriptorSet(
						descSets[0], 0, 0, 1, vk::DescriptorType::eSampledImage
					).setImageInfo(mipUpImageInfo),
					vk::WriteDescriptorSet(
						descSets[0], 1, 0, 1, vk::DescriptorType::eStorageImage
					).setImageInfo(mipDownImageInfo),
					vk::WriteDescriptorSet(
						descSets[0], 2, 0, 1, vk::DescriptorType::eSampler
					).setImageInfo(sampler),
				};
				mDevice->updateDescriptorSets(wdesc, {});
				cmdBuf.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute, *mPipelineLayoutEnvFilter, 0, descSets[0], {}
				);
				cmdBuf.dispatch((threadNum + 7) / 8, (threadNum + 7) / 8, 6);

				cmdBuf.pipelineBarrier(
					vk::PipelineStageFlagBits::eComputeShader,
					vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eFragmentShader, {},
					{}, {}, vk::ImageMemoryBarrier(
						vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
						vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
						mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mEnvMapImg,
						vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, i, 1, 0, 6)
					));
			}

			// Generate a illuminance map
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::PipelineStageFlagBits::eComputeShader, {},
				{}, {}, vk::ImageMemoryBarrier(
					{/*no access*/ }, vk::AccessFlagBits::eShaderWrite,
					vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mEnvDiffuseImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
				));

			cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOEnvDiffuse);
			{
				auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
					*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutEnvFilter
				));
				auto envMapImageInfo = vk::DescriptorImageInfo({}, *mEnvMapView, vk::ImageLayout::eShaderReadOnlyOptimal);
				auto envDiffuseImageInfo = vk::DescriptorImageInfo({}, *mEnvDiffuseArrayView, vk::ImageLayout::eGeneral);
				auto sampler = vk::DescriptorImageInfo(*mEnvMapSampler, {}, {});
				auto wdesc = {
					vk::WriteDescriptorSet(
						descSets[0], 0, 0, 1, vk::DescriptorType::eSampledImage
					).setImageInfo(envMapImageInfo),
					vk::WriteDescriptorSet(
						descSets[0], 1, 0, 1, vk::DescriptorType::eStorageImage
					).setImageInfo(envDiffuseImageInfo),
					vk::WriteDescriptorSet(
						descSets[0], 2, 0, 1, vk::DescriptorType::eSampler
					).setImageInfo(sampler),
				};
				mDevice->updateDescriptorSets(wdesc, {});
				cmdBuf.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute, *mPipelineLayoutEnvFilter, 0, descSets[0], {}
				);
			}
			cmdBuf.dispatch((mEnvDiffuseExtent.width + 7) / 8, (mEnvDiffuseExtent.height + 7) / 8, 6);

			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eFragmentShader, {},
				{}, {}, vk::ImageMemoryBarrier(
					vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
					vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx, *mEnvDiffuseImg,
					vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)
				));

			// Calculate SH factors for the irradiance map
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTopOfPipe,
				vk::PipelineStageFlagBits::eTransfer, {},
				{}, vk::BufferMemoryBarrier(
					{/*no access*/ }, vk::AccessFlagBits::eTransferWrite,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mSHBuf, 0, VK_WHOLE_SIZE
				), {});
			cmdBuf.fillBuffer(*mSHBuf, 0, VK_WHOLE_SIZE, 0u);
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eTransfer,
				vk::PipelineStageFlagBits::eComputeShader, {},
				{}, vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eTransferWrite,
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mSHBuf, 0, VK_WHOLE_SIZE
				), {});

			cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOProjSH);
			{
				auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
					*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutProjSH
				));
				auto envDiffuseImageInfo = vk::DescriptorImageInfo({}, *mEnvDiffuseArrayView, vk::ImageLayout::eShaderReadOnlyOptimal);
				auto shBufInfo = vk::DescriptorBufferInfo(*mSHBuf, 0, VK_WHOLE_SIZE);
				auto wdesc = {
					vk::WriteDescriptorSet(
						descSets[0], 0, 0, 1, vk::DescriptorType::eSampledImage
					).setImageInfo(envDiffuseImageInfo),
					vk::WriteDescriptorSet(
						descSets[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer
					).setBufferInfo(shBufInfo),
				};
				mDevice->updateDescriptorSets(wdesc, {});
				cmdBuf.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute, *mPipelineLayoutProjSH, 0, descSets[0], {}
				);
			}
			cmdBuf.dispatch((mEnvDiffuseExtent.width + 7) / 8, (mEnvDiffuseExtent.height + 7) / 8, 6);

			// Convert SH factors from fixed point value to float value
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eComputeShader, {},
				{}, vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mSHBuf, 0, VK_WHOLE_SIZE
				), {});

			cmdBuf.bindPipeline(vk::PipelineBindPoint::eCompute, *mPSOConvSH);
			{
				auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
					*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutConvSH
				));
				auto shBufInfo = vk::DescriptorBufferInfo(*mSHBuf, 0, VK_WHOLE_SIZE);
				auto wdesc = {
					vk::WriteDescriptorSet(
						descSets[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer
					).setBufferInfo(shBufInfo),
				};
				mDevice->updateDescriptorSets(wdesc, {});
				cmdBuf.bindDescriptorSets(
					vk::PipelineBindPoint::eCompute, *mPipelineLayoutConvSH, 0, descSets[0], {}
				);
			}
			cmdBuf.dispatch(1, 1, 1);

			// Wait for previous shaders
			cmdBuf.pipelineBarrier(
				vk::PipelineStageFlagBits::eComputeShader,
				vk::PipelineStageFlagBits::eFragmentShader, {},
				{}, vk::BufferMemoryBarrier(
					vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
					vk::AccessFlagBits::eShaderRead,
					mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
					*mSHBuf, 0, VK_WHOLE_SIZE
				), {});
		}

		auto srgbToLinear = [](std::array<float, 4> srgb) {
			auto srgbToLinearUnit = [](float c) {
				return (c <= 0.04045f) ? (c / 12.92f) : std::powf((c + 0.055f) / 1.055f, 2.4f);
			};
			return std::array<float, 4>({
				srgbToLinearUnit(srgb[0]), srgbToLinearUnit(srgb[1]), srgbToLinearUnit(srgb[2]), srgb[3]
			});
		};

		const std::array<vk::ClearValue, 2> sceneClearValue = {
			vk::ClearColorValue(srgbToLinear({0.1f,0.2f,0.4f,1.0f})),
			vk::ClearDepthStencilValue(0.0f)
		};
		const auto renderPassInfo = vk::RenderPassBeginInfo(
			*mRenderPass, *mSceneFramebuffer[mBackBufferIdx], vk::Rect2D({}, mSceneExtent), sceneClearValue
		);

		// The initial layout of the render pass are "Undefined", so any layout can be accepted
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
		auto descLightBufInfo = vk::DescriptorBufferInfo(*mUniformBuffers[mFrameCount % 2], 2048, 2048);
		auto descSHBufInfo = vk::DescriptorBufferInfo(*mSHBuf, 0, VK_WHOLE_SIZE);
		auto descTexInfo = vk::DescriptorImageInfo({}, *mSailboatView, vk::ImageLayout::eShaderReadOnlyOptimal);
		auto descSamplerInfo = vk::DescriptorImageInfo(*mSampler, {}, {});
		const auto descEnvMapInfo = vk::DescriptorImageInfo({}, *mEnvMapView, vk::ImageLayout::eShaderReadOnlyOptimal);
		const auto descEnvDiffuseInfo = vk::DescriptorImageInfo({}, *mEnvDiffuseView, vk::ImageLayout::eShaderReadOnlyOptimal);
		const auto descEnvMapSamplerInfo = vk::DescriptorImageInfo(*mEnvMapSampler, {}, {});
		const auto descABRDFInfo = vk::DescriptorImageInfo({}, *mAmbientBrdfView, vk::ImageLayout::eShaderReadOnlyOptimal);
		const auto descABRDFSamplerInfo = vk::DescriptorImageInfo(*mAmbientBrdfSampler, {}, {});
		wdescSets[0] = vk::WriteDescriptorSet(
			mSphereDescSetBuf[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eUniformBuffer
			).setBufferInfo(descBufInfo);
		wdescSets[1] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descTexInfo);
		wdescSets[2] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 1, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descSamplerInfo);
		wdescSets[3] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 2, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descEnvMapInfo);
		wdescSets[4] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 3, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descEnvDiffuseInfo);
		wdescSets[5] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 4, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descEnvMapSamplerInfo);
		wdescSets[6] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 5, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descABRDFInfo);
		wdescSets[7] = vk::WriteDescriptorSet(
			mSphereDescSetTex[mFrameCount % 2], 6, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descABRDFSamplerInfo);
		wdescSets[8] = vk::WriteDescriptorSet(
			mSphereDescSetBuf[mFrameCount % 2], 10, 0, 1, vk::DescriptorType::eUniformBuffer
		).setBufferInfo(descLightBufInfo);
		wdescSets[9] = vk::WriteDescriptorSet(
			mSphereDescSetBuf[mFrameCount % 2], 11, 0, 1, vk::DescriptorType::eStorageBuffer
		).setBufferInfo(descSHBufInfo);
		mDevice->updateDescriptorSets(10, wdescSets, 0, nullptr);
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
		cmdBuf.drawIndexed(6 * SphereStacks * SphereSlices, 18, 0, 0, 1);

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
		descLightBufInfo = vk::DescriptorBufferInfo(*mUniformBuffers[mFrameCount % 2], 2048, 2048);
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
		wdescSets[3] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 2, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descEnvMapInfo);
		wdescSets[4] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 3, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descEnvDiffuseInfo);
		wdescSets[5] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 4, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descEnvMapSamplerInfo);
		wdescSets[6] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 5, 0, 1, vk::DescriptorType::eSampledImage
		).setImageInfo(descABRDFInfo);
		wdescSets[7] = vk::WriteDescriptorSet(
			mPlaneDescSetTex[mFrameCount % 2], 6, 0, 1, vk::DescriptorType::eSampler
		).setImageInfo(descABRDFSamplerInfo);
		wdescSets[8] = vk::WriteDescriptorSet(
			mPlaneDescSetBuf[mFrameCount % 2], 10, 0, 1, vk::DescriptorType::eUniformBuffer
		).setBufferInfo(descLightBufInfo);
		wdescSets[9] = vk::WriteDescriptorSet(
			mPlaneDescSetBuf[mFrameCount % 2], 11, 0, 1, vk::DescriptorType::eStorageBuffer
		).setBufferInfo(descSHBufInfo);
		mDevice->updateDescriptorSets(10, wdescSets, 0, nullptr);
		cmdBuf.bindVertexBuffers(0, *mPlaneVB, { 0 });
		cmdBuf.bindIndexBuffer(*mPlaneIB, 0, vk::IndexType::eUint16);
		const auto planeDescSets = {
			mPlaneDescSetBuf[mFrameCount % 2], mPlaneDescSetTex[mFrameCount % 2]
		};
		cmdBuf.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, *mPipelineLayout, 0, planeDescSets, {}
		);
		cmdBuf.drawIndexed(6, 1, 0, 0, 0);

		// Post pass
		cmdBuf.nextSubpass(vk::SubpassContents::eInline);
		if (!(mPostDescSet[mFrameCount % 2]))
		{
			auto descSets = mDevice->allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
				*mDescPools[mFrameCount % 2], *mDescriptorSetLayoutPost
			));
			mPostDescSet[mFrameCount % 2] = descSets[0];
		}
		auto descImgInfo = vk::DescriptorImageInfo({}, *mSceneColorView, vk::ImageLayout::eShaderReadOnlyOptimal);
		wdescSets[0] = vk::WriteDescriptorSet(
			mPostDescSet[mFrameCount % 2], 0, 0, 1, vk::DescriptorType::eInputAttachment
		).setImageInfo(descImgInfo);
		mDevice->updateDescriptorSets(1, wdescSets, 0, nullptr);
		cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *mPSOPost);
		cmdBuf.bindDescriptorSets(
			vk::PipelineBindPoint::eGraphics, *mPipelineLayoutPost, 0, mPostDescSet[mFrameCount % 2], {}
		);
		cmdBuf.draw(3, 1, 0, 0);

		cmdBuf.endRenderPass();

		cmdBuf.end();

		// Execute a command

		const vk::PipelineStageFlags submitPipelineStage = vk::PipelineStageFlagBits::eBottomOfPipe;
		const auto submitInfo = vk::SubmitInfo(*drawingSema, submitPipelineStage, cmdBuf, *swapchainSema);
		mQueue.submit(submitInfo, fence);
	}

	void Present()
	{
		const auto presentInfo = vk::PresentInfoKHR(*mSwapchainSema[mBackBufferIdx], *mSwapchain, mBackBufferIdx);
		const auto r = mQueuePresent.presentKHR(presentInfo);
		ASSERT(r == vk::Result::eSuccess || r == vk::Result::eSuboptimalKHR, "Presentation failed");
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
	DirectX::XMVECTOR mCameraPos = DirectX::XMVectorSet(0.0f, 2.5f, -5.0f, 0);
	DirectX::XMVECTOR mCameraTarget = DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0);
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

