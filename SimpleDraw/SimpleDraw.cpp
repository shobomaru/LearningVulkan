#define _WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string_view>
#include <exception>
#define VK_USE_PLATFORM_WIN32_KHR
#include <Vulkan/vulkan.hpp>

#pragma comment(lib, "vulkan-1.lib")

using namespace std;

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
	void ASSERT(bool r, const char* msg)
	{
		if (!r)
			throw runtime_error(msg);
	}

	vk::UniqueInstance mInstance;
	VkDebugUtilsMessengerEXT mDebugUtilsMessenger = {};
	destructor_callback mDebugUtilsDeleter;
	vk::UniqueSurfaceKHR mSurface;
	vk::UniqueDevice mDevice;
	uint32_t mQueueFamilyGfxIdx = 0;
	vk::Queue mQueue = {};
	vk::Queue mQueuePresent = {};
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

		// Check device extension capabilities
		for (auto& ext : deviceExtensions)
		{
			const auto supportedExts = physDevice.enumerateDeviceExtensionProperties();
			ASSERT(find_if(begin(supportedExts), end(supportedExts),
					[&](const vk::ExtensionProperties& s) { return string_view(s.extensionName) == ext; }
				) != end(supportedExts), "Extension not available");
		}

		// Create a device
		float queueGfxPriority = 1.0f;
		const vk::DeviceQueueCreateInfo deviceQueueInfo[] = {
			{{}, queueGfxIdx, 1, &queueGfxPriority},
		};
		const auto deviceCreateInfo = vk::DeviceCreateInfo(
			{}, _countof(deviceQueueInfo), deviceQueueInfo,
			0, nullptr, _countof(deviceExtensions), deviceExtensions
		);
		mDevice = physDevice.createDeviceUnique(deviceCreateInfo);

		// Get device queues
		mQueue = mDevice->getQueue(queueGfxIdx, 0);
		mQueuePresent = mDevice->getQueue(queueGfxIdx, 0);

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

		int a = 0;
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

		// Start drawing

		auto cmdBuf = *mCmdBuf[mFrameCount % 2];
		cmdBuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
			vk::ImageMemoryBarrier(
				// Image layout can automatically be changed to UNDEFINED if previous content is not refer.
				{ /*no access*/ }, vk::AccessFlagBits::eTransferWrite,
				vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
				mQueueFamilyGfxIdx, mQueueFamilyGfxIdx,
				mBackBuffers[mBackBufferIdx],
				vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

		cmdBuf.clearColorImage(
			mBackBuffers[mBackBufferIdx],
			vk::ImageLayout::eTransferDstOptimal,
			vk::ClearColorValue().setFloat32({ 0.1f, 0.2f, 0.4f, 1.0f }),
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		cmdBuf.pipelineBarrier(
			vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
			vk::ImageMemoryBarrier(
				vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
				vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
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
	AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
	const int windowWidth = (rect.right - rect.left);
	const int windowHeight = (rect.bottom - rect.top);

	HWND hWnd = CreateWindow(L"WindowClass", L"Window",
		WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, 0, windowWidth, windowHeight,
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

