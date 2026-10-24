/* upbench.c (2026-10-24): host -> device-local upload paths of colibri's staged uploader
 * (backend_vulkan.c up_add / up_import), measured outside the engine.
 *
 * For a run of G tensors of B bytes each (one commit), on one physical device:
 *   staged : copy through 2 x 16 MiB host-visible staging slots, double buffered (up_add)
 *   import : each tensor's pages imported (VK_EXT_external_memory_host), all copies in one
 *            command buffer, submitted at the end (dev's up_add_host: u->used never grows)
 *   importp: the same imports, the slot submitted once 16 MiB of imports are recorded in it,
 *            so the next imports overlap the copies (the candidate fix)
 * Sources are fresh posix_memalign'd + memset images per tensor in every mode (the engine
 * converts each expert into a fresh image), so the page faults are outside the timing.
 * Build: cc -O2 -o upbench upbench.c -lvulkan
 * Run:   ./upbench <device index> [B MiB] [G] [reps]   (device -1 lists them) */
#define _GNU_SOURCE
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define SLOT ((size_t)16 << 20)
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s: %d (line %d)\n", #x, r_, __LINE__); exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }

static VkPhysicalDevice phys; static VkDevice dev; static VkQueue q; static uint32_t fam;
static PFN_vkGetMemoryHostPointerPropertiesEXT getHP;
static size_t imp_align;
static VkCommandPool pool; static VkCommandBuffer cmd[2]; static VkFence fence[2]; static int pending[2];
static VkBuffer sbuf; static VkDeviceMemory smem; static uint8_t *sptr;
static VkBuffer dbuf; static VkDeviceMemory dmem;
typedef struct { VkBuffer b; VkDeviceMemory m; } Imp;
static Imp imps[2][1024]; static int nimp[2];
static int cur, open_; static size_t used, imp_used;
static double t_import, t_memcpy, t_wait, t_release;

static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want && !(mp.memoryTypes[i].propertyFlags & avoid)) return i;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type\n"); exit(1);
}
static void release(int s) {
    double t0 = now();
    for (int i = 0; i < nimp[s]; i++) { vkDestroyBuffer(dev, imps[s][i].b, NULL); vkFreeMemory(dev, imps[s][i].m, NULL); }
    nimp[s] = 0;
    t_release += now() - t0;
}
static void waitslot(int s) {
    if (pending[s]) { double t0 = now(); CK(vkWaitForFences(dev, 1, &fence[s], VK_TRUE, UINT64_MAX)); t_wait += now() - t0; pending[s] = 0; }
    release(s);
}
static void sopen(void) {
    if (open_) return;
    waitslot(cur);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    CK(vkResetCommandBuffer(cmd[cur], 0)); CK(vkBeginCommandBuffer(cmd[cur], &bi));
    open_ = 1; used = 0; imp_used = 0;
}
static void ssubmit(void) {
    if (!open_) return;
    open_ = 0;
    CK(vkEndCommandBuffer(cmd[cur])); CK(vkResetFences(dev, 1, &fence[cur]));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd[cur]};
    CK(vkQueueSubmit(q, 1, &si, fence[cur]));
    pending[cur] = 1; cur ^= 1;
}
static void finish(void) { ssubmit(); waitslot(0); waitslot(1); }
static void add_staged(const uint8_t *src, size_t bytes, size_t doff) {
    for (size_t off = 0; off < bytes;) {
        sopen();
        size_t room = SLOT - used, n = bytes - off < room ? bytes - off : room, so = (size_t)cur * SLOT + used;
        double t0 = now(); memcpy(sptr + so, src + off, n); t_memcpy += now() - t0;
        VkBufferCopy c = {so, doff + off, n}; vkCmdCopyBuffer(cmd[cur], sbuf, dbuf, 1, &c);
        used = (used + n + 255) & ~(size_t)255; off += n;
        if (used >= SLOT) ssubmit();
    }
}
static void add_import(const uint8_t *src, size_t bytes, size_t doff, int pipelined) {
    sopen();
    double t0 = now();
    uintptr_t p = (uintptr_t)src, base = p / imp_align * imp_align, end = (p + bytes + imp_align - 1) / imp_align * imp_align;
    size_t sz = end - base;
    VkMemoryHostPointerPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    CK(getHP(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, (void *)base, &mp));
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = sz, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VkBuffer b; CK(vkCreateBuffer(dev, &bi, NULL, &b));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, b, &req);
    uint32_t bits = mp.memoryTypeBits & req.memoryTypeBits, mt = 0;
    while (!(bits & (1u << mt))) mt++;
    VkImportMemoryHostPointerInfoEXT im = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, .pHostPointer = (void *)base};
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &im, .allocationSize = sz, .memoryTypeIndex = mt};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    t_import += now() - t0;
    imps[cur][nimp[cur]].b = b; imps[cur][nimp[cur]].m = m; nimp[cur]++;
    VkBufferCopy c = {p - base, doff, bytes}; vkCmdCopyBuffer(cmd[cur], b, dbuf, 1, &c);
    imp_used += bytes;
    if (pipelined && imp_used >= SLOT) ssubmit();
}

int main(int argc, char **argv) {
    int want = argc > 1 ? atoi(argv[1]) : -1;
    double mib = argc > 2 ? atof(argv[2]) : 4.0;
    int G = argc > 3 ? atoi(argv[3]) : 3, R = argc > 4 ? atoi(argv[4]) : 20;
    size_t B = (size_t)(mib * (1 << 20));
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pd[16]; if (n > 16) n = 16; vkEnumeratePhysicalDevices(inst, &n, pd);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pd[i], &p);
        if (want < 0) printf("%u: %s (type %d, driver 0x%x)\n", i, p.deviceName, p.deviceType, p.driverVersion);
    }
    if (want < 0 || want >= (int)n) return 0;
    phys = pd[want];
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(phys, &pp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties qf[32]; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    fam = 0;   /* a transfer-only family if there is one, as the uploader prefers its own queue */
    for (uint32_t i = 0; i < nq; i++) if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) { fam = i; break; }
    float pr = 1.f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = fam, .queueCount = 1, .pQueuePriorities = &pr};
    const char *ext[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = 2, .ppEnabledExtensionNames = ext};
    CK(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, fam, 0, &q);
    getHP = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp};
    vkGetPhysicalDeviceProperties2(phys, &p2);
    imp_align = hp.minImportedHostPointerAlignment ? hp.minImportedHostPointerAlignment : 4096;
    VkCommandPoolCreateInfo cp = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = fam};
    CK(vkCreateCommandPool(dev, &cp, NULL, &pool));
    VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2};
    CK(vkAllocateCommandBuffers(dev, &ca, cmd));
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    for (int s = 0; s < 2; s++) CK(vkCreateFence(dev, &fi, NULL, &fence[s]));
    VkBufferCreateInfo sb = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * SLOT, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    CK(vkCreateBuffer(dev, &sb, NULL, &sbuf));
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(dev, sbuf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
        .memoryTypeIndex = memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    CK(vkAllocateMemory(dev, &ai, NULL, &smem)); CK(vkBindBufferMemory(dev, sbuf, smem, 0));
    CK(vkMapMemory(dev, smem, 0, VK_WHOLE_SIZE, 0, (void **)&sptr));
    size_t total = B * G;
    VkBufferCreateInfo db = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = total, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    CK(vkCreateBuffer(dev, &db, NULL, &dbuf));
    vkGetBufferMemoryRequirements(dev, dbuf, &req);
    ai.allocationSize = req.size; ai.memoryTypeIndex = memtype(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    CK(vkAllocateMemory(dev, &ai, NULL, &dmem)); CK(vkBindBufferMemory(dev, dbuf, dmem, 0));
    printf("device %d: %s, queue family %u, import alignment %zu, %d x %.2f MiB per commit, %d reps\n",
           want, pp.deviceName, fam, imp_align, G, mib, R);
    uint8_t **img = calloc(G, sizeof *img);
    size_t isz = (B + imp_align - 1) / imp_align * imp_align;
    const char *names[3] = {"staged", "import", "importp"};
    double best[3] = {1e30, 1e30, 1e30}, med[3][256];
    double bi_[3] = {0}, bm_[3] = {0}, bw_[3] = {0}, br_[3] = {0};
    for (int r = 0; r < R + 1; r++)
        for (int mode = 0; mode < 3; mode++) {
            for (int g = 0; g < G; g++) { if (posix_memalign((void **)&img[g], imp_align, isz)) return 1; memset(img[g], (r + g) & 0xff, isz); }
            t_import = t_memcpy = t_wait = t_release = 0;
            double t0 = now();
            for (int g = 0; g < G; g++)
                if (mode == 0) add_staged(img[g], B, (size_t)g * B);
                else add_import(img[g], B, (size_t)g * B, mode == 2);
            finish();
            double ms = now() - t0;
            for (int g = 0; g < G; g++) free(img[g]);
            if (r == 0) continue;   /* warm-up */
            med[mode][r - 1 < 256 ? r - 1 : 255] = ms;
            if (ms < best[mode]) { best[mode] = ms; bi_[mode] = t_import; bm_[mode] = t_memcpy; bw_[mode] = t_wait; br_[mode] = t_release; }
        }
    for (int mode = 0; mode < 3; mode++) {
        int k = R < 256 ? R : 256;
        for (int i = 1; i < k; i++) for (int j = i; j > 0 && med[mode][j] < med[mode][j - 1]; j--) { double t = med[mode][j]; med[mode][j] = med[mode][j - 1]; med[mode][j - 1] = t; }
        double m = med[mode][k / 2];
        printf("%-8s best %8.2f ms (%5.2f GB/s)  median %8.2f ms (%5.2f GB/s)  [best: import %.2f memcpy %.2f wait %.2f release %.2f ms]\n",
               names[mode], best[mode], total / best[mode] / 1e6, m, total / m / 1e6, bi_[mode], bm_[mode], bw_[mode], br_[mode]);
    }
    return 0;
}
