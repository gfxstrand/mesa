#ifndef NVK_CMD_BUFFER_H
#define NVK_CMD_BUFFER_H 1

#include "nvk_private.h"

#include "nouveau_push.h"

#include "vulkan/runtime/vk_command_buffer.h"
#include "vulkan/runtime/vk_command_pool.h"

#define NVK_CMD_BUF_SIZE 64*1024

struct nvk_cmd_pool {
   struct vk_command_pool vk;
   struct list_head cmd_buffers;
   struct list_head free_cmd_buffers;

   struct nvk_device *dev;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(nvk_cmd_pool, vk.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)

/** Root descriptor table.  This gets pushed to the GPU directly */
struct nvk_root_descriptor_table {
   union {
      struct {
         uint32_t block_size[3];
         uint32_t grid_size[3];
         uint32_t _pad[2];
      } cs;
   };

   /* Client push constants */
   uint8_t push[128];

   /* Descriptor set base addresses */
   uint64_t sets[NVK_MAX_SETS];

   /* TODO: Dynamic buffer bindings */
};

struct nvk_descriptor_state {
   struct nvk_root_descriptor_table root;
   struct nvk_descriptor_set *sets[NVK_MAX_SETS];
   uint32_t sets_dirty;
};

struct nvk_compute_state {
   struct nvk_compute_pipeline *pipeline;
   struct nvk_descriptor_state descriptors;
};

struct nvk_cmd_buffer_upload {
   uint8_t *map;
   unsigned offset;
   uint64_t size;
   struct nouveau_ws_bo *upload_bo;
   struct list_head list;
};

struct nvk_cmd_buffer {
   struct vk_command_buffer vk;

   struct nvk_cmd_pool *pool;
   struct list_head pool_link;

   struct {
      struct nvk_compute_state cs;
   } state;

   struct nouveau_ws_push *push;
   bool reset_on_submit;

   struct nvk_cmd_buffer_upload upload;

   VkResult record_result;
};

VkResult nvk_reset_cmd_buffer(struct nvk_cmd_buffer *cmd_buffer);
void nvk_cmd_buffer_begin_compute(struct nvk_cmd_buffer *cmd,
                                  const VkCommandBufferBeginInfo *pBeginInfo);


VK_DEFINE_HANDLE_CASTS(nvk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

static inline struct nvk_descriptor_state *
nvk_get_descriptors_state(struct nvk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   switch (bind_point) {
   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmd->state.cs.descriptors;
   default:
      unreachable("Unhandled bind point");
   }
};

bool
nvk_cmd_buffer_upload_alloc(struct nvk_cmd_buffer *cmd_buffer, unsigned size,
                            uint64_t *addr, void **ptr);

#define SUBC_CP(m) 1, (m)
#define NVE4_CP(n) SUBC_CP(NVE4_COMPUTE_##n)
#define SUBC_M2MF(m) 2, (m)
#define SUBC_P2MF(m) 2, (m)
#define NVC0_M2MF(n) SUBC_M2MF(NVC0_M2MF_##n)
#define NV01_SUBCHAN_OBJECT                                      0x00000000

/* legacy pushes */
static inline void
LPUSH_DATA(struct nvk_cmd_buffer *cmd, uint32_t data)
{
   *cmd->push->map++ = data;
}

static inline void
LPUSH_DATAh(struct nvk_cmd_buffer *cmd, uint64_t data)
{
   *cmd->push->map++ = (uint32_t)(data >> 32);
}

static inline void
LPUSH_DATAp(struct nvk_cmd_buffer *cmd, const void *data, uint32_t size)
{
   memcpy(cmd->push->map, data, size * 4);
   cmd->push->map += size;
}

static inline void
BEGIN_NVC0(struct nvk_cmd_buffer *cmd, int subc, int mthd, unsigned size)
{
   LPUSH_DATA (cmd, NVC0_FIFO_PKHDR_SQ(subc, mthd, size));
}

static inline void
BEGIN_1IC0(struct nvk_cmd_buffer *cmd, int subc, int mthd, unsigned size)
{
   LPUSH_DATA (cmd, NVC0_FIFO_PKHDR_1I(subc, mthd, size));
}

#endif
