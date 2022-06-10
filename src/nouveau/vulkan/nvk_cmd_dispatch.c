#include "nvk_cmd_buffer.h"
#include "nvk_descriptor_set.h"
#include "nvk_device.h"
#include "nvk_physical_device.h"
#include "nvk_pipeline.h"

#include "classes/cla0b5.h"

#include "nvk_cla0c0.h"
#include "cla1c0.h"
#include "nvk_clc3c0.h"

#include "drf.h"
#include "cla0c0qmd.h"
#include "clc0c0qmd.h"
#include "clc3c0qmd.h"

#define NVA0C0_QMDV00_06_VAL_SET(p,a...) NVVAL_MW_SET((p), NVA0C0, QMDV00_06, ##a)
#define NVA0C0_QMDV00_06_DEF_SET(p,a...) NVDEF_MW_SET((p), NVA0C0, QMDV00_06, ##a)
#define NVC0C0_QMDV02_01_VAL_SET(p,a...) NVVAL_MW_SET((p), NVC0C0, QMDV02_01, ##a)
#define NVC0C0_QMDV02_01_DEF_SET(p,a...) NVDEF_MW_SET((p), NVC0C0, QMDV02_01, ##a)
#define NVC3C0_QMDV02_02_VAL_SET(p,a...) NVVAL_MW_SET((p), NVC3C0, QMDV02_02, ##a)
#define NVC3C0_QMDV02_02_DEF_SET(p,a...) NVDEF_MW_SET((p), NVC3C0, QMDV02_02, ##a)

void
nvk_cmd_buffer_begin_compute(struct nvk_cmd_buffer *cmd,
                             const VkCommandBufferBeginInfo *pBeginInfo)
{
   struct nvk_device *dev = (struct nvk_device *)cmd->vk.base.device;
   struct nvk_physical_device *pdev = dev->pdev;

   if (pdev->dev->chipset < 0xe0)
      return;

   nouveau_ws_push_ref(cmd->push, dev->tls, NOUVEAU_WS_BO_RDWR);
   P_MTHD(cmd->push, NVA0C0, SET_SHADER_LOCAL_MEMORY_A);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_A(cmd->push, dev->tls->offset >> 32);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_B(cmd->push, dev->tls->offset & 0xffffffff);

   /* No idea why there are 2. Divide size by 2 to be safe.
    * Actually this might be per-MP TEMP size and looks like I'm only using
    * 2 MPs instead of all 8.
    */
   uint64_t temp_size = dev->tls->size / dev->pdev->dev->mp_count;
   P_MTHD(cmd->push, NVA0C0, SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A(cmd->push, temp_size >> 32);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B(cmd->push, temp_size & ~0x7fff);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C(cmd->push, 0xff);

   if (pdev->compute_class < VOLTA_COMPUTE_A) {
      P_MTHD(cmd->push, NVA0C0, SET_SHADER_LOCAL_MEMORY_THROTTLED_A);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_A(cmd->push, temp_size >> 32);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_B(cmd->push, temp_size & ~0x7fff);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_THROTTLED_C(cmd->push, 0xff);
   }

   if (pdev->compute_class < VOLTA_COMPUTE_A) {
      P_MTHD(cmd->push, NVA0C0, SET_SHADER_LOCAL_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_LOCAL_MEMORY_WINDOW(cmd->push, 0xff << 24);

      P_MTHD(cmd->push, NVA0C0, SET_SHADER_SHARED_MEMORY_WINDOW);
      P_NVA0C0_SET_SHADER_SHARED_MEMORY_WINDOW(cmd->push, 0xfe << 24);

      // TODO CODE_ADDRESS_HIGH
   } else {
      uint64_t temp = 0xfeULL << 24;

      P_MTHD(cmd->push, NVC3C0, SET_SHADER_SHARED_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_A(cmd->push, temp >> 32);
      P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_B(cmd->push, temp & 0xffffffff);

      temp = 0xffULL << 24;
      P_MTHD(cmd->push, NVC3C0, SET_SHADER_LOCAL_MEMORY_WINDOW_A);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A(cmd->push, temp >> 32);
      P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B(cmd->push, temp & 0xffffffff);
   }

   P_MTHD(cmd->push, NVA0C0, SET_SPA_VERSION);
   P_NVA0C0_SET_SPA_VERSION(cmd->push, { .major = pdev->compute_class >= KEPLER_COMPUTE_B ? 0x4 : 0x3 });

   P_MTHD(cmd->push, NVA0C0, INVALIDATE_SHADER_CACHES_NO_WFI);
   P_NVA0C0_INVALIDATE_SHADER_CACHES_NO_WFI(cmd->push, { .constant = CONSTANT_TRUE });
}

static void
gv100_compute_setup_launch_desc(uint32_t *qmd,
                                uint32_t x, uint32_t y, uint32_t z)
{
   NVC3C0_QMDV02_02_VAL_SET(qmd, CTA_RASTER_WIDTH, x);
   NVC3C0_QMDV02_02_VAL_SET(qmd, CTA_RASTER_HEIGHT, y);
   NVC3C0_QMDV02_02_VAL_SET(qmd, CTA_RASTER_DEPTH, z);
}

static inline void
gp100_cp_launch_desc_set_cb(uint32_t *qmd, unsigned index,
                            uint32_t size, uint64_t address)
{
   NVC0C0_QMDV02_01_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_LOWER, index, address);
   NVC0C0_QMDV02_01_VAL_SET(qmd, CONSTANT_BUFFER_ADDR_UPPER, index, address >> 32);
   NVC0C0_QMDV02_01_VAL_SET(qmd, CONSTANT_BUFFER_SIZE_SHIFTED4, index,
                                 DIV_ROUND_UP(size, 16));
   NVC0C0_QMDV02_01_DEF_SET(qmd, CONSTANT_BUFFER_VALID, index, TRUE);
}

VKAPI_ATTR void VKAPI_CALL
nvk_CmdDispatch(VkCommandBuffer commandBuffer,
                uint32_t groupCountX,
                uint32_t groupCountY,
                uint32_t groupCountZ)
{
   VK_FROM_HANDLE(nvk_cmd_buffer, cmd, commandBuffer);
   const struct nvk_compute_pipeline *pipeline = cmd->state.cs.pipeline;
   const struct nvk_shader *shader =
      &pipeline->base.shaders[MESA_SHADER_COMPUTE];
   struct nvk_descriptor_state *desc = &cmd->state.cs.descriptors;

   desc->root.cs.block_size[0] = shader->cp.block_size[0];
   desc->root.cs.block_size[1] = shader->cp.block_size[1];
   desc->root.cs.block_size[2] = shader->cp.block_size[2];
   desc->root.cs.grid_size[0] = groupCountX;
   desc->root.cs.grid_size[1] = groupCountY;
   desc->root.cs.grid_size[2] = groupCountZ;

   uint32_t root_table_size = sizeof(desc->root);
   void *root_table_map;
   uint64_t root_table_addr;
   if (!nvk_cmd_buffer_upload_alloc(cmd, root_table_size, &root_table_addr,
                                    &root_table_map))
      return; /* TODO: Error */

   P_MTHD(cmd->push, NVA0C0, OFFSET_OUT_UPPER);
   P_NVA0C0_OFFSET_OUT_UPPER(cmd->push, root_table_addr >> 32);
   P_NVA0C0_OFFSET_OUT(cmd->push, root_table_addr & 0xffffffff);
   P_MTHD(cmd->push, NVA0C0, LINE_LENGTH_IN);
   P_NVA0C0_LINE_LENGTH_IN(cmd->push, root_table_size);
   P_NVA0C0_LINE_COUNT(cmd->push, 0x1);

   P_1INC(cmd->push, NVA0C0, LAUNCH_DMA);
   P_NVA0C0_LAUNCH_DMA(cmd->push,
                       { .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
                         .sysmembar_disable = SYSMEMBAR_DISABLE_TRUE });
   P_INLINE_ARRAY(cmd->push, (uint32_t *)&desc->root, root_table_size / 4);

   uint32_t *qmd;
   uint64_t qmd_addr;
   if (!nvk_cmd_buffer_upload_alloc(cmd, 512, &qmd_addr, (void **)&qmd))
      return; /* TODO: Error */

   memcpy(qmd, pipeline->qmd_template, 256);
   gv100_compute_setup_launch_desc(qmd, groupCountX, groupCountY, groupCountZ);

   gp100_cp_launch_desc_set_cb(qmd, 1, 256, root_table_addr);

   P_MTHD(cmd->push, NVA0C0, INVALIDATE_SHADER_CACHES_NO_WFI);
   P_NVA0C0_INVALIDATE_SHADER_CACHES_NO_WFI(cmd->push, { .constant = CONSTANT_TRUE });

   P_MTHD(cmd->push, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(cmd->push, qmd_addr >> 8);
   P_IMMD(cmd->push, NVA0C0, SEND_SIGNALING_PCAS_B,
          { .invalidate = INVALIDATE_TRUE,
            .schedule = SCHEDULE_TRUE });
}
