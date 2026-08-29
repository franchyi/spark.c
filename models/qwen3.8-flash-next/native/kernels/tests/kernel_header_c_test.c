#include "flash/fabric_api.h"
#include "flash/kernel_api.h"
#include "flash/qwen_expert_pack_api.h"
#include "flash/qwen_gdn_aux_api.h"
#include "flash/qwen_qsa_block_api.h"
#include "flash/qwen_ple_block_api.h"
#include "flash/qwen_decode_glue_api.h"

#include <stddef.h>

_Static_assert(FLASH_KERNEL_ABI_VERSION == 1u, "unexpected ABI version");
_Static_assert(sizeof(FlashDenseNvfp4Plan) == 80u,
               "C plan layout drifted");
_Static_assert(sizeof(FlashGroupedNvfp4Plan) == 80u,
               "C grouped plan layout drifted");
_Static_assert(sizeof(FlashGroupedNvfp4WeightView) == 32u,
               "C grouped weight view layout drifted");
_Static_assert(sizeof(FlashGroupedNvfp4Args) == 224u,
               "C grouped arguments layout drifted");
_Static_assert(sizeof(FlashSiluNvfp4Plan) == 48u,
               "C fused SiLU plan layout drifted");
_Static_assert(sizeof(FlashSiluNvfp4Args) == 128u,
               "C fused SiLU arguments layout drifted");
_Static_assert(sizeof(FlashSegmentedSiluNvfp4Plan) == 56u,
               "C segmented SiLU plan layout drifted");
_Static_assert(sizeof(FlashSegmentedSiluNvfp4Args) == 152u,
               "C segmented SiLU arguments layout drifted");
_Static_assert(sizeof(FlashSegmentedNvfp4QuantizePlan) == 56u,
               "C segmented quantize plan layout drifted");
_Static_assert(sizeof(FlashSegmentedNvfp4QuantizeArgs) == 152u,
               "C segmented quantize arguments layout drifted");
_Static_assert(sizeof(FlashMoeRoutePlan) == 40u,
               "C MoE route plan layout drifted");
_Static_assert(sizeof(FlashMoeRouteArgs) == 128u,
               "C MoE route arguments layout drifted");
_Static_assert(sizeof(FlashMoeGatePlan) == 48u,
               "C MoE gate plan layout drifted");
_Static_assert(sizeof(FlashMoeGateArgs) == 112u,
               "C MoE gate arguments layout drifted");
_Static_assert(sizeof(FlashSharedExpertPlan) == 48u,
               "C shared expert plan layout drifted");
_Static_assert(sizeof(FlashSharedExpertArgs) == 136u,
               "C shared expert arguments layout drifted");
_Static_assert(sizeof(FlashQsaExpandPlan) == 40u,
               "C QSA expansion plan layout drifted");
_Static_assert(sizeof(FlashQsaExpandArgs) == 88u,
               "C QSA expansion arguments layout drifted");
_Static_assert(sizeof(FlashQsaScorePlan) == 48u,
               "C QSA score plan layout drifted");
_Static_assert(sizeof(FlashQsaScoreArgs) == 112u,
               "C QSA score arguments layout drifted");
_Static_assert(sizeof(FlashQsaKvPackPlan) == 48u,
               "C QSA K/V-pack plan layout drifted");
_Static_assert(sizeof(FlashQsaKvPackArgs) == 136u,
               "C QSA K/V-pack arguments layout drifted");
_Static_assert(sizeof(FlashQsaDecodePlan) == 56u,
               "C QSA decode plan layout drifted");
_Static_assert(sizeof(FlashQsaDecodeArgs) == 144u,
               "C QSA decode arguments layout drifted");
_Static_assert(sizeof(FlashGdnBlockPlan) == 48u,
               "C GDN block plan layout drifted");
_Static_assert(sizeof(FlashGdnBlockArgs) == 216u,
               "C GDN block arguments layout drifted");
_Static_assert(sizeof(FlashQwenQsaProjectArgs) == 168u,
               "C Qwen QSA projection arguments layout drifted");
_Static_assert(sizeof(FlashQwenQsaFinishArgs) == 72u,
               "C Qwen QSA finish arguments layout drifted");
_Static_assert(sizeof(FlashQwenPleBlockArgs) == 144u,
               "C Qwen PLE block arguments layout drifted");
_Static_assert(sizeof(FlashQwenDecodeGlueArgs) == 32u,
               "C Qwen decode glue arguments layout drifted");
_Static_assert(sizeof(FlashQwenLmHeadArgs) == 56u,
               "C Qwen LM-head arguments layout drifted");
_Static_assert(sizeof(FlashCoherentRegionConfig) == 48u,
               "C coherent region config layout drifted");
_Static_assert(sizeof(FlashCoherentRegionView) == 80u,
               "C coherent region view layout drifted");
int main(void) {
  FlashDenseNvfp4Plan plan = {0};
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = FLASH_KERNEL_ABI_VERSION;
  plan.output_dtype = FLASH_DTYPE_BF16;
  return plan.output_dtype == FLASH_DTYPE_BF16 ? 0 : 1;
}
