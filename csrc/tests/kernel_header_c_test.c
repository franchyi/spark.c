#include "sparkserve/fabric_api.h"
#include "sparkserve/ggml_quant_api.h"
#include "sparkserve/glm_dsa_api.h"
#include "sparkserve/glm_kda_api.h"
#include "sparkserve/kernel_api.h"
#include "sparkserve/glm_mqa_api.h"
#include "sparkserve/glm_sparse_mla_api.h"
#include "sparkserve/qwen_expert_pack_api.h"
#include "sparkserve/qwen_gdn_aux_api.h"
#include "sparkserve/qwen_qsa_block_api.h"
#include "sparkserve/qwen_ple_block_api.h"
#include "sparkserve/qwen_decode_glue_api.h"

#include <stddef.h>

_Static_assert(SPARKSERVE_KERNEL_ABI_VERSION == 1u, "unexpected ABI version");
_Static_assert(sizeof(SparkServeDenseNvfp4Plan) == 80u,
               "C plan layout drifted");
_Static_assert(sizeof(SparkServeGroupedNvfp4Plan) == 80u,
               "C grouped plan layout drifted");
_Static_assert(sizeof(SparkServeGroupedNvfp4WeightView) == 32u,
               "C grouped weight view layout drifted");
_Static_assert(sizeof(SparkServeGroupedNvfp4Args) == 224u,
               "C grouped arguments layout drifted");
_Static_assert(sizeof(SparkServeSiluNvfp4Plan) == 48u,
               "C fused SiLU plan layout drifted");
_Static_assert(sizeof(SparkServeSiluNvfp4Args) == 128u,
               "C fused SiLU arguments layout drifted");
_Static_assert(sizeof(SparkServeSegmentedSiluNvfp4Plan) == 56u,
               "C segmented SiLU plan layout drifted");
_Static_assert(sizeof(SparkServeSegmentedSiluNvfp4Args) == 152u,
               "C segmented SiLU arguments layout drifted");
_Static_assert(sizeof(SparkServeSegmentedNvfp4QuantizePlan) == 56u,
               "C segmented quantize plan layout drifted");
_Static_assert(sizeof(SparkServeSegmentedNvfp4QuantizeArgs) == 152u,
               "C segmented quantize arguments layout drifted");
_Static_assert(sizeof(SparkServeMoeRoutePlan) == 40u,
               "C MoE route plan layout drifted");
_Static_assert(sizeof(SparkServeMoeRouteArgs) == 128u,
               "C MoE route arguments layout drifted");
_Static_assert(sizeof(SparkServeMoeGatePlan) == 48u,
               "C MoE gate plan layout drifted");
_Static_assert(sizeof(SparkServeMoeGateArgs) == 112u,
               "C MoE gate arguments layout drifted");
_Static_assert(sizeof(SparkServeSharedExpertPlan) == 48u,
               "C shared expert plan layout drifted");
_Static_assert(sizeof(SparkServeSharedExpertArgs) == 136u,
               "C shared expert arguments layout drifted");
_Static_assert(sizeof(SparkServeQsaExpandPlan) == 40u,
               "C QSA expansion plan layout drifted");
_Static_assert(sizeof(SparkServeQsaExpandArgs) == 88u,
               "C QSA expansion arguments layout drifted");
_Static_assert(sizeof(SparkServeQsaScorePlan) == 48u,
               "C QSA score plan layout drifted");
_Static_assert(sizeof(SparkServeQsaScoreArgs) == 112u,
               "C QSA score arguments layout drifted");
_Static_assert(sizeof(SparkServeQsaKvPackPlan) == 48u,
               "C QSA K/V-pack plan layout drifted");
_Static_assert(sizeof(SparkServeQsaKvPackArgs) == 136u,
               "C QSA K/V-pack arguments layout drifted");
_Static_assert(sizeof(SparkServeQsaDecodePlan) == 56u,
               "C QSA decode plan layout drifted");
_Static_assert(sizeof(SparkServeQsaDecodeArgs) == 144u,
               "C QSA decode arguments layout drifted");
_Static_assert(sizeof(SparkServeGdnBlockPlan) == 48u,
               "C GDN block plan layout drifted");
_Static_assert(sizeof(SparkServeGdnBlockArgs) == 216u,
               "C GDN block arguments layout drifted");
_Static_assert(sizeof(SparkServeQwenQsaProjectArgs) == 168u,
               "C Qwen QSA projection arguments layout drifted");
_Static_assert(sizeof(SparkServeQwenQsaFinishArgs) == 72u,
               "C Qwen QSA finish arguments layout drifted");
_Static_assert(sizeof(SparkServeQwenPleBlockArgs) == 144u,
               "C Qwen PLE block arguments layout drifted");
_Static_assert(sizeof(SparkServeQwenDecodeGlueArgs) == 32u,
               "C Qwen decode glue arguments layout drifted");
_Static_assert(sizeof(SparkServeQwenLmHeadArgs) == 56u,
               "C Qwen LM-head arguments layout drifted");
_Static_assert(sizeof(SparkServeCoherentRegionConfig) == 48u,
               "C coherent region config layout drifted");
_Static_assert(sizeof(SparkServeCoherentRegionView) == 80u,
               "C coherent region view layout drifted");
_Static_assert(sizeof(SparkServeGgmlQuantDenseArgs) == 88u,
               "C GGML quant dense arguments layout drifted");
_Static_assert(sizeof(SparkServeGgmlQuantRoutedArgs) == 120u,
               "C GGML quant routed arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKdaArgs) == 104u,
               "C GLM KDA arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKdaConvArgs) == 80u,
               "C GLM KDA conv arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKdaPrepareArgs) == 128u,
               "C GLM KDA prepare arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKdaGateArgs) == 80u,
               "C GLM KDA gate arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKPoolCompressArgs) == 104u,
               "C GLM KPool compress arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmKPoolDecodeArgs) == 168u,
               "C GLM KPool decode arguments layout drifted");
_Static_assert(sizeof(SparkServeGlmIndexerPrepArgs) == 112u,
               "C GLM indexer prep arguments layout drifted");

int main(void) {
  _Static_assert(sizeof(SparkServeGlmPagedMqaArgs) == 136,
                 "GLM paged-MQA ABI size changed");
  _Static_assert(SPARKSERVE_GLM_MQA_SCHEDULE_WORDS == 98,
                 "GB10 paged-MQA metadata size changed");
  _Static_assert(sizeof(SparkServeGlmSparseMlaPackKvArgs) == 72,
                 "GLM sparse-MLA KV-pack ABI size changed");
  _Static_assert(sizeof(SparkServeGlmSparseMlaPadQueryArgs) == 56,
                 "GLM sparse-MLA query-pad ABI size changed");
  _Static_assert(sizeof(SparkServeGlmSparseMlaDecodeArgs) == 224,
                 "GLM sparse-MLA decode ABI size changed");
  SparkServeDenseNvfp4Plan plan = {0};
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  plan.output_dtype = SPARKSERVE_DTYPE_BF16;
  return plan.output_dtype == SPARKSERVE_DTYPE_BF16 ? 0 : 1;
}
