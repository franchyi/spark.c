#include "sparkserve/fabric_api.h"
#include "sparkserve/kernel_api.h"

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
_Static_assert(sizeof(SparkServeCoherentRegionConfig) == 48u,
               "C coherent region config layout drifted");
_Static_assert(sizeof(SparkServeCoherentRegionView) == 80u,
               "C coherent region view layout drifted");

int main(void) {
  SparkServeDenseNvfp4Plan plan = {0};
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = SPARKSERVE_KERNEL_ABI_VERSION;
  plan.output_dtype = SPARKSERVE_DTYPE_BF16;
  return plan.output_dtype == SPARKSERVE_DTYPE_BF16 ? 0 : 1;
}
