#ifndef SPARKSERVE_DS4_GLM53_API_H
#define SPARKSERVE_DS4_GLM53_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKSERVE_DS4_GLM53_ABI_VERSION 1u

typedef struct sparkserve_ds4_glm53_engine sparkserve_ds4_glm53_engine;
typedef struct sparkserve_ds4_glm53_session sparkserve_ds4_glm53_session;

typedef struct {
    int32_t code;
    const char *message;
} sparkserve_ds4_glm53_status;

enum {
    SPARKSERVE_DS4_GLM53_WARM_WEIGHTS = 1u << 0,
    SPARKSERVE_DS4_GLM53_QUALITY = 1u << 1,
    SPARKSERVE_DS4_GLM53_SSD_STREAMING = 1u << 2,
    SPARKSERVE_DS4_GLM53_SSD_STREAMING_COLD = 1u << 3,
    SPARKSERVE_DS4_GLM53_MTP = 1u << 4,
};

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    const char *model_path;
    uint32_t context_size;
    uint32_t prefill_chunk;
    uint32_t threads;
    uint32_t power_percent;
    uint32_t flags;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_cache_experts;
    uint32_t reserved;
} sparkserve_ds4_glm53_config;

typedef struct {
    const char *role;
    const char *content;
} sparkserve_ds4_glm53_message;

typedef struct {
    int32_t *data;
    uint32_t len;
    uint32_t reserved;
} sparkserve_ds4_glm53_tokens;

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_open(
    sparkserve_ds4_glm53_engine **out,
    const sparkserve_ds4_glm53_config *config);
void sparkserve_ds4_glm53_close(sparkserve_ds4_glm53_engine *engine);
const char *sparkserve_ds4_glm53_model_name(
    const sparkserve_ds4_glm53_engine *engine);
int32_t sparkserve_ds4_glm53_eos_token(
    const sparkserve_ds4_glm53_engine *engine);
uint32_t sparkserve_ds4_glm53_is_stop_token(
    const sparkserve_ds4_glm53_engine *engine,
    int32_t token);

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_encode_messages(
    sparkserve_ds4_glm53_engine *engine,
    const sparkserve_ds4_glm53_message *messages,
    uint32_t message_count,
    uint32_t enable_thinking,
    sparkserve_ds4_glm53_tokens *out);
sparkserve_ds4_glm53_status sparkserve_ds4_glm53_encode_text(
    sparkserve_ds4_glm53_engine *engine,
    const char *text,
    sparkserve_ds4_glm53_tokens *out);
void sparkserve_ds4_glm53_tokens_free(
    sparkserve_ds4_glm53_tokens *tokens);

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_create(
    sparkserve_ds4_glm53_session **out,
    sparkserve_ds4_glm53_engine *engine);
void sparkserve_ds4_glm53_session_free(
    sparkserve_ds4_glm53_session *session);
sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_sync(
    sparkserve_ds4_glm53_session *session,
    const int32_t *tokens,
    uint32_t token_count);
int32_t sparkserve_ds4_glm53_session_sample(
    sparkserve_ds4_glm53_session *session,
    float temperature,
    int32_t top_k,
    float top_p,
    float min_p,
    uint64_t *rng);
sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_eval(
    sparkserve_ds4_glm53_session *session,
    int32_t token);
int32_t sparkserve_ds4_glm53_session_position(
    const sparkserve_ds4_glm53_session *session);

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_token_piece(
    sparkserve_ds4_glm53_engine *engine,
    int32_t token,
    uint8_t **out,
    size_t *out_len);
void sparkserve_ds4_glm53_text_free(uint8_t *text);

#ifdef __cplusplus
}
#endif

#endif
