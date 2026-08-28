#include "sparkserve/ds4_glm53_api.h"

#include "ds4.h"

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct sparkserve_ds4_glm53_engine {
    ds4_engine *inner;
    int context_size;
};

struct sparkserve_ds4_glm53_session {
    ds4_session *inner;
    sparkserve_ds4_glm53_engine *engine;
};

static _Thread_local char sparkserve_ds4_glm53_error[512];

static sparkserve_ds4_glm53_status status_ok(void) {
    sparkserve_ds4_glm53_status status = {0, NULL};
    return status;
}

static sparkserve_ds4_glm53_status status_error(const char *message) {
    if (!message) message = "unknown ds4 GLM-5.3 error";
    snprintf(sparkserve_ds4_glm53_error,
             sizeof(sparkserve_ds4_glm53_error), "%s", message);
    sparkserve_ds4_glm53_status status = {
        .code = -1,
        .message = sparkserve_ds4_glm53_error,
    };
    return status;
}

static bool valid_config(const sparkserve_ds4_glm53_config *config) {
    return config &&
           config->struct_size == sizeof(*config) &&
           config->abi_version == SPARKSERVE_DS4_GLM53_ABI_VERSION &&
           config->model_path && config->model_path[0] &&
           config->context_size > 0 && config->context_size <= INT_MAX &&
           config->threads <= INT_MAX &&
           config->power_percent <= 100;
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_open(
        sparkserve_ds4_glm53_engine **out,
        const sparkserve_ds4_glm53_config *config) {
    if (!out) return status_error("GLM-5.3 engine output pointer is null");
    *out = NULL;
    if (!valid_config(config)) {
        return status_error("invalid GLM-5.3 engine configuration");
    }

    sparkserve_ds4_glm53_engine *engine = calloc(1, sizeof(*engine));
    if (!engine) return status_error("cannot allocate GLM-5.3 engine wrapper");

    ds4_engine_options options = {
        .model_path = config->model_path,
        .backend = DS4_BACKEND_CUDA,
        .n_threads = (int)config->threads,
        .context_size = (int)config->context_size,
        .prefill_chunk = config->prefill_chunk,
        .mtp_draft_tokens = 1,
        .mtp_margin = 3.0f,
        .power_percent = (int)config->power_percent,
        .ssd_streaming_cache_experts = config->ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = config->ssd_streaming_cache_bytes,
        .warm_weights = (config->flags & SPARKSERVE_DS4_GLM53_WARM_WEIGHTS) != 0,
        .quality = (config->flags & SPARKSERVE_DS4_GLM53_QUALITY) != 0,
        .glm_mtp = (config->flags & SPARKSERVE_DS4_GLM53_MTP) != 0,
        .ssd_streaming =
            (config->flags & SPARKSERVE_DS4_GLM53_SSD_STREAMING) != 0,
        .ssd_streaming_cold =
            (config->flags & SPARKSERVE_DS4_GLM53_SSD_STREAMING_COLD) != 0,
        .placement_ctx_hint = (int)config->context_size,
        .placement_session_count_hint = 1,
    };
    if (ds4_engine_open(&engine->inner, &options) != 0) {
        free(engine);
        return status_error("ds4 could not open the GLM-5.3 model");
    }
    if (!ds4_engine_is_glm53(engine->inner)) {
        ds4_engine_close(engine->inner);
        free(engine);
        return status_error("the loaded GGUF is not a ds4 GLM-5.3 model");
    }
    engine->context_size = (int)config->context_size;
    *out = engine;
    return status_ok();
}

void sparkserve_ds4_glm53_close(sparkserve_ds4_glm53_engine *engine) {
    if (!engine) return;
    ds4_engine_close(engine->inner);
    free(engine);
}

const char *sparkserve_ds4_glm53_model_name(
        const sparkserve_ds4_glm53_engine *engine) {
    if (!engine) return NULL;
    return ds4_engine_model_name(engine->inner);
}

int32_t sparkserve_ds4_glm53_eos_token(
        const sparkserve_ds4_glm53_engine *engine) {
    if (!engine) return -1;
    return ds4_token_eos(engine->inner);
}

uint32_t sparkserve_ds4_glm53_is_stop_token(
        const sparkserve_ds4_glm53_engine *engine,
        int32_t token) {
    return engine && ds4_token_is_stop(engine->inner, token) ? 1u : 0u;
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_encode_messages(
        sparkserve_ds4_glm53_engine *engine,
        const sparkserve_ds4_glm53_message *messages,
        uint32_t message_count,
        uint32_t enable_thinking,
        sparkserve_ds4_glm53_tokens *out) {
    if (!engine || !out || (!messages && message_count != 0)) {
        return status_error("invalid GLM-5.3 message encoding arguments");
    }
    memset(out, 0, sizeof(*out));

    ds4_tokens encoded = {0};
    ds4_chat_begin(engine->inner, &encoded);
    for (uint32_t i = 0; i < message_count; i++) {
        if (!messages[i].role || !messages[i].content) {
            ds4_tokens_free(&encoded);
            return status_error("GLM-5.3 chat message has a null role or content");
        }
        ds4_chat_append_message(engine->inner, &encoded,
                                messages[i].role, messages[i].content);
    }
    ds4_chat_append_assistant_prefix(
        engine->inner, &encoded,
        enable_thinking ? DS4_THINK_HIGH : DS4_THINK_NONE);

    if (encoded.len < 0 || (uint64_t)encoded.len > UINT32_MAX) {
        ds4_tokens_free(&encoded);
        return status_error("GLM-5.3 prompt token count exceeds the adapter ABI");
    }
    if (encoded.len > 0) {
        out->data = malloc((size_t)encoded.len * sizeof(*out->data));
        if (!out->data) {
            ds4_tokens_free(&encoded);
            return status_error("cannot allocate GLM-5.3 prompt token buffer");
        }
        for (int i = 0; i < encoded.len; i++) out->data[i] = encoded.v[i];
    }
    out->len = (uint32_t)encoded.len;
    ds4_tokens_free(&encoded);
    return status_ok();
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_encode_text(
        sparkserve_ds4_glm53_engine *engine,
        const char *text,
        sparkserve_ds4_glm53_tokens *out) {
    if (!engine || !text || !out) {
        return status_error("invalid GLM-5.3 text encoding arguments");
    }
    memset(out, 0, sizeof(*out));
    ds4_tokens encoded = {0};
    ds4_tokenize_text(engine->inner, text, &encoded);
    if (encoded.len < 0 || (uint64_t)encoded.len > UINT32_MAX) {
        ds4_tokens_free(&encoded);
        return status_error("GLM-5.3 text token count exceeds the adapter ABI");
    }
    if (encoded.len > 0) {
        out->data = malloc((size_t)encoded.len * sizeof(*out->data));
        if (!out->data) {
            ds4_tokens_free(&encoded);
            return status_error("cannot allocate GLM-5.3 text token buffer");
        }
        for (int i = 0; i < encoded.len; i++) out->data[i] = encoded.v[i];
    }
    out->len = (uint32_t)encoded.len;
    ds4_tokens_free(&encoded);
    return status_ok();
}

void sparkserve_ds4_glm53_tokens_free(
        sparkserve_ds4_glm53_tokens *tokens) {
    if (!tokens) return;
    free(tokens->data);
    memset(tokens, 0, sizeof(*tokens));
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_create(
        sparkserve_ds4_glm53_session **out,
        sparkserve_ds4_glm53_engine *engine) {
    if (!out || !engine) return status_error("invalid GLM-5.3 session arguments");
    *out = NULL;
    sparkserve_ds4_glm53_session *session = calloc(1, sizeof(*session));
    if (!session) return status_error("cannot allocate GLM-5.3 session wrapper");
    if (ds4_session_create(&session->inner, engine->inner,
                           engine->context_size) != 0) {
        free(session);
        return status_error("ds4 could not create a GLM-5.3 session");
    }
    session->engine = engine;
    *out = session;
    return status_ok();
}

void sparkserve_ds4_glm53_session_free(
        sparkserve_ds4_glm53_session *session) {
    if (!session) return;
    ds4_session_free(session->inner);
    free(session);
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_sync(
        sparkserve_ds4_glm53_session *session,
        const int32_t *tokens,
        uint32_t token_count) {
    if (!session || (!tokens && token_count != 0) || token_count > INT_MAX) {
        return status_error("invalid GLM-5.3 session sync arguments");
    }
    ds4_tokens prompt = {
        .v = (int *)(uintptr_t)tokens,
        .len = (int)token_count,
        .cap = (int)token_count,
    };
    char error[512] = {0};
    int rc = ds4_session_sync(session->inner, &prompt, error, sizeof(error));
    if (rc != 0) {
        return status_error(error[0] ? error : "ds4 GLM-5.3 prefill failed");
    }
    return status_ok();
}

int32_t sparkserve_ds4_glm53_session_sample(
        sparkserve_ds4_glm53_session *session,
        float temperature,
        int32_t top_k,
        float top_p,
        float min_p,
        uint64_t *rng) {
    if (!session || !rng) return -1;
    return ds4_session_sample(session->inner, temperature, top_k,
                              top_p, min_p, rng);
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_session_eval(
        sparkserve_ds4_glm53_session *session,
        int32_t token) {
    if (!session) return status_error("GLM-5.3 session is null");
    char error[512] = {0};
    if (ds4_session_eval(session->inner, token, error, sizeof(error)) != 0) {
        return status_error(error[0] ? error : "ds4 GLM-5.3 decode failed");
    }
    return status_ok();
}

int32_t sparkserve_ds4_glm53_session_position(
        const sparkserve_ds4_glm53_session *session) {
    if (!session) return -1;
    return ds4_session_pos(session->inner);
}

sparkserve_ds4_glm53_status sparkserve_ds4_glm53_token_piece(
        sparkserve_ds4_glm53_engine *engine,
        int32_t token,
        uint8_t **out,
        size_t *out_len) {
    if (!engine || !out || !out_len) {
        return status_error("invalid GLM-5.3 token piece arguments");
    }
    *out = NULL;
    *out_len = 0;
    char *piece = ds4_token_text(engine->inner, token, out_len);
    if (!piece) return status_error("ds4 could not decode a GLM-5.3 token");
    *out = (uint8_t *)piece;
    return status_ok();
}

void sparkserve_ds4_glm53_text_free(uint8_t *text) {
    free(text);
}
