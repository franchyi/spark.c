#ifndef DS4_KVSTORE_H
#define DS4_KVSTORE_H

#include "engine.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_KVSTORE_FIXED_HEADER 48u
#define DS4_KVSTORE_DEFAULT_MB 4096
#define DS4_KVSTORE_HIT_HALF_LIFE_SECONDS (6ull * 60ull * 60ull)

#define DS4_KVSTORE_EXT_TOOL_MAP          (1u << 0)
#define DS4_KVSTORE_EXT_RESPONSES_VISIBLE (1u << 1)
#define DS4_KVSTORE_EXT_THINKING_VISIBLE  (1u << 2)
#define DS4_KVSTORE_EXT_SESSION_TITLE     (1u << 3)

typedef enum {
    DS4_KVSTORE_REASON_UNKNOWN   = 0,
    DS4_KVSTORE_REASON_COLD      = 1,
    DS4_KVSTORE_REASON_CONTINUED = 2,
    DS4_KVSTORE_REASON_EVICT     = 3,
    DS4_KVSTORE_REASON_SHUTDOWN  = 4,
    DS4_KVSTORE_REASON_AGENT_SYSTEM  = 5,
    DS4_KVSTORE_REASON_AGENT_SESSION = 6,
} kv_store_reason;

typedef enum {
    DS4_KVSTORE_LOG_DEFAULT,
    DS4_KVSTORE_LOG_KVCACHE,
    DS4_KVSTORE_LOG_WARNING,
} kv_store_log_type;

typedef struct {
    /* The file name is the rendered byte prefix, not the token sequence. The
     * payload still carries the exact tokens and graph state; the hash only
     * answers "does this checkpoint represent the bytes at the front of the
     * incoming prompt?" */
    char sha[41];
    char *path;
    uint8_t quant_bits;
    /* Stored in header byte 7.  Flash is 0 for backward compatibility with
     * older cache files where this reserved byte was always written as zero. */
    uint8_t model_id;
    uint8_t reason;
    uint32_t tokens;
    uint32_t hits;
    uint32_t ctx_size;
    uint8_t ext_flags;
    uint64_t created_at;
    uint64_t last_used;
    uint64_t payload_bytes;
    uint64_t text_bytes;
    uint64_t file_size;
} kv_store_entry;

typedef struct {
    int min_tokens;
    int cold_max_tokens;
    int continued_interval_tokens;
    int boundary_trim_tokens;
    int boundary_align_tokens;
} kv_store_options;

typedef struct {
    bool enabled;
    char *dir;
    uint64_t budget_bytes;
    bool reject_different_quant;
    kv_store_options opt;
    int continued_last_store_tokens;
    kv_store_entry *entry;
    int len;
    int cap;
    const char *log_name;
    void *log_ud;
    void (*log)(void *ud, kv_store_log_type type, const char *msg);
} kv_store;

typedef struct {
    const char *text;
    size_t text_len;
    uint8_t model_id;
    uint8_t quant_bits;
    uint32_t ctx_size;
    bool reject_different_quant;
} kv_store_eviction_context;

typedef struct {
    void *ud;
    uint8_t ext_flag;
    bool (*serialized_size)(void *ud, const char *text, uint64_t *bytes_out);
    bool (*write)(void *ud, FILE *fp, const char *text, uint64_t *written_bytes);
    int (*load)(void *ud, FILE *fp, const void *wanted);
    const void *load_wanted;
} kv_store_trailer_hooks;

typedef struct {
    int tokens;
    uint32_t text_bytes;
    uint8_t quant_bits;
    uint8_t ext_flags;
    double load_ms;
    char *path;
} kv_store_load_result;

kv_store_options kv_store_default_options(void);
uint8_t kv_store_reason_code(const char *reason);
const char *kv_store_key_kind(uint8_t ext_flags);

bool kv_store_open(kv_store *kc, const char *dir, uint64_t budget_mb,
                      bool reject_different_quant, kv_store_options opt,
                      const char *log_name,
                      void (*log)(void *ud, kv_store_log_type type, const char *msg),
                      void *log_ud);
void kv_store_close(kv_store *kc);
void kv_store_clear(kv_store *kc);
void kv_store_entry_free(kv_store_entry *e);

char *kv_store_render_tokens_text(ds4_engine *engine,
                                     const ds4_tokens *tokens,
                                     size_t *out_len);
bool kv_store_byte_prefix_match(const char *text, size_t text_len,
                                   const char *prefix, size_t prefix_len);
void kv_store_tokens_copy_prefix(ds4_tokens *dst, const ds4_tokens *src, int n);
void kv_store_build_prompt_from_exact_prefix_and_text_suffix(
        ds4_engine *engine,
        const ds4_tokens *exact_prefix,
        const char *suffix_text,
        ds4_tokens *out);

int kv_store_store_len(const kv_store *kc, int tokens);
int kv_store_chat_anchor_pos(const kv_store *kc,
                                const ds4_tokens *prompt,
                                int user_token_id,
                                int assistant_token_id);
int kv_store_continued_store_target(const kv_store *kc, int live_tokens);
void kv_store_note_store(kv_store *kc, int tokens);
int kv_store_suppress_continued_store(kv_store *kc, int tokens);
void kv_store_restore_suppressed_continued(kv_store *kc,
                                              int old_tokens,
                                              int suppressed_tokens);

bool kv_store_file_size_fits(const kv_store *kc,
                                uint64_t text_bytes,
                                uint64_t payload_bytes,
                                uint64_t trailer_bytes,
                                uint64_t *file_bytes_out,
                                uint64_t *required_bytes_out);
double kv_store_entry_eviction_score(const kv_store_entry *e,
                                        const ds4_tokens *live,
                                        uint64_t now,
                                        const kv_store_eviction_context *incoming);
void kv_store_evict(kv_store *kc, const ds4_tokens *live,
                       uint64_t extra_bytes,
                       const kv_store_eviction_context *incoming);
int kv_store_find_text_prefix(kv_store *kc, const char *prompt_text,
                                 int model_id, int quant_bits, int ctx_size);

bool kv_store_store_live_prefix_text(kv_store *kc,
                                        ds4_engine *engine,
                                        ds4_session *session,
                                        const ds4_tokens *tokens,
                                        int store_len,
                                        const char *reason,
                                        const char *cache_text_override,
                                        uint8_t cache_text_ext,
                                        const char *cache_text_key,
                                        const kv_store_trailer_hooks *hooks,
                                        char *err,
                                        size_t err_len);
bool kv_store_store_live_prefix(kv_store *kc,
                                   ds4_engine *engine,
                                   ds4_session *session,
                                   const ds4_tokens *tokens,
                                   int store_len,
                                   const char *reason,
                                   const kv_store_trailer_hooks *hooks,
                                   char *err,
                                   size_t err_len);
bool kv_store_maybe_store_continued(kv_store *kc,
                                       ds4_engine *engine,
                                       ds4_session *session,
                                       const kv_store_trailer_hooks *hooks,
                                       char *err,
                                       size_t err_len);
int kv_store_try_load_text(kv_store *kc,
                              ds4_engine *engine,
                              ds4_session *session,
                              const char *prompt_text,
                              ds4_tokens *effective_prompt,
                              kv_store_load_result *result,
                              const kv_store_trailer_hooks *hooks,
                              bool responses_protocol);
void kv_store_load_result_free(kv_store_load_result *result);

bool kv_store_read_header(FILE *fp, kv_store_entry *e,
                             uint32_t *text_bytes);
bool kv_store_read_entry_file(const char *path, const char sha[41],
                                 kv_store_entry *out);
void kv_store_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes);
bool kv_store_touch_file(const char *path, uint32_t hits);
bool kv_store_sha_hex_name(const char *name, char sha[41]);
void kv_store_sha1_bytes_hex(const void *ptr, size_t len, char out[41]);
char *kv_store_path_join(const char *dir, const char *name);
char *kv_store_path_for_sha(kv_store *kc, const char sha[41]);
void kv_store_le_put32(uint8_t *p, uint32_t v);
uint32_t kv_store_le_get32(const uint8_t *p);

#endif
