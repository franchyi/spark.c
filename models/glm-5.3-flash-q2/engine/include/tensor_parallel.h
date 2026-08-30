#ifndef DS4_TP_H
#define DS4_TP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "engine.h"

/* Tensor-parallel transport and lockstep protocol.
 *
 * Two ranks run the same logical model, each with one contiguous half of the
 * routed experts resident. Rank 0 (leader) is a normal frontend session that
 * mirrors every ds4_session_sync()/ds4_session_eval() call to rank 1 (worker)
 * over a TCP control socket, so both engines execute the identical graph
 * sequence.
 * Inside each decoded token, partial block outputs are exchanged through a
 * registered memory slab: two-sided RDMA SEND/RECV when RDMA over
 * Thunderbolt is available, or a full-duplex TCP exchange as fallback.
 *
 * Layering: engine.c calls the session-mirroring and slab entry points;
 * ds4_metal.m only ever sees tensor_parallel_gate_exchange() through a callback
 * registered with the GPU gate machinery.  Nothing here touches tensors.
 */

typedef struct tensor_parallel tensor_parallel;

enum {
    DS4_TP_GATE_ATTN = 0,
    DS4_TP_GATE_FFN = 1,
    DS4_TP_GATES_PER_LAYER = 2,
    /* Max rows in a verify-block batch gate (speculative blocks are <=5). */
    DS4_TP_BATCH_MAX_ROWS = 8,
};

/* Engine identity exchanged in the hello so a mismatched pair aborts before
 * any inference runs. */
typedef struct {
    uint64_t gguf_bytes;
    uint32_t model_id;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t quant_bits;
    uint32_t ctx_size;
    /* Decode gate schedule, used to place RDMA recvs into the right slab
     * slot. A nonempty mask lists the exact slots in firing order; otherwise
     * slot(seq) = start + ((seq-1) % per_token) * step.
     * per_token 0 falls back to the identity mapping over all slots
     * (DS4: every layer fires ATTN then FFN). GLM fires one FFN gate per
     * sparse layer in streaming mode. Hybrid GLM-5.3 uses the mask because
     * DSA layers also fire ATTN while KDA layers do not. Exchanged in the
     * hello; both sides must agree. */
    uint32_t gate_slot_start;
    uint32_t gate_slot_step;
    uint32_t gates_per_token;
    uint64_t gate_slot_mask[DS4_TP_GATE_MASK_WORDS];
} tensor_parallel_identity;

bool tensor_parallel_enabled(const tensor_parallel_options *opt);

typedef enum {
    DS4_TP_CLI_ERROR = -1,
    DS4_TP_CLI_NOT_MATCHED = 0,
    DS4_TP_CLI_MATCHED = 1,
} tensor_parallel_cli_parse_result;

/* CLI parsing, same contract as ds4_dist_parse_cli_arg(): returns 1 when the
 * argument was consumed, 0 when not matched, -1 on error (err filled). */
int tensor_parallel_parse_cli_arg(
        const char *arg,
        int *index,
        int argc,
        char **argv,
        tensor_parallel_options *opt,
        char *err,
        size_t errlen);
int tensor_parallel_adopt_distributed_options(
        tensor_parallel_options *tp,
        distributed_options *dist,
        char *err,
        size_t errlen);
void tensor_parallel_usage(FILE *fp);

/* Validates option combinations that TP cannot run with (SSD streaming,
 * distributed mode, MTP drafting, CPU backend). */
int tensor_parallel_validate_engine_options(
        const ds4_engine_options *opt,
        char *err,
        size_t errlen);

/* Connection bring-up.  The leader listens and accepts one worker; the
 * worker dials with retry.  Both then exchange and validate identities.
 * Blocking; call after the engine is loaded (identity needs the shape). */
int tensor_parallel_create(
        tensor_parallel **out,
        const tensor_parallel_options *opt,
        const tensor_parallel_identity *id,
        char *err,
        size_t errlen);
void tensor_parallel_free(tensor_parallel *tp);

int tensor_parallel_rank(const tensor_parallel *tp);
bool tensor_parallel_is_rdma(const tensor_parallel *tp);
uint32_t tensor_parallel_peer_ctx(const tensor_parallel *tp);
bool tensor_parallel_failed(const tensor_parallel *tp);
void tensor_parallel_mark_failed(tensor_parallel *tp);

/* Gate slab.  The engine allocates one shared GPU-visible block and hands
 * its base VA here; tensor_parallel registers it with the NIC (RDMA) and exchanges
 * remote keys.  Layout, all offsets from base, S = n_layer * 2 slots:
 *
 *   out vectors   S * vec_bytes   written by local GPU kernels
 *   in  vectors   S * vec_bytes   RDMA/TCP-written with the peer partials
 *   in  seq flags S * 8           written strictly after each in vector
 *   token slot    16              {seq u64, token i32, pad} leader->worker
 *   (gpu flags, then batch out/in: n_layer * BATCH_MAX_ROWS * vec_bytes
 *    each, row partials for the speculative verify-block gates)
 *
 * vec_bytes = n_embd * 4 (f32 partials, never quantized on the wire). */
uint64_t tensor_parallel_slab_bytes(uint32_t n_layer, uint32_t n_embd);
uint64_t tensor_parallel_slab_out_offset(const tensor_parallel *tp, uint32_t layer, uint32_t gate);
uint64_t tensor_parallel_slab_in_offset(const tensor_parallel *tp, uint32_t layer, uint32_t gate);
uint64_t tensor_parallel_slab_batch_out_offset(const tensor_parallel *tp, uint32_t layer);
uint64_t tensor_parallel_slab_batch_in_offset(const tensor_parallel *tp, uint32_t layer);
uint64_t tensor_parallel_slab_gpu_flags_offset(const tensor_parallel *tp);
int tensor_parallel_attach_slab(tensor_parallel *tp, void *base, char *err, size_t errlen);

/* Exchange one gate: send out[layer][gate] to the peer's in[layer][gate]
 * and wait until the peer's partial for `seq` has fully landed locally.
 * Called from the GPU gate service thread.  Returns 0 on failure. */
int tensor_parallel_gate_exchange(tensor_parallel *tp, uint32_t layer, uint32_t gate, uint64_t seq);

/* Verify-block batch gate: exchange `rows` row partials for one layer in one
 * bulk RDMA transfer, with a symmetric TCP transfer as fallback. Called from
 * the GPU gate service thread. */
int tensor_parallel_batch_gate_exchange(tensor_parallel *tp, uint32_t layer, uint32_t rows,
                               uint64_t seq);

/* Prefill batch gate: arbitrary-size symmetric payload exchange over bulk
 * RDMA, with interleaved 2MB TCP rounds as fallback (see tensor_parallel.c). */
int tensor_parallel_big_gate_exchange(tensor_parallel *tp, uint32_t layer, uint64_t seq,
                             const void *out, void *in, uint64_t bytes);

/* Lockstep mirroring (leader side) and worker loop primitives. */
typedef struct {
    uint64_t session_id;
    int32_t token;
    uint32_t reserved;
} tensor_parallel_batch_item;

int tensor_parallel_send_session_create(tensor_parallel *tp, uint64_t session_id, int ctx_size);
int tensor_parallel_send_session_destroy(tensor_parallel *tp, uint64_t session_id);
int tensor_parallel_send_sync(tensor_parallel *tp, uint64_t session_id,
                     const int *tokens, uint32_t n_tokens);
int tensor_parallel_send_eval(tensor_parallel *tp, uint64_t session_id,
                     uint64_t seq, int token);
int tensor_parallel_send_rewind(tensor_parallel *tp, uint64_t session_id, int pos);
int tensor_parallel_send_invalidate(tensor_parallel *tp, uint64_t session_id);
int tensor_parallel_send_eval_batch(tensor_parallel *tp, const tensor_parallel_batch_item *items,
                           uint32_t count);
int tensor_parallel_send_mixed_batch(tensor_parallel *tp, uint64_t prefill_session_id,
                            const int *prompt, uint32_t prompt_count,
                            const tensor_parallel_batch_item *items,
                            uint32_t count);
int tensor_parallel_send_command_ack(tensor_parallel *tp, uint64_t session_id, int status);
int tensor_parallel_wait_command_ack(tensor_parallel *tp, uint64_t session_id,
                            const char *operation, char *err, size_t errlen);
int tensor_parallel_send_stop(tensor_parallel *tp);

/* Worker: blocks for the next mirrored command.  Frame types below; for
 * DS4_TP_FRAME_SYNC the token array is returned in *tokens / *n_tokens
 * (malloc'd, caller frees), for DS4_TP_FRAME_EVAL seq/token are filled. */
typedef enum {
    DS4_TP_FRAME_ERROR = -1,
    DS4_TP_FRAME_SYNC = 1,
    DS4_TP_FRAME_EVAL = 2,
    DS4_TP_FRAME_REWIND = 3,
    DS4_TP_FRAME_INVALIDATE = 4,
    DS4_TP_FRAME_STOP = 5,
    DS4_TP_FRAME_HASH = 6,
    DS4_TP_FRAME_RDMA_INFO = 7,
    DS4_TP_FRAME_SYNC_ACK = 8,
    DS4_TP_FRAME_RDMA_READY = 9,
    DS4_TP_FRAME_LOGITS = 10,
    DS4_TP_FRAME_VERIFY = 11,
    DS4_TP_FRAME_VERIFY_COMMIT = 12,
    DS4_TP_FRAME_SESSION_CREATE = 13,
    DS4_TP_FRAME_SESSION_DESTROY = 14,
    DS4_TP_FRAME_EVAL_BATCH = 15,
    DS4_TP_FRAME_MIXED_BATCH = 16,
    DS4_TP_FRAME_COMMAND_ACK = 17,
} tensor_parallel_frame_type;

typedef struct {
    tensor_parallel_frame_type type;
    uint64_t session_id;
    uint64_t seq;
    int value;
    int *tokens;
    uint32_t n_tokens;
    tensor_parallel_batch_item *items;
    uint32_t n_items;
} tensor_parallel_command;

int tensor_parallel_recv_command(
        tensor_parallel *tp,
        tensor_parallel_command *command,
        char *err,
        size_t errlen);
void tensor_parallel_command_free(tensor_parallel_command *command);

/* Debug lockstep check: both sides send their hidden-state hash for a token
 * and compare.  Returns 0 on transport failure, -1 on hash mismatch. */
int tensor_parallel_hash_check(tensor_parallel *tp, uint64_t seq, uint64_t hash, char *err, size_t errlen);

/* Vocab-split output head: the worker ships its logits half to the leader
 * after every eval (and after a sync) on the control socket. */
int tensor_parallel_send_logits_half(tensor_parallel *tp, const float *half, uint32_t count);
int tensor_parallel_recv_logits_half(tensor_parallel *tp, float *half, uint32_t count);

/* Speculative verify mirroring.  The leader announces a draft block right
 * before both ranks run the expert-split batch verify; the worker then blocks
 * on the commit frame, which carries the leader's decision: full_accept keeps
 * the pushed rows, otherwise both sides roll back and replay replay_n tokens
 * through the gated single-token decode in lockstep. */
int tensor_parallel_send_verify(tensor_parallel *tp, uint64_t session_id,
                       const int *drafts, uint32_t n);
int tensor_parallel_send_verify_commit(tensor_parallel *tp, int32_t full_accept, int32_t replay_n);
int tensor_parallel_recv_verify_commit(tensor_parallel *tp, int32_t *full_accept, int32_t *replay_n);

/* Standalone worker mode entry. Loads nothing itself: the engine is already
 * open. */
int tensor_parallel_worker_run(ds4_engine *engine, const tensor_parallel_options *opt);

#endif
