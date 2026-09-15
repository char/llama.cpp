#include "models.h"

#include <type_traits>

void llama_model_aliceai_t5_moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_DECODER_BLOCK_COUNT, hparams.dec_n_layer);
    uint32_t decoder_start_token_id;
    ml.get_key(LLM_KV_DECODER_START_TOKEN_ID, decoder_start_token_id);
    hparams.dec_start_token_id = decoder_start_token_id;
    ml.get_key(LLM_KV_ENCODER_ATTENTION_HEAD_COUNT_KV, n_head_kv_enc);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE, hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM, hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);

    if (hparams.dec_n_layer == 0 || hparams.dec_n_layer > hparams.n_layer()) {
        throw std::runtime_error("AliceAI requires 0 < decoder_block_count <= block_count");
    }
    if (hparams.n_head() == 0 || n_head_kv_enc == 0 || hparams.n_head() % n_head_kv_enc != 0 ||
            hparams.n_head_kv() == 0 || hparams.n_head() % hparams.n_head_kv() != 0) {
        throw std::runtime_error("AliceAI KV head counts must divide the query head count");
    }
    if (hparams.n_expert == 0 || hparams.n_ff_exp() == 0 ||
            hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID) {
        throw std::runtime_error("AliceAI requires sigmoid routing and nonempty experts");
    }

    if (hparams.rope_scaling_type_train == LLAMA_ROPE_SCALING_TYPE_YARN) {
        ml.get_key(LLM_KV_ROPE_SCALING_YARN_ATTN_FACTOR, hparams.rope_attn_factor);
        // GGML applies the default YaRN magnitude inside RoPE; metadata gives the total magnitude.
        hparams.rope_attn_factor /= 1.0f + 0.1f * logf(1.0f / hparams.rope_freq_scale_train);
    }

    // The published causal flag describes the encoder. llama_encode disables causality separately.
    hparams.causal_attn = true;
    type = LLM_TYPE_UNKNOWN;
}

void llama_model_aliceai_t5_moe::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.dec_start_token_id < 0 || hparams.dec_start_token_id >= n_vocab) {
        throw std::runtime_error("AliceAI decoder start token is outside the vocabulary");
    }
    if (n_embd_head_k != n_embd_head_v || n_embd_head_k == 0) {
        throw std::runtime_error("AliceAI requires equal, nonzero key and value head dimensions");
    }

    const int64_t n_ff_exp = hparams.n_ff_exp();
    const int64_t n_embd_q = n_embd_head_k * n_head;
    const int64_t n_embd_k_enc = n_embd_head_k * n_head_kv_enc;
    const int64_t n_embd_v_enc = n_embd_head_v * n_head_kv_enc;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm_enc = create_tensor(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm     = create_tensor(tn(LLM_TENSOR_DEC_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    layers_enc.resize(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers_enc[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ENC_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_ENC_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ENC_ATTN_K, "weight", i), {n_embd, n_embd_k_enc}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ENC_ATTN_V, "weight", i), {n_embd, n_embd_v_enc}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ENC_ATTN_OUT, "weight", i), {n_embd_head_v * n_head, n_embd}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_ENC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_ENC_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_ENC_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_ENC_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_ENC_EXP_PROBS_B, i), {n_expert}, 0);
        layer.ffn_down_b      = create_tensor(tn(LLM_TENSOR_ENC_FFN_OUTPUT_B, i), {n_embd}, 0);
    }

    for (uint32_t i = 0; i < hparams.dec_n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_DEC_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq = create_tensor(tn(LLM_TENSOR_DEC_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_DEC_ATTN_K, "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv = create_tensor(tn(LLM_TENSOR_DEC_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_DEC_ATTN_OUT, "weight", i), {n_embd_head_v * n_head, n_embd}, 0);

        layer.attn_norm_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.wq_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
        layer.wk_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_K, "weight", i), {n_embd, n_embd_k_gqa}, 0);
        layer.wv_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_V, "weight", i), {n_embd, n_embd_v_gqa}, 0);
        layer.wo_cross = create_tensor(tn(LLM_TENSOR_DEC_CROSS_ATTN_OUT, "weight", i), {n_embd_head_v * n_head, n_embd}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_DEC_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_DEC_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_DEC_FFN_UP_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_DEC_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_DEC_EXP_PROBS_B, i), {n_expert}, 0);
        layer.ffn_down_b      = create_tensor(tn(LLM_TENSOR_DEC_FFN_OUTPUT_B, i), {n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_aliceai_t5_moe::build_arch_graph(const llm_graph_params & params) const {
    switch (params.gtype) {
        case LLM_GRAPH_TYPE_ENCODER:
            return std::make_unique<graph<true>>(*this, params);
        case LLM_GRAPH_TYPE_DEFAULT:
        case LLM_GRAPH_TYPE_DECODER:
            return std::make_unique<graph<false>>(*this, params);
        default:
            GGML_ABORT("invalid graph type");
    }
}

template <bool is_enc>
llama_model_aliceai_t5_moe::graph<is_enc>::graph(const llama_model_aliceai_t5_moe & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_head_kv_self = is_enc ? model.n_head_kv_enc : n_head_kv;
    const int n_layers = is_enc ? n_layer : hparams.dec_n_layer;
    const auto & layers = is_enc ? model.layers_enc : model.layers;
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head));

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = is_enc ? nullptr : build_inp_out_ids();
    ggml_tensor * embd_enc = nullptr;
    llm_graph_input_attn_cross * inp_cross = nullptr;

    using inp_attn_type = std::conditional_t<is_enc, llm_graph_input_attn_no_cache, llm_graph_input_attn_kv>;
    inp_attn_type * inp_attn;
    if constexpr (is_enc) {
        inp_attn = build_attn_inp_no_cache();
    } else {
        inp_attn = build_attn_inp_kv();
        embd_enc = build_inp_cross_embd();
        inp_cross = build_attn_inp_cross();
    }

    for (int il = 0; il < n_layers; ++il) {
        const auto & layer = layers[il];
        ggml_tensor * residual = inpL;
        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head_kv_self, il);
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);

        cur = build_attn(inp_attn, layer.wo, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "attn_out", il);

        if constexpr (!is_enc) {
            residual = cur;
            cur = build_norm(cur, layer.attn_norm_cross, nullptr, LLM_NORM_RMS, il);
            cb(cur, "attn_norm_cross", il);

            Qcur = build_lora_mm(layer.wq_cross, cur);
            Kcur = build_lora_mm(layer.wk_cross, embd_enc);
            Vcur = build_lora_mm(layer.wv_cross, embd_enc);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, embd_enc->ne[1]);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, embd_enc->ne[1]);
            cur = build_attn(inp_cross, layer.wo_cross, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
            cur = ggml_add(ctx0, cur, residual);
            cb(cur, "cross_attn_out", il);
        }

        if (il == n_layers - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
        }
        residual = cur;
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);
        cur = build_moe_ffn(cur, layer.ffn_gate_inp, layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps,
                layer.ffn_exp_probs_b, n_expert, hparams.n_expert_used(il), LLM_FFN_SILU,
                hparams.expert_weights_norm, hparams.expert_weights_scale, LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il);
        cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        cb(cur, "ffn_out", il);
        cur = ggml_add(ctx0, cur, residual);
        inpL = build_cvec(cur, il);
        cb(inpL, "l_out", il);
    }

    ggml_tensor * cur = build_norm(inpL, is_enc ? model.output_norm_enc : model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    if constexpr (!is_enc) {
        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
    }
    ggml_build_forward_expand(gf, cur);
}

template struct llama_model_aliceai_t5_moe::graph<true>;
template struct llama_model_aliceai_t5_moe::graph<false>;
