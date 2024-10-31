// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_for_speculative_decoding_impl.hpp"

namespace ov::genai {
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::ContinuousBatchingForSpeculativeDecodingImpl(
    ov::Core& core,
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const DeviceConfig& device_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled) {
    m_tokenizer = tokenizer;
    m_is_validation_mode_enabled = is_validation_mode_enabled;
    init(model, scheduler_config, plugin_config,  device_config, core);
}

void
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_request(SequenceGroup::Ptr request) {
    for (const auto& sequence: request->get_sequences()) {
        if (m_scheduler->has_block_table(sequence->get_id())) {
            m_scheduler->free_sequence(sequence->get_id());
        }
    }
    m_sampler->clear_request_info(request->get_request_id());
    request->set_generation_status(GenerationStatus::DROPPED_BY_HANDLE);
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_request(int64_t request_id) {
    for (size_t i = 0; i < m_requests.size(); ++i) {
        auto& request = m_requests[i];
        if (request->get_request_id() != request_id) {
            continue;
        }
        finish_request(request);
        m_requests.erase(m_requests.begin() + i);
        break;
    }
}

GeneratedRequests
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_generated_requests() {
    GeneratedRequests result;
    for (const auto& request : m_requests) {
        const auto& request_id = request->get_request_id();
        if (!result.count(request_id)) {
            result.insert({request_id, {{}} });
        }
        auto& generated_request = result[request_id];
        for (const auto& sequence : request->get_running_sequences()) {
            const auto& sequence_id = sequence->get_grouped_id();
            OPENVINO_ASSERT(!generated_request.count(sequence_id));
            generated_request.insert({{sequence_id, { sequence->get_generated_ids(), sequence->get_generated_log_probs() } }});
        }
    }
    return result;
}

// { min_len_of_prefix, min_length_of_candidate }
std::pair<size_t, size_t>
get_prefix_len(
    const std::vector<Sequence::Ptr>& running_sequences,
    const GeneratedSequences& candidates) {
    size_t min_generated_tokens = std::numeric_limits<size_t>::max(),
           min_candidate_len = std::numeric_limits<size_t>::max();
    for (const auto& running_sequence : running_sequences) {
        const auto& sequence_id = running_sequence->get_grouped_id();
        if (!candidates.count(sequence_id)) {
            continue;
        }

        const auto& candidate_sequence = candidates.at(sequence_id);

        const std::vector<int64_t>& candidate_token_ids = candidate_sequence.token_ids,
                                    running_token_ids = running_sequence->get_generated_ids();

        const size_t candidate_sequence_gen_len = candidate_token_ids.size(),
                     running_sequence_gen_len = running_sequence->get_generated_len();
        
        // to find the len of prefix
        size_t sequence_prefix_len = std::min(candidate_sequence_gen_len, running_sequence_gen_len);
        for (size_t i = 0; i < sequence_prefix_len; ++i) {
            if (candidate_token_ids[i] != running_token_ids[i]) {
                sequence_prefix_len = i;
                break;
            }
        }

        min_generated_tokens = std::min(sequence_prefix_len, min_generated_tokens);
        min_candidate_len = std::min(candidate_sequence_gen_len, min_candidate_len);
    }
    return { min_generated_tokens, min_candidate_len };
}

size_t
remove_tokens_from_sequence(Sequence::Ptr& sequence,
                            size_t min_generated_tokens,
                            LogitProcessor& logit_proccessor) {
    const auto generated_token_ids = sequence->get_generated_ids(); 
    const auto sequence_generated_len = generated_token_ids.size();
    OPENVINO_ASSERT(sequence_generated_len >= min_generated_tokens);

    size_t removed_token_cnt = sequence_generated_len - min_generated_tokens;
    for (size_t i = min_generated_tokens; i < sequence_generated_len; ++i) {
        logit_proccessor.decrease_generated_token_occurance(generated_token_ids[i]);
    }
    sequence->remove_last_tokens(removed_token_cnt);
    return (sequence_generated_len - min_generated_tokens);
}

size_t
insert_tokens_to_sequence(Sequence::Ptr& sequence,
                          const std::vector<int64_t>& token_ids,
                          const std::vector<float>& token_log_probs,
                          LogitProcessor& logit_proccessor,
                          bool is_update_sampler) {
    size_t generated_len = sequence->get_generated_len(), candidate_len = token_ids.size();
    OPENVINO_ASSERT(generated_len <= candidate_len);
    for (size_t i = generated_len; i < candidate_len; ++i) {
        sequence->append_token(token_ids[i], token_log_probs[i]);
        if (is_update_sampler) {
            logit_proccessor.register_new_generated_token(token_ids[i]);
        }
    }
    return (candidate_len - generated_len);
}


// `is_init_all_sequences_in_request` is flag to enable initialization of all sequences in case of `num_return_sequences > 1`.
// Only first sequence from all will be initialized if flag is set to `false` state.
// This approach helped to process prompt once in speculative decoding multisequence case.
size_t
init_request(
    SequenceGroup::Ptr request,
    const GeneratedSequences& candidates,
    LogitProcessor& logit_processor,
    bool is_update_logit_processor,
    bool is_init_all_sequences_in_request = false) {
    OPENVINO_ASSERT(request->get_sampling_parameters().is_speculative_decoding(),
                    "Speculative decoding should have initialized options `assistant_confidence_threshold` xor `num_assistant_tokens` in `GenerationConfig`.");
    if (candidates.begin()->second.token_ids.empty() && !is_init_all_sequences_in_request) {
        return 0;
    }
    size_t min_candidate_len = std::numeric_limits<size_t>::max();
    if (is_init_all_sequences_in_request) {
        for (const auto& candidate_sequence : candidates) {
            min_candidate_len = std::min(candidate_sequence.second.token_ids.size(), min_candidate_len);
        }
    } else {
        // place only one token to first sequence in case of multisequence generation.
        // Left sequences in request will be initialized in sampler and validated after (only one token).
        min_candidate_len = request->get_sampling_parameters().num_return_sequences == 1 ? candidates.begin()->second.token_ids.size() : 1;
    }
    for (const auto& candidate_sequence : candidates) {
        Sequence::Ptr sequence;
        if (is_init_all_sequences_in_request && candidate_sequence.first > 0) {
            sequence = Sequence::Ptr(new Sequence(candidate_sequence.first));
            sequence->set_status(ov::genai::SequenceStatus::RUNNING);
            request->add_sequence(sequence);
        } else {
            auto running_sequences = request->get_running_sequences();
            OPENVINO_ASSERT(!running_sequences.empty());
            sequence = request->get_running_sequences()[0];
        }
        auto token_ids = candidate_sequence.second.token_ids;
        auto log_probs = candidate_sequence.second.log_probs;
        token_ids.resize(min_candidate_len);
        log_probs.resize(min_candidate_len);

        for (size_t i = 0; i < min_candidate_len; ++i) {
            sequence->append_token(token_ids[i], log_probs[i]);
            if (is_update_logit_processor) {
                logit_processor.register_new_generated_token(token_ids[i]);
            }
        }

        if (!is_init_all_sequences_in_request) {
            break;
        }
    }
    return min_candidate_len;
}

UpdateRequestResult 
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::init_request_by_candidate(
    uint64_t request_id,
    const GeneratedSequences& candidates) {
    for (auto& request : m_requests) {
        if (request->get_request_id() != request_id) {
            continue;
        }
        
        UpdateRequestResult result;
        m_sampler->create_logit_processor(request_id, request->get_sampling_parameters(), request->get_prompt_ids());
        auto& logit_processor = m_sampler->get_logit_processor(request_id);
        result.inserted_tokens_cnt = init_request(request, candidates, logit_processor, true, true);
        request->set_num_validated_tokens(result.inserted_tokens_cnt);
        return result;
    }
    return {0, 0};
}

UpdateRequestResult
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_request(uint64_t request_id,
                                                                                         const GeneratedSequences& candidates,
                                                                                         bool is_update_logit_processor) {
    UpdateRequestResult result{0, 0};
    for (auto& request : m_requests) {
        if (request_id != request->get_request_id()) {
            continue;
        }

        std::vector<Sequence::Ptr> running_sequences = request->get_running_sequences();
        OPENVINO_ASSERT(running_sequences.size() > 0);
        size_t min_generated_tokens, min_candidate_len;
        if (running_sequences.front()->get_generated_len() == 0 && !request->get_num_tokens_to_validate()) {
            m_sampler->create_logit_processor(request_id, request->get_sampling_parameters(), request->get_prompt_ids());
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            result.inserted_tokens_cnt = init_request(request, candidates, logit_processor, is_update_logit_processor);
            min_generated_tokens = result.inserted_tokens_cnt;
            running_sequences = request->get_running_sequences();
            min_candidate_len = result.inserted_tokens_cnt;
        } else {
            // update existing sequences by the candidates
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            std::tie(min_generated_tokens, min_candidate_len) = get_prefix_len(running_sequences, candidates);

            for (auto& running_sequence : running_sequences) {
                if (!candidates.count(running_sequence->get_grouped_id())) {
                    continue;
                }

                result.removed_tokens_cnt = remove_tokens_from_sequence(running_sequence, min_generated_tokens, logit_processor);

                auto candidate_sequence = candidates.at(running_sequence->get_grouped_id());
                std::vector<int64_t> candidate_token_ids = candidate_sequence.token_ids;
                std::vector<float> candidate_token_log_probs = candidate_sequence.log_probs;
                candidate_token_ids.resize(min_candidate_len);
                candidate_token_log_probs.resize(min_candidate_len);
                result.inserted_tokens_cnt = insert_tokens_to_sequence(running_sequence, candidate_token_ids, candidate_token_log_probs, logit_processor, is_update_logit_processor);
            }
            // we should update a logit processor just for draft model to generate the same tokens
            // logit processors of main model will be updated in sampler while validation mode
            if (is_update_logit_processor) {
                logit_processor.update_generated_len(min_candidate_len);
            }
        }

        // update request context information to provide correct scheduling phase
        const size_t num_processed_tokens = request->get_num_processed_tokens(),
                     prompt_len = request->get_prompt_len(),
                     updated_context_len = min_candidate_len + prompt_len,
                     max_new_tokens = request->get_sampling_parameters().max_new_tokens;
        size_t generated_len = request->get_context_len() - request->get_prompt_len();
        if (num_processed_tokens > 0) {
            request->update_processed_tokens_num(num_processed_tokens - result.removed_tokens_cnt);
            generated_len -= result.removed_tokens_cnt;
        }
        request->set_num_validated_tokens(result.inserted_tokens_cnt);
        request->pause_generation(false);
        generated_len += result.inserted_tokens_cnt;

        // to pause `draft_model` generation in case of `generated_len >= max_new_tokens - 1` to generate last token by `main_model`
        if (!m_is_validation_mode_enabled && (generated_len >= max_new_tokens - 1 || result.inserted_tokens_cnt == 0)) {
            request->pause_generation(true);
        }
        break;
    }

    return result;
}

void
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::pull_awaiting_requests() {
    ContinuousBatchingImpl::_pull_awaiting_requests();
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::multistep() {
    size_t generated_tokens_cnt = 0;
    // cycle to generate several tokens per one iteration for speculative decoding case
    bool to_generate = true;
    while (to_generate) {
        generated_tokens_cnt++;
        auto time0 = std::chrono::high_resolution_clock::now();
        step();
        auto time1 = std::chrono::high_resolution_clock::now();
        auto time_res0 = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();
        std::cout << "Draft model time = " << time_res0 << " ms\n";

        to_generate = false;
        for (auto& request : m_requests) {
            const auto& sampling_params = request->get_sampling_parameters();
            if (!sampling_params.is_speculative_decoding()) {
                // generate only one token in case of non speculative decoding
                request->pause_generation(true);
            } else if (request->get_num_processed_tokens() == 0 && sampling_params.num_return_sequences > 1) {
                request->pause_generation(true);
            } else if (sampling_params.num_assistant_tokens <= generated_tokens_cnt && sampling_params.assistant_confidence_threshold == 0.f) {
                request->pause_generation(true);
            } else if (request->get_context_len() >= request->get_prompt_len() &&
                (request->get_context_len() - request->get_prompt_len()) >= sampling_params.max_new_tokens - 1) {
                request->pause_generation(true);
            } else if (sampling_params.max_new_tokens == 0) {
                request->pause_generation(true);
            }
            to_generate |= request->can_generate_tokens();
        }
    }
}
}
