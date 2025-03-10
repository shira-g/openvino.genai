// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_pipeline_static.hpp"

#include <chrono>
#include <regex>

#include "debug_utils.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "utils.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper.hpp"
#include "whisper/whisper_config.hpp"
#include "whisper/whisper_utils.hpp"

#include "openvino/core/layout.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"

using ov::genai::MicroSeconds;

namespace {

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val) {
    auto* tensor_data = tensor.data<T>();
    std::fill(tensor_data, tensor_data + tensor.get_size(), fill_val);
}

template <typename T>
void copy_to_tensor(const std::vector<T>& src_vec, ov::Tensor dst_tensor) {
    auto* dst_ptr = dst_tensor.data<T>();
    OPENVINO_ASSERT(src_vec.size() == dst_tensor.get_size());
    std::copy(src_vec.begin(), src_vec.end(), dst_ptr);
}

ov::Tensor encode(ov::InferRequest& request,
                  std::vector<float>& mel_data,
                  const size_t feature_size,
                  const size_t nb_max_frames,
                  ov::genai::RawPerfMetrics& raw_metrics) {
    OPENVINO_ASSERT(mel_data.size() == feature_size * nb_max_frames,
                    "Mel spectrogram required size: ",
                    feature_size,
                    " * ",
                    nb_max_frames,
                    ". Actual size: ",
                    mel_data.size(),
                    ".");
    copy_to_tensor(mel_data, request.get_tensor("input_features"));

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    return request.get_tensor("last_hidden_state");
}

// FIXME: Duplicate from llm_pipeline_static.cpp - need to reuse instead of copy-paste
ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

void set_cross_attn_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
    // NB: Source outputs:
    // present.0.encoder.key
    // present.0.encoder.value

    // NB: Dest inputs:
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("encoder") == std::string::npos) {
            continue;
        }
        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");
        dest.set_tensor(with_past_input_name, source.get_tensor(source_output_name));
    }
}

void update_past_key_value(ov::InferRequest& source, ov::InferRequest& dest, const size_t kv_pos = 0u) {
    // NB: Source outputs:
    // present.0.decoder.key
    // present.0.decoder.value

    // NB: Dest inputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("decoder") == std::string::npos) {
            continue;
        }

        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past_key_values");

        auto src_kv_tensor = source.get_tensor(source_output_name);
        auto dst_kv_tensor = dest.get_tensor(with_past_input_name);
        auto kv_size = src_kv_tensor.get_shape()[2];
        // NB: Copy src_kv_tensor into dst_kv_tensor[:, :, kv_pos:kv_pos+kv_size, :]
        auto dst_kv_tensor_slice = make_tensor_slice(dst_kv_tensor, 2u, kv_pos, kv_pos + kv_size);
        src_kv_tensor.copy_to(dst_kv_tensor_slice);
    }
}

void set_decoder_input_ids(ov::InferRequest& decoder,
                           const std::vector<int32_t>& init_ids) {
    auto input_ids_tensor = decoder.get_tensor("input_ids");
    const size_t seq_length = input_ids_tensor.get_shape()[1];

    OPENVINO_ASSERT(seq_length >= init_ids.size());

    auto input_ids_data = input_ids_tensor.data<int32_t>();
    std::copy(init_ids.begin(), init_ids.end(), input_ids_data);
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               const std::vector<int32_t>& init_ids,
               const ov::genai::WhisperGenerationConfig& config,
               ov::genai::RawPerfMetrics& raw_metrics,
               const bool apply_logit_processors = true,
               const bool return_timestamps = false) {
    // NB: Fill decoder inputs
    encoder_hidden_state.copy_to(decoder.get_tensor("encoder_hidden_states"));
    set_decoder_input_ids(decoder, init_ids);

    ov::genai::utils::infer_with_perf_metrics(decoder, raw_metrics);

    auto output_tensor = decoder.get_tensor("logits");

    if (apply_logit_processors) {
        ov::genai::do_suppress_tokens(output_tensor, 0, config.begin_suppress_tokens);
        ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

        if (return_timestamps) {
            ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, {}, true);
        }
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);
    return output_token;
}

int64_t decode_with_past(ov::InferRequest& decoder_with_past,
                         const int64_t input_id,
                         const int64_t position_id,
                         const ov::genai::WhisperGenerationConfig& config,
                         ov::genai::RawPerfMetrics& raw_metrics,
                         const bool return_timestamps,
                         const std::vector<int64_t>& generated_tokens) {
    // FIXME: Avoid this cast to i32. Why it's not i64 precision in model?
    decoder_with_past.get_tensor("input_ids").data<int32_t>()[0] = static_cast<int32_t>(input_id);
    decoder_with_past.get_tensor("cache_position").data<int64_t>()[0] = position_id;
    // FIXME: Is "attention_mask" supposed to be f16?
    decoder_with_past.get_tensor("attention_mask").data<ov::float16>()[position_id - 1] = 0u;

    ov::genai::utils::infer_with_perf_metrics(decoder_with_past, raw_metrics);

    auto output_tensor = decoder_with_past.get_tensor("logits");
    ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

    if (return_timestamps) {
        ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, generated_tokens);
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);
    return output_token;
}

void zero_past_key_values(ov::InferRequest& request) {
    for (auto& input : request.get_compiled_model().inputs()) {
        std::string past_key_value_decoder_name = input.get_any_name();
        if (past_key_value_decoder_name.find("decoder") == std::string::npos ||
            past_key_value_decoder_name.find("past_key_values") == std::string::npos) {
            continue;
        }
        fill_tensor<ov::float16>(request.get_tensor(past_key_value_decoder_name), 0);
    }
}

void prepare_decoder_with_past(ov::InferRequest& decoder_with_past, ov::InferRequest& decoder, const size_t init_ids_size) {
    // NB: Prepare attetion mask to be in a format [0, 0, 0, 1, 1, 1, 1, ..., 0, 1]
    // Mask should be inverted for decoder_with_past 
    auto attention_mask = decoder_with_past.get_tensor("attention_mask");
    auto* attention_mask_ptr = attention_mask.data<ov::float16>();
    std::fill(attention_mask_ptr, attention_mask_ptr + init_ids_size, 0);
    std::fill(attention_mask_ptr + init_ids_size, attention_mask_ptr + attention_mask.get_size() - 2, 1);
    attention_mask_ptr[attention_mask.get_size() - 2] = 0;
    attention_mask_ptr[attention_mask.get_size() - 1] = 1;
    // NB: Zero past_key_values.*.decoder.value tensors
    zero_past_key_values(decoder_with_past);
    // NB: Copy KV-caches from decoder
    set_cross_attn_key_value(decoder, decoder_with_past);
    update_past_key_value(decoder, decoder_with_past);
};

int64_t detect_language(ov::Tensor& encoder_hidden_state,
                        ov::genai::DecoderCache& decoder_cache,
                        const ov::genai::WhisperGenerationConfig& config,
                        ov::genai::RawPerfMetrics& raw_metrics) {
    auto decoder = decoder_cache.get_model(1);

    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int32_t> init_ids{static_cast<int32_t>(config.decoder_start_token_id)};
    set_decoder_input_ids(decoder, init_ids);

    const auto infer_start = std::chrono::steady_clock::now();
    decoder.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    auto output_tensor = decoder.get_tensor("logits");

    auto logits_data = output_tensor.data<float>();

    int64_t output_token;
    float max_prob = -std::numeric_limits<float>::infinity();

    for (auto [_, lang_token] : config.lang_to_id) {
        auto prob = logits_data[lang_token];
        if (prob > max_prob) {
            max_prob = prob;
            output_token = lang_token;
        }
    }

    return output_token;
}

std::vector<int32_t> prepare_init_ids(ov::Tensor& encoder_hidden_state,
                                      ov::genai::DecoderCache& decoder_cache,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const bool return_timestamps,
                                      ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        if (return_timestamps) {
            return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id)};
        } else {
            return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                        static_cast<int32_t>(config.no_timestamps_token_id)};
        }
    }

    int32_t language_token_id;
    if (config.language.has_value()) {
        std::string language = *config.language;
        if (config.lang_to_id.count(language)) {
            language_token_id = static_cast<int32_t>(config.lang_to_id.at(language));
        }
    } else {
        language_token_id = detect_language(encoder_hidden_state, decoder_cache, config, raw_metrics);
    }

    int32_t task_token_id = static_cast<int32_t>(config.transcribe_token_id);
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = static_cast<int32_t>(config.translate_token_id);
    }

    if (return_timestamps) {
        return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                    language_token_id,
                                    task_token_id};
    }

    return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                language_token_id,
                                task_token_id,
                                static_cast<int32_t>(config.no_timestamps_token_id)};
}

std::pair<bool, std::vector<int64_t>> full_decode(ov::Tensor& encoder_hidden_state,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::WhisperInitializedModels& models,
                                                  std::vector<int32_t> init_ids,
                                                  const size_t max_new_tokens,
                                                  const bool return_timestamps,
                                                  ov::genai::RawPerfMetrics& raw_metrics,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    int64_t output_token = decode(encoder_hidden_state, models.decoder, init_ids, config, raw_metrics, true, return_timestamps);
    std::vector<int64_t> output_tokens{output_token};

    if (!return_timestamps && streamer && streamer->write(output_token) != ov::genai::StreamingStatus::RUNNING) {
        return {true, output_tokens};
    }

    if (max_new_tokens == 1) {
        return {false, output_tokens};
    }

    prepare_decoder_with_past(models.decoder_with_past, models.decoder, init_ids.size());

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token = decode_with_past(models.decoder_with_past,
                                             output_tokens.back(),
                                             i + init_ids.size(),
                                             config,
                                             raw_metrics,
                                             return_timestamps,
                                             output_tokens);
        update_past_key_value(models.decoder_with_past, models.decoder_with_past, i + init_ids.size());

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);

        if (!return_timestamps && streamer && streamer->write(output_token) != ov::genai::StreamingStatus::RUNNING) {
            return {true, output_tokens};
        }
    }

    return {false, output_tokens};
}

bool check_decoder_model_compatibility(const std::shared_ptr<ov::Model>& decoder) {
    for (auto input : decoder->inputs()) {
        if (input.get_any_name() == "attention_mask") {
            return true;
        }
    }
    return false;
}

void add_attention_mask_input(std::shared_ptr<ov::Model> model) {
    using namespace ov::pass::pattern;
    using namespace ov::op;
    class AttentionMaskInput : public ov::pass::MatcherPass {
    public:
        OPENVINO_MATCHER_PASS_RTTI("AttentionMaskInput");

        AttentionMaskInput(std::shared_ptr<ov::Model> model) {
            auto range = wrap_type<v4::Range>();
            auto convert1 = wrap_type<v0::Convert>({range});
            auto greater = wrap_type<v1::Greater>({convert1, any_input()});
            auto convert2 = wrap_type<v0::Convert>({greater});

            register_matcher(std::make_shared<Matcher>(convert2, this->get_type_info().name), [model](Matcher& m) {
                auto node = m.get_match_root();
                auto attention_mask = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, -1});
                attention_mask->get_output_tensor(0).set_names({"attention_mask"});
                model->add_parameters({attention_mask});
                ov::replace_node(node, attention_mask);
                return false;
            });
        }
    };

    ov::pass::Manager pm;
    pm.register_pass<AttentionMaskInput>(model);
    pm.run_passes(model);
}


ov::PartialShape get_encoder_hidden_state_shape(const std::shared_ptr<ov::Model>& encoder) {
    return encoder->output("last_hidden_state").get_partial_shape();
}

void reshape_to_static(std::shared_ptr<ov::Model> model, const uint32_t input_size, const uint32_t kvcache_size, const ov::PartialShape& lhstate_shape) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size + 1});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("cache_position") != std::string::npos) {
            new_shape = ov::PartialShape({1});
        } else if (input_name.find("encoder_hidden_states") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;     // batch_dim
            new_shape[1] = lhstate_shape[1];  // from encoder output{'last_hidden_state'}
            new_shape[2] = lhstate_shape[2];
        } else if (input_name.find("past_key_values") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[0] = 1;  // Use batch dim here
            new_shape[2] = input_name.find(".decoder") != std::string::npos
                               ? kvcache_size - input_size // kv_size for decoder
                               : lhstate_shape[1];  // hidden state size for encoder
        }
        new_shapes.emplace(input_name, new_shape);
    }

    model->reshape(new_shapes);
}

void reshape_to_static_encoder(std::shared_ptr<ov::Model> model, const size_t feature_size) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_features") != std::string::npos) {
            const auto& partial_shape = input.get_partial_shape();
            OPENVINO_ASSERT(partial_shape.size() >= 3);
            new_shape = partial_shape;
            new_shape[0] = 1;  // batch_dim
            new_shape[1] = feature_size;
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

void reshape_input_ids(std::shared_ptr<ov::Model> model, const uint32_t input_size) {
    model->reshape({{"input_ids", ov::PartialShape({1, input_size})}});
}

void preprocess_encoder(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor preprocessor(model);

    preprocessor.input("input_features").tensor().set_element_type(ov::element::Type_t::f32);
    preprocessor.input("input_features").preprocess().convert_element_type(ov::element::Type_t::f32);
    preprocessor.output("last_hidden_state").tensor().set_element_type(ov::element::Type_t::f16);

    model = preprocessor.build();
}

void preprocess_decoder(std::shared_ptr<ov::Model> model) {
    ov::preprocess::PrePostProcessor preprocessor(model);

    for (auto tensor : model->inputs()) {
        if (tensor.get_any_name().find("input_ids") != std::string::npos) {
            preprocessor.input("input_ids").tensor().set_element_type(ov::element::Type_t::i32);
            preprocessor.input("input_ids").preprocess().convert_element_type(ov::element::Type_t::i32);
        } else if (tensor.get_any_name().find("attention_mask") != std::string::npos) {
            preprocessor.input("attention_mask").tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.input("attention_mask").preprocess().convert_element_type();
        } else if (tensor.get_any_name().find("encoder_hidden_states") != std::string::npos) {
            preprocessor.input("encoder_hidden_states").tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.input("encoder_hidden_states").preprocess().convert_element_type(ov::element::Type_t::f32);
        } else if (tensor.get_any_name().find("past_key_values") != std::string::npos) {
            preprocessor.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.input(tensor.get_any_name()).preprocess().convert_element_type();

            // if (tensor.get_any_name().find(".value") != std::string::npos) {
            //    preprocessor.output(tensor.get_any_name()).tensor().set_layout(ov::Layout("NCWH"));
            //    preprocessor.output(tensor.get_any_name()).model().set_layout(ov::Layout("NCHW"));
            //} else if (tensor.get_any_name().find(".key") != std::string::npos) {
            //    preprocessor.output(tensor.get_any_name()).tensor().set_layout(ov::Layout("NCHW"));
            //    preprocessor.output(tensor.get_any_name()).model().set_layout(ov::Layout("NCHW"));
            //}
        }
    }

    for (auto tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            preprocessor.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
            preprocessor.output(tensor.get_any_name()).postprocess().convert_element_type();

            // if (tensor.get_any_name().find(".value") != std::string::npos) {
            //    preprocessor.output(tensor.get_any_name()).tensor().set_layout(ov::Layout("NCWH"));
            //    preprocessor.output(tensor.get_any_name()).model().set_layout(ov::Layout("NCHW"));
            //} else if (tensor.get_any_name().find(".key") != std::string::npos) {
            //    preprocessor.output(tensor.get_any_name()).tensor().set_layout(ov::Layout("NCHW"));
            //    preprocessor.output(tensor.get_any_name()).model().set_layout(ov::Layout("NCHW"));
            //}
        }
    }

    model = preprocessor.build();
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (int i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
    model->validate_nodes_and_infer_types();
    return model;
}

}  // namespace

namespace ov {
namespace genai {

ov::InferRequest DecoderCache::get_model(uint8_t input_ids_size) {
    if (m_cache.find(input_ids_size) == m_cache.cend()) {
        reshape_input_ids(m_decoder_model, input_ids_size);

        ov::Core core = utils::singleton_core();
        ov::CompiledModel compiled_model = core.compile_model(m_decoder_model, "NPU", m_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper decoder model");
        m_cache.emplace(input_ids_size, compiled_model.create_infer_request());
    }

    return m_cache.at(input_ids_size);
}

WhisperPipeline::StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& models_path,
                                                              const ov::AnyMap& properties)
    : WhisperPipelineImplBase{models_path} {
    ov::Core core = utils::singleton_core();

    auto encoder_model = core.read_model(models_path / "openvino_encoder_model.xml", {}, properties);
    auto decoder_model = core.read_model(models_path / "openvino_decoder_model.xml", {}, properties);
    auto decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml", {}, properties);

    add_attention_mask_input(decoder_with_past_model);

    size_t max_sequence_length = 448;

    reshape_to_static_encoder(encoder_model, m_feature_extractor.feature_size);

    auto last_hidden_state_shape = get_encoder_hidden_state_shape(encoder_model);
    reshape_to_static(decoder_model, 1, 1, last_hidden_state_shape);
    reshape_to_static(decoder_with_past_model, 1, max_sequence_length, last_hidden_state_shape);

    // Replace KV-tensors for the entire cache to tensors only for new token
    decoder_with_past_model = redirect_new_kv_to_output(decoder_with_past_model);

    preprocess_encoder(encoder_model);
    preprocess_decoder(decoder_model);
    preprocess_decoder(decoder_with_past_model);

    ov::CompiledModel compiled_model;
    compiled_model = core.compile_model(encoder_model, "NPU", properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper encoder model");
    m_models.encoder = compiled_model.create_infer_request();

    // Will compile decoder model when it's needed 
    m_decoder_cache = DecoderCache(decoder_model, properties);

    compiled_model = core.compile_model(decoder_with_past_model, "NPU", properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Static Whisper decoder with past model");
    m_models.decoder_with_past = compiled_model.create_infer_request();

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1) {
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
    }
}

WhisperDecodedResults WhisperPipeline::StaticWhisperPipeline::generate(
    const RawSpeechInput& raw_speech_input,
    OptionalWhisperGenerationConfig generation_config,
    const std::shared_ptr<StreamerBase> streamer_ptr) {
    auto start_time = std::chrono::steady_clock::now();
    WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    
    // If stop_token_ids were not provided, take value from default m_generation_config
    if (config.stop_token_ids.empty())
        config.stop_token_ids = m_generation_config.stop_token_ids;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    OPENVINO_ASSERT(!config.initial_prompt.has_value(), "'initial_prompt' parameter is not supported on NPU device.");
    OPENVINO_ASSERT(!config.hotwords.has_value(), "'hotwords' parameter is not supported on NPU device.");

    size_t max_new_tokens = config.get_max_new_tokens();

    WhisperPerfMetrics perf_metrics;
    perf_metrics.num_input_tokens = 0;
    RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const auto extract_start = std::chrono::steady_clock::now();
    auto input_features = m_feature_extractor.extract(raw_speech_input);
    const auto extract_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start);
    perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(extract_ms);

    const bool is_shortform = input_features.n_frames <= m_feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    std::vector<int32_t> init_ids;
    std::vector<int64_t> output_tokens;
    std::vector<Segment> segments;

    // 0.02 by default
    const float time_precision =
        static_cast<float>(m_feature_extractor.chunk_length) / m_model_config.max_source_positions;
    size_t segment_offset = 0;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
        if (output_tokens.size() >= max_new_tokens) {
            break;
        }

        auto input_features_chunk =
            input_features.get_data_with_offset(chunk_offset, m_feature_extractor.nb_max_frames);

        ov::Tensor hidden_state_tensor = encode(m_models.encoder,
                                                input_features_chunk,
                                                m_feature_extractor.feature_size,
                                                m_feature_extractor.nb_max_frames,
                                                raw_metrics);

        // prepare init_ids just once for whole input
        if (init_ids.empty()) {
            init_ids = prepare_init_ids(hidden_state_tensor, m_decoder_cache, config, return_timestamps, raw_metrics);

            // Get decoder with size of input_ids
            m_models.decoder = m_decoder_cache.get_model(init_ids.size());
        }

        auto [cancelled, chunk_output_tokens] = full_decode(hidden_state_tensor,
                                                            config,
                                                            m_models,
                                                            init_ids,
                                                            max_new_tokens - output_tokens.size(),
                                                            return_timestamps,
                                                            raw_metrics,
                                                            streamer_ptr);

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  m_feature_extractor.nb_max_frames,
                                                                  time_precision);

            ov::genai::utils::filter_non_segment_metrics(raw_metrics, output_tokens.size(), extracted_segments.segment_ranges);

            segments.insert(segments.end(), extracted_segments.segments.begin(), extracted_segments.segments.end());

            output_tokens.insert(output_tokens.end(),
                                 extracted_segments.non_timestamp_tokens.begin(),
                                 extracted_segments.non_timestamp_tokens.end());

            if (streamer_ptr && streamer_ptr->write(extracted_segments.non_timestamp_tokens) != StreamingStatus::RUNNING) {
                cancelled = true;
                break;
            }

            segment_offset = extracted_segments.last_offset;
        } else {
            output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());
        }

        if (is_shortform) {
            segment_offset = input_features.n_frames;
        }

        if (cancelled) {
            break;
        }
    }

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    auto decode_start_time = std::chrono::steady_clock::now();
    WhisperDecodedResults result{std::vector{m_tokenizer.decode(output_tokens)}, std::vector{1.f}};
    result.perf_metrics = perf_metrics;
    result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));

    // if return_timestamps wasn't enabled by user
    if (!config.return_timestamps) {
        return result;
    }

    if (segments.size()) {
        std::vector<WhisperDecodedResultChunk> chunks;
        chunks.reserve(segments.size());

        for (auto& segment : segments) {
            decode_start_time = std::chrono::steady_clock::now();
            chunks.push_back(
                WhisperDecodedResultChunk{segment.m_start, segment.m_end, m_tokenizer.decode(segment.m_tokens)});
            result.perf_metrics.raw_metrics.detokenization_durations.emplace_back(
                    PerfMetrics::get_microsec(std::chrono::steady_clock::now() - decode_start_time));
        }

        result.chunks = chunks;
    }

    auto& metrics = result.perf_metrics;
    metrics.load_time = this->m_load_time_ms;
    auto stop_time = std::chrono::steady_clock::now();
    metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    metrics.raw_metrics.tokenization_durations.emplace_back(MicroSeconds(0.0f));
    metrics.evaluate_statistics(start_time);

    return result;
}

}  // namespace genai
}  // namespace ov
