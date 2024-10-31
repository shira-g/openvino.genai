// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <ctime>
#include <cassert>
#include <filesystem>

#include "image_generation/diffusion_pipeline.hpp"

#include "openvino/genai/image_generation/autoencoder_kl.hpp"
#include "openvino/genai/image_generation/clip_text_model.hpp"
#include "openvino/genai/image_generation/clip_text_model_with_projection.hpp"
#include "openvino/genai/image_generation/unet2d_condition_model.hpp"

#include "json_utils.hpp"

namespace ov {
namespace genai {

class StableDiffusionXLPipeline : public DiffusionPipeline {
public:
    StableDiffusionXLPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir) :
        DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(root_dir / "text_encoder");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(root_dir / "text_encoder_2");
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet");
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder");
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder");
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());
    }

    StableDiffusionXLPipeline(PipelineType pipeline_type, const std::filesystem::path& root_dir, const std::string& device, const ov::AnyMap& properties) :
        DiffusionPipeline(pipeline_type) {
        const std::filesystem::path model_index_path = root_dir / "model_index.json";
        std::ifstream file(model_index_path);
        OPENVINO_ASSERT(file.is_open(), "Failed to open ", model_index_path);

        nlohmann::json data = nlohmann::json::parse(file);
        using utils::read_json_param;

        set_scheduler(Scheduler::from_config(root_dir / "scheduler/scheduler_config.json"));

        const std::string text_encoder = data["text_encoder"][1].get<std::string>();
        if (text_encoder == "CLIPTextModel") {
            m_clip_text_encoder = std::make_shared<CLIPTextModel>(
                root_dir / "text_encoder",
                device,
                properties_for_text_encoder(properties, "lora_te1")
            );
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string text_encoder_2 = data["text_encoder_2"][1].get<std::string>();
        if (text_encoder_2 == "CLIPTextModelWithProjection") {
            m_clip_text_encoder_with_projection = std::make_shared<CLIPTextModelWithProjection>(
                root_dir / "text_encoder_2",
                device,
                properties_for_text_encoder(properties, "lora_te2")
            );
        } else {
            OPENVINO_THROW("Unsupported '", text_encoder, "' text encoder type");
        }

        const std::string unet = data["unet"][1].get<std::string>();
        if (unet == "UNet2DConditionModel") {
            m_unet = std::make_shared<UNet2DConditionModel>(root_dir / "unet", device, properties);
        } else {
            OPENVINO_THROW("Unsupported '", unet, "' UNet type");
        }

        const std::string vae = data["vae"][1].get<std::string>();
        if (vae == "AutoencoderKL") {
            if (m_pipeline_type == PipelineType::TEXT_2_IMAGE)
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_decoder", device, properties);
            else if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
                m_vae = std::make_shared<AutoencoderKL>(root_dir / "vae_encoder", root_dir / "vae_decoder", device, properties);
            } else {
                OPENVINO_ASSERT("Unsupported pipeline type");
            }
        } else {
            OPENVINO_THROW("Unsupported '", vae, "' VAE decoder type");
        }

        // initialize generation config
        initialize_generation_config(data["_class_name"].get<std::string>());

        update_adapters_from_properties(properties, m_generation_config.adapters);
    }

    StableDiffusionXLPipeline(
        PipelineType pipeline_type,
        const CLIPTextModel& clip_text_model,
        const CLIPTextModelWithProjection& clip_text_model_with_projection,
        const UNet2DConditionModel& unet,
        const AutoencoderKL& vae)
        : DiffusionPipeline(pipeline_type),
          m_clip_text_encoder(std::make_shared<CLIPTextModel>(clip_text_model)),
          m_clip_text_encoder_with_projection(std::make_shared<CLIPTextModelWithProjection>(clip_text_model_with_projection)),
          m_unet(std::make_shared<UNet2DConditionModel>(unet)),
          m_vae(std::make_shared<AutoencoderKL>(vae)) {
        initialize_generation_config("StableDiffusionXLPipeline");
    }

    void reshape(const int num_images_per_prompt, const int height, const int width, const float guidance_scale) override {
        check_image_size(height, width);

        const size_t batch_size_multiplier = do_classifier_free_guidance(guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        m_clip_text_encoder->reshape(batch_size_multiplier);
        m_clip_text_encoder_with_projection->reshape(batch_size_multiplier);
        m_unet->reshape(num_images_per_prompt * batch_size_multiplier, height, width, m_clip_text_encoder->get_config().max_position_embeddings);
        m_vae->reshape(num_images_per_prompt, height, width);
    }

    void compile(const std::string& device, const ov::AnyMap& properties) override {
        update_adapters_from_properties(properties, m_generation_config.adapters);

        m_clip_text_encoder->compile(device, properties);
        m_clip_text_encoder_with_projection->compile(device, properties);
        m_unet->compile(device, properties);
        m_vae->compile(device, properties);
    }

    ov::Tensor prepare_latents(ov::Tensor initial_image, const ImageGenerationConfig& generation_config) const override {
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        ov::Shape latent_shape{generation_config.num_images_per_prompt, unet_config.in_channels,
                               generation_config.height / vae_scale_factor, generation_config.width / vae_scale_factor};
        ov::Tensor latent(ov::element::f32, {});

        if (initial_image) {
            latent = m_vae->encode(initial_image);
            m_scheduler->add_noise(latent, generation_config.generator);
        } else {
            latent = generation_config.generator->randn_tensor(latent_shape);

            // latents are multiplied by 'init_noise_sigma'
            float * latent_data = latent.data<float>();
            for (size_t i = 0; i < latent.get_size(); ++i)
                latent_data[i] *= m_scheduler->get_init_noise_sigma();
        }

        return latent;
    }

    ov::Tensor generate(const std::string& positive_prompt,
                        ov::Tensor initial_image,
                        const ov::AnyMap& properties) override {
        ImageGenerationConfig generation_config = m_generation_config;
        generation_config.update_generation_config(properties);

        // Stable Diffusion pipeline
        // see https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline#deconstruct-the-stable-diffusion-pipeline

        const auto& unet_config = m_unet->get_config();
        const size_t batch_size_multiplier = do_classifier_free_guidance(generation_config.guidance_scale) ? 2 : 1;  // Unet accepts 2x batch in case of CFG
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        if (generation_config.height < 0)
            generation_config.height = unet_config.sample_size * vae_scale_factor;
        if (generation_config.width < 0)
            generation_config.width = unet_config.sample_size * vae_scale_factor;
        check_inputs(generation_config, initial_image);

        m_clip_text_encoder->set_adapters(generation_config.adapters);
        m_clip_text_encoder_with_projection->set_adapters(generation_config.adapters);
        m_unet->set_adapters(generation_config.adapters);

        if (generation_config.generator == nullptr) {
            uint32_t seed = time(NULL);
            generation_config.generator = std::make_shared<CppStdGenerator>(seed);
        }

        std::vector<float> time_ids = {static_cast<float>(generation_config.width),
                                       static_cast<float>(generation_config.height),
                                       0,
                                       0,
                                       static_cast<float>(generation_config.width),
                                       static_cast<float>(generation_config.height),
                                       };
        ov::Tensor add_time_ids(ov::element::f32, {batch_size_multiplier, time_ids.size()});
        float* add_time_ids_data = add_time_ids.data<float>();
        std::copy(time_ids.begin(), time_ids.end(), add_time_ids_data);

        if (batch_size_multiplier > 1) {
            std::copy(time_ids.begin(), time_ids.end(), add_time_ids_data + time_ids.size());
        }

        ov::Tensor add_text_embeds = m_clip_text_encoder_with_projection->infer(positive_prompt, generation_config.negative_prompt, batch_size_multiplier > 1);
        m_clip_text_encoder->infer(positive_prompt, generation_config.negative_prompt, batch_size_multiplier > 1);

        // prompt_embeds = prompt_embeds.hidden_states[-2]
        size_t idx_hidden_state_1 = m_clip_text_encoder->get_config().num_hidden_layers;
        ov::Tensor encoder_hidden_states_1 = m_clip_text_encoder->get_output_tensor(idx_hidden_state_1);
        size_t idx_hidden_state_2 = m_clip_text_encoder_with_projection->get_config().num_hidden_layers;
        ov::Tensor encoder_hidden_states_2 = m_clip_text_encoder_with_projection->get_output_tensor(idx_hidden_state_2);

        ov::Shape ehs_1_shape = encoder_hidden_states_1.get_shape();
        ov::Shape ehs_2_shape = encoder_hidden_states_2.get_shape();

        OPENVINO_ASSERT(ehs_1_shape[0] == ehs_2_shape[0] && ehs_1_shape[1] == ehs_2_shape[1],
                        "Tensors for concatenation must have the same dimensions");

        // concatenate hidden_states from two encoders
        ov::Shape encoder_hidden_states_shape = {ehs_1_shape[0], ehs_1_shape[1], ehs_1_shape[2] + ehs_2_shape[2]};
        ov::Tensor encoder_hidden_states(encoder_hidden_states_1.get_element_type(), encoder_hidden_states_shape);

        const float* ehs_1_data = encoder_hidden_states_1.data<const float>();
        const float* ehs_2_data = encoder_hidden_states_2.data<const float>();
        float* encoder_hidden_states_data = encoder_hidden_states.data<float>();

        for (size_t i = 0; i < ehs_1_shape[0]; ++i) {
            for (size_t j = 0; j < ehs_1_shape[1]; ++j) {
                size_t offset_1 = (i * ehs_1_shape[1] + j) * ehs_1_shape[2];
                size_t offset_2 = (i * ehs_2_shape[1] + j) * ehs_2_shape[2];

                size_t step = (i * ehs_1_shape[1] + j) * (ehs_1_shape[2] + ehs_2_shape[2]);

                std::memcpy(encoder_hidden_states_data + step, ehs_1_data + offset_1, ehs_1_shape[2] * sizeof(float));
                std::memcpy(encoder_hidden_states_data + step + ehs_1_shape[2], ehs_2_data + offset_2, ehs_2_shape[2] * sizeof(float));
            }
        }

        // replicate encoder hidden state to UNet model
        if (generation_config.num_images_per_prompt == 1) {
            // reuse output of text encoder directly w/o extra memory copy
            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states);
            m_unet->set_hidden_states("text_embeds", add_text_embeds);
            m_unet->set_hidden_states("time_ids", add_time_ids);

        } else {
            ov::Shape enc_shape = encoder_hidden_states.get_shape();
            enc_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor encoder_hidden_states_repeated(encoder_hidden_states.get_element_type(), enc_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                batch_copy(encoder_hidden_states, encoder_hidden_states_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    batch_copy(encoder_hidden_states, encoder_hidden_states_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("encoder_hidden_states", encoder_hidden_states_repeated);

            ov::Shape t_emb_shape = add_text_embeds.get_shape();
            t_emb_shape[0] *= generation_config.num_images_per_prompt;

            ov::Tensor add_text_embeds_repeated(add_text_embeds.get_element_type(), t_emb_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                batch_copy(add_text_embeds, add_text_embeds_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    batch_copy(add_text_embeds, add_text_embeds_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("text_embeds", add_text_embeds_repeated);

            ov::Shape t_ids_shape = add_time_ids.get_shape();
            t_ids_shape[0] *= generation_config.num_images_per_prompt;
            ov::Tensor add_time_ids_repeated(add_time_ids.get_element_type(), t_ids_shape);
            for (size_t n = 0; n < generation_config.num_images_per_prompt; ++n) {
                batch_copy(add_time_ids, add_time_ids_repeated, 0, n);
                if (batch_size_multiplier > 1) {
                    batch_copy(add_time_ids, add_time_ids_repeated,
                        1, generation_config.num_images_per_prompt + n);
                }
            }

            m_unet->set_hidden_states("time_ids", add_time_ids_repeated);
        }

        m_scheduler->set_timesteps(generation_config.num_inference_steps, generation_config.strength);
        std::vector<std::int64_t> timesteps = m_scheduler->get_timesteps();

        // preparate initial latents
        ov::Tensor latent = prepare_latents(initial_image, generation_config);

        ov::Shape latent_shape_cfg = latent.get_shape();
        latent_shape_cfg[0] *= batch_size_multiplier;
        ov::Tensor latent_cfg(ov::element::f32, latent_shape_cfg);

        ov::Tensor denoised, noisy_residual_tensor(ov::element::f32, {});
        for (size_t inference_step = 0; inference_step < generation_config.num_inference_steps; inference_step++) {
            // concat the same latent twice along a batch dimension in case of CFG
            if (batch_size_multiplier > 1) {
                batch_copy(latent, latent_cfg, 0, 0, generation_config.num_images_per_prompt);
                batch_copy(latent, latent_cfg, 0, generation_config.num_images_per_prompt, generation_config.num_images_per_prompt);
            } else {
                std::memcpy(latent_cfg.data<float>(), latent.data<float>(), latent_cfg.get_size() * sizeof(float));
            }

            m_scheduler->scale_model_input(latent_cfg, inference_step);

            ov::Tensor timestep(ov::element::i64, {1}, &timesteps[inference_step]);
            ov::Tensor noise_pred_tensor = m_unet->infer(latent_cfg, timestep);

            ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
            noise_pred_shape[0] /= batch_size_multiplier;
            noisy_residual_tensor.set_shape(noise_pred_shape);

            if (batch_size_multiplier > 1) {
                // perform guidance
                float* noisy_residual = noisy_residual_tensor.data<float>();
                const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
                const float* noise_pred_text = noise_pred_uncond + noisy_residual_tensor.get_size();

                for (size_t i = 0; i < noisy_residual_tensor.get_size(); ++i) {
                    noisy_residual[i] = noise_pred_uncond[i] +
                        generation_config.guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
                }
            } else {
                noisy_residual_tensor = noise_pred_tensor;
            }

            auto scheduler_step_result = m_scheduler->step(noisy_residual_tensor, latent, inference_step, generation_config.generator);
            latent = scheduler_step_result["latent"];

            // check whether scheduler returns "denoised" image, which should be passed to VAE decoder
            const auto it = scheduler_step_result.find("denoised");
            denoised = it != scheduler_step_result.end() ? it->second : latent;
        }

        return m_vae->decode(denoised);
    }

private:
    bool do_classifier_free_guidance(float guidance_scale) const {
        return guidance_scale > 1.0f && m_unet->get_config().time_cond_proj_dim < 0;
    }

    void initialize_generation_config(const std::string& class_name) override {
        assert(m_unet != nullptr);
        assert(m_vae != nullptr);
        const auto& unet_config = m_unet->get_config();
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();

        m_generation_config.height = unet_config.sample_size * vae_scale_factor;
        m_generation_config.width = unet_config.sample_size * vae_scale_factor;

        if (class_name == "StableDiffusionXLPipeline") {
            m_generation_config.guidance_scale = 5.0f;
            m_generation_config.num_inference_steps = 50;
            m_generation_config.strength = m_pipeline_type == PipelineType::IMAGE_2_IMAGE ? 0.3f : 1.0f;
        } else {
            OPENVINO_THROW("Unsupported class_name '", class_name, "'. Please, contact OpenVINO GenAI developers");
        }
    }

    void check_image_size(const int height, const int width) const override {
        assert(m_vae != nullptr);
        const size_t vae_scale_factor = m_vae->get_vae_scale_factor();
        OPENVINO_ASSERT((height % vae_scale_factor == 0 || height < 0) &&
            (width % vae_scale_factor == 0 || width < 0), "Both 'width' and 'height' must be divisible by",
            vae_scale_factor);
    }

    void check_inputs(const ImageGenerationConfig& generation_config, ov::Tensor initial_image) const override {
        check_image_size(generation_config.width, generation_config.height);

        const bool is_classifier_free_guidance = do_classifier_free_guidance(generation_config.guidance_scale);
        const char * const pipeline_name = "Stable Diffusion XL";

        OPENVINO_ASSERT(generation_config.prompt_3 == std::nullopt, "Prompt 3 is not used by ", pipeline_name);
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt.empty(), "Negative prompt is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(is_classifier_free_guidance || generation_config.negative_prompt_2 == std::nullopt, "Negative prompt 2 is not used when guidance scale <= 1.0");
        OPENVINO_ASSERT(generation_config.negative_prompt_3 == std::nullopt, "Negative prompt 3 is not used by ", pipeline_name);

        if (m_pipeline_type == PipelineType::IMAGE_2_IMAGE) {
            if (initial_image) {
                ov::Shape initial_image_shape = initial_image.get_shape();
                size_t height = initial_image_shape[1], width = initial_image_shape[2];

                OPENVINO_ASSERT(generation_config.height == height,
                    "Height for initial (", height, ") and generated (", generation_config.height,") images must be the same");
                OPENVINO_ASSERT(generation_config.width == width,
                    "Width for initial (", width, ") and generated (", generation_config.width,") images must be the same");
            }

            OPENVINO_ASSERT(generation_config.strength >= 0.0f && generation_config.strength <= 1.0f,
                "'Strength' generation parameter must be withion [0, 1] range");
        } else {
            OPENVINO_ASSERT(generation_config.strength == 1.0f, "'Strength' generation parameter must be 1.0f for Text 2 image pipeline");
            OPENVINO_ASSERT(!initial_image, "Internal error: initial_image must be empty for Text 2 image pipeline");
        }
    }

    ov::AnyMap properties_for_text_encoder(ov::AnyMap properties, const std::string& tensor_name_prefix) {
        std::optional<AdapterConfig> adapters;
        if(update_adapters_from_properties(properties, adapters) && !adapters->get_tensor_name_prefix()) {
            adapters->set_tensor_name_prefix(tensor_name_prefix);
            properties[ov::genai::adapters.name()] = *adapters;
        }
        return properties;
    }

    friend class Text2ImagePipeline;
    friend class Image2ImagePipeline;

    std::shared_ptr<CLIPTextModel> m_clip_text_encoder;
    std::shared_ptr<CLIPTextModelWithProjection> m_clip_text_encoder_with_projection;
    std::shared_ptr<UNet2DConditionModel> m_unet;
    std::shared_ptr<AutoencoderKL> m_vae;
};

}  // namespace genai
}  // namespace ov
