// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/llm_pipeline.hpp"

int main(int argc, char* argv[]) try {
    if (4 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> <DRAFT_MODEL_DIR> '<PROMPT>'");
    }

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    // add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5;
    // add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
    // config.assistant_confidence_threshold = 0.4

    std::string main_model_path = argv[1];
    std::string draft_model_path = argv[2];
    std::string prompt = argv[3];
    // std::string prompt = "Article: While Google and Apple are developing self driving vehicles, Nasa has revealed its vision - and its a lot more fun. Called the Modular Robotic Vehicle, or MRV, it was developed at NASA's Johnson Space Center to show off the technologies that could let man move across other planets. On Earth, however, it also has some neat tricks - from the ability to park sideways to being able to spin on the spot, as well as being driven remotely at up to 70mph. Watch the buggy in action below . Unlike a normal car, the buggy has no mechanical linkages to the propulsion, steering, or brake actuators, the driver of an MRV relies completely on control inputs being converted to electrical signals and then transmitted by wires to the vehicle's motors. A turn of the steering wheel, for instance, is recorded by sensors and sent to computers at the rear of the vehicle. These computers interpret that signal and instruct motors at one or all four of the wheels to move at the appropriate rate, causing the vehicle to turn as commanded. Nasa says the vehicles was buit 'in order to advance technologies that have applications for future vehicles both in space and on Earth.' 'With seating for two people, MRV is a fully electric vehicle well-suited for busy urban environments, it says. MRV is driven by four independent wheel modules called e-corners. Each e-corner consists of a redundant steering actuator, a passive trailing arm suspension, an in-wheel propulsion motor, and a motor-driven friction braking system. Each e-corner can be controlled independently and rotated ±180 degrees about its axis. This allows for a suite of driving modes allowing MRV to maneuver unlike any traditional vehicle on the road. In addition to conventional front two wheel steering, the back wheels can also articulate allowing for turning radiuses as tight as zero. The driving mode can be switched so that all four wheels point and move in the same direction achieving an omni-directional, crab-like motion. This makes a maneuver such as parallel parking as easy as driving next to an available spot, stopping, and then operating sideways to slip directly in between two cars. 'This two-seater vehicle was designed to meet the growing challenges and demands of urban transportation,' said Mason Markee, also with Johnson. 'The MRV would be ideal for daily transportation in an urban environment with a designed top speed of 70 km/hr and range of 100 km of city driving on a single charge of the battery. 'The size and maneuverability of MRV gives it an advantage in navigating and parking in tight quarters.' With a designed top speed of around 70 km/hr, test driving proved to be a bit of fun. Justin Ridley told an Nasa magazine: 'It's like driving on ice but having complete control. 'It's a blast to ride in and even more fun to drive. We've talked about it being like an amusement park ride. 'The 'fun' of driving was not something we tried to design for, just something that came out of the design. 'Once we got it running many of us commented that we had no idea it was going to be able to do the things it does.' 'This work also allowed us to develop some technologies we felt were needed for our future rovers,' said Ridley. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. The driving mode can be switched so that all four wheels point and move in the same direction - perfect for parallel parking. 'These include redundant by-wire systems, liquid cooling, motor technology, advanced vehicle control algorithms. 'We were able to learn a lot about these and other technologies by building this vehicle.' The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. Both of these interfaces were specially designed to mimic the feel of the mechanical/hydraulic systems that people are used to feeling when driving their own cars. The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. 'While the vehicle as a whole is designed around operating in an urban environment, the core technologies are advancements used in many of our robotic systems and rovers,' said Mason. 'Actuators, motor controllers, sensors, batteries, BMS, component cooling, sealing, and software are all examples of technologies that are being devel oped and tested in MRV that will be used in next generation rover systems. 'The technologies developed in MRV have direct application in future manned vehicles undertaking missions on the surface of Earth's moon, on Mars, or even an asteroid. 'Additionally, MRV provides a platform to learn lessons that could drive the next generation of automobiles.";
//     std::string prompt = R"(<|user|> ###
// Article: John Carver will turn to Siem de Jong in a bid to save Newcastle’s sorry season. The £6million summer signing suffered a collapsed lung in February and it was initially feared he would miss the remainder of the campaign, especially as it was the second time he had fallen victim to the problem. De Jong has started just one Premier League game since arriving from Ajax and he was only days away from a return to action following a five-month layoff with a torn thigh muscle when he was diagnosed with the collapsed lung. Newcastle attacking midfielder Siem de Jong could return to action before the end of the current campaign . De Jong, pictured in training on April 9 with his team-mates, will feature for Newcastle's reserve side . However, the 26-year-old returned to training earlier this month and will now feature for United’s reserves when they entertain Derby County at St James’ Park on Wednesday night, as will England Under 19 winger Rolando Aarons. Sunday’s visit of Spurs will probably come too soon for De Jong and hamstring absentee Aarons – who has not played since November - but Carver will be desperately hoping to have them available for the final five matches of a season which is in danger of ending on a sour note. Newcastle have lost five on the spin and their head coach has admitted that he does not know where the next point is coming from. They have scored just once in eight hours and would be fighting relegation had it not been for a five-match winning run under Alan Pardew last autumn. Supporters are organising a boycott ahead of the televised clash with Spurs, where thousands are expected to stay away in protest at Mike Ashley’s running of the club. Newcastle boss John Carver will be hoping his side will improve following the return of De Jong .

// Summarize the above article in 5 sentence.
// <|end|><|assistant|>)";


    
    // User can run main and draft model on different devices.
    // Please, set device for main model in `LLMPipeline` constructor and in in `ov::genai::draft_model` for draft.
    std::string main_device = "GPU", draft_device = main_device;

    // Perform the inference
    auto get_default_block_size = [](const std::string& device) {
        const size_t cpu_block_size = 32;
        const size_t gpu_block_size = 16;

        bool is_gpu = device.find("GPU") != std::string::npos;

        return is_gpu ? gpu_block_size : cpu_block_size;
    };

    ov::genai::SchedulerConfig scheduler_config;
    scheduler_config.cache_size = 5;
    scheduler_config.block_size = get_default_block_size(main_device);

    // Example to run main_model on GPU and draft_model on CPU:
    // ov::genai::LLMPipeline pipe(main_model_path, "GPU", ov::genai::draft_model(draft_model_path, "CPU"), ov::genai::scheduler_config(scheduler_config));
    ov::genai::LLMPipeline pipe(main_model_path, main_device, ov::genai::draft_model(draft_model_path, draft_device), ov::genai::scheduler_config(scheduler_config));

    // auto streamer = [](std::string subword) {
    //     std::cout << subword << std::flush;
    //     return false;
    // };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    auto time0 = std::chrono::high_resolution_clock::now();
    std::string result = pipe.generate(prompt, config);
    auto time1 = std::chrono::high_resolution_clock::now();
    auto time_res0 = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();
    std::cout << "Answer: " << result << std::endl;
    std::cout << "\n";
    std::cout << "\n";
    std::cout << "************** Final statistics **************\n";
    std::cout << "Total execution time = " << time_res0 << " ms\n";
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
