#!/usr/bin/env python3
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai
import queue
import threading

def streamer(subword): 
        print(subword, end='', flush=True) 
        # Return flag corresponds whether generation should be stopped. 
        # False means continue generation. 
        return False 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('draft_model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    # User can run main and draft model on different devices.
    # Please, set device for main model in `openvino_genai.LLMPipeline` constructor and in openvino_genai.draft_model` for draft.
    main_device = 'GPU'  # GPU can be used as well
    draft_device = main_device

    scheduler_config = openvino_genai.SchedulerConfig()
    # cache params
    scheduler_config.cache_size = 2
    scheduler_config.block_size = 16

    draft_model = openvino_genai.draft_model(args.draft_model_dir, draft_device)

    pipe = openvino_genai.LLMPipeline(args.model_dir, main_device, scheduler_config=scheduler_config, draft_model=draft_model)
    
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    # Speculative decoding generation parameters like `num_assistant_tokens` and `assistant_confidence_threshold` are mutually excluded
    # add parameter to enable speculative decoding to generate `num_assistant_tokens` candidates by draft_model per iteration
    config.num_assistant_tokens = 5
    # add parameter to enable speculative decoding to generate candidates by draft_model while candidate probability is higher than `assistant_confidence_threshold`
    # config.assistant_confidence_threshold = 0.4

    # Since the streamer is set, the results will be printed 
    # every time a new token is generated and put into the streamer queue.
    import time
    
    args.prompt = "Article: While Google and Apple are developing self driving vehicles, Nasa has revealed its vision - and its a lot more fun. Called the Modular Robotic Vehicle, or MRV, it was developed at NASA's Johnson Space Center to show off the technologies that could let man move across other planets. On Earth, however, it also has some neat tricks - from the ability to park sideways to being able to spin on the spot, as well as being driven remotely at up to 70mph. Watch the buggy in action below . Unlike a normal car, the buggy has no mechanical linkages to the propulsion, steering, or brake actuators, the driver of an MRV relies completely on control inputs being converted to electrical signals and then transmitted by wires to the vehicle's motors. A turn of the steering wheel, for instance, is recorded by sensors and sent to computers at the rear of the vehicle. These computers interpret that signal and instruct motors at one or all four of the wheels to move at the appropriate rate, causing the vehicle to turn as commanded. Nasa says the vehicles was buit 'in order to advance technologies that have applications for future vehicles both in space and on Earth.' 'With seating for two people, MRV is a fully electric vehicle well-suited for busy urban environments, it says. MRV is driven by four independent wheel modules called e-corners. Each e-corner consists of a redundant steering actuator, a passive trailing arm suspension, an in-wheel propulsion motor, and a motor-driven friction braking system. Each e-corner can be controlled independently and rotated ±180 degrees about its axis. This allows for a suite of driving modes allowing MRV to maneuver unlike any traditional vehicle on the road. In addition to conventional front two wheel steering, the back wheels can also articulate allowing for turning radiuses as tight as zero. The driving mode can be switched so that all four wheels point and move in the same direction achieving an omni-directional, crab-like motion. This makes a maneuver such as parallel parking as easy as driving next to an available spot, stopping, and then operating sideways to slip directly in between two cars. 'This two-seater vehicle was designed to meet the growing challenges and demands of urban transportation,' said Mason Markee, also with Johnson. 'The MRV would be ideal for daily transportation in an urban environment with a designed top speed of 70 km/hr and range of 100 km of city driving on a single charge of the battery. 'The size and maneuverability of MRV gives it an advantage in navigating and parking in tight quarters.' With a designed top speed of around 70 km/hr, test driving proved to be a bit of fun. Justin Ridley told an Nasa magazine: 'It's like driving on ice but having complete control. 'It's a blast to ride in and even more fun to drive. We've talked about it being like an amusement park ride. 'The 'fun' of driving was not something we tried to design for, just something that came out of the design. 'Once we got it running many of us commented that we had no idea it was going to be able to do the things it does.' 'This work also allowed us to develop some technologies we felt were needed for our future rovers,' said Ridley. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. The driving mode can be switched so that all four wheels point and move in the same direction - perfect for parallel parking. 'These include redundant by-wire systems, liquid cooling, motor technology, advanced vehicle control algorithms. 'We were able to learn a lot about these and other technologies by building this vehicle.' The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. The driver controls MRV with a conventional looking steering wheel and accelerator/brake pedal assembly. Both of these interfaces were specially designed to mimic the feel of the mechanical/hydraulic systems that people are used to feeling when driving their own cars. The buggy can also be controlled remotely, and in the future Nasa says this system can be expanded to allow for autonomous driving. 'While the vehicle as a whole is designed around operating in an urban environment, the core technologies are advancements used in many of our robotic systems and rovers,' said Mason. 'Actuators, motor controllers, sensors, batteries, BMS, component cooling, sealing, and software are all examples of technologies that are being devel oped and tested in MRV that will be used in next generation rover systems. 'The technologies developed in MRV have direct application in future manned vehicles undertaking missions on the surface of Earth's moon, on Mars, or even an asteroid. 'Additionally, MRV provides a platform to learn lessons that could drive the next generation of automobiles."
    start = time.time()
    result = pipe.generate(args.prompt, config)
    print(f"Result: {result}")
    end = time.time()
    print(end - start)

if '__main__' == __name__:
    main()
