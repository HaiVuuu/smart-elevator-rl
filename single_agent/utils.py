import pygame
import time
import numpy as np
from .normal_algorithm import NormalAlgorithm

def evaluate_model(model, env, max_steps, deterministic=False,delay = 0.1,render = True):
    is_rule_based = isinstance(model, NormalAlgorithm)


    obs, _ = env.reset()
    done, steps, total_reward = False, 0, 0

    info = {}
    while not done and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        if done:
            break
        if is_rule_based:
            action = model.predict(env.building)
            obs, reward, done, _, info = env.step(action)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(action)

        if render:
            env.render()
        time.sleep(delay)

        total_reward += reward
        steps += 1

    # If loop ended due to max_steps, get one last info read
    # if not done:
    #     _, _, _, _, info = env.step(action)

    return info
