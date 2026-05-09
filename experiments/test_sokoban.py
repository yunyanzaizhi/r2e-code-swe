#!/usr/bin/env python3
"""Quick smoke test for Sokoban environment."""
from sokoban.env import SokobanEnv
e = SokobanEnv(mode="tiny_rgb_array", dim_room=(6,6), num_boxes=1)
obs, info = e.reset(seed=42)
print("Sokoban env OK!")
print("Observation:")
print(obs)
print()
obs2, r, done, info2 = e.step(1)  # Up
print("After Up: reward=", r, "done=", done)
print(obs2)

# Test structured summary memory
from agent_system.memory.structured_summary import StructuredSummaryMemory
mem = StructuredSummaryMemory()
mem.reset(batch_size=1)
mem.store({'text_obs': [obs], 'action': ['Up']}, rewards=[r])
contexts, lengths = mem.fetch()
print("\nStructured Summary:")
print(contexts[0])
print("OK - All tests passed!")
