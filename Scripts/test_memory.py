from BaseClasses import Memory
import numpy as np
import copy

mem = Memory(5, 2, 1)

print(mem.data)

print("-" * 10)

for i in range(5):
    mem.append(
        [3,3], 77,228,[7,3], 0
    )


print(mem.data)

print("-" * 10)
mem.append(
    [4, 4], 69,228,[4, 4], 0
)

print(mem.data)
print("-" * 10)

print(mem.get_observation())

print("-" * 10)


print(mem.get_action())
print("-" * 10)

print(mem.get_reward())


print("-" * 10)

print(mem.get_next_observation())
print('*' * 10)

batch_size = 2
selected_rows = np.random.choice( 
    np.arange( mem.get_size() ), batch_size, replace=False
) # Рандомные строки в количесве batch size

miniBatch = copy.deepcopy( mem )

miniBatch.data = miniBatch.data[selected_rows] # Slice

print( miniBatch.get_next_observation() )
print( miniBatch.get_dones() )

