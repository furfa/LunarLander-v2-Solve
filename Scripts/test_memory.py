from BaseClasses import Memory

mem = Memory(5, 2, 1)

print(mem.data)

print("-" * 10)

for i in range(5):
    mem.append(
        [3,3], 77,228,[7,3]
    )


print(mem.data)

print("-" * 10)
mem.append(
    [4, 4], 69,228,[4, 4]
)

print(mem.data)
print("-" * 10)

print(mem.get_observation())

print("-" * 10)


print(mem.get_action())
print("-" * 10)

print(mem.get_reward())


print("-" * 10)

print(mem.get_prev_observation())

