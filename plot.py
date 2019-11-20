import matplotlib.pyplot as plt
import re


game_time = []
entropy = []
policy = []
value = []

numbers = re.compile(r'-?\d+(?:\.\d+)?')

with open('train.log', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if 'Average game time' in line:
            val = numbers.findall(line)[0]
            game_time.append(float(val))
        elif 'Average value loss' in line:
            val = numbers.findall(line)[0]
            value.append(float(val))
        elif 'Average policy loss' in line:
            val = numbers.findall(line)[0]
            policy.append(float(val))
        elif 'Average entropy' in line:
            val = numbers.findall(line)[0]
            entropy.append(float(val))

plt.plot(range(len(game_time)), game_time, 'bo')
plt.ylabel('Mean Episode Time')
plt.xlabel('Generation')
plt.title('Episode Time')
plt.show()

plt.plot(range(len(entropy)), [-e for e in entropy], 'bo')
plt.ylabel('Mean Entropy')
plt.xlabel('Generation')
plt.title('Entropy')
plt.show()

plt.plot(range(len(policy)), policy, 'bo')
plt.plot(range(len(value)), value, 'ro')
plt.ylim(bottom=0)
plt.ylabel('Loss')
plt.xlabel('Generation')
plt.title('Loss')
plt.legend(['Policy', 'Value'])
plt.show()
