import matplotlib.pyplot as plt
import re


game_time = []
entropy = []
policy = []
value = []
gradients = []

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
        elif 'Average gradient' in line:
            val = numbers.findall(line)[0]
            gradients.append(float(val))

plt.plot(range(len(game_time)), game_time, 'bo')
plt.ylabel('Mean Episode Time')
plt.xlabel('Generation')
plt.title('Episode Time')
plt.show()

plt.plot(range(len(entropy)), [-e for e in entropy], 'bo')
plt.ylabel('Gradient')
plt.xlabel('Generation')
plt.title('Gradients')
plt.show()

plt.plot(range(len(entropy)), [e for e in entropy], 'bo')
plt.ylabel('Mean Entropy')
plt.xlabel('Generation')
plt.title('Entropy')
plt.show()

plt.plot(range(len(policy)), policy, 'ro')
plt.ylim(bottom=0)
plt.ylabel('Policy Loss')
plt.xlabel('Generation')
plt.title('Policy Loss')
plt.show()

plt.plot(range(len(value)), value, 'bo')
plt.ylim(bottom=0)
plt.ylabel('Value Loss')
plt.xlabel('Generation')
plt.title('Value Loss')
plt.show()
