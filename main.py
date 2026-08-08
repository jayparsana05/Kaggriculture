from kaggle_environments import make
from submission import agent

env = make("kaggriculture", debug=True)
env.run([agent, "random"])

final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# html_content = env.render(mode="html", width=800, height=600)
# with open("replay.html", "w") as f:
#     f.write(html_content)

# print("Saved replay to replay.html")