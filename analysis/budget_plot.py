import seaborn as sns
import matplotlib.pyplot as plt

# Define x and y values
budget = [0.1, 1, 2.2, 5, 10, 25, 50]
miou = [69.6, 73.1, 73.5, 74.1, 73.4, 72.8, 72.3]
fully_sup = 71.9
budget_str = [str(y)+'%' for y in budget]
miou_str = [str(y) for y in miou]
miou[2] += 0.1

# Set a pastel color palette
# colors = ["#FACFD7", "#FFE4C4", "#E1D5E7", "#D0F0C0", "#FFD1DC", "#BFEFFF", "#FFDAB9", "#FFBBCC"]
colors = ["red", "blue", "green", "purple", "orange", "magenta", "cyan", "brown"]
sns.set_palette(colors)

# Create a Seaborn line plot
sns.set_style("white")
sns.lineplot(x=budget, y=miou)

xmin = -7
xmax = 55

for i in range(len(budget)):
    plt.plot(budget[i], miou[i], 'o', color="grey", markersize=7)
    plt.hlines(y=miou[i], xmin=xmin, xmax=budget[i], color="grey", alpha=0.5, linestyles='dotted')
    plt.text(budget[i]+1.2, miou[i]+0.05, budget_str[i], fontsize=9)

# Plot fully supervised
plt.hlines(y=fully_sup, xmin=xmin, xmax=xmax, color="black",
           alpha=0.5, linestyles='dashed', label="Fully supervised")
plt.legend()

# Set y-axis tick locations and labels
plt.xticks(fontsize=8)
plt.yticks([fully_sup]+miou,[str(fully_sup)]+ miou_str, fontsize=8)

# Other params
# plt.xscale("log")
plt.ylim(min(miou)-0.5, max(miou)+0.5)
plt.xlim(xmin, xmax) # -20, 105

# Set title and axis labels
# plt.title("Performance variation on GTAV → Cityscapes\n with different budgets for pixels labeling", fontdict={'fontsize': 10})
plt.xlabel("Budget (%)")
plt.ylabel("mIoU (%)")

# Display the plot
plt.show()
plt.savefig("visualizations/budget_plot.png", dpi=300)
