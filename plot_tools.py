import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew  # for some statistics
from matplotlib import colors

class PlotRelations:
    def __init__(self, df: pd.DataFrame, save_path: str):
        self.__df = df
        self.__save_path = save_path

    def corrdot(self, *args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}"
        ax = plt.gca()
        ax.annotate(corr_text, [.5, .5, ], xycoords="axes fraction",
                    ha='center', va='center', fontsize=20)

    def corrfunc(self, x, y, **kws):
        r, p = stats.pearsonr(x, y)
        p_stars = ''
        if p <= 0.05:
            p_stars = '*'
        if p <= 0.01:
            p_stars = '**'
        if p <= 0.001:
            p_stars = '***'
        corr_text = f"{r:2.2f} {p_stars}"
        ax = plt.gca()
        ax.annotate(corr_text, [.5, .5, ], xycoords="axes fraction",
                    ha='center', va='center', fontsize=20)

    def plot_graph(self):
        g = sns.PairGrid(self.__df, aspect=1.5,
                         diag_sharey=False, despine=False)
        g.map_lower(sns.scatterplot, color='limegreen',
                    alpha=0.4, s=30, edgecolor='black')
        g.map_diag(sns.histplot, kde=True,
                   linewidth=0.5, bins=15, edgecolor='k')
        g.map_upper(self.corrfunc)
        g.fig.subplots_adjust(wspace=0, hspace=0)

        # Add titles to the diagonal axes/subplots
        for ax, col in zip(np.diag(g.axes), self.__df.columns):
            ax.set_title(col, y=0.82, fontsize=15)
        for ax in g.axes.flatten():
            ax.grid(which = "major", axis='both', color='#758D99', zorder=1, linewidth = 0.5, alpha = 0.4,linestyle='-')
            ax.grid(which = "minor", axis='both', color='#758D99', zorder=1, linewidth = 0.3, alpha = 0.2,linestyle='-')
            ax.minorticks_on()
            ax.spines[['top','right','bottom', 'left']].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.__save_path,
                    transparent=True,
                    bbox_inches='tight',
                    pad_inches=0,
                    format='png',
                    dpi=300)
        plt.show()


def plot_distribution(df: pd.DataFrame, var: str):
    skew =  df[var].skew()
    fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
    xx = np.linspace(df[var].min(), df[var].max(), 100)
    axs[0].plot(xx, norm.pdf(xx, *norm.fit(df[var])), color="r", lw=2.5)
    sns.histplot(
    df[var], kde=True,
    stat="density", kde_kws=dict(cut=3),
    alpha=.4, edgecolor=(1, 1, 1, .4),ax=axs[0]
    )
    (mu, sigma) = norm.fit(df[var])
    text = f'Normal dist. ($\mu=$ {mu:.2f}, $\sigma=$ {sigma:.2f})'
    axs[0].legend([text],loc='best', fontsize=10)
    property = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[0].text(0.05, 0.90, f"Skewness={skew:.2f}", transform=axs[0].transAxes, fontsize=10,
        verticalalignment='top', bbox=property)
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'{var} distribution')
    #QQ-plot
    res = stats.probplot(df[var],plot=axs[1]) 
    axs[0].tick_params(axis='x', rotation=45)
    axs[1].tick_params(axis='x', rotation=45)
    for ax in axs:
        ax.grid(which = "major", axis='both', color='#758D99', zorder=1, linewidth = 0.5, alpha = 0.4,linestyle='-')
        ax.grid(which = "minor", axis='both', color='#758D99', zorder=1, linewidth = 0.3, alpha = 0.2,linestyle='-')
        ax.minorticks_on()
        ax.spines[['top','right','bottom', 'left']].set_visible(False)
        # ax.grid(axis='y')
    # Set source text
    # fig.text(x=0.12, y=0.01, s="""Source: "Describe Source""", transform=fig.transFigure, ha='left', fontsize=9, alpha=.7)  
    plt.tight_layout()
    plt.show()



def plot_corration_map(df: pd.DataFrame)->None:
    # sns.set(font_scale=.8)
    # Create the correlation matrix
    corr = df.corr().round(2)

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(9, 11))
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = colors.ListedColormap(["navy", "royalblue", "lightsteelblue", 
                           "beige", "peachpuff", "salmon", "darkred"])
    bounds = [-1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1]
    cb_ticks=[-1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1]

    norm = colors.BoundaryNorm(bounds, cmap.N)

    g = sns.heatmap(
        corr,          # The data to plot
        mask=mask,     # Mask some cells
        cmap=cmap,     # What colors to plot the heatmap as
        annot=False,    # Should the values be plotted in the cells?
        # vmax=.5,       # The maximum value of the legend. All higher vals will be same color
        # vmin=-.5,      # The minimum value of the legend. All lower vals will be same color
        center=0,      # The center value of the legend. With divergent cmap, where white is
        square=True,   # Force cells to be square
        linewidths=.5, # Width of lines that divide cells
        cbar_kws={"shrink": .3, 'label': 'Correlation'},  # Extra kwargs for the legend; in this case, shrink by 50%
        annot_kws = {'size': 5},
    )
    
    ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Correlation Matrix', fontsize=12)
    plt.tight_layout()
    plt.show()