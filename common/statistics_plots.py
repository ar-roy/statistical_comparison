import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import OrderedDict


def plot_change(vals_first, vals_last, names, plot_title, dir_output, fn_output, y_label):
    fig, ax = plt.subplots(figsize=(4, 6), dpi=300, facecolor='w', edgecolor='k')
    x_first = np.zeros(len(vals_first))
    x_last = np.ones(len(vals_first))
    color = iter(plt.cm.winter(np.linspace(0, 1, 2)))
    ax.scatter(x_first, vals_first, c=next(color), marker='o', alpha=0.5, label=names[0], zorder=10)
    ax.scatter(x_last, vals_last, c=next(color), marker='o', alpha=0.5, label=names[1], zorder=10)
    for val_first, val_last in zip(vals_first, vals_last):
        ax.plot([0,1], [val_first, val_last], color='lightgrey', linewidth=0.75, zorder=-10)
    # Plot formatting
    ax.legend(framealpha=0.5, loc='upper right', fontsize=10)
    ax.set_title('{}'.format(plot_title))
    ax.set_ylabel('{}'.format(y_label))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(np.asarray([vals_first, vals_last]).min() - abs(np.asarray([vals_first, vals_last]).min() * 0.1),
                np.asarray([vals_first, vals_last]).max() + abs(np.asarray([vals_first, vals_last]).max() * 0.1))
    plt.xticks(np.arange(2), names, rotation=45, ha='right')
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/scatter_change_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_scatter(ref_vals, test_vals, test_names, pearson_r, slope_intercept, plot_title, y_label,
                 x_label, dir_output='', fn_output='', mean_error=np.nan, mean_bias=np.nan, bPlot_diagonal=True,
                 circle_area=None, marker_colors=None, x_lim=None, y_lim=None, bPlot_axes=False):
    n_sets = len(test_names)
    print(f'statistics_plots::plot_scatter: Plotting {n_sets} evaluation(s)')
    n_plot_rows = int(np.ceil(n_sets / 3))
    if n_sets >= 3:
        n_plot_cols = 3
    else:
        n_plot_cols = n_sets
    fig, axs = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols, figsize=(5 * n_plot_cols, 5 * n_plot_rows),
                            dpi=300, facecolor='w', edgecolor='k', constrained_layout=True)
    for idx, ax in enumerate(fig.axes):
        if idx >= n_sets:
            break

        try:
            # marker color given & circle area given
            unique_colors = set(marker_colors['restenosed_colors'])
            for uc in unique_colors:
                mask = marker_colors['restenosed_colors'] == uc
                ax.scatter(ref_vals[mask], test_vals[idx][mask], s=circle_area[mask],
                           color=uc, alpha=0.5, zorder=10,
                           label=marker_colors.at[mask[::1].idxmax(), 'restenosed_labels'])
            ax.plot([], [], ' ', label='Smallest: {}'.format((circle_area.min() / 10).round(2)))
            ax.plot([], [], ' ', label='Largest: {}'.format((circle_area.max() / 10).round(2)))
        except:
            pass

        try:
            # marker color given & circle area not given
            unique_colors = set(marker_colors['restenosed_colors'])
            for uc in unique_colors:
                mask = marker_colors['restenosed_colors'] == uc
                ax.scatter(ref_vals[mask], test_vals[idx][mask], color=uc, alpha=0.5, zorder=10,
                           label=marker_colors.at[mask[::1].idxmax(), 'restenosed_labels'])
        except:
            pass

        try:
            # marker color not given & circle area given
            ax.plot([], [], ' ', label='Smallest: {}'.format((circle_area.min() / 10).round(2)))
            ax.plot([], [], ' ', label='Largest: {}'.format((circle_area.max() / 10).round(2)))
            ax.scatter(ref_vals, test_vals[idx], s=circle_area, color='blue', alpha=0.5, zorder=10)

        except:
            pass

        try:
            # marker color not given & circle area not given
            ax.scatter(ref_vals, test_vals[idx], color='blue', alpha=0.5, zorder=10)
        except:
            pass

        # x_lin_regress = np.linspace(test_vals[idx].min(), test_vals[idx].max())
        x_lin_regress = np.linspace(ref_vals.min(), ref_vals.max())
        y_lin_regress = slope_intercept[0][idx] * x_lin_regress + slope_intercept[1][idx]
        ax.plot(x_lin_regress, y_lin_regress, c='black', lw=0.5, label='linear regression')
        if bPlot_diagonal:
            diagonal_line = np.linspace(0, max(ref_vals.max(), x_lim[1], y_lim[1]))
            ax.plot(diagonal_line, diagonal_line, c='green', lw=0.5, ls='--', alpha=0.5, label='y=x')
        if bPlot_axes:
            ax.axhline(0.8, color='red', lw=0.5, ls='--', alpha=0.5)
            ax.axvline(0.8, color='red', lw=0.5, ls='--', alpha=0.5)
            # ax.axhline(0, color='red', lw=0.5, ls='--', alpha=0.5)
            # ax.axvline(0, color='red', lw=0.5, ls='--', alpha=0.5)

        # Plot formatting
        ax.legend(framealpha=0.5, loc='lower right', fontsize=10)
        try:
            # mean error and mean bias are given
            text_statistics = '\n'.join((
                r'$r=%.3f$' % (pearson_r[idx]),
                r'$y=%.3fx+%.3f$' % (slope_intercept[0][idx], slope_intercept[1][idx]),
                r'$MAE:%0.2f$' % (mean_error[idx]),
                r'$Bias:%0.2f$' % (mean_bias[idx])))
        except:
            # mean error and mean bias are not given
            text_statistics = '\n'.join((
                r'$r=%.3f$' % (pearson_r[idx]),
                r'$y=%.3fx+%.3f$' % (slope_intercept[0][idx], slope_intercept[1][idx])))
        ax.text(0.03, 0.97, text_statistics, transform=ax.transAxes,  verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.set_title('{}'.format(test_names[idx]))
        ax.set_xlabel('{}'.format(x_label))
        ax.set_ylabel('{}'.format(y_label))
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
        ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    fig.suptitle(plot_title)
    try:
        # directory and filename output given, save figure
        Path(dir_output).mkdir(parents=True, exist_ok=True)
        plt.savefig('{}/scatter_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
        plt.close()
    except:
        # directory and filename not given, return the figure
        return fig


def plot_scatter_stacked(ref_vals, test_vals, test_names, plot_title, dir_output,
                         fn_output, y_label, x_label, bPlot_change=True):
    n_sets = len(test_names)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    color = iter(plt.cm.winter(np.linspace(0, 1, n_sets)))
    for i_test, i_test_name in zip(test_vals, test_names):
        c = next(color)
        ax.scatter(ref_vals, i_test, marker='o', color=c, alpha=0.75, label=i_test_name, zorder=10)
    if bPlot_change:
        ax.vlines(ref_vals, ymin=test_vals[0], ymax=test_vals[1],
                  colors='lightgrey', linewidth=0.75, zorder=-10)
    # Plot formatting
    ax.legend(framealpha=0.5, loc='upper right', fontsize=10)
    ax.set_title('{}'.format(plot_title))
    ax.set_xlabel('{}'.format(x_label))
    ax.set_ylabel('{}'.format(y_label))
    ax.set_ylim(test_vals.min() - abs(test_vals.min() * 0.1),
                test_vals.max() + abs(test_vals.max() * 0.1))
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/scatter_stacked_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_scatter_stacked_unique(ref_vals, test_vals, test_names, markers, plot_title, dir_output,
                                fn_output, y_label, x_label, bPlot_change=True):
    n_sets = len(test_names)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    color = iter(plt.cm.winter(np.linspace(0, 1, n_sets)))
    unique_markers = set(markers)
    c = next(color)
    for um in unique_markers:
        # FIXME: Generalize to multiple use cases
        mask = markers == um
        if um == 'o':
            unique_label = 'Non-Restenosed: {}'.format(test_names[0])
        elif um == 'x':
            unique_label = 'Restenosed: {}'.format(test_names[0])
        ax.scatter(ref_vals[mask], test_vals[0][mask], marker=um, color=c, alpha=0.75,
                   label=unique_label, zorder=10)
    if bPlot_change:
        # FIXME: Generalize to multiple use cases
        # FOR change in protocol
        test_vals_change = test_vals[0] - test_vals[1]
        # Treat 0 as < 0, 1 > 0
        b_positive_neg = test_vals_change > 0
        unique_change = set(b_positive_neg)
        for um in unique_markers:
            mask_restenosis_type = markers == um
            # Treat 'o' as non-restenosed, 'x' as restenosed
            for uc in unique_change:
                mask_geom_change = b_positive_neg == uc
                mask_geom_change = pd.Series(mask_geom_change)
                mask = mask_restenosis_type & mask_geom_change
                if 'o' in um and uc == False:
                    ax.vlines(ref_vals[mask], ymin=test_vals[0][mask], ymax=test_vals[1][mask],
                              colors='red', label='Decay: Commercial {} NewProtocol'.format('\u2192'),
                              linewidth=0.75)
                elif 'o' in um and uc == True:
                    ax.vlines(ref_vals[mask], ymin=test_vals[0][mask], ymax=test_vals[1][mask],
                              colors='green', label='Improvement: Commercial {} NewProtocol'.format('\u2192'),
                              linewidth=0.75)
                elif 'x' in um and uc == False:
                    ax.vlines(ref_vals[mask], ymin=test_vals[0][mask], ymax=test_vals[1][mask],
                              colors='green', label='Improvement: Commercial {} NewProtocol'.format('\u2192'),
                              linewidth=0.75)
                elif 'x' in um and uc == True:
                    ax.vlines(ref_vals[mask], ymin=test_vals[0][mask], ymax=test_vals[1][mask],
                              colors='red', label='Decay: Commercial {} NewProtocol'.format('\u2192'),
                              linewidth=0.75)
    # Label formatting (stop repetition)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), framealpha=0.5, loc='upper right', fontsize=8)
    # Plot formatting
    ax.set_title('{}'.format(plot_title))
    ax.set_xlabel('{}'.format(x_label))
    ax.set_ylabel('{}'.format(y_label))
    ax.set_ylim(test_vals.min() - abs(test_vals.min() * 0.1),
                test_vals.max() + abs(test_vals.max() * 0.1))
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/scatter_stacked_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_bland_altman(ref_vals, test_vals, ref_name, test_names, mean_error, mean_bias, pearson_r,
                      slope_intercept, CI_pop, z_value, plot_title, dir_output, fn_output,
                      x_label, y_label, x_lim=None, y_lim=None):
    n_sets = len(test_names)
    n_plot_rows = int(np.ceil(n_sets / 3))
    if n_sets >= 3:
        n_plot_cols = 3
    else:
        n_plot_cols = n_sets
    fig, axs = plt.subplots(nrows=n_plot_rows, ncols=n_plot_cols,
                            figsize=(5 * n_plot_cols, 5 * n_plot_rows),
                            dpi=300, facecolor='w', edgecolor='k', constrained_layout=True)
    for idx, ax in enumerate(fig.axes):
        ax.scatter(ref_vals, test_vals[idx], marker='o', c='blue', alpha=0.5)
        try:
            x_lin_regress = np.linspace(ref_vals.min(), ref_vals.max())
            y_lin_regress = slope_intercept[0][idx] * x_lin_regress + slope_intercept[1][idx]
            ax.plot(x_lin_regress, y_lin_regress, c='black', lw=0.5, label='linear regression')
        except:
            pass
        # Plot confidence interval of the population
        ax.hlines(mean_bias[idx], xmin=0, xmax=ref_vals.max(), color='red',
                  linestyles='dashed', lw=0.5,
                  label='mean (' + str(mean_bias[idx].round(3)) + ')')
        ax.hlines(mean_bias[idx] + CI_pop[idx], xmin=0, xmax=ref_vals.max(), color='green',
                  linestyles='dashed', lw=0.5,
                  label='mean + {}*std ({})'.format(z_value, (mean_bias[idx] + CI_pop[idx]).round(3)))
        ax.hlines(mean_bias[idx] - CI_pop[idx], xmin=0, xmax=ref_vals.max(), color='green',
                  linestyles='dashed', lw=0.5,
                  label='mean - {}*std ({})'.format(z_value, (mean_bias[idx] - CI_pop[idx]).round(3)))
        # Plot formatting
        ax.legend(framealpha=0.5, loc='upper right', fontsize=10)
        try:
            plot_text = '\n'.join((
                r'$r=%.3f$' % (pearson_r[idx]),
                r'$y=%.2fx+%.2f$' % (slope_intercept[0][idx], slope_intercept[1][idx])))
            ax.text(0.03, 0.97, plot_text, transform=ax.transAxes, verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        except:
            print('statistics_plots::plot_bland_altman: Not plotting slope/intercept and correlation')
        ax.set_title('{}'.format(test_names[idx]))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        min_y = min(test_vals.min(), (mean_bias - CI_pop).min())
        max_y = max(test_vals.max(), (mean_bias + CI_pop).max())
        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])
        else:
            ax.set_ylim(min_y-abs(min_y)*0.1, max_y+abs(max_y)*0.1)
        ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
        ax.xaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.5, zorder=-10)
    fig.suptitle(plot_title)
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/bland_altman_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_boxplot(test_names, data, plot_title, dir_output, fn_output, y_label, y_lim=None, rounding_value=2):
    n_sets = len(test_names)
    fig, ax = plt.subplots(figsize=(n_sets * 1.5, 8), dpi=300, facecolor='w', edgecolor='k')
    bp = plt.boxplot(data,
                     sym='+',
                     widths=0.25,
                     showmeans=True,
                     meanprops=dict(label='Mean'),
                     )
    plt.violinplot(data,
                   showmeans=False,
                   showmedians=True,
                   showextrema=False)
    # Label formatting (stop repetition of 'Mean')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), framealpha=0.5, loc='upper right', fontsize=10)
    # Plot formatting
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title('{}'.format(plot_title))
    ax.set_xlabel('Processes')
    ax.set_ylabel('{}'.format(y_label))
    ax.set_xticklabels(test_names, rotation=45, ha='right')
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    # Text labels for each boxplot
    for idx, b in enumerate(bp['medians']):
        bp_median = b.get_ydata()
        bp_mean = data.transpose()[idx].mean()
        ax.annotate('Median: {}\nMean: {}'.format(bp_median[0].round(rounding_value),
                                                  bp_mean.round(rounding_value)),
                    xy=(b.get_xdata().mean(), ax.get_ylim()[0]),
                    xytext=(0,2),
                    textcoords="offset points",
                    ha='center',
                    va='bottom')
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/boxplot_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_barchart(test_names, mean_values, CI_mean, plot_title, dir_output, fn_output, y_label,
                  y_lim=None, rounding_value=2):
    n_sets = len(test_names)
    fig, ax = plt.subplots(figsize=(n_sets * 1.5, 8), dpi=300, facecolor='w', edgecolor='k')
    bars = ax.bar(np.arange(n_sets),
                  mean_values,
                  yerr=CI_mean,
                  align='center',
                  alpha=0.5,
                  error_kw=dict(lw=0.5, capsize=5, ecolor='red'))
    # Plot formatting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax.set_title('{}'.format(plot_title))
    ax.set_xlabel('Processes')
    ax.set_ylabel('{}'.format(y_label))
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    plt.xticks(np.arange(n_sets), test_names, rotation=45, ha='right')
    # Text labels for each bar
    for idx, bar in enumerate(bars):
        i_height = bar.get_height()
        try:
            i_CI = CI_mean[idx]
        except:
            i_CI = CI_mean
        ax.annotate('{} +/- {}'.format(i_height.round(rounding_value), i_CI.round(rounding_value)),
                    xy=(bar.get_x() + bar.get_width() / 2, i_height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center',
                    va='bottom')
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/barchart_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_joint_density(data, x_col_name, y_col_name, dir_output, fn_output, kind='scatter', xylim=None):
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    sns.jointplot(x=x_col_name, y=y_col_name, data=data, kind=kind, xlim=xylim, ylim=xylim)
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/pop_densities/density_{}.png'.format(dir_output, fn_output),
                bbox_inches='tight')
    plt.close()


def plot_pop_density_all(test_names, bias, plot_title, dir_output, fn_output, x_lim=np.nan,
                         x_label='', y_label='', hist=False, kde=True, norm_hist=True, bins=None):
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    if kde:
        kde_kws = {'shade': True, 'linewidth': 3}
    else:
        kde_kws = None
    for i_test, i_bias in zip(test_names, bias):
        sns.distplot(i_bias,
                     hist=hist,
                     kde=kde,
                     kde_kws=kde_kws,
                     label=i_test,
                     norm_hist=norm_hist,
                     bins=bins)
    # Plot formatting
    plt.legend(prop={'size': 10})
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    try:
        plt.xlim(x_lim)
    except:
        pass
    # Save
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/pop_densities/density_{}.png'.format(dir_output, fn_output), bbox_inches='tight')
    plt.close()


def plot_pop_density(test_name, bias, mean_bias, CI_pop, z_value, plot_title, dir_output,
                     fn_output, x_label, pvalue=np.nan, x_lim=np.nan, y_lim=np.nan, bins=None):
    plt.figure(num=None, figsize=(8, 6), dpi=300, facecolor='w', edgecolor='k')
    # Plot population density
    if bins is not None:
        sns.distplot(bias,
                     bins=bins,
                     hist=True,
                     kde=True,
                     kde_kws={'linewidth': 3},
                     label=test_name)
    else:
        sns.distplot(bias,
                     hist=True,
                     kde=True,
                     kde_kws={'linewidth': 3},
                     label=test_name)
    # Plot confidence interval of the population
    y_max = plt.gca().get_ylim()[1]
    plt.vlines(mean_bias, ymin=0, ymax=y_max, color='red', alpha=0.5,
               label='mean (' + str(mean_bias.round(3)) + ')')
    plt.vlines(mean_bias + CI_pop, ymin=0, ymax=y_max, color='green', alpha=0.5,
               label='mean + {}*std ({})'.format(z_value, (mean_bias + CI_pop).round(2)))
    plt.vlines(mean_bias - CI_pop, ymin=0, ymax=y_max, color='green', alpha=0.5,
               label='mean - {}*std ({})'.format(z_value, (mean_bias - CI_pop).round(2)))
    # Write pvalue
    try:
        plt.plot([], [], ' ', label='p-value ({})'.format(pvalue.round(3)))
    except:
        pass
    # Plot formatting
    plt.legend(prop={'size': 10})
    plt.title(plot_title + ': ' + test_name + ' (n={})'.format(len(bias)))
    plt.xlabel(x_label)
    plt.ylabel('Density')
    try:
        plt.xlim(x_lim)
        plt.ylim(y_lim)
    except:
        pass
    # Save
    Path('{}/pop_densities'.format(dir_output)).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/pop_densities/density_{}_{}.png'.format(dir_output, fn_output, test_name),
                bbox_inches='tight')
    plt.close()


def plot_diagnostic_column_scatter(test_name, bool_diagnostic, bias, diagnosis_name, dir_output, fn_output):
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    nRestenosis = sum(bool_diagnostic)
    for idx, b in enumerate(bias):
        bRestenosis = bool_diagnostic[idx]
        if bRestenosis:
            plt.plot(['{} (n={})'.format(diagnosis_name, nRestenosis)],
                     [b], marker='o', markersize=4, color='blue', alpha=0.3)
        else:
            plt.plot(['Not {} (n={})'.format(diagnosis_name, len(bool_diagnostic) - nRestenosis)],
                     [b], marker='o', markersize=3, color='blue')
    # Plot formatting
    plt.title('{} detection'.format(diagnosis_name))
    plt.xlabel('{} Ground Truth'.format(diagnosis_name))
    plt.ylabel('Bias')
    # Save
    Path('{}/diagnostic_accuracy'.format(dir_output)).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/diagnostic_accuracy/diagnostic_column_scatter_{}_{}.png'.format(dir_output, fn_output, test_name),
                bbox_inches='tight')
    plt.close()


def plot_conf_matrix(conf_matrix, x_label, y_label):
    fig_conf_matrix = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = fig_conf_matrix.add_axes([0, 0, 1, 1])
    sns.heatmap(conf_matrix, annot=True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['False', 'True'])
    ax.set_yticklabels(['False', 'True'])
    plt.close()


def plot_diagnostic_heatmap(test_name, diagnostic_thresholds, sens, spec, ppv, npv, fpr, tpr,
                            dir_output, fn_output):
    n_thresholds = len(diagnostic_thresholds)
    diagnostic_vals = pd.DataFrame({'Sensitivity': sens,
                                    'Specificity': spec,
                                    'PPV': ppv,
                                    'NPV': npv,
                                    'TPR': [x[1] for x in tpr],
                                    'FPR': [x[1] for x in fpr]})
    plt.figure(num=None, figsize=(n_thresholds * 2.5, 10), dpi=100, facecolor='w', edgecolor='k')
    sns.heatmap(diagnostic_vals.transpose(), annot=True, xticklabels=diagnostic_thresholds)
    # Plot formatting
    plt.xticks(size=10)
    plt.yticks(rotation=0, size=10)
    plt.xlabel("Bias threshold", size=10)
    plt.title("Diagnostic Heatmap: {}".format(test_name), size=15)
    # Save
    Path('{}/diagnostic_accuracy'.format(dir_output)).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/diagnostic_accuracy/diagnostic_heatmap_{}_{}.png'.format(dir_output, fn_output, test_name),
                bbox_inches='tight')
    plt.close()


def plot_diagnostic_roc(test_names, fpr, tpr, auc, dir_output, fn_output):
    n_sets = len(test_names)
    color = iter(plt.cm.winter(np.linspace(0, 1, n_sets)))
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    for test_names_i, fpr_i, tpr_i, auc_i, color_i in zip(test_names, fpr, tpr, auc, color):
        plt.plot(fpr_i, tpr_i, color=color_i, label=f'{test_names_i} (AUC={auc_i.round(3)})')
    plt.plot([0, 1.05], [0, 1.05], color='navy', lw=2, linestyle='--')
    # Plot formatting
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (FPR)')
    plt.ylabel('Sensitivity (TPR)')
    plt.title('ROC')
    plt.legend(loc='lower right')
    # Save
    Path('{}/diagnostic_accuracy'.format(dir_output)).mkdir(parents=True, exist_ok=True)
    plt.savefig('{}/diagnostic_accuracy/ROC_{}.png'.format(dir_output, fn_output),
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print('This is a module with lots of different plots. Call them individually...')
    pass
