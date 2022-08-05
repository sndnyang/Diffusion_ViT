import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def expected_calibration_error(predictions, truths, confidences, bin_size=0.1, title='demo'):

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []

    # Compute empirical probability for each bin
    plot_x = []
    ece = 0
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf = compute_accuracy(conf_thresh - bin_size, conf_thresh, confidences, predictions, truths)
        plot_x.append(avg_conf)
        accs.append(acc)
        ece += abs(avg_conf - acc) * perc_pred
    return ece


def reliability_diagrams(predictions, truths, confidences, bin_size=0.1, title='demo', args=None):

    # import seaborn as sns
    # sns.set()
    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)
    accs = []

    # Compute empirical probability for each bin
    conf_x = []
    ece = 0
    for conf_thresh in upper_bounds:
        acc, perc_pred, avg_conf = compute_accuracy(conf_thresh - bin_size, conf_thresh, confidences, predictions, truths)
        conf_x.append(avg_conf)
        accs.append(acc)
        temp = abs(avg_conf - acc) * perc_pred
        print('m %.2f, B_m %d, acc(B_m) %.4f, conf = %.4f, |B_m||acc(B_m) - conf(B_m)|/n = %.5f' % (conf_thresh, int(perc_pred * len(predictions)), acc, avg_conf, temp))
        ece += temp

    # Produce error bars for each bin
    upper_bound_to_bootstrap_est = {x: [] for x in upper_bounds}
    for i in range(1):

        # Generate bootstrap
        boot_strap_outcomes = []
        boot_strap_confs = random.sample(confidences, len(confidences))
        for samp_conf in boot_strap_confs:
            correct = 0
            if random.random() < samp_conf:
                correct = 1
            boot_strap_outcomes.append(correct)

        # Compute error frequency in each bin
        for upper_bound in upper_bounds:
            conf_thresh_upper = upper_bound
            conf_thresh_lower = upper_bound - bin_size

            filtered_tuples = [x for x in zip(boot_strap_outcomes, boot_strap_confs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
            correct = len([x for x in filtered_tuples if x[0] == 1])
            acc = float(correct) / len(filtered_tuples) if len(filtered_tuples) > 0 else 0

            upper_bound_to_bootstrap_est[upper_bound].append(acc)

    upper_bound_to_bootstrap_upper_bar = {}
    upper_bound_to_bootstrap_lower_bar = {}
    for upper_bound, freqs in upper_bound_to_bootstrap_est.items():
        top_95_quintile_i = int(0.975 * len(freqs))
        lower_5_quintile_i = int(0.025 * len(freqs))

        upper_bar = sorted(freqs)[top_95_quintile_i]
        lower_bar = sorted(freqs)[lower_5_quintile_i]

        upper_bound_to_bootstrap_upper_bar[upper_bound] = upper_bar
        upper_bound_to_bootstrap_lower_bar[upper_bound] = lower_bar

    upper_bars = []
    lower_bars = []
    for i, upper_bound in enumerate(upper_bounds):
        if upper_bound_to_bootstrap_upper_bar[upper_bound] == 0:
            upper_bars.append(0)
            lower_bars.append(0)
        else:
            # The error bar arguments need to be the distance from the data point, not the y-value
            upper_bars.append(abs(conf_x[i] - upper_bound_to_bootstrap_upper_bar[upper_bound]))
            lower_bars.append(abs(conf_x[i] - upper_bound_to_bootstrap_lower_bar[upper_bound]))

    # sns.set(font_scale=2)
    fig, ax = plt.subplots()
    ax.errorbar(conf_x, conf_x, label="Perfect classifier calibration")

    new_conf_x = []
    new_accs = []
    for i, bars in enumerate(zip(lower_bars, upper_bars)):
        if bars[0] == 0 and bars[1] == 0:
            continue
        new_conf_x.append(conf_x[i])
        new_accs.append(accs[i])

    print("ECE: %g" % ece)
    ax.plot(new_conf_x, new_accs, '-o', label="Accuracy", color="red")
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.title(title + " ECE: %.2f%%" % (ece * 100))
    plt.ylabel('Empirical probability')
    plt.xlabel('Estimated probability')

    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar([0, 1], [0, 1], label="Perfect classifier calibration")
    # ax.plot(new_conf_x, new_accs, '-o', label="Accuracy", color="black")
    ax.bar(upper_bounds - 0.025, accs, width=bin_size, label="Accuracy", color="red", edgecolor='gray', align='center')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    plt.title(title + " ECE: %.2f%%" % (ece * 100), fontsize=20)
    plt.ylabel('Empirical probability', fontsize=20)
    plt.xlabel('Estimated probability', fontsize=16)
    # fig.savefig("reliability.tif", format='tif', bbox_inches='tight', dpi=1200)

    print(args.resume)
    if args is not None and args.resume:
        fig.savefig(args.resume + "_calibration.png") # , bbox_inches='tight')# , dpi=1200)
        # fig.savefig(args.load_path + "_calibration.eps", format='eps', bbox_inches='tight', dpi=1200)

    plt.show()
    plt.close()


def compute_accuracy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):

    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        accuracy = float(correct) / len(filtered_tuples)
        perc_of_data = float(len(filtered_tuples)) / len(conf)
        return accuracy, perc_of_data, avg_conf


def compute_accuracy2(conf_thresh_lower, conf_thresh_upper, conf, pred, true):

    num_classes = max(true)
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        corrects = []
        acc = []
        for i in range(num_classes):
            predict = len([x for x in filtered_tuples if x[0] == i])
            category = len([x for x in filtered_tuples if x[1] == i])
            correct = len([x for x in filtered_tuples if x[0] == i and x[0] == x[1]])
            if category == 0:
                accuracy = 0
            else:
                accuracy = float(correct) / category
            acc.append(accuracy)
            print("category %d: predict num: %d, ground truth num: %d, correct: %d, %.4f" % (i, predict, category, correct, accuracy))
        avg_conf = sum([x[2] for x in filtered_tuples]) / len(filtered_tuples)
        perc_of_data = float(len(filtered_tuples)) / len(conf)
        accuracy = sum(acc) / num_classes
        return accuracy, perc_of_data, avg_conf


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
