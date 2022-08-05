#######################################################
# ## BPDA +EOT PGD WHITEBOX ATTACK FOR SAVED NETS ## #
#######################################################
import os
import sys
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import json
import datetime
import argparse

import torch
from models.DifViT import ViT
from diffusion import GaussianDiffusion
from Task.attack_utils import setup_exp, import_data

# json file with experiment config
CONFIG_FILE = './bpda_eot_attack.json'
model_path = sys.argv[1]
print(model_path)
eps = float(sys.argv[3])


###############
# ## SETUP ## #
###############

# load experiment config
with open(CONFIG_FILE) as file:
    config = json.load(file)
# directory for experiment results
config['adv_norm'] = sys.argv[2]
exp_dir = model_path + '_robust_' + datetime.datetime.now().strftime('%d%m%H%M') + '_%s_%d/' % (config['adv_norm'], int(eps))
# setup folders, save code, set seed and get device
setup_exp(exp_dir, config['seed'], ['log'], ['bpda_eot_attack.py', CONFIG_FILE])

print('Loading data and nets.')
config['adv_eps'] = eps
# data loader
data, num_classes = import_data(config['data_type'], False, False)
attack_loader = DataLoader(data, batch_size=config['batch_size'], shuffle=config['subset_shuffle'], num_workers=0)

arg = argparse.Namespace()
arg.workers = 8
img_size = 32
n_classes = 10
channel = 3
SPT = False
LSA = False
patch_size = 4
heads = 12
depth = 9
dim = 384
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
arg.device = device

model = ViT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=dim,
            mlp_dim_ratio=2, depth=depth, heads=heads, dim_head=dim // heads, channels=channel,
            stochastic_depth=0, is_SPT=SPT, is_LSA=LSA).to(device)

diffusion_model = GaussianDiffusion(
    model,
    image_size=img_size,
    channels=channel,
).to(device)


ckpt_dict = t.load(model_path, map_location=lambda storage, loc: storage.cuda())
try:
    diffusion_model.load_state_dict(ckpt_dict["model_state_dict"])
    clf = diffusion_model.denoise_fn
except:
    model.load_state_dict(ckpt_dict["model_state_dict"])
    clf = model.to(device)

clf.eval()

# cross-entropy loss function to generate attack gradients
criterion = t.nn.CrossEntropyLoss()

# rescale adversarial parameters for attacks on images with pixel intensities in the range [-1, 1]
config['adv_eps'] *= 2.0 / 255.0
config['adv_eta'] *= 2.0 / 255.0


#############################################
# ## FUNCTIONS FOR ATTACK, DEFENSE, EVAL ## #
#############################################

def eot_defense_prediction(logits, reps=1, eot_defense_ave=None):
    # finite-sample approximation of stochastic classifier for EOT defense averaging different methods
    # for deterministic logits with reps=1, this is just standard prediction for any eot_defense_ave
    if eot_defense_ave == 'logits':
        logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    elif eot_defense_ave == 'softmax':
        logits_pred = F.softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
    elif eot_defense_ave == 'logsoftmax':
        logits_pred = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
    elif reps == 1:
        logits_pred = logits
    else:
        raise RuntimeError('Invalid ave_method_pred (use "logits" or "softmax" or "logsoftmax")')
    # finite sample approximation of stochastic classifier prediction
    _, y_pred = t.max(logits_pred, 1)
    return y_pred

def eot_attack_loss(logits, y, reps=1, eot_attack_ave='loss'):
    # finite-sample approximation of stochastic classifier loss for different EOT attack averaging methods
    # for deterministic logits with reps=1 this is just standard cross-entropy loss for any eot_attack_ave
    if eot_attack_ave == 'logits':
        logits_loss = logits.view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        y_loss = y
    elif eot_attack_ave == 'softmax':
        logits_loss = t.log(F.softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0))
        y_loss = y
    elif eot_attack_ave == 'logsoftmax':
        logits_loss = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        y_loss = y
    elif eot_attack_ave == 'loss':
        logits_loss = logits
        y_loss = y.repeat(reps)
    else:
        raise RuntimeError('Invalid ave_method_eot ("logits", "softmax", "logsoftmax", "loss")')
    # final cross-entropy loss to generate attack grad
    loss = criterion(logits_loss, y_loss)
    return loss

def predict(X, y, requires_grad=True, reps=1, eot_defense_ave=None, eot_attack_ave='loss'):
    if requires_grad:
        logits = clf(X)
    else:
        with t.no_grad():
            logits = clf(X.data)

    # finite-sample approximation of stochastic classifier prediction
    y_pred = eot_defense_prediction(logits.detach(), reps, eot_defense_ave)
    correct = t.eq(y_pred, y)
    # loss for specified EOT attack averaging method
    loss = eot_attack_loss(logits, y, reps, eot_attack_ave)

    return correct.detach(), loss

def rand_init_l_p(X_adv, adv_norm, adv_eps):
    # random initialization in l_inf or l_2 ball
    if adv_norm == 'l_inf':
        X_adv.data = t.clamp(X_adv.data + adv_eps * (2 * t.rand_like(X_adv) - 1), min=-1, max=1)
    elif adv_norm == 'l_2':
        r = t.randn_like(X_adv)
        r_unit = r / r.view(r.shape[0], -1).norm(p=2, dim=1).view(r.shape[0], 1, 1, 1)
        X_adv.data += adv_eps * t.rand(X_adv.shape[0], 1, 1, 1).cuda() * r_unit
    else:
        raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
    return X_adv

def pgd_update(X_adv, grad, X, adv_norm, adv_eps, adv_eta, eps=1e-10):
    if adv_norm == 'l_inf':
        # l_inf steepest ascent update
        X_adv.data += adv_eta * t.sign(grad)
        # project to l_inf ball
        X_adv = t.clamp(t.min(X + adv_eps, t.max(X - adv_eps, X_adv)), min=-1, max=1)
    elif adv_norm == 'l_2':
        # l_2 steepest ascent update
        X_adv.data += adv_eta * grad / grad.view(X.shape[0], -1).norm(p=2, dim=1).view(X.shape[0], 1, 1, 1)
        # project to l_2 ball
        dists = (X_adv - X).view(X.shape[0], -1).norm(dim=1, p=2).view(X.shape[0], 1, 1, 1)
        X_adv = t.clamp(X + t.min(dists, adv_eps*t.ones_like(dists))*(X_adv-X)/(dists+eps), min=-1, max=1)
    else:
        raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
    return X_adv

def purify_and_predict(X, y, purify_reps=1, requires_grad=True):
    # parallel states for either EOT attack grad or EOT defense with large-sample evaluation of stochastic classifier
    X_repeat = X.repeat([purify_reps, 1, 1, 1])
    # predict labels of purified states
    correct, loss = predict(X_repeat, y, requires_grad, purify_reps,
                            config['eot_defense_ave'], config['eot_attack_ave'])
    if requires_grad:
        # get BPDA gradients with respect to purified states
        X_grads = t.autograd.grad(loss, [X_repeat])[0]
        # average gradients over parallel samples for EOT attack
        attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)
        return correct, attack_grad
    else:
        return correct, None

def eot_defense_verification(X_adv, y, correct, defended):
    # confirm that images are broken using a large sample size to evaluate the stochastic classifier
    for verify_ind in range(correct.nelement()):
        if correct[verify_ind] == 0 and defended[verify_ind] == 1:
            defended[verify_ind] = purify_and_predict(X_adv[verify_ind].unsqueeze(0), y[verify_ind].view([1]),
                                                      config['eot_defense_reps'], requires_grad=False)[0]
    return defended

def eval_and_bpda_eot_grad(X_adv, y, defended, requires_grad=True):
    # forward pass to identify candidates for breaks (and backward pass to get BPDA + EOT grad if requires_grad==True)
    correct, attack_grad = purify_and_predict(X_adv, y, config['eot_attack_reps'], requires_grad)
    # evaluate candidates for breaks using a large number of parallel MCMC samples
    if config['langevin_steps'] > 0 and config['eot_defense_reps'] > 0:
        defended = eot_defense_verification(X_adv, y, correct, defended)
    else:
        defended *= correct
    return defended, attack_grad

def eval_and_clf_pgd_grad(X_adv, y, requires_grad=True):
    X_pgd = t.autograd.Variable(X_adv.clone(), requires_grad=requires_grad)
    # forward pass to get prediction for current adversaries
    correct, loss = predict(X_pgd, y, requires_grad=requires_grad)
    # backward pass to get pgd attack gradient
    if requires_grad:
        attack_grad = t.autograd.grad(loss, [X_pgd])[0]
    else:
        attack_grad = None
    return correct, attack_grad

def eval_and_attack_grad(X_adv, y, defended, step, requires_grad=True):
    if config['langevin_steps'] > 0 and config['use_bpda_eot']:
        # stochastic classifier eval and BPDA + EOT attack gradient if requires_grad
        defended, attack_grad = eval_and_bpda_eot_grad(X_adv, y, defended, requires_grad)
    else:
        # pgd attack vs. deterministic classifier network followed by stochastic classification
        # this is just pgd vs. deterministic classifier network if langevin_steps=0
        if step == -1 or step == config['adv_steps']:
            # stochastic classification for baseline/final eval (or deterministic classifier eval if langevin_steps=0)
            defended, attack_grad = eval_and_bpda_eot_grad(X_adv, y, defended, False)
        else:
            # deterministic classifier network eval and PGD attack gradient from unpurified samples if requires_grad
            correct, attack_grad = eval_and_clf_pgd_grad(X_adv, y, requires_grad)
            if config['langevin_steps'] == 0:
                defended *= correct
    return defended, attack_grad

def attack_batch(X, y, batch_num):
    # get baseline accuracy for natural images
    defended = eval_and_attack_grad(X, y, t.ones_like(y).bool(), -1, False)[0]
    print('Batch {} of {} Baseline: {} of {}'.
          format(batch - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
                 defended.sum(), len(defended)))

    # record of defense over attacks
    class_batch = t.zeros([config['adv_steps'] + 2, X.shape[0]]).bool()
    class_batch[0] = defended.cpu()
    # record for adversarial images for verified breaks
    ims_adv_batch = t.zeros(X.shape)
    for ind in range(defended.nelement()):
        if defended[ind] == 0:
            # record mis-classified natural images as adversarial states
            ims_adv_batch[ind] = X[ind].cpu()

    # initialize adversarial image as natural image
    X_adv = X.clone()
    # start in random location of l_p ball
    if config['adv_rand_start']:
        X_adv = rand_init_l_p(X_adv, config['adv_norm'], config['adv_eps'])

    # adversarial attacks on a single batch of images
    for step in range(config['adv_steps'] + 1):

        # get attack gradient and update defense record
        defended, attack_grad = eval_and_attack_grad(X_adv, y, defended, step)

        # update step-by-step defense record
        class_batch[step+1] = defended.cpu()
        # add adversarial images for newly broken images to list
        for ind in range(defended.nelement()):
            if class_batch[step, ind] == 1 and defended[ind] == 0:
                ims_adv_batch[ind] = X_adv[ind].cpu()

        # update adversarial images (except on final iteration so final adv images match final eval)
        if step < config['adv_steps']:
            X_adv = pgd_update(X_adv, attack_grad, X, config['adv_norm'], config['adv_eps'], config['adv_eta'])

        if step == 1 or step % config['log_freq'] == 0 or step == config['adv_steps']:
            # print attack info
            print('Batch {} of {}, Attack {} of {}   Batch defended: {} of {}'.
                  format(batch_num - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
                         step, config['adv_steps'], int(t.sum(defended).cpu().numpy()), X_adv.shape[0]))

    # record final adversarial image for unbroken states
    for ind in range(defended.nelement()):
        if defended[ind] == 1:
            ims_adv_batch[ind] = X_adv[ind].cpu()

    return class_batch, ims_adv_batch


########################################
# ## ATTACK CLASSIFIER AND PURIFIER ## #
########################################

# defense record for over attacks
class_path = t.zeros([config['adv_steps'] + 2, 0]).bool()
# record of original images, adversarial images, and labels
labs = t.zeros(0).long()
ims_orig = t.zeros(0)
ims_adv = t.zeros(0)

# run adversarial attacks on samples from image bank in small batches
print('\nAttack has begun.\n----------')
for batch, (X_batch, y_batch) in enumerate(attack_loader):
    if (batch + 1) < config['start_batch']:
        continue
    elif (batch + 1) > config['end_batch']:
        break
    else:
        # record original states and labels
        ims_orig = t.cat((ims_orig, X_batch), dim=0)
        labs = t.cat((labs, y_batch), dim=0)

        # load to gpu
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

        # attack images using setting in config
        class_batch, ims_adv_batch = attack_batch(X_batch, y_batch, batch)

        # update defense records
        class_path = t.cat((class_path, class_batch), dim=1)
        # record adversarial images
        ims_adv = t.cat((ims_adv, ims_adv_batch), dim=0)
        print('Attack concluded on Batch {} of {}. Total Secure Images: {} of {}\n-----------'.
              format(batch - config['start_batch'] + 2, config['end_batch'] - config['start_batch'] + 1,
                     class_path[config['adv_steps']+1, :].sum(), class_path.shape[1]))
        # save attack info
        t.save({'ims_orig': ims_orig, 'ims_adv': ims_adv, 'labs': labs, 'class_path': class_path},
               exp_dir + 'log/results.pth')

# final defense accuracy
accuracy_baseline = float(class_path[0, :].sum()) / class_path.shape[1]
accuracy_adv = float(class_path[config['adv_steps']+1, :].sum()) / class_path.shape[1]
print('\n{} Attack Results for {} samples: Non-Adversarial {}    Adversarial: {}'.format(config['adv_norm'], class_path.shape[1], accuracy_baseline, accuracy_adv))
# plot accuracy over attacks
plt.plot(class_path.float().mean(1).numpy())
plt.table(cellText=[[accuracy_baseline, accuracy_adv, class_path.shape[1]]],
          colLabels=['baseline', 'secure', 'total images'], bbox=[0.0, -0.35, 1, 0.125])
plt.xlabel('attack')
plt.ylabel('accuracy')
plt.subplots_adjust(bottom=0.25)
plt.savefig(exp_dir + 'log/accuracy_over_attack.png')
plt.close()
