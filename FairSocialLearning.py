#!/usr/bin/env python
# coding: utf-8

# What is this?
## Simulation of the paper "On Statistical Discrimination as a Failure of Social Learning: A Multi-Armed Bandit Approach" (Management Science, to appear). https://arxiv.org/abs/2010.01079
## To replicate the results with symmetric models, run
##   python FairSocialLearning.py --size=full 
## To replicate the results with asymmtric models, run 
##   python FairSocialLearning.py --size=full --asymmetric

import argparse

# ArgumentParser
parser = argparse.ArgumentParser(description="Your script description.")
parser.add_argument("--asymmetric", action='store_true', default=False,
                    help="Enable asymmetric mode. Default is False.")
parser.add_argument("--size", choices=["full", "middle", "debug"], default="middle",
                    help="Set the size. Choices are ['full', 'middle', 'debug']. Default is 'middle'.")
parser.add_argument("--save_pickles", action='store_true', default=False,
                    help="Enable save pickles. Default is False.")
args = parser.parse_args()

print(f"Asymmetric: {args.asymmetric}")
print(f"Size: {args.size}")
print(f"Save Pickles: {args.save_pickles}")

import numpy as np
np.random.seed(1)
import os
import matplotlib
try:
    from google.colab import files
    #files.download(filename)  
except: # https://stackoverflow.com/questions/35737116/runtimeerror-invalid-display-variable
    matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial import distance #cosine similarity
from scipy import stats
import copy
import pickle
from matplotlib.colors import LinearSegmentedColormap
import sys

# Save all figures to this directory
if args.asymmetric:
    fig_dir = "figures_asymmetric/"
else:
    fig_dir = "figures/"
# Save all pickles to this directory. If "" then pickle is not saved
if args.save_pickles:
    pickle_dir = "pickles/"
else:
    pickle_dir = ""

import multiprocessing
N_JOBS = max(1, int(multiprocessing.cpu_count()*0.8)) # Number of processes. 0.8 is for not using all processors

# model parameters
mu_x = 1.5 # 3
group_theta_var = False # Not used
if args.asymmetric:
    group_mux_var = True # True for Asymmetric model, False for Symmetric model
else:
    group_mux_var = False
d = 5
sigma_x = 1 # 2
group_x_var = False # Not used
sigma_eps = 0.5 * sigma_x #* np.sqrt(d)
lmd = 1 #lambda
N_SWITCH = 50 #rooney->greedy: switch timing
IUCB_THRESHOLD = 0.5 #Hybrid parameter
run_full = False
run_middle = False
save_img = True
def is_colab():
    try:
        from google.colab import files
        return True
    except:
        return False

# Scale of simulation
if args.size == "full": # full run
    print("full simulation")
    R = 4000
    N = 1000
    run_full = True
elif args.size == "middle": # middle-size run
    print("middle-scale simulation")
    R = 1000
    N = 1000
    run_middle = True
else: # very short run for debug
    print("debug (short) simulation")
    R = 10
    N = 300

# matplotlib params for plotting
Alpha = 0.2
Capsize = 10
COLOR_UCB = "tab:blue"
COLOR_CS_UCB = "navy"
COLOR_HYBRID = "tab:orange"
COLOR_CS_HYBRID = "#cc4c0b" 
COLOR_GREEDY = "tab:red"
COLOR_ROONEY = "lightblue"
COLOR_ROONEY_SWITCH = "tab:green" 
COLOR_ROONEY_GREEDY = "tab:red"
HATCH_BLUE = '/'
HATCH_RED = '\\'
HATCH_GREEN = '//'
# linestyle
linestyle_tuple = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),

     'loosely dashed':        (0, (5, 10)),
     'moderately loosely dashed':        (0, (5, 7)),
     'dashed':                (0, (5, 5)),
     'moderately densely dashed':        (0, (5, 3)),
     'densely dashed':        (0, (5, 1)),

     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),

     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
     }
LINESTYLE_UCB = "solid"
LINESTYLE_CS_UCB = linestyle_tuple["densely dashed"]
LINESTYLE_HYBRID = "dashdot"
LINESTYLE_CS_HYBRID = "dashed" #modified from dashed
LINESTYLE_GREEDY = linestyle_tuple["densely dotted"]
LINESTYLE_CS_GREEDY = "dotted"
LINESTYLE_ROONEY = "dashdot"
LINESTYLE_ROONEY_SWITCH = linestyle_tuple["densely dashdotted"]
LINESTYLE_ROONEY_GREEDY = linestyle_tuple["densely dotted"]

Figsize = (6,4)
#Fontsize = 18 
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["figure.subplot.bottom"] = 0.14

# Main simulation class
class Simulation:
    def __init__(self, Kg, N0, N=N, theta_pars = None, sigma_eta = 1, iucb_threshold = IUCB_THRESHOLD,
                 policy = "greedy", count_ws_regret = False, n_switch = -1, mu_x_diff = False):
        self.rng = np.random
        self.Kg = Kg
        self.N0 = N0
        self.N = N
        self.Ng = len(self.Kg) # Num of groups.
        self.K = np.sum(self.Kg)
        self.iucb_threshold = iucb_threshold
        self.initial_order = self.rng.permutation(range(self.K)) #initial sample order (shuffled)
        if theta_pars == None:
            self.thetas = [1 * np.ones(d) for g in range(self.Ng)] # theta = (1,1,..) 
            if group_theta_var:
                self.thetas[0] *= 1.5
        else:
            self.thetas = [theta_pars[g] * np.ones(d) for g in range(self.Ng)]
        G_tmp = []
        for g,ng in enumerate(self.Kg):
            for i in range(ng):
                G_tmp.append(g)
        self.gs = np.array(G_tmp) #group attribution
        self.policy = policy
        self.count_ws_regret = count_ws_regret
        self.n_switch = n_switch
        self.mu_x_diff = mu_x_diff
        self.Vs = [lmd * np.identity(d) for g in range(self.Ng)]
        self.xys = [np.zeros(d) for g in range(self.Ng)]
        self.regret = 0 #total regret
        self.strong_regret = 0 #Unconstrained regret in the paper
        self.draws = np.zeros(self.Ng) # num of draws for each group
        self.best = np.zeros(self.Ng) # num of true best cand for each group
        self.nonbest = np.zeros(self.Ng) # num of true non-best cand for each group
        self.best_draws = np.zeros(self.Ng) # draws with true best
        self.subsidy = 0 #total subsidy
        self.regret_seq = np.zeros(self.N)
        self.strong_regret_seq = np.zeros(self.N)
        self.subsidycs = 0 #total subsidy (cost-saving scheme)
        self.draws_seq = np.zeros((self.N, self.Ng))
        self.best_seq = np.zeros((self.N, self.Ng))
        self.nonbest_seq = np.zeros((self.N, self.Ng))
        self.best_draws_seq = np.zeros((self.N, self.Ng)) # draws_seq with true best
        self.best_group_seq = np.zeros(self.N) #group of the best cand for each round
        self.subsidy_seq = np.zeros(self.N)
        self.subsidycs_seq = np.zeros(self.N)
        self.sigma_eta = sigma_eta
        self.n = 0
    def draw_x(self, g): 
        mu = mu_x
        if g==1:
            if type(self.mu_x_diff) != bool: 
                mu += self.mu_x_diff
            elif group_mux_var: #smaller mean in group 2
                mu /= 1.5
        sigma = sigma_x
        if g==1 and group_x_var: #larger variance in group 2
            sigma = sigma * np.sqrt(2)
        return np.array([mu + sigma * self.rng.normal() for f in range(d)])
    def get_htheta(self, g): # hat{theta} = Vg^-1 xy
        return np.linalg.solve(self.Vs[g], self.xys[g])
    def get_ttheta(self, x, g, htheta): # UCB index
        alpha = 1 /np.sqrt(d) #scale param
        delta = 0.1
        V = self.Vs[g]
        w,P = np.linalg.eig(V)
        s = np.sum(np.log(w)) - d * np.log(lmd) + 2 * np.log(1./delta)  
        width = sigma_eps * alpha * np.sqrt(s) #confidence diwth
        P_inv = np.linalg.inv(P) 
        D = P_inv @ V @ P
        xd = P_inv @ x
        normalized_xd = np.array([xd[f] / np.sqrt(D[f,f]) for f in range(d)]) 
        my_norm = np.sqrt(np.inner(normalized_xd, D @ normalized_xd))
        xd_ex = normalized_xd * (width / my_norm)
        xd_ex_rec = P @ xd_ex 
        return xd_ex_rec + htheta
    def get_bestarm(self, xs): #best arm
        istar = np.argmax([self.get_q(x,i) for i,x in enumerate(xs)])
        return istar
    def select_arm(self, xs):
        if self.n == self.N0:
            for g in range(self.Ng):
                pass
        if self.n < self.N0: # initial samples
            vs_htheta = np.zeros(len(xs))
            vs_htheta_group = np.zeros(len(xs)) #not used
            iota = self.initial_order[ self.n % self.K ]
            g_iota = self.gs[iota] #group of iota
            for i,x in enumerate(xs):
                g = self.gs[i]
                htheta = self.get_htheta(g)
                vs_htheta[i] = np.inner(x, htheta)
                if g == g_iota:
                    vs_htheta_group[i] = np.inner(x, htheta)
                else:
                    vs_htheta_group[i] = - 10000000
            self.subsidycs_seq[self.n] = self.subsidycs
            if self.count_ws_regret:
                istar = np.argmax(vs_htheta) #best arm
                self.subsidycs += vs_htheta[istar] - vs_htheta[iota]
            self.subsidycs_seq[self.n] = self.subsidycs
            return iota
        elif self.policy == "greedy": #LF
            vs = np.zeros(len(xs))
            for i,x in enumerate(xs):
                g = self.gs[i]
                htheta = self.get_htheta(g)
                vs[i] = np.inner(x, htheta)
            self.subsidycs_seq[self.n] = self.subsidycs
            return np.argmax(vs)
        elif self.policy == "ucb" or self.policy == "improved_ucb": #imporoved ucb = hybrid in the paper
            vs = np.zeros(len(xs))
            vs_htheta = np.zeros(len(xs))
            for i,x in enumerate(xs):
                g = self.gs[i]
                htheta = self.get_htheta(g)
                ttheta = self.get_ttheta(x, g, htheta)
                vs_htheta[i] = np.inner(x, htheta)
                if self.policy == "ucb":
                    vs[i] = np.inner(x, ttheta)
                else: # iucb (hybrid)
                    Delta = np.inner(x, ttheta - htheta)
                    if Delta <= self.iucb_threshold * sigma_x * np.linalg.norm(htheta): 
                        vs[i] = np.inner(x, htheta) #no subsidize
                    else:
                        vs[i] = np.inner(x, ttheta) #subsidize
            istar = np.argmax(vs) # best arm
            self.subsidy += vs[istar] - vs_htheta[istar]
            self.subsidycs += np.max(vs_htheta) - vs_htheta[istar]
            self.subsidy_seq[self.n] = self.subsidy
            self.subsidycs_seq[self.n] = self.subsidycs
            return istar
        elif self.policy == "rooney": # and (self.n % 10) == 0:
            # first stage
            vs = np.zeros(len(xs))
            vs_true = np.zeros(len(xs))
            for i,x in enumerate(xs):
                g = self.gs[i]
                htheta = self.get_htheta(g)
                vs[i] = np.inner(x, htheta)
                vs_true[i] = np.inner(x, self.thetas[g])
            i_gs = np.array([-1 for g in range(self.Ng)]) #finalist for each group
            for g in range(self.Ng):
                vs_filt = np.zeros(len(xs))
                for i,x in enumerate(xs):
                    if g == self.gs[i]:
                        vs_filt[i] = vs[i]
                    else:
                        vs_filt[i] = -10000000
                i_gs[g] = np.argmax(vs_filt)
            self.etas = np.array([self.sigma_eta * self.rng.normal() for i in xs]) #2nd stage information (additional signals)
            finalist_vs = np.array([vs[i] + self.etas[i] for i in i_gs])
            g = np.argmax(finalist_vs)
            return i_gs[g]
        elif self.policy in ["rooney", "rooney_greedy"]: #rooney-greedy = LF (2-stage)
            # first stage
            vs = np.zeros(len(xs))
            vs_true = np.zeros(len(xs))
            for i,x in enumerate(xs):
                g = self.gs[i]
                htheta = self.get_htheta(g)
                vs[i] = np.inner(x, htheta)
                vs_true[i] = np.inner(x, self.thetas[g])
            i_gs = np.array([-1 for g in range(self.Ng)]) 
            vs_filt = copy.deepcopy(vs)
            for g in range(self.Ng):
                i_gs[g] = np.argmax(vs_filt)
                vs_filt[ i_gs[g] ] = -10000000
            self.etas = np.array([self.sigma_eta * self.rng.normal() for i in xs]) #addiional signal eta
            finalist_vs = np.array([vs[i] + self.etas[i] for i in i_gs])
            finalist_vs_true = np.array([vs_true[i] + self.etas[i] for i in i_gs])
            g = np.argmax(finalist_vs)
            return i_gs[g]
        else:
            print("Unknown policy");assert(False)
    def get_q(self, x, i):
        g = self.gs[i]
        theta_g = self.thetas[g]
        if not self.policy in ["rooney", "rooney_greedy"] or self.n < self.N0:
            return np.inner(x, theta_g)
        else:
            return np.inner(x, theta_g) + self.etas[i]
    def get_reward(self, x, i):
        eps = sigma_eps * self.rng.normal()
        return self.get_q(x, i) + eps
    def calculate_rooney_best(self, xs): 
        # calc Rooney Regret
        qs_base = [np.inner(x, self.thetas[self.gs[i]]) for i,x in enumerate(xs)]
        i_gs = np.array([-1 for g in range(self.Ng)]) 
        for g in range(self.Ng):
            qs_filt = np.zeros(len(xs))
            for i,x in enumerate(xs):
                if g == self.gs[i]:
                    qs_filt[i] = qs_base[i]
                else:
                    qs_filt[i] = -10000000
            i_gs[g] = np.argmax(qs_filt)
        finalist_qs = np.array([qs_base[i] + self.etas[i] for i in i_gs])
        rooney_best_q = np.max(finalist_qs)
        qs_filt = copy.deepcopy(qs_base)
        i_gs = np.array([-1 for g in range(self.Ng)])
        for g in range(self.Ng):
            i_gs[g] = np.argmax(qs_filt)
            qs_filt[ i_gs[g] ] = -10000000
        finalist_qs = np.array([qs_base[i] + self.etas[i] for i in i_gs])
        omni_best_q = np.max(finalist_qs)
        return (rooney_best_q, omni_best_q) 
    def update_reward(self, xs, y, i, g):
        x = xs[i]
        xx = np.outer(x, x)
        xy = x*y
        self.Vs[g] += xx
        self.xys[g] += xy
        # update regret
        if self.n >= self.N0 or self.count_ws_regret:
            if not self.policy in ["rooney", "rooney_greedy"]:
                istar = self.get_bestarm(xs)
                self.regret += self.get_q(xs[istar], istar) - self.get_q(xs[i], i)
            else: #rooney
                rooney_best_q, omni_best_q = self.calculate_rooney_best(xs)
                self.regret += rooney_best_q - self.get_q(xs[i], i)
                self.strong_regret += omni_best_q - self.get_q(xs[i], i)
        self.draws[g] += 1
        istar = self.get_bestarm(xs)
        gstar = self.gs[istar]
        self.nonbest += self.Kg
        self.best[gstar] += 1
        self.nonbest[gstar] -= 1        
        if istar == i: # true best = selected arm
            self.best_draws[g] += 1
        self.best_group_seq[self.n] = gstar
        self.regret_seq[self.n] = self.regret
        if self.policy in ["rooney", "rooney_greedy"]:
            self.strong_regret_seq[self.n] = self.strong_regret
        self.draws_seq[self.n] = self.draws
        self.best_seq[self.n] = self.best
        self.nonbest_seq[self.n] = self.nonbest
        self.best_draws_seq[self.n] = self.best_draws
        self.n = self.n + 1
        self.eta = None
    def is_perpetunderest(self):
        #self.perpunderestimate = True
        for g in range(self.Ng):
            if self.draws[g] == self.draws_seq[self.N0, g]: 
                return True 
        return False
    def round(self):
        # draw context
        xs = [self.draw_x(self.gs[i]) for i in range(self.K)]
        # select arm
        iota = self.select_arm(xs)
        # observe reward
        x = xs[iota]
        y = self.get_reward(x, iota)
        # update model parameters
        self.update_reward(xs, y, iota, self.gs[iota])
    def get_cand(self, g):
        # draw context
        x = self.draw_x(self.gs[g])
        iota = g
        # observe reward
        y = self.get_reward(x, iota)
        return y
    def run(self, rng):
        self.rng = rng
        for n in range(self.N):
            if self.n_switch == self.n and self.policy == "rooney": #if n_switch is set then turns to greedy (LF) at some timestep
                self.policy = "rooney_greedy" 
            self.round()

def my_show():
    try:
        from google.colab import files
        plt.show()
    except:
        pass

def my_savefig(fig, filename):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(os.path.join(fig_dir, filename), dpi=fig.dpi, bbox_inches='tight')
    
def colab_save(filename):
    try:
        from google.colab import files
        files.download(filename)  
    except:
        pass

# parallel computation.
# avoiding randomness with https://joblib.readthedocs.io/en/latest/auto_examples/parallel_random_state.html
def run_sim(sim, random_state):
    rng = np.random.RandomState(random_state)
    sim.run(rng)
    return sim
from joblib import Parallel, delayed

def output_to_pickle(filename, data):
    if len(pickle_dir) == 0:
        pass
    else:
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)
        pickle.dump(data, open( os.path.join(pickle_dir, filename), "wb" ) )


# LF versus UCB
def experiment1():
    Kg = (10, 2)
    sims = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "greedy") for r in range(R)]
    sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "ucb") for r in range(R)]

    rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
    sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_ucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_ucb[r], rss[r+R]) for r in range(R)] )
    output_to_pickle("experiment1.pickle", (sims, sims_ucb))

    all_regret = np.zeros( (R, N) )
    all_regret_ucb = np.zeros( (R, N) )
    all_draw2 = np.zeros( (R, N) )
    all_draw2_ucb = np.zeros( (R, N) )
    for r in range(R):
        all_regret[r] = sims[r].regret_seq
        all_regret_ucb[r] = sims_ucb[r].regret_seq
        all_draw2[r] = sims[r].draws_seq[:,1]
        all_draw2_ucb[r] = sims_ucb[r].draws_seq[:,1]

    #plotting
    fig = plt.figure(figsize=Figsize)
    avg_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)
    lower_quantile_regret = np.quantile(all_regret, 0.05, axis=0)
    upper_quantile_regret = np.quantile(all_regret, 0.95, axis=0)
    avg_regret_ucb = np.mean(all_regret_ucb, axis=0)
    std_regret_ucb = np.std(all_regret_ucb, axis=0)
    lower_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.05, axis=0)
    upper_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.errorbar([N-1], avg_regret[-1], yerr=2*std_regret[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret, upper_quantile_regret, alpha=0.3, color = COLOR_GREEDY)
    plt.plot(range(N), avg_regret_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_regret_ucb[-1], yerr=2*std_regret_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret_ucb, upper_quantile_regret_ucb, alpha=0.3, color = COLOR_UCB)
    plt.ylabel("Regret")
    plt.ylim(0, avg_regret[-1]*1.2)
    plt.xlabel("Round (n)")
    plt.legend()
    if save_img:
        my_savefig(fig, 'policycomp_regret.pdf')
        colab_save('policycomp_regret.pdf')
    my_show()
    plt.clf()

np.random.seed(2)
experiment1()

# UCB versus Hybrid
def experiment2():
    Kg = (10, 2)
    sims = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "greedy") for r in range(R)]
    sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "ucb") for r in range(R)]
    sims_iucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb") for r in range(R)]
    rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
    sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_ucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_ucb[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_iucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb[r], rss[r+R]) for r in range(R)] ) #parallel computation
    output_to_pickle("experiment2.pickle", (sims, sims_ucb, sims_iucb))

    all_regret = np.zeros( (R,N) )
    all_draw2 = np.zeros( (R,N) )
    all_subsidy = np.zeros( (R,N) )
    all_subsidycs = np.zeros( (R,N) )
    all_regret_ucb = np.zeros( (R,N) )
    all_draw2_ucb = np.zeros( (R,N) )
    all_subsidy_ucb = np.zeros( (R,N) )
    all_subsidycs_ucb = np.zeros( (R,N) )
    all_regret_iucb = np.zeros( (R,N) )
    all_draw2_iucb = np.zeros( (R,N) )
    all_subsidy_iucb = np.zeros( (R,N) )
    all_subsidycs_iucb = np.zeros( (R,N) )
    for r in range(R):
        all_regret[r] += sims[r].regret_seq
        all_draw2[r] += sims[r].draws_seq[:,1]
        all_subsidy[r] += sims[r].subsidy_seq
        all_subsidycs[r] += sims[r].subsidycs_seq
        all_regret_ucb[r] += sims_ucb[r].regret_seq
        all_draw2_ucb[r] += sims_ucb[r].draws_seq[:,1]
        all_subsidy_ucb[r] += sims_ucb[r].subsidy_seq
        all_subsidycs_ucb[r] += sims_ucb[r].subsidycs_seq
        all_regret_iucb[r] += sims_iucb[r].regret_seq
        all_draw2_iucb[r] += sims_iucb[r].draws_seq[:,1]
        all_subsidy_iucb[r] += sims_iucb[r].subsidy_seq
        all_subsidycs_iucb[r] += sims_iucb[r].subsidycs_seq
    output_to_pickle("out/experiment2d.pickle", (all_regret_ucb, all_draw2_ucb, all_subsidy_ucb, all_subsidycs_ucb, all_regret_iucb, all_draw2_iucb, all_subsidy_iucb, all_subsidycs_iucb))

    #plotting
    fig = plt.figure(figsize=Figsize)
    avg_regret_ucb = np.mean(all_regret_ucb, axis=0)
    std_regret_ucb = np.std(all_regret_ucb, axis=0)
    lower_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.05, axis=0)
    upper_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_regret_ucb[-1], yerr=2*std_regret_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret_ucb, upper_quantile_regret_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_regret_iucb = np.mean(all_regret_iucb, axis=0)
    std_regret_iucb = np.std(all_regret_iucb, axis=0)
    lower_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.05, axis=0)
    upper_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_regret_iucb[-1], yerr=2*std_regret_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret_iucb, upper_quantile_regret_iucb, alpha=Alpha, color = COLOR_HYBRID)
    plt.ylabel("Regret")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_regret.pdf')
        colab_save('iucb_regret.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_draw2_ucb = np.mean(all_draw2_ucb, axis=0)
    std_draw2_ucb = np.std(all_draw2_ucb, axis=0)
    lower_quantile_draw2_ucb = np.quantile(all_draw2_ucb, 0.05, axis=0)
    upper_quantile_draw2_ucb = np.quantile(all_draw2_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_draw2_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_draw2_ucb[-1], yerr=2*std_draw2_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_ucb, upper_quantile_draw2_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_draw2_iucb = np.mean(all_draw2_iucb, axis=0)
    std_draw2_iucb = np.std(all_draw2_iucb, axis=0)
    lower_quantile_draw2_iucb = np.quantile(all_draw2_iucb, 0.05, axis=0)
    upper_quantile_draw2_iucb = np.quantile(all_draw2_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_draw2_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_draw2_iucb[-1], yerr=2*std_draw2_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_iucb, upper_quantile_draw2_iucb, alpha=Alpha, color = COLOR_HYBRID)
    plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])

    plt.ylabel("# of minorities hired")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_draw2.pdf')
        colab_save('iucb_draw2.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_subsidy_ucb = np.mean(all_subsidy_ucb, axis=0)
    std_subsidy_ucb = np.std(all_subsidy_ucb, axis=0)
    lower_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.05, axis=0)
    upper_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_ucb, label = "UCB", color=COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_subsidy_ucb[-1], yerr=2*std_subsidy_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_ucb, upper_quantile_subsidy_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_subsidy_iucb = np.mean(all_subsidy_iucb, axis=0)
    std_subsidy_iucb = np.std(all_subsidy_iucb, axis=0)
    lower_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.05, axis=0)
    upper_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_subsidy_iucb[-1], yerr=2*std_subsidy_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_iucb, upper_quantile_subsidy_iucb, alpha=Alpha, color = COLOR_HYBRID)

    plt.ylabel("Subsidy")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_subsidy.pdf')
        colab_save('iucb_subsidy.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_subsidy_iucb = np.mean(all_subsidy_iucb, axis=0)
    std_subsidy_iucb = np.std(all_subsidy_iucb, axis=0)
    lower_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.05, axis=0)
    upper_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_subsidy_iucb[-1], yerr=2*std_subsidy_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_iucb, upper_quantile_subsidy_iucb, alpha=Alpha, color = COLOR_HYBRID)

    avg_subsidycs_ucb = np.mean(all_subsidycs_ucb, axis=0)
    std_subsidycs_ucb = np.std(all_subsidycs_ucb, axis=0)
    lower_quantile_subsidycs_ucb = np.quantile(all_subsidycs_ucb, 0.05, axis=0)
    upper_quantile_subsidycs_ucb = np.quantile(all_subsidycs_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidycs_ucb, label = "CS-UCB", color = COLOR_CS_UCB, linestyle = LINESTYLE_CS_UCB)
    plt.errorbar([N-1], avg_subsidycs_ucb[-1], yerr=2*std_subsidycs_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_CS_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidycs_ucb, upper_quantile_subsidycs_ucb, alpha=Alpha, color = COLOR_CS_UCB)

    avg_subsidycs_iucb = np.mean(all_subsidycs_iucb, axis=0)
    std_subsidycs_iucb = np.std(all_subsidycs_iucb, axis=0)
    lower_quantile_subsidycs_iucb = np.quantile(all_subsidycs_iucb, 0.05, axis=0)
    upper_quantile_subsidycs_iucb = np.quantile(all_subsidycs_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidycs_iucb, label = "CS-Hybrid", color = COLOR_CS_HYBRID, linestyle = LINESTYLE_CS_HYBRID)
    plt.errorbar([N-1], avg_subsidycs_iucb[-1], yerr=2*std_subsidycs_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_CS_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidycs_iucb, upper_quantile_subsidycs_iucb, alpha=Alpha, color = COLOR_CS_HYBRID)

    plt.ylabel("Subsidy")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_subsidy_cs.pdf')
        colab_save('iucb_subsidy_cs.pdf')
    plt.clf()

    # fairness measure plotting starts here
    all_disparity = np.zeros( (R, N) )
    all_disparity_ucb = np.zeros( (R, N) )
    all_disparity_iucb = np.zeros( (R, N) )
    for r in range(R):
        all_disparity[r] = np.abs( (sims[r].draws_seq[:,0]/Kg[0] - sims[r].draws_seq[:,1]/Kg[1])/(sims[r].draws_seq[:,0] + sims[r].draws_seq[:,1]) )
        all_disparity_ucb[r] = np.abs( (sims_ucb[r].draws_seq[:,0]/Kg[0] - sims_ucb[r].draws_seq[:,1]/Kg[1])/(sims_ucb[r].draws_seq[:,0] + sims_ucb[r].draws_seq[:,1]) )
        all_disparity_iucb[r] = np.abs( (sims_iucb[r].draws_seq[:,0]/Kg[0] - sims_iucb[r].draws_seq[:,1]/Kg[1])/(sims_iucb[r].draws_seq[:,0] + sims_iucb[r].draws_seq[:,1]) )
    fig = plt.figure(figsize=Figsize)
    avg_disparity = np.mean(all_disparity, axis=0)
    std_disparity = np.std(all_disparity, axis=0)
    lower_quantile_disparity = np.quantile(all_disparity, 0.05, axis=0)
    upper_quantile_disparity = np.quantile(all_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_disparity, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.errorbar([N-1], avg_disparity[-1], yerr=2*std_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity, upper_quantile_disparity, alpha=0.3, color = COLOR_GREEDY)
    
    avg_disparity_ucb = np.mean(all_disparity_ucb, axis=0)
    std_disparity_ucb = np.std(all_disparity_ucb, axis=0)
    lower_quantile_disparity_ucb = np.quantile(all_disparity_ucb, 0.05, axis=0)
    upper_quantile_disparity_ucb = np.quantile(all_disparity_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_disparity_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_disparity_ucb[-1], yerr=2*std_disparity_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity_ucb, upper_quantile_disparity_ucb, alpha=0.3, color = COLOR_UCB)

    avg_disparity_iucb = np.mean(all_disparity_iucb, axis=0)
    std_disparity_iucb = np.std(all_disparity_iucb, axis=0)
    lower_quantile_disparity_iucb = np.quantile(all_disparity_iucb, 0.05, axis=0)
    upper_quantile_disparity_iucb = np.quantile(all_disparity_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_disparity_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_disparity_iucb[-1], yerr=2*std_disparity_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity_iucb, upper_quantile_disparity_iucb, alpha=0.3, color = COLOR_HYBRID)
    plt.ylabel("Demographic disparity")
    plt.xlabel("Round (n)")
    plt.ylim(-0.00, 0.1)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_disparity.pdf')
        colab_save('iucb_disparity.pdf')
    plt.clf()


    all_eo_disparity = np.zeros(  (R, N) )
    all_eo_disparity_ucb = np.zeros(  (R, N) )
    all_eo_disparity_iucb = np.zeros( (R, N) )
    EPS = 0.000001
    for r in range(R):
        all_eo_disparity[r] = np.abs(sims[r].best_draws_seq[:,0]/(sims[r].best_seq[:,0] + EPS) -  sims[r].best_draws_seq[:,1]/(sims[r].best_seq[:,1] + EPS) )
        all_eo_disparity_ucb[r] = np.abs(sims_ucb[r].best_draws_seq[:,0]/(sims_ucb[r].best_seq[:,0] + EPS) -  sims_ucb[r].best_draws_seq[:,1]/(sims_ucb[r].best_seq[:,1] + EPS) )
        all_eo_disparity_iucb[r] = np.abs(sims_iucb[r].best_draws_seq[:,0]/(sims_iucb[r].best_seq[:,0] + EPS) -  sims_iucb[r].best_draws_seq[:,1]/(sims_iucb[r].best_seq[:,1] + EPS) )
    fig = plt.figure(figsize=Figsize)

    avg_eo_disparity = np.mean(all_eo_disparity, axis=0)
    std_eo_disparity = np.std(all_eo_disparity, axis=0)
    lower_quantile_eo_disparity = np.quantile(all_eo_disparity, 0.05, axis=0)
    upper_quantile_eo_disparity = np.quantile(all_eo_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.errorbar([N-1], avg_eo_disparity[-1], yerr=2*std_eo_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity, upper_quantile_eo_disparity, alpha=0.3, color = COLOR_GREEDY)
    
    avg_eo_disparity_ucb = np.mean(all_eo_disparity_ucb, axis=0)
    std_eo_disparity_ucb = np.std(all_eo_disparity_ucb, axis=0)
    lower_quantile_eo_disparity_ucb = np.quantile(all_eo_disparity_ucb, 0.05, axis=0)
    upper_quantile_eo_disparity_ucb = np.quantile(all_eo_disparity_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_eo_disparity_ucb[-1], yerr=2*std_eo_disparity_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity_ucb, upper_quantile_eo_disparity_ucb, alpha=0.3, color = COLOR_UCB)

    avg_eo_disparity_iucb = np.mean(all_eo_disparity_iucb, axis=0)
    std_eo_disparity_iucb = np.std(all_eo_disparity_iucb, axis=0)
    lower_quantile_eo_disparity_iucb = np.quantile(all_eo_disparity_iucb, 0.05, axis=0)
    upper_quantile_eo_disparity_iucb = np.quantile(all_eo_disparity_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_eo_disparity_iucb[-1], yerr=2*std_eo_disparity_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity_iucb, upper_quantile_eo_disparity_iucb, alpha=0.3, color = COLOR_HYBRID)
    #plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])
    plt.ylabel("Disparity on equal opportunity")
    plt.xlabel("Round (n)")
    plt.ylim(-0.0, 0.5)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_eo_disparity.pdf')
        colab_save('iucb_eo_disparity.pdf')
    plt.clf()
    
    all_pe_disparity = np.zeros(  (R, N) )
    all_pe_fpr0 = np.zeros( (R, N) )
    all_pe_fpr1 = np.zeros( (R, N) )
    all_pe_disparity_ucb = np.zeros(  (R, N) )
    all_pe_fpr0_ucb = np.zeros( (R, N) )
    all_pe_fpr1_ucb = np.zeros( (R, N) )
    all_pe_disparity_iucb = np.zeros( (R, N) )
    all_pe_fpr0_iucb = np.zeros( (R, N) )
    all_pe_fpr1_iucb = np.zeros( (R, N) )
    EPS = 0.000001
    for r in range(R):
        nonbest_draws_seq = sims[r].draws_seq - sims[r].best_draws_seq
        nonbest_seq = sims[r].nonbest_seq
        all_pe_disparity[r] = np.abs(nonbest_draws_seq[:,0]/(nonbest_seq[:,0] + EPS) 
                                         -  nonbest_draws_seq[:,1]/(nonbest_seq[:,1] + EPS) )
        all_pe_fpr0[r] = nonbest_draws_seq[:,0]/(nonbest_seq[:,0] + EPS)
        all_pe_fpr1[r] = nonbest_draws_seq[:,1]/(nonbest_seq[:,1] + EPS)

        nonbest_draws_seq_ucb = sims_ucb[r].draws_seq - sims_ucb[r].best_draws_seq
        nonbest_seq_ucb = sims_ucb[r].nonbest_seq
        all_pe_disparity_ucb[r] = np.abs(nonbest_draws_seq_ucb[:,0]/(nonbest_seq_ucb[:,0] + EPS) 
                                         -  nonbest_draws_seq_ucb[:,1]/(nonbest_seq_ucb[:,1] + EPS) )
        all_pe_fpr0_ucb[r] = nonbest_draws_seq_ucb[:,0]/(nonbest_seq_ucb[:,0] + EPS)
        all_pe_fpr1_ucb[r] = nonbest_draws_seq_ucb[:,1]/(nonbest_seq_ucb[:,1] + EPS)

        nonbest_draws_seq_iucb = sims_iucb[r].draws_seq - sims_iucb[r].best_draws_seq
        nonbest_seq_iucb = sims_iucb[r].nonbest_seq
        all_pe_disparity_iucb[r] = np.abs(nonbest_draws_seq_iucb[:,0]/(nonbest_seq_iucb[:,0] + EPS) 
                                         -  nonbest_draws_seq_iucb[:,1]/(nonbest_seq_iucb[:,1] + EPS) )
        all_pe_fpr0_iucb[r] = nonbest_draws_seq_iucb[:,0]/(nonbest_seq_iucb[:,0] + EPS)
        all_pe_fpr1_iucb[r] = nonbest_draws_seq_iucb[:,1]/(nonbest_seq_iucb[:,1] + EPS)
    fig = plt.figure(figsize=Figsize)

    avg_pe_disparity = np.mean(all_pe_disparity, axis=0)
    std_pe_disparity = np.std(all_pe_disparity, axis=0)
    lower_quantile_pe_disparity = np.quantile(all_pe_disparity, 0.05, axis=0)
    upper_quantile_pe_disparity = np.quantile(all_pe_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.errorbar([N-1], avg_pe_disparity[-1], yerr=2*std_pe_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity, upper_quantile_pe_disparity, alpha=0.3, color = COLOR_GREEDY)
    avg_pe_fpr0 = np.mean(all_pe_fpr0, axis=0)
    avg_pe_fpr1 = np.mean(all_pe_fpr1, axis=0)
#    plt.plot(range(N), avg_pe_fpr0, label = "LF FPR1")
#    plt.plot(range(N), avg_pe_fpr1, label = "LF FPR2")    
    
    avg_pe_disparity_ucb = np.mean(all_pe_disparity_ucb, axis=0)
    std_pe_disparity_ucb = np.std(all_pe_disparity_ucb, axis=0)
    lower_quantile_pe_disparity_ucb = np.quantile(all_pe_disparity_ucb, 0.05, axis=0)
    upper_quantile_pe_disparity_ucb = np.quantile(all_pe_disparity_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_pe_disparity_ucb[-1], yerr=2*std_pe_disparity_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity_ucb, upper_quantile_pe_disparity_ucb, alpha=0.3, color = COLOR_UCB)
    avg_pe_fpr0_ucb = np.mean(all_pe_fpr0_ucb, axis=0)
    avg_pe_fpr1_ucb = np.mean(all_pe_fpr1_ucb, axis=0)
#    plt.plot(range(N), avg_pe_fpr0_ucb, label = "UCB FPR1")
#    plt.plot(range(N), avg_pe_fpr1_ucb, label = "UCB FPR2")

    avg_pe_disparity_iucb = np.mean(all_pe_disparity_iucb, axis=0)
    std_pe_disparity_iucb = np.std(all_pe_disparity_iucb, axis=0)
    lower_quantile_pe_disparity_iucb = np.quantile(all_pe_disparity_iucb, 0.05, axis=0)
    upper_quantile_pe_disparity_iucb = np.quantile(all_pe_disparity_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_pe_disparity_iucb[-1], yerr=2*std_pe_disparity_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity_iucb, upper_quantile_pe_disparity_iucb, alpha=0.3, color = COLOR_HYBRID)
    avg_pe_fpr0_iucb = np.mean(all_pe_fpr0_iucb, axis=0)
    avg_pe_fpr1_iucb = np.mean(all_pe_fpr1_iucb, axis=0)
#    plt.plot(range(N), avg_pe_fpr0_iucb, label = "Hybrid FPR1")
#    plt.plot(range(N), avg_pe_fpr1_iucb, label = "Hybrid FPR2")

    
    #plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])
    plt.ylabel("Disparity on predictive equality")
    plt.xlabel("Round (n)")
    plt.ylim(-0.0, 0.1)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_pe_disparity.pdf')
        colab_save('iucb_pe_disparity.pdf')
    plt.clf()

    avg_eosum_disparity = np.mean(all_eo_disparity + all_pe_disparity, axis=0)
    std_eosum_disparity = np.std(all_eo_disparity + all_pe_disparity, axis=0)
    #lower_quantile_eosum_disparity = np.quantile(all_eosum_disparity, 0.05, axis=0)
    #upper_quantile_eosum_disparity = np.quantile(all_eosum_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.errorbar([N-1], avg_eosum_disparity[-1], yerr=2*std_eosum_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity, upper_quantile_eosum_disparity, alpha=0.3, color = COLOR_GREEDY)
    
    avg_eosum_disparity_ucb = np.mean(all_eo_disparity_ucb + all_pe_disparity_ucb, axis=0)
    std_eosum_disparity_ucb = np.std(all_eo_disparity_ucb + all_pe_disparity_ucb, axis=0)
    #lower_quantile_eosum_disparity_ucb = np.quantile(all_eosum_disparity_ucb, 0.05, axis=0)
    #upper_quantile_eosum_disparity_ucb = np.quantile(all_eosum_disparity_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_eosum_disparity_ucb[-1], yerr=2*std_eosum_disparity_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity_ucb, upper_quantile_eosum_disparity_ucb, alpha=0.3, color = COLOR_UCB)

    avg_eosum_disparity_iucb = np.mean(all_eo_disparity_iucb + all_pe_disparity_iucb, axis=0)
    std_eosum_disparity_iucb = np.std(all_eo_disparity_iucb + all_pe_disparity_iucb, axis=0)
    #lower_quantile_eosum_disparity_iucb = np.quantile(all_eosum_disparity_iucb, 0.05, axis=0)
    #upper_quantile_eosum_disparity_iucb = np.quantile(all_eosum_disparity_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_eosum_disparity_iucb[-1], yerr=2*std_eosum_disparity_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity_iucb, upper_quantile_eosum_disparity_iucb, alpha=0.3, color = COLOR_HYBRID)
    plt.ylabel("Disparity on equalized odds")
    plt.xlabel("Round (n)")
    plt.ylim(-0.0, 0.5)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_eosum_disparity.pdf')
        colab_save('iucb_eosum_disparity.pdf')
    plt.clf()
    
np.random.seed(20)
experiment2()



# In[5]:


# 2stage Rooney versus LF
def experiment3():
    sigma_eta = np.round(6.0, 1)
    pu_count, pu_count_switch, pu_count_greedy = 0, 0, 0 
    all_regret = np.zeros((R, N))
    all_strong_regret = np.zeros((R, N))
    all_regret_switch = np.zeros((R, N))
    all_strong_regret_switch = np.zeros((R, N))
    all_regret_greedy = np.zeros((R, N))
    all_strong_regret_greedy = np.zeros((R, N))
    all_draw2 = np.zeros((R, N))
    all_draw2_switch = np.zeros((R, N))
    all_draw2_greedy = np.zeros((R, N))
    k_list = (10, 2)
    Kg = k_list
    rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
    sims = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, N=N, sigma_eta = sigma_eta, policy = "rooney") for r in range(R)] 
    sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_switch = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, N=N, sigma_eta = sigma_eta, policy = "rooney", n_switch = N_SWITCH) for r in range(R)] 
    sims_switch = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_switch[r], rss[r+R]) for r in range(R)] ) #parallel computation
    sims_greedy = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, N=N, sigma_eta = sigma_eta, policy = "rooney_greedy") for r in range(R)]
    sims_greedy = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_greedy[r], rss[r+2*R]) for r in range(R)] ) #parallel computation
    output_to_pickle("experiment3.pickle", (sims, sims_switch, sims_greedy))
    for r in range(R):
        if sims[r].is_perpetunderest():
            pu_count += 1
        if sims_switch[r].is_perpetunderest():
            pu_count_switch += 1
        if sims_greedy[r].is_perpetunderest():
            pu_count_greedy += 1
        all_regret[r,:] = sims[r].regret_seq
        all_strong_regret[r,:] = sims[r].strong_regret_seq
        all_draw2[r,:] = sims[r].draws_seq[:,1]
        all_regret_switch[r,:] = sims_switch[r].regret_seq
        all_strong_regret_switch[r,:] = sims_switch[r].strong_regret_seq
        all_draw2_switch[r,:] = sims_switch[r].draws_seq[:,1]
        all_regret_greedy[r,:] = sims_greedy[r].regret_seq
        all_strong_regret_greedy[r,:] = sims_greedy[r].strong_regret_seq
        all_draw2_greedy[r,:] = sims_greedy[r].draws_seq[:,1]

    # plotting start
    fig = plt.figure(figsize=Figsize)
    #plt.plot(range(N), avg_draw2, label = "LF")
    #plt.bar(range(len(sigma_etas)), avg_regret, tick_label=labels, align="center")
    avg_regret = np.mean(all_regret, axis=0)
    std_regret = np.std(all_regret, axis=0)
    avg_regret_switch = np.mean(all_regret_switch, axis=0)
    std_regret_switch = np.std(all_regret_switch, axis=0)
    avg_regret_greedy = np.mean(all_regret_greedy, axis=0)
    std_regret_greedy = np.std(all_regret_greedy, axis=0)
    avg_strong_regret = np.mean(all_strong_regret, axis=0)
    std_strong_regret = np.std(all_strong_regret, axis=0)
    avg_strong_regret_switch = np.mean(all_strong_regret_switch, axis=0)
    std_strong_regret_switch = np.std(all_strong_regret_switch, axis=0)
    avg_strong_regret_greedy = np.mean(all_strong_regret_greedy, axis=0)
    std_strong_regret_greedy = np.std(all_strong_regret_greedy, axis=0)
    #plt.plot(range(N), avg_regret, label = "sigma_eta = "+str(sigma_eta))

    lower_quantile_regret = np.quantile(all_regret, 0.05, axis=0)
    upper_quantile_regret = np.quantile(all_regret, 0.95, axis=0)
    plt.plot(range(N), avg_regret, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY) #
    plt.errorbar([N-1], avg_regret[-1], yerr=2*std_regret[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret, upper_quantile_regret, alpha=0.3, color = COLOR_ROONEY) 
    plt.ylabel("Regret")
#    plt.xlabel("sigma_eta")
    plt.xlabel("Round (n)")
    plt.legend()
    
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_regret.pdf')
        colab_save('rooney_largeg2_regret.pdf')
    plt.clf()
    
    fig = plt.figure(figsize=Figsize)

    lower_quantile_strong_regret = np.quantile(all_strong_regret, 0.05, axis=0)
    upper_quantile_strong_regret = np.quantile(all_strong_regret, 0.95, axis=0)
    plt.plot(range(N), avg_strong_regret, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_strong_regret[-1], yerr=2*std_strong_regret[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    plt.fill_between(range(N), lower_quantile_strong_regret, upper_quantile_strong_regret, alpha=0.3, color = COLOR_ROONEY)

    lower_quantile_strong_regret_switch = np.quantile(all_strong_regret_switch, 0.05, axis=0)
    upper_quantile_strong_regret_switch = np.quantile(all_strong_regret_switch, 0.95, axis=0)
    plt.plot(range(N), avg_strong_regret_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_strong_regret_switch[-1], yerr=2*std_strong_regret_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    plt.fill_between(range(N), lower_quantile_strong_regret_switch, upper_quantile_strong_regret_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)

    lower_quantile_strong_regret_greedy = np.quantile(all_strong_regret_greedy, 0.05, axis=0)
    upper_quantile_strong_regret_greedy = np.quantile(all_strong_regret_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_strong_regret_greedy, label = "LF", color= COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_strong_regret_greedy[-1], yerr=2*std_strong_regret_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    plt.fill_between(range(N), lower_quantile_strong_regret_greedy, upper_quantile_strong_regret_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)

    plt.ylim((0,avg_strong_regret_greedy[-1]*1.5))
    plt.ylabel("U2S-Reg")
#    plt.xlabel("sigma_eta")
    plt.xlabel("Round (n)")
    plt.legend(loc='lower right')
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_strong_regret.pdf')
        colab_save('rooney_largeg2_strong_regret.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    #plt.plot(range(N), avg_draw2, label = "LF")
    avg_draw2 = np.mean(all_draw2, axis=0)
    std_draw2 = np.std(all_draw2, axis=0)
    lower_quantile_draw2 = np.quantile(all_draw2, 0.05, axis=0)
    upper_quantile_draw2 = np.quantile(all_draw2, 0.95, axis=0)
    avg_draw2_switch = np.mean(all_draw2_switch, axis=0)
    std_draw2_switch = np.std(all_draw2_switch, axis=0)
    lower_quantile_draw2_switch = np.quantile(all_draw2_switch, 0.05, axis=0)
    upper_quantile_draw2_switch = np.quantile(all_draw2_switch, 0.95, axis=0)
    avg_draw2_greedy = np.mean(all_draw2_greedy, axis=0)
    std_draw2_greedy = np.std(all_draw2_greedy, axis=0)
    lower_quantile_draw2_greedy = np.quantile(all_draw2_greedy, 0.05, axis=0)
    upper_quantile_draw2_greedy = np.quantile(all_draw2_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_draw2, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_draw2[-1], yerr=2*std_draw2[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2, upper_quantile_draw2, alpha=0.3, color = COLOR_ROONEY)
    plt.plot(range(N), avg_draw2_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_draw2_switch[-1], yerr=2*std_draw2_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_switch, upper_quantile_draw2_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)
    plt.plot(range(N), avg_draw2_greedy, label = "LF", color= COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_draw2_greedy[-1], yerr=2*std_draw2_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_greedy, upper_quantile_draw2_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)
    plt.plot(range(N), [i*k_list[1]/np.sum(k_list) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])
    plt.ylabel("# of minorities hired")
    plt.xlabel("Round n")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_draw2.pdf')
        colab_save('rooney_largeg2_draw2.pdf')
    plt.clf()

    # plotting fairness measure 
    # self.draws_seq[self.n] = self.draws
    # Kg = (10, 2)
    all_disparity = np.zeros( (R, N) ) #Rooney
    all_disparity_switch = np.zeros( (R, N) ) #Rooney-LF
    all_disparity_greedy = np.zeros( (R, N) ) #LF
    for r in range(R):
        all_disparity[r] = np.abs( (sims[r].draws_seq[:,0]/Kg[0] - sims[r].draws_seq[:,1]/Kg[1])/(sims[r].draws_seq[:,0] + sims[r].draws_seq[:,1]) )
        all_disparity_switch[r] = np.abs( (sims_switch[r].draws_seq[:,0]/Kg[0] - sims_switch[r].draws_seq[:,1]/Kg[1])/(sims_switch[r].draws_seq[:,0] + sims_switch[r].draws_seq[:,1]) )
        all_disparity_greedy[r] = np.abs( (sims_greedy[r].draws_seq[:,0]/Kg[0] - sims_greedy[r].draws_seq[:,1]/Kg[1])/(sims_greedy[r].draws_seq[:,0] + sims_greedy[r].draws_seq[:,1]) )
    fig = plt.figure(figsize=Figsize)
    avg_disparity = np.mean(all_disparity, axis=0)
    std_disparity = np.std(all_disparity, axis=0)
    lower_quantile_disparity = np.quantile(all_disparity, 0.05, axis=0)
    upper_quantile_disparity = np.quantile(all_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_disparity, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_disparity[-1], yerr=2*std_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity, upper_quantile_disparity, alpha=0.3, color = COLOR_ROONEY)
    avg_disparity_switch = np.mean(all_disparity_switch, axis=0)
    std_disparity_switch = np.std(all_disparity_switch, axis=0)
    lower_quantile_disparity_switch = np.quantile(all_disparity_switch, 0.05, axis=0)
    upper_quantile_disparity_switch = np.quantile(all_disparity_switch, 0.95, axis=0)
    plt.plot(range(N), avg_disparity_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_disparity_switch[-1], yerr=2*std_disparity_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity_switch, upper_quantile_disparity_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)
    avg_disparity_greedy = np.mean(all_disparity_greedy, axis=0)
    std_disparity_greedy = np.std(all_disparity_greedy, axis=0)
    lower_quantile_disparity_greedy = np.quantile(all_disparity_greedy, 0.05, axis=0)
    upper_quantile_disparity_greedy = np.quantile(all_disparity_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_disparity_greedy, label = "LF", color = COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_disparity_greedy[-1], yerr=2*std_disparity_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_disparity_greedy, upper_quantile_disparity_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)
    plt.ylabel("Demographic disparity")
    plt.xlabel("Round (n)")
    plt.ylim(0.0, 0.2)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_disparity.pdf')
        colab_save('rooney_largeg2_disparity.pdf')
    plt.clf()

    # equal opportunity
    all_eo_disparity = np.zeros( (R, N) ) #Rooney
    all_eo_disparity_switch = np.zeros( (R, N) ) #Rooney-LF
    all_eo_disparity_greedy = np.zeros( (R, N) ) #LF
    EPS = 0.000001
    for r in range(R):
        all_eo_disparity[r] = np.abs(sims[r].best_draws_seq[:,0]/(sims[r].best_seq[:,0] + EPS) -  sims[r].best_draws_seq[:,1]/(sims[r].best_seq[:,1] + EPS) )
        all_eo_disparity_switch[r] = np.abs(sims_switch[r].best_draws_seq[:,0]/(sims_switch[r].best_seq[:,0] + EPS) - sims_switch[r].best_draws_seq[:,1]/(sims_switch[r].best_seq[:,1] + EPS) )
        all_eo_disparity_greedy[r] = np.abs(sims_greedy[r].best_draws_seq[:,0]/(sims_greedy[r].best_seq[:,0] + EPS) -  sims_greedy[r].best_draws_seq[:,1]/(sims_greedy[r].best_seq[:,1] + EPS) )
    fig = plt.figure(figsize=Figsize)
    avg_eo_disparity = np.mean(all_eo_disparity, axis=0)
    std_eo_disparity = np.std(all_eo_disparity, axis=0)
    lower_quantile_eo_disparity = np.quantile(all_eo_disparity, 0.05, axis=0)
    upper_quantile_eo_disparity = np.quantile(all_eo_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_eo_disparity[-1], yerr=2*std_eo_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity, upper_quantile_eo_disparity, alpha=0.3, color = COLOR_ROONEY)

    avg_eo_disparity_switch = np.mean(all_eo_disparity_switch, axis=0)
    std_eo_disparity_switch = np.std(all_eo_disparity_switch, axis=0)
    lower_quantile_eo_disparity_switch = np.quantile(all_eo_disparity_switch, 0.05, axis=0)
    upper_quantile_eo_disparity_switch = np.quantile(all_eo_disparity_switch, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_eo_disparity_switch[-1], yerr=2*std_eo_disparity_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity_switch, upper_quantile_eo_disparity_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)

    avg_eo_disparity_greedy = np.mean(all_eo_disparity_greedy, axis=0)
    std_eo_disparity_greedy = np.std(all_eo_disparity_greedy, axis=0)
    lower_quantile_eo_disparity_greedy = np.quantile(all_eo_disparity_greedy, 0.05, axis=0)
    upper_quantile_eo_disparity_greedy = np.quantile(all_eo_disparity_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_eo_disparity_greedy, label = "LF", color = COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_eo_disparity_greedy[-1], yerr=2*std_eo_disparity_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eo_disparity_greedy, upper_quantile_eo_disparity_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)
    
    #plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])
    plt.ylabel("EO")
    plt.xlabel("Round (n)")
    plt.ylim(-0.0, 0.5)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_eo_disparity.pdf')
        colab_save('rooney_largeg2_eo_disparity.pdf')
    plt.clf()

    # predictive equality
    all_pe_disparity = np.zeros( (R, N) ) #Rooney
    all_pe_disparity_switch = np.zeros( (R, N) ) #Rooney-LF
    all_pe_disparity_greedy = np.zeros( (R, N) ) #LF
    EPS = 0.000001
    for r in range(R):
        nonbest_draws_seq = sims[r].draws_seq - sims[r].best_draws_seq
        nonbest_seq = sims[r].nonbest_seq
        all_pe_disparity[r] = np.abs(nonbest_draws_seq[:,0]/(nonbest_seq[:,0] + EPS) 
                                     -  nonbest_draws_seq[:,1]/(nonbest_seq[:,1] + EPS) )
        
        nonbest_draws_seq_switch = sims_switch[r].draws_seq - sims_switch[r].best_draws_seq
        nonbest_seq_switch = sims_switch[r].nonbest_seq
        all_pe_disparity_switch[r] = np.abs(nonbest_draws_seq_switch[:,0]/(nonbest_seq_switch[:,0] + EPS) 
                                     -  nonbest_draws_seq_switch[:,1]/(nonbest_seq_switch[:,1] + EPS) )

        nonbest_draws_seq_greedy = sims_greedy[r].draws_seq - sims_greedy[r].best_draws_seq
        nonbest_seq_greedy = sims_greedy[r].nonbest_seq
        all_pe_disparity_greedy[r] = np.abs(nonbest_draws_seq_greedy[:,0]/(nonbest_seq_greedy[:,0] + EPS) 
                                     -  nonbest_draws_seq_greedy[:,1]/(nonbest_seq_greedy[:,1] + EPS) )
    fig = plt.figure(figsize=Figsize)
    avg_pe_disparity = np.mean(all_pe_disparity, axis=0)
    std_pe_disparity = np.std(all_pe_disparity, axis=0)
    lower_quantile_pe_disparity = np.quantile(all_pe_disparity, 0.05, axis=0)
    upper_quantile_pe_disparity = np.quantile(all_pe_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_pe_disparity[-1], yerr=2*std_pe_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity, upper_quantile_pe_disparity, alpha=0.3, color = COLOR_ROONEY)

    avg_pe_disparity_switch = np.mean(all_pe_disparity_switch, axis=0)
    std_pe_disparity_switch = np.std(all_pe_disparity_switch, axis=0)
    lower_quantile_pe_disparity_switch = np.quantile(all_pe_disparity_switch, 0.05, axis=0)
    upper_quantile_pe_disparity_switch = np.quantile(all_pe_disparity_switch, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_pe_disparity_switch[-1], yerr=2*std_pe_disparity_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity_switch, upper_quantile_pe_disparity_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)

    avg_pe_disparity_greedy = np.mean(all_pe_disparity_greedy, axis=0)
    std_pe_disparity_greedy = np.std(all_pe_disparity_greedy, axis=0)
    lower_quantile_pe_disparity_greedy = np.quantile(all_pe_disparity_greedy, 0.05, axis=0)
    upper_quantile_pe_disparity_greedy = np.quantile(all_pe_disparity_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_pe_disparity_greedy, label = "LF", color = COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_pe_disparity_greedy[-1], yerr=2*std_pe_disparity_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_pe_disparity_greedy, upper_quantile_pe_disparity_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)

    plt.ylabel("Disparity on predictive equality")
    plt.xlabel("Round (n)")
    plt.ylim(-0.0, 0.3)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_pe_disparity.pdf')
        colab_save('rooney_largeg2_pe_disparity.pdf')
    plt.clf()

    # Equal opportunity + Predictive equality = Equalized Odds
    fig = plt.figure(figsize=Figsize)
    avg_eosum_disparity = np.mean(all_eo_disparity + all_pe_disparity, axis=0)
    std_eosum_disparity = np.std(all_eo_disparity + all_pe_disparity, axis=0)
    #lower_quantile_eosum_disparity = np.quantile(all_eosum_disparity, 0.05, axis=0)
    #upper_quantile_eosum_disparity = np.quantile(all_eosum_disparity, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity, label = "Rooney", color = COLOR_ROONEY, linestyle = LINESTYLE_ROONEY)
    plt.errorbar([N-1], avg_eosum_disparity[-1], yerr=2*std_eosum_disparity[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity, upper_quantile_eosum_disparity, alpha=0.3, color = COLOR_ROONEY)

    avg_eosum_disparity_switch = np.mean(all_eo_disparity_switch + all_pe_disparity_switch, axis=0)
    std_eosum_disparity_switch = np.std(all_eo_disparity_switch + all_pe_disparity_switch, axis=0)
    #lower_quantile_eosum_disparity_switch = np.quantile(all_eosum_disparity_switch, 0.05, axis=0)
    #upper_quantile_eosum_disparity_switch = np.quantile(all_eosum_disparity_switch, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity_switch, label = "Rooney-LF", color = COLOR_ROONEY_SWITCH, linestyle = LINESTYLE_ROONEY_SWITCH)
    plt.errorbar([N-1], avg_eosum_disparity_switch[-1], yerr=2*std_eosum_disparity_switch[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_SWITCH) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity_switch, upper_quantile_eosum_disparity_switch, alpha=0.3, color = COLOR_ROONEY_SWITCH)

    avg_eosum_disparity_greedy = np.mean(all_eo_disparity_greedy + all_pe_disparity_greedy, axis=0)
    std_eosum_disparity_greedy = np.std(all_eo_disparity_greedy + all_pe_disparity_greedy, axis=0)
    #lower_quantile_eosum_disparity_greedy = np.quantile(all_eosum_disparity_greedy, 0.05, axis=0)
    #upper_quantile_eosum_disparity_greedy = np.quantile(all_eosum_disparity_greedy, 0.95, axis=0)
    plt.plot(range(N), avg_eosum_disparity_greedy, label = "LF", color = COLOR_ROONEY_GREEDY, linestyle = LINESTYLE_ROONEY_GREEDY)
    plt.errorbar([N-1], avg_eosum_disparity_greedy[-1], yerr=2*std_eosum_disparity_greedy[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_ROONEY_GREEDY) #2 sigma
    #plt.fill_between(range(N), lower_quantile_eosum_disparity_greedy, upper_quantile_eosum_disparity_greedy, alpha=0.3, color = COLOR_ROONEY_GREEDY)
    
    #plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])
    plt.ylabel("Disparity on equalized odds")
    plt.xlabel("Round (n)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_largeg2_eosum_disparity.pdf')
        colab_save('rooney_largeg2_eosum_disparity.pdf')
    plt.clf()
    
np.random.seed(7)
experiment3()

# UCB versus Hybrid, Several different parameters of Hybrid
def experiment4():
    Kg = (10, 2)
    #sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "ucb") for r in range(R)]
    sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=0.0000001) for r in range(R)]
    sims_iucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=0.5) for r in range(R)]
    sims_iucb2 = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=0.1) for r in range(R)]
    sims_iucb3 = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=0.25) for r in range(R)]
    sims_iucb4 = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=1.0) for r in range(R)]
#    sims_iucb5 = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", iucb_threshold=2.0) for r in range(R)]
    # iucb_threshold
    rss = np.random.randint(np.iinfo(np.int32).max, size=R*50)
    sims_ucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_ucb[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_iucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb[r], rss[r+R]) for r in range(R)] ) #parallel computation
    sims_iucb2 = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb2[r], rss[r+2*R]) for r in range(R)] ) #parallel computation
    sims_iucb3 = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb3[r], rss[r+3*R]) for r in range(R)] ) #parallel computation
    sims_iucb4 = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb4[r], rss[r+4*R]) for r in range(R)] ) #parallel computation
#    sims_iucb5 = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb5[r], rss[r+5*R]) for r in range(R)] ) #parallel computation
    output_to_pickle("experiment4.pickle", (sims_ucb, sims_iucb, sims_iucb2, sims_iucb3, sims_iucb4))

    all_regret_ucb = np.zeros( (R,N) )
    all_draw2_ucb = np.zeros( (R,N) )
    all_subsidy_ucb = np.zeros( (R,N) )
    all_subsidycs_ucb = np.zeros( (R,N) )
    all_regret_iucb = np.zeros( (R,N) )
    all_draw2_iucb = np.zeros( (R,N) )
    all_subsidy_iucb = np.zeros( (R,N) )
    all_subsidycs_iucb = np.zeros( (R,N) )
    all_regret_iucb2 = np.zeros( (R,N) )
    all_draw2_iucb2 = np.zeros( (R,N) )
    all_subsidy_iucb2 = np.zeros( (R,N) )
    all_subsidycs_iucb2 = np.zeros( (R,N) )
    all_regret_iucb3 = np.zeros( (R,N) )
    all_draw2_iucb3 = np.zeros( (R,N) )
    all_subsidy_iucb3 = np.zeros( (R,N) )
    all_subsidycs_iucb3 = np.zeros( (R,N) )
    all_regret_iucb4 = np.zeros( (R,N) )
    all_draw2_iucb4 = np.zeros( (R,N) )
    all_subsidy_iucb4 = np.zeros( (R,N) )
    all_subsidycs_iucb4 = np.zeros( (R,N) )
    for r in range(R):
        all_regret_ucb[r] += sims_ucb[r].regret_seq
        all_draw2_ucb[r] += sims_ucb[r].draws_seq[:,1]
        all_subsidy_ucb[r] += sims_ucb[r].subsidy_seq
        all_subsidycs_ucb[r] += sims_ucb[r].subsidycs_seq
        all_regret_iucb[r] += sims_iucb[r].regret_seq
        all_draw2_iucb[r] += sims_iucb[r].draws_seq[:,1]
        all_subsidy_iucb[r] += sims_iucb[r].subsidy_seq
        all_subsidycs_iucb[r] += sims_iucb[r].subsidycs_seq
        all_regret_iucb2[r] += sims_iucb2[r].regret_seq
        all_draw2_iucb2[r] += sims_iucb2[r].draws_seq[:,1]
        all_subsidy_iucb2[r] += sims_iucb2[r].subsidy_seq
        all_subsidycs_iucb2[r] += sims_iucb2[r].subsidycs_seq
        all_regret_iucb3[r] += sims_iucb3[r].regret_seq
        all_draw2_iucb3[r] += sims_iucb3[r].draws_seq[:,1]
        all_subsidy_iucb3[r] += sims_iucb3[r].subsidy_seq
        all_subsidycs_iucb3[r] += sims_iucb3[r].subsidycs_seq
        all_regret_iucb4[r] += sims_iucb4[r].regret_seq
        all_draw2_iucb4[r] += sims_iucb4[r].draws_seq[:,1]
        all_subsidy_iucb4[r] += sims_iucb4[r].subsidy_seq
        all_subsidycs_iucb4[r] += sims_iucb4[r].subsidycs_seq
    
    #plotting starts here
    
    # color gradation
    colors = ["tab:blue", "navy", "tab:orange", "black"]
    cmap = LinearSegmentedColormap.from_list("blueGradient", colors)

    fig = plt.figure(figsize=Figsize)
    avg_regret_ucb = np.mean(all_regret_ucb, axis=0)
    std_regret_ucb = np.std(all_regret_ucb, axis=0)
    lower_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.05, axis=0)
    upper_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_ucb, label = "a = 0 (UCB)", color = cmap(0), linestyle = linestyle_tuple["dashed"])
#    plt.errorbar([N-1], avg_regret_ucb[-1], yerr=2*std_regret_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    #plt.fill_between(range(N), lower_quantile_regret_ucb, upper_quantile_regret_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_regret_iucb2 = np.mean(all_regret_iucb2, axis=0)
    std_regret_iucb2 = np.std(all_regret_iucb2, axis=0)
    lower_quantile_regret_iucb2 = np.quantile(all_regret_iucb2, 0.05, axis=0)
    upper_quantile_regret_iucb2 = np.quantile(all_regret_iucb2, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb2, label = "a = 0.1", color = cmap(0.2), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_regret_iucb2[-1], yerr=2*std_regret_iucb2[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.25)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_regret_iucb2, upper_quantile_regret_iucb2, alpha=Alpha, color = COLOR_HYBRID)

    avg_regret_iucb3 = np.mean(all_regret_iucb3, axis=0)
    std_regret_iucb3 = np.std(all_regret_iucb3, axis=0)
    lower_quantile_regret_iucb3 = np.quantile(all_regret_iucb3, 0.05, axis=0)
    upper_quantile_regret_iucb3 = np.quantile(all_regret_iucb3, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb3, label = "a = 0.25", color = cmap(0.5), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_regret_iucb3[-1], yerr=2*std_regret_iucb3[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.5)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_regret_iucb3, upper_quantile_regret_iucb3, alpha=Alpha, color = COLOR_HYBRID)

    avg_regret_iucb = np.mean(all_regret_iucb, axis=0)
    std_regret_iucb = np.std(all_regret_iucb, axis=0)
    lower_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.05, axis=0)
    upper_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb, label = "a = 0.5", color = cmap(0.67), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_regret_iucb[-1], yerr=2*std_regret_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.75)) #2 sigma
    #plt.fill_between(range(N), lower_quantile_regret_iucb, upper_quantile_regret_iucb, alpha=Alpha, color = COLOR_HYBRID)

    avg_regret_iucb4 = np.mean(all_regret_iucb4, axis=0)
    std_regret_iucb4 = np.std(all_regret_iucb4, axis=0)
    lower_quantile_regret_iucb4 = np.quantile(all_regret_iucb4, 0.05, axis=0)
    upper_quantile_regret_iucb4 = np.quantile(all_regret_iucb4, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb4, label = "a = 1.0", color = cmap(1.0), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_regret_iucb4[-1], yerr=2*std_regret_iucb4[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(1.0)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_regret_iucb4, upper_quantile_regret_iucb4, alpha=Alpha, color = COLOR_HYBRID)
    
    plt.ylabel("Regret")
    plt.xlabel("Round (n)")
    plt.legend(loc = 'upper left')
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_regret_ex.pdf')
        colab_save('iucb_regret_ex.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_subsidy_ucb = np.mean(all_subsidy_ucb, axis=0)
    std_subsidy_ucb = np.std(all_subsidy_ucb, axis=0)
    lower_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.05, axis=0)
    upper_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_ucb, label = "a = 0 (UCB))", color = cmap(0), linestyle = linestyle_tuple["dashed"])
#    plt.errorbar([N-1], avg_subsidy_ucb[-1], yerr=2*std_subsidy_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0)) #2 sigma
    #plt.fill_between(range(N), lower_quantile_subsidy_ucb, upper_quantile_subsidy_ucb, alpha=Alpha, color = COLOR_UCB)


    avg_subsidy_iucb2 = np.mean(all_subsidy_iucb2, axis=0)
    std_subsidy_iucb2 = np.std(all_subsidy_iucb2, axis=0)
    lower_quantile_subsidy_iucb2 = np.quantile(all_subsidy_iucb2, 0.05, axis=0)
    upper_quantile_subsidy_iucb2 = np.quantile(all_subsidy_iucb2, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb2, label = "a = 0.1", color = cmap(0.2), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_subsidy_iucb2[-1], yerr=2*std_subsidy_iucb2[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.25)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_subsidy_iucb2, upper_quantile_subsidy_iucb2, alpha=Alpha, color = COLOR_HYBRID)

    avg_subsidy_iucb3 = np.mean(all_subsidy_iucb3, axis=0)
    std_subsidy_iucb3 = np.std(all_subsidy_iucb3, axis=0)
    lower_quantile_subsidy_iucb3 = np.quantile(all_subsidy_iucb3, 0.05, axis=0)
    upper_quantile_subsidy_iucb3 = np.quantile(all_subsidy_iucb3, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb3, label = "a = 0.25", color = cmap(0.5), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_subsidy_iucb3[-1], yerr=2*std_subsidy_iucb3[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.5)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_subsidy_iucb3, upper_quantile_subsidy_iucb3, alpha=Alpha, color = COLOR_HYBRID)

    avg_subsidy_iucb = np.mean(all_subsidy_iucb, axis=0)
    std_subsidy_iucb = np.std(all_subsidy_iucb, axis=0)
    lower_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.05, axis=0)
    upper_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb, label = "a = 0.5", color = cmap(0.67), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_subsidy_iucb[-1], yerr=2*std_subsidy_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(0.75)) #2 sigma
    #plt.fill_between(range(N), lower_quantile_subsidy_iucb, upper_quantile_subsidy_iucb, alpha=Alpha, color = COLOR_HYBRID)

    avg_subsidy_iucb4 = np.mean(all_subsidy_iucb4, axis=0)
    std_subsidy_iucb4 = np.std(all_subsidy_iucb4, axis=0)
    lower_quantile_subsidy_iucb4 = np.quantile(all_subsidy_iucb4, 0.05, axis=0)
    upper_quantile_subsidy_iucb4 = np.quantile(all_subsidy_iucb4, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb4, label = "a = 1", color = cmap(1.0), linestyle = LINESTYLE_UCB)
#    plt.errorbar([N-1], avg_subsidy_iucb4[-1], yerr=2*std_subsidy_iucb4[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = cmap(1.0)) #2 sigma
#    plt.fill_between(range(N), lower_quantile_subsidy_iucb4, upper_quantile_subsidy_iucb4, alpha=Alpha, color = COLOR_HYBRID)

    plt.ylabel("Subsidy")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_subsidy_ex.pdf')
        colab_save('iucb_subsidy_ex.pdf')
    plt.clf()


np.random.seed(2000)
experiment4()


# In[ ]:


# LF, UCB versus Hybrid, varying \mu_x (addtional)
def experiment5():
    mu_x_diff_cand = -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75 # -1.5, -1.0, -0.5 , 0.0, 0.5, 1.0, 1.5 
    C = len(mu_x_diff_cand)
    if True:
        all_regret_greedy = np.zeros( (R,C) )
        all_regret_ucb = np.zeros( (R,C) )
        all_regret_iucb = np.zeros( (R,C) )
        all_count_group2best_greedy = np.zeros( (R,C) )
        pu_count_greedy = np.zeros(C)
        pu_count_ucb = np.zeros(C)
        pu_count_iucb = np.zeros(C)
        for c_cur, mu_x_diff in enumerate(mu_x_diff_cand):
            sys.stdout.flush()
            Kg = (10, 2)
            sims_greedy = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "greedy", mu_x_diff = mu_x_diff) for r in range(R)]
            sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "ucb", mu_x_diff = mu_x_diff) for r in range(R)]
            sims_iucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", mu_x_diff = mu_x_diff) for r in range(R)]
            rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
            sims_greedy = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_greedy[r], rss[r]) for r in range(R)] ) #parallel computation
            sims_ucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_ucb[r], rss[r+R]) for r in range(R)] ) #parallel computation
            sims_iucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb[r], rss[r+2*R]) for r in range(R)] ) #parallel computation
            for r in range(R):
                all_regret_greedy[r][c_cur] += sims_greedy[r].regret_seq[-1]
                all_regret_ucb[r][c_cur] += sims_ucb[r].regret_seq[-1]
                all_regret_iucb[r][c_cur] += sims_iucb[r].regret_seq[-1]
                all_count_group2best_greedy[r][c_cur] = np.count_nonzero(sims_greedy[r].best_group_seq == 1) #1はgroup 2
            for r in range(R):
                if sims_greedy[r].is_perpetunderest():
                    pu_count_greedy[c_cur] += 1
                if sims_ucb[r].is_perpetunderest():
                    pu_count_ucb[c_cur] += 1
                if sims_iucb[r].is_perpetunderest():
                    pu_count_iucb[c_cur] += 1
    output_to_pickle("experiment5.pickle", (sims_greedy, sims_ucb, sims_iucb))

    #plotting part starts here
    fig = plt.figure(figsize=Figsize)
    avg_regret_greedy = np.mean(all_regret_greedy, axis=0)
    std_regret_greedy = np.std(all_regret_greedy, axis=0)
    avg_regret_ucb = np.mean(all_regret_ucb, axis=0)
    std_regret_ucb = np.std(all_regret_ucb, axis=0)
    avg_regret_iucb = np.mean(all_regret_iucb, axis=0)
    std_regret_iucb = np.std(all_regret_iucb, axis=0)
    plt.plot(mu_x_diff_cand, avg_regret_greedy, label = "LF", color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    plt.plot(mu_x_diff_cand, avg_regret_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.plot(mu_x_diff_cand, avg_regret_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    for c, mu_x_diff in enumerate(mu_x_diff_cand):
        plt.errorbar([mu_x_diff], avg_regret_greedy[c], yerr=2*std_regret_greedy[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
        plt.errorbar([mu_x_diff], avg_regret_ucb[c], yerr=2*std_regret_ucb[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
        plt.errorbar([mu_x_diff], avg_regret_iucb[c], yerr=2*std_regret_iucb[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.ylabel("Regret")
    plt.xlabel("Gap: $\overline{\mu}_{x,2}-\overline{\mu}_{x,1}$")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'mux_regret.pdf')
        colab_save('mux_regret.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    #count = np.count_nonzero(arr == 2)
    avg_grp2_greedy = np.mean(all_count_group2best_greedy, axis=0) / N
    std_grp2_greedy = np.std(all_count_group2best_greedy, axis=0) / N
    plt.plot(mu_x_diff_cand, avg_grp2_greedy, color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
    for c, mu_x_diff in enumerate(mu_x_diff_cand):
        plt.errorbar([mu_x_diff], avg_grp2_greedy[c], yerr=2*std_grp2_greedy[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_GREEDY) #2 sigma
#        plt.errorbar([mu_x_diff], avg_regret_ucb[c], yerr=2*std_regret_ucb[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
#        plt.errorbar([mu_x_diff], avg_regret_iucb[c], yerr=2*std_regret_iucb[c]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.ylabel("$\mathrm{\mathbb{P}}[g(i^*(n)) = 2]$")
    plt.xlabel("Gap: $\overline{\mu}_{x,2}-\overline{\mu}_{x,1}$")
    #plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'mux_group2best.pdf')
        colab_save('mux_group2best.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    w = 1 
    # iucb
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count_iucb)]
    pu_count_intervals = ([pu_count_iucb[i]-conf[0] for i,conf in enumerate(confs)],
                          [conf[1]-pu_count_iucb[i] for i,conf in enumerate(confs)])
    pu_count_intervals_tail = (pu_count_intervals[0], pu_count_intervals[1])
    plt.bar(np.array(range(len(mu_x_diff_cand)))+7*w/6, pu_count_iucb, width=w/6, tick_label=mu_x_diff_cand, yerr = pu_count_intervals_tail, align="center",\
            capsize = Capsize, label = "HYBRID", edgecolor='black', color=COLOR_ROONEY, hatch=HATCH_GREEN) 
    # greedy
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count_greedy)]
    pu_count_intervals = ([pu_count_greedy[i]-conf[0] for i,conf in enumerate(confs)],
                          [conf[1]-pu_count_greedy[i] for i,conf in enumerate(confs)])
    pu_count_intervals_tail = (pu_count_intervals[0], pu_count_intervals[1])
    plt.bar(np.array(range(len(mu_x_diff_cand)))+5*w/6, pu_count_greedy, width=w/6, tick_label=mu_x_diff_cand, yerr = pu_count_intervals_tail, align="center",\
            capsize = Capsize, label = "LF", edgecolor='black', color=COLOR_GREEDY, hatch=HATCH_RED) 
    # ucb
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count_ucb)]
    pu_count_intervals = ([pu_count_ucb[i]-conf[0] for i,conf in enumerate(confs)],
                          [conf[1]-pu_count_ucb[i] for i,conf in enumerate(confs)])
    pu_count_intervals_tail = (pu_count_intervals[0], pu_count_intervals[1])
    plt.bar(np.array(range(len(mu_x_diff_cand)))+6*w/6, pu_count_ucb, width=w/6, tick_label=mu_x_diff_cand, yerr = pu_count_intervals_tail, align="center",\
            capsize = Capsize, label = "UCB", edgecolor='black', color=COLOR_UCB, hatch=HATCH_BLUE) 

    
    # fill=False
    #plt.ylim(0, 100)
    plt.ylabel("# of PU")
    plt.xlabel("Gap: $\overline{\mu}_{x,2}-\overline{\mu}_{x,1}$")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'mux_pu.pdf')
        colab_save('mux_pu.pdf')
    plt.clf()

    
np.random.seed(2)
experiment5()
# sys.exit()


# In[ ]:


# 2-stage dependence on eta
def experiment6():
#    sigma_etas = np.round( np.sqrt(d) * np.array([0, 0.6, 1.2, 1.8, 2.4]), 1) 
    sigma_etas = np.round( np.array([0, 1.5, 3.0, 4.5, 6.0]), 1) 
    pu_count = np.zeros(len(sigma_etas))
    pu_count_greedy = np.zeros(len(sigma_etas))
    all_regret = np.zeros((R, N, len(sigma_etas)))
    all_strong_regret = np.zeros((R, N, len(sigma_etas)))
    all_regret_greedy = np.zeros((R, N, len(sigma_etas)))
    all_strong_regret_greedy = np.zeros((R, N, len(sigma_etas)))
    all_draw2 = np.zeros((R, N,len(sigma_etas)))
    all_draw2_greedy = np.zeros((R, N,len(sigma_etas)))
    k_list = (10, 2) 
    for i,sigma_eta in enumerate(sigma_etas):
        sims = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, sigma_eta = sigma_eta, policy = "rooney") for r in range(R)] 
        sims_greedy = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, sigma_eta = sigma_eta, policy = "rooney_greedy") for r in range(R)]
        rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
        sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
        sims_greedy = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_greedy[r], rss[r+R]) for r in range(R)] ) #parallel computation
        for r in range(R):
            if sims[r].is_perpetunderest():
                pu_count[i] += 1
            if sims_greedy[r].is_perpetunderest():
                pu_count_greedy[i] += 1
            all_regret[r,:,i] = sims[r].regret_seq
            all_strong_regret[r,:,i] = sims[r].strong_regret_seq
            all_regret_greedy[r,:,i] = sims_greedy[r].regret_seq
            all_strong_regret_greedy[r,:,i] = sims_greedy[r].strong_regret_seq
            all_draw2[r,:,i] = sims[r].draws_seq[:,1]
            all_draw2_greedy[r,:,i] = sims_greedy[r].draws_seq[:,1]
    output_to_pickle("experiment6.pickle", (sims, sims_greedy))

    # plotting part starts here
    labels = [sigma_eta for sigma_eta in sigma_etas]
    fig = plt.figure(figsize=Figsize)
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count)]
    pu_count_intervals = ([pu_count[i]-conf[0] for i,conf in enumerate(confs)],  [conf[1]-pu_count[i] for i,conf in enumerate(confs)])
    w = 1 #plot bar width
    plt.bar(np.array(range(len(sigma_etas)))-w/6, pu_count, width=w/3, yerr = pu_count_intervals,\
            align="center", capsize = Capsize, label="Rooney", edgecolor='black', color=COLOR_ROONEY, hatch=HATCH_BLUE)
    confs_greedy = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count_greedy)]
    pu_count_intervals_greedy = ([pu_count_greedy[i]-conf[0] for i,conf in enumerate(confs_greedy)],  [conf[1]-pu_count_greedy[i] for i,conf in enumerate(confs_greedy)])
    plt.bar(np.array(range(len(sigma_etas)))+w/6, pu_count_greedy, width=w/3, yerr = pu_count_intervals_greedy,\
            align="center", capsize = Capsize, label="LF", edgecolor='black', color=COLOR_GREEDY, hatch=HATCH_RED)
    plt.xticks(ticks = np.array(range(len(sigma_etas))), labels = labels)
    plt.ylabel("# of PU")
    plt.xlabel("sigma_eta")
    plt.legend(loc='lower right')
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_pu.pdf')
        colab_save('rooney_pu.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    #plt.plot(range(N), avg_draw2, label = "LF")
    #plt.bar(range(len(sigma_etas)), avg_regret, tick_label=labels, align="center")
    avg_regret = np.mean(all_regret, axis=0)
    avg_regret_greedy = np.mean(all_regret_greedy, axis=0)
    avg_strong_regret = np.mean(all_strong_regret, axis=0)
    avg_strong_regret_greedy = np.mean(all_strong_regret_greedy, axis=0)
    for i,sigma_eta in enumerate(sigma_etas):
        if i==1:
            #plt.plot(range(N), avg_regret[:,i], label = "sigma_eta = "+str(sigma_eta))
            plt.plot(range(N), avg_strong_regret[:,i], label = "strong_sigma_eta = "+str(sigma_eta))
            plt.plot(range(N), avg_strong_regret_greedy[:,i], label = "greedy strong_sigma_eta = "+str(sigma_eta))
            #plt.plot(range(N), avg_regret_greedy[:,i], label = "greedy sigma_eta = "+str(sigma_eta))
    plt.ylabel("Regret")
#    plt.xlabel("sigma_eta")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_regret.pdf')
        colab_save('rooney_regret.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    #plt.plot(range(N), avg_draw2, label = "LF")
    avg_draw2 = np.mean(all_draw2, axis=0)
    avg_draw2_greedy = np.mean(all_draw2_greedy, axis=0)
    for i,sigma_eta in enumerate(sigma_etas):
        plt.plot(range(N), avg_draw2[:,i], label = "sigma_eta = "+str(sigma_eta))
        plt.plot(range(N), avg_draw2_greedy[:,i], label = "sigma_eta_greedy = "+str(sigma_eta))
    plt.ylabel("# of minorities hired")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'rooney_draw2.pdf')
        colab_save('rooney_draw2.pdf')
    plt.clf()

np.random.seed(6)
experiment6()

# warm-start LF versus Hybrid
def experiment7():
    Kg = (10, 2)
    #N0_list = [10, 20, 50, 100]
    N0_list = [10, 15, 20, 50]
    pu_count_greedy = np.zeros(len(N0_list))
    pu_count_iucb = 0
    sims_greedy_list = [None for N0 in N0_list]
    
    # simulation
    for k,N0 in enumerate(N0_list):
        rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
        sims = [Simulation(Kg = Kg, N0 = N0, policy = "greedy", count_ws_regret = True) for r in range(R)]
        sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
        for r in range(R):
            if sims[r].is_perpetunderest():
                pu_count_greedy[k] += 1
        sims_greedy_list[k] = sims
    rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
    sims_iucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, policy = "improved_ucb", count_ws_regret = True) for r in range(R)]
    sims_iucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb[r], rss[r]) for r in range(R)] ) #parallel computation
    for r in range(R):
        if sims_iucb[r].is_perpetunderest():
            pu_count_iucb += 1
    output_to_pickle("experiment7.pickle", (sims_iucb))

    # plotting
    all_regret_greedy = np.zeros( (R,N,len(N0_list)) )
    all_draw2_greedy = np.zeros( (R,N,len(N0_list)) )
    all_subsidy_greedy = np.zeros( (R,N,len(N0_list)) )
    all_subsidycs_greedy = np.zeros( (R,N,len(N0_list)) )
    all_regret_iucb = np.zeros( (R,N) )
    all_draw2_iucb = np.zeros( (R,N) )
    all_subsidy_iucb = np.zeros( (R,N) )
    all_subsidycs_iucb = np.zeros( (R,N) )
    for r in range(R):
        for k,N0 in enumerate(N0_list):
            all_regret_greedy[r,:,k] += sims_greedy_list[k][r].regret_seq
            all_draw2_greedy[r,:,k] += sims_greedy_list[k][r].draws_seq[:,1]
            all_subsidy_greedy[r,:,k] += sims_greedy_list[k][r].subsidy_seq
            all_subsidycs_greedy[r,:,k] += sims_greedy_list[k][r].subsidycs_seq
        all_regret_iucb[r] += sims_iucb[r].regret_seq
        all_draw2_iucb[r] += sims_iucb[r].draws_seq[:,1]
        all_subsidy_iucb[r] += sims_iucb[r].subsidy_seq
        all_subsidycs_iucb[r] += sims_iucb[r].subsidycs_seq
    output_to_pickle("out/experiment8.pickle", (all_regret_greedy, all_draw2_greedy, all_subsidy_greedy, all_subsidycs_greedy, all_regret_iucb, all_draw2_iucb, all_subsidy_iucb, all_subsidycs_iucb))

    fig = plt.figure(figsize=Figsize)
    avg_regret_greedy = np.mean(all_regret_greedy, axis=0) #N x k
    std_regret_greedy = np.std(all_regret_greedy, axis=0) #N x k
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:brown']
    linestyles = [(0,(1,1)),(0,(2,1)),(0,(3,1)),(0,(4,1))]
    for k,N0 in enumerate(N0_list):
        plt.plot(range(N), avg_regret_greedy[:,k], label = f"LF with N0={N0}", color = colors[k], linestyle = linestyles[k]) #, color = COLOR_GREEDY, linestyle = LINESTYLE_GREEDY)
        #plt.fill_between(range(N), lower_quantile_regret_ucb, upper_quantile_regret_ucb, alpha=Alpha, color = COLOR_UCB)
        plt.errorbar([N-1], avg_regret_greedy[-1,k], yerr=2*std_regret_greedy[-1,k]/np.sqrt(R), fmt='o', color = colors[k], linestyle = linestyles[k], capsize = Capsize) #2 sigma

    avg_regret_iucb = np.mean(all_regret_iucb, axis=0)
    std_regret_iucb = np.std(all_regret_iucb, axis=0)
    plt.plot(range(N), avg_regret_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_regret_iucb[-1], yerr=2*std_regret_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.ylabel("Regret")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'ws_compare_regret.pdf')
        colab_save('ws_compare_regret.pdf')
    plt.clf()

    pu_count_all = [pu_count_iucb] + list(pu_count_greedy)
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count_all)]
    pu_count_intervals = ([pu_count_all[i]-conf[0] for i,conf in enumerate(confs)],  [conf[1]-pu_count_all[i] for i,conf in enumerate(confs)])
    colors = ["tab:blue"] + ["tab:red" for N0 in N0_list] 
    hatches = [HATCH_BLUE] + [HATCH_RED for N0 in N0_list] 
    labels = ["Hybrid"] + [f"N0={N0}" for N0 in N0_list]
    fig = plt.figure(figsize=Figsize)
    # hatch does not allow different hatches... it is too bad.
    pu_count_intervals_head = (pu_count_intervals[0][:1], pu_count_intervals[1][:1])
    plt.bar(range(len(pu_count_all))[:1], pu_count_all[:1], tick_label=labels[:1], yerr = pu_count_intervals_head, align="center",\
            capsize = Capsize, edgecolor='black', color=COLOR_ROONEY, hatch=HATCH_BLUE)#, hatch=['--', '+', 'x', '\\','--']) #color = colors, hatch='--') #['--', '+', 'x', '\\','--'])
    # this adds a dummy line to print "Hybrid" label
    pu_count_intervals_tail = ([0]+pu_count_intervals[0][1:], [0]+pu_count_intervals[1][1:])
    plt.bar(range(len(pu_count_all)), [0]+pu_count_all[1:], tick_label=labels[:], yerr = pu_count_intervals_tail, align="center",\
            capsize = Capsize, edgecolor='black', color=COLOR_GREEDY, hatch=HATCH_RED)#, hatch=['--', '+', 'x', '\\','--']) #color = colors, hatch='--') #['--', '+', 'x', '\\','--'])
    # fill=False
    #plt.ylim(0, 100)
    plt.ylabel("# of PU")
    plt.xlabel("Decision Rules")
    #plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'ws_compare_pu.pdf')
        colab_save('ws_compare_pu.pdf')
    plt.clf()


    fig = plt.figure(figsize=Figsize)
    avg_subsidycs_greedy = np.mean(all_subsidycs_greedy, axis=0) #N x k
    avg_subsidycs_iucb = np.mean(all_subsidycs_iucb, axis=0)
    std_subsidycs_greedy = np.std(all_subsidycs_greedy, axis=0) #N 
    std_subsidycs_iucb = np.std(all_subsidycs_iucb, axis=0)
    avg_subsidycs_all = [avg_subsidycs_iucb[-1]] + [avg_subsidycs_greedy[-1,k] for k,N0 in enumerate(N0_list)] 
    std_subsidycs_all = [std_subsidycs_iucb[-1]] + [std_subsidycs_greedy[-1,k] for k,N0 in enumerate(N0_list)] 
    confs = [stats.norm.interval(0.95, loc=c, scale=std_subsidycs_all[i]) for i,c in enumerate(avg_subsidycs_all)]
    subsidycs_intervals = ([avg_subsidycs_all[i]-conf[0] for i,conf in enumerate(confs)],  [conf[1]-avg_subsidycs_all[i] for i,conf in enumerate(confs)])
    subsidycs_intervals_head = (subsidycs_intervals[0][:1], subsidycs_intervals[1][:1])
    # this adds a dummy line to print "Hybrid" label, which is hacky...
    subsidycs_intervals_tail = ([0]+subsidycs_intervals[0][1:], [0]+subsidycs_intervals[1][1:])
    labels = ["Hybrid"] + [f"N0={N0}" for N0 in N0_list]
    plt.bar(range(len(avg_subsidycs_all))[:1], avg_subsidycs_all[:1], tick_label=labels[:1], yerr = subsidycs_intervals_head,\
            align="center", capsize = Capsize, edgecolor='black', color=COLOR_ROONEY, hatch=HATCH_BLUE) 
    plt.bar(range(len(avg_subsidycs_all)), [0]+avg_subsidycs_all[1:], tick_label=labels, yerr = subsidycs_intervals_tail,\
            align="center", capsize = Capsize, edgecolor='black', color=COLOR_GREEDY, hatch=HATCH_RED) 
    plt.ylabel("Subsidy")
    plt.xlabel("Decision Rules")

    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'ws_compare_subsidy.pdf')
        colab_save('ws_compare_subsidy.pdf')
    plt.clf()

np.random.seed(8)
experiment7()


# PU as a function of K1 (majority pop)
def experiment8():
    k_list_all = [(2,2),(10,2),(30,2),(100,2)]
    pu_count = np.zeros(len(k_list_all))
    
    # simulation
    for i,k_list in enumerate(k_list_all):
        sims = [Simulation(Kg = k_list, N0 = np.sum(k_list)*1, policy = "greedy") for r in range(R)]
        rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
        sims = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims[r], rss[r]) for r in range(R)] ) #parallel computation
        for r in range(R):
            #sims[r].run()
            if sims[r].is_perpetunderest():
                pu_count[i] += 1
    output_to_pickle("experiment8.pickle", (sims))
    
    #plotting
    confs = [stats.binom.interval(0.95, n=R, p=min(max(0.0001,c/R),0.9999)) for i,c in enumerate(pu_count)]
    pu_count_intervals = ([pu_count[i]-conf[0] for i,conf in enumerate(confs)],  [conf[1]-pu_count[i] for i,conf in enumerate(confs)])
    labels = [k_list[0] for k_list in k_list_all]
    fig = plt.figure(figsize=Figsize)
    plt.bar(range(len(k_list_all)), pu_count, tick_label=labels, yerr = pu_count_intervals,\
            align="center", capsize = Capsize, edgecolor='black', color=COLOR_ROONEY, hatch=HATCH_BLUE)
    plt.ylabel("# of PU")
    plt.xlabel("# of majority candidates K_1")
    #plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'groupsize_pu.pdf')
        colab_save('groupsize_pu.pdf')
    plt.clf()

np.random.seed(3)
experiment8()

# version of experiment 2 with N=10000
def experiment9():
    global N, R
    Kg = (10, 2)
    if run_full: # or (not is_colab()): # outside colab
        N = 10000
        R = 50
    elif run_middle:
        N = 1000
        R = 50        
    else:
        N = 5000
        R = 5
    #print(f"N={N}")

    # simulation
    rss = np.random.randint(np.iinfo(np.int32).max, size=R*10)
    sims_ucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, N=N, policy = "ucb") for r in range(R)]
    sims_ucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_ucb[r], rss[r]) for r in range(R)] ) #parallel computation
    sims_iucb = [Simulation(Kg = Kg, N0 = np.sum(Kg)*1, N=N, policy = "improved_ucb") for r in range(R)]
    sims_iucb = Parallel(n_jobs=N_JOBS)( [delayed(run_sim)(sims_iucb[r], rss[r+R]) for r in range(R)] ) #parallel computation
    output_to_pickle("experiment9.pickle", (sims_ucb, sims_iucb))

    # plotting starts here
    all_regret_ucb = np.zeros( (R,N) )
    all_draw2_ucb = np.zeros( (R,N) )
    all_subsidy_ucb = np.zeros( (R,N) )
    all_subsidycs_ucb = np.zeros( (R,N) )
    all_regret_iucb = np.zeros( (R,N) )
    all_draw2_iucb = np.zeros( (R,N) )
    all_subsidy_iucb = np.zeros( (R,N) )
    all_subsidycs_iucb = np.zeros( (R,N) )
    for r in range(R):
        all_regret_ucb[r] += sims_ucb[r].regret_seq
        all_draw2_ucb[r] += sims_ucb[r].draws_seq[:,1]
        all_subsidy_ucb[r] += sims_ucb[r].subsidy_seq
        all_subsidycs_ucb[r] += sims_ucb[r].subsidycs_seq
        all_regret_iucb[r] += sims_iucb[r].regret_seq
        all_draw2_iucb[r] += sims_iucb[r].draws_seq[:,1]
        all_subsidy_iucb[r] += sims_iucb[r].subsidy_seq
        all_subsidycs_iucb[r] += sims_iucb[r].subsidycs_seq

    fig = plt.figure(figsize=Figsize)
    avg_regret_ucb = np.mean(all_regret_ucb, axis=0)
    std_regret_ucb = np.std(all_regret_ucb, axis=0)
    lower_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.05, axis=0)
    upper_quantile_regret_ucb = np.quantile(all_regret_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_regret_ucb[-1], yerr=2*std_regret_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret_ucb, upper_quantile_regret_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_regret_iucb = np.mean(all_regret_iucb, axis=0)
    std_regret_iucb = np.std(all_regret_iucb, axis=0)
    lower_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.05, axis=0)
    upper_quantile_regret_iucb = np.quantile(all_regret_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_regret_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_regret_iucb[-1], yerr=2*std_regret_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_regret_iucb, upper_quantile_regret_iucb, alpha=Alpha, color = COLOR_HYBRID)
    plt.ylabel("Regret")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_regret_long.pdf')
        colab_save('iucb_regret_long.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_draw2_ucb = np.mean(all_draw2_ucb, axis=0)
    std_draw2_ucb = np.std(all_draw2_ucb, axis=0)
    lower_quantile_draw2_ucb = np.quantile(all_draw2_ucb, 0.05, axis=0)
    upper_quantile_draw2_ucb = np.quantile(all_draw2_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_draw2_ucb, label = "UCB", color = COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_draw2_ucb[-1], yerr=2*std_draw2_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_ucb, upper_quantile_draw2_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_draw2_iucb = np.mean(all_draw2_iucb, axis=0)
    std_draw2_iucb = np.std(all_draw2_iucb, axis=0)
    lower_quantile_draw2_iucb = np.quantile(all_draw2_iucb, 0.05, axis=0)
    upper_quantile_draw2_iucb = np.quantile(all_draw2_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_draw2_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_draw2_iucb[-1], yerr=2*std_draw2_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_draw2_iucb, upper_quantile_draw2_iucb, alpha=Alpha, color = COLOR_HYBRID)
    plt.plot(range(N), [i*Kg[1]/np.sum(Kg) for i in range(N)], label = "Optimal", color = "black", linestyle=linestyle_tuple["loosely dashed"])

    plt.ylabel("# of minorities hired")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_draw2_long.pdf')
        colab_save('iucb_draw2_long.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_subsidy_ucb = np.mean(all_subsidy_ucb, axis=0)
    std_subsidy_ucb = np.std(all_subsidy_ucb, axis=0)
    lower_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.05, axis=0)
    upper_quantile_subsidy_ucb = np.quantile(all_subsidy_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_ucb, label = "UCB", color=COLOR_UCB, linestyle = LINESTYLE_UCB)
    plt.errorbar([N-1], avg_subsidy_ucb[-1], yerr=2*std_subsidy_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_ucb, upper_quantile_subsidy_ucb, alpha=Alpha, color = COLOR_UCB)

    avg_subsidy_iucb = np.mean(all_subsidy_iucb, axis=0)
    std_subsidy_iucb = np.std(all_subsidy_iucb, axis=0)
    lower_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.05, axis=0)
    upper_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_subsidy_iucb[-1], yerr=2*std_subsidy_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_iucb, upper_quantile_subsidy_iucb, alpha=Alpha, color = COLOR_HYBRID)

    plt.ylabel("Subsidy")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_subsidy_long.pdf')
        colab_save('iucb_subsidy_long.pdf')
    plt.clf()

    fig = plt.figure(figsize=Figsize)
    avg_subsidy_iucb = np.mean(all_subsidy_iucb, axis=0)
    std_subsidy_iucb = np.std(all_subsidy_iucb, axis=0)
    lower_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.05, axis=0)
    upper_quantile_subsidy_iucb = np.quantile(all_subsidy_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidy_iucb, label = "Hybrid", color = COLOR_HYBRID, linestyle = LINESTYLE_HYBRID)
    plt.errorbar([N-1], avg_subsidy_iucb[-1], yerr=2*std_subsidy_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidy_iucb, upper_quantile_subsidy_iucb, alpha=Alpha, color = COLOR_HYBRID)

    avg_subsidycs_ucb = np.mean(all_subsidycs_ucb, axis=0)
    std_subsidycs_ucb = np.std(all_subsidycs_ucb, axis=0)
    lower_quantile_subsidycs_ucb = np.quantile(all_subsidycs_ucb, 0.05, axis=0)
    upper_quantile_subsidycs_ucb = np.quantile(all_subsidycs_ucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidycs_ucb, label = "CS-UCB", color = COLOR_CS_UCB, linestyle = LINESTYLE_CS_UCB)
    plt.errorbar([N-1], avg_subsidycs_ucb[-1], yerr=2*std_subsidycs_ucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_CS_UCB) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidycs_ucb, upper_quantile_subsidycs_ucb, alpha=Alpha, color = COLOR_CS_UCB)

    avg_subsidycs_iucb = np.mean(all_subsidycs_iucb, axis=0)
    std_subsidycs_iucb = np.std(all_subsidycs_iucb, axis=0)
    lower_quantile_subsidycs_iucb = np.quantile(all_subsidycs_iucb, 0.05, axis=0)
    upper_quantile_subsidycs_iucb = np.quantile(all_subsidycs_iucb, 0.95, axis=0)
    plt.plot(range(N), avg_subsidycs_iucb, label = "CS-Hybrid", color = COLOR_CS_HYBRID, linestyle = LINESTYLE_CS_HYBRID)
    plt.errorbar([N-1], avg_subsidycs_iucb[-1], yerr=2*std_subsidycs_iucb[-1]/np.sqrt(R), fmt='o', capsize = Capsize, color = COLOR_CS_HYBRID) #2 sigma
    plt.fill_between(range(N), lower_quantile_subsidycs_iucb, upper_quantile_subsidycs_iucb, alpha=Alpha, color = COLOR_CS_HYBRID)

    plt.ylabel("Subsidy")
    plt.xlabel("Round (n)")
    plt.legend()
    my_show()
    if save_img:
        my_savefig(fig, 'iucb_subsidy_cs_long.pdf')
        colab_save('iucb_subsidy_cs_long.pdf')
    plt.clf()


np.random.seed(200)
experiment9()

