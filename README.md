# What is this?

Simulation of the paper "On Statistical Discrimination as a Failure of Social Learning: A Multi-Armed Bandit Approach" (Management Science, to appear). https://arxiv.org/abs/2010.01079

# Faster replication

To replicate the results with symmetric models (Figs 1--14) with fewer trials, run
```
 python FairSocialLearning.py --size=middle
```
To replicate the results with asymmetric models (Figs 15--) , run 
```
python FairSocialLearning.py --size=middle --asymmetric
```

PDF files are created at figures/ and figures/asymmetric directories.

# Full replication

To replicate the results with symmetric models (Figs 1--14), run
```
 python FairSocialLearning.py --size=full
```
To replicate the results with asymmetric models (Figs 15--) , run 
```
python FairSocialLearning.py --size=full --asymmetric
```

PDF files are created at figures/ and figures/asymmetric directories.
