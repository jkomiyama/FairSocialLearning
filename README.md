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

# Result files

PDF files are created at figures/ and figures/asymmetric directories.

After running full simulation (output PDFs are created in directories "figures" and "figures_asymmetric"), run
```
python RenameCopy.py
```
to match the name of the files to the name of the pdfs in the paper. Figures[1--20].pdf will be in the current directory

# Environments

The codes are tested on Python 3.10.6. It takes 5 hours to run each symmetric or asymmetric simulation of full scale on a modern desktop with 24-core CPU (48-core hyperthreading). We do not use any GPGPU.
