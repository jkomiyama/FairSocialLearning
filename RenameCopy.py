#!/usr/bin/env python
# coding: utf-8

# What is this?
## Simulation of the paper "On Statistical Discrimination as a Failure of Social Learning: A Multi-Armed Bandit Approach" (Management Science, to appear). https://arxiv.org/abs/2010.01079
## To replicate the results with symmetric models, run
##   python FairSocialLearning.py --size=full 
## To replicate the results with asymmtric models, run 
##   python FairSocialLearning.py --size=full --asymmetric
## After running these codes (output PDFs are created in directories "figures" and "figures_asymmetric"), run
##  python RenameCopy.py
## to match the name of the files to the name of the pdfs in the paper. Figure[1--20].pdf will be in the current directory
import shutil
shutil.copy("figures/groupsize_pu.pdf", "Figure1.pdf")
shutil.copy("figures/policycomp_regret.pdf", "Figure2.pdf")
shutil.copy("figures/iucb_regret.pdf", "Figure3.pdf")
shutil.copy("figures/iucb_subsidy_long.pdf", "Figure4.pdf")
shutil.copy("figures/iucb_eosum_disparity.pdf", "Figure5.pdf")
shutil.copy("figures/iucb_disparity.pdf", "Figure6.pdf")
shutil.copy("figures/rooney_pu.pdf", "Figure7.pdf")
shutil.copy("figures/rooney_largeg2_strong_regret.pdf", "Figure8.pdf")
shutil.copy("figures/rooney_largeg2_eosum_disparity.pdf", "Figure9.pdf")
shutil.copy("figures/rooney_largeg2_disparity.pdf", "Figure10.pdf")
shutil.copy("figures/iucb_subsidy_cs.pdf", "Figure11_1.pdf")
shutil.copy("figures/iucb_subsidy_cs_long.pdf", "Figure11_2.pdf")
shutil.copy("figures/iucb_regret_ex.pdf", "Figure12.pdf")
shutil.copy("figures/iucb_subsidy_ex.pdf", "Figure13.pdf")
shutil.copy("figures/ws_compare_pu.pdf", "Figure14_1.pdf")
shutil.copy("figures/ws_compare_subsidy.pdf", "Figure14_2.pdf")
shutil.copy("figures_asymmetric/mux_regret.pdf", "Figure15.pdf")
shutil.copy("figures_asymmetric/mux_group2best.pdf", "Figure16.pdf")
shutil.copy("figures_asymmetric/groupsize_pu.pdf", "Figure17.pdf")
shutil.copy("figures_asymmetric/policycomp_regret.pdf", "Figure18.pdf")
shutil.copy("figures_asymmetric/iucb_regret.pdf", "Figure19.pdf")
shutil.copy("figures_asymmetric/iucb_subsidy.pdf", "Figure20.pdf")