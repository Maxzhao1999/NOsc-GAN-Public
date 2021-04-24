#$-wd /vols/t2k/users/lsw1117/logs
#$-q gpu.q -l h_rt=12:0:0
#$-m ea -M lsw1117@ic.ac.uk
cd /vols/t2k/users/lsw1117/NOsc-GAN
python3 batch_run.py ${SGE_TASK_ID}
