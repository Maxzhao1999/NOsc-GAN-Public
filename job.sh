#$-wd /vols/dune/zz5617/logs
#$-q gpu.q -l h_rt=12:0:0
#$-m ea -M zz5617@ic.ac.uk
cd /vols/dune/zz5617/NOsc-GAN
python3 batch_run.py ${SGE_TASK_ID}
