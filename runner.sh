. ~/.zshrc
conda activate MFA
 /home/jeffeuxmartin/miniconda3/envs/MFA/bin/mfa \
   align -j 8 --clean \
   -t /home/jeffeuxmartin/ReUnits/mfaout_new \
   /home/jeffeuxmartin/ReUnits/NewPipeline/total \
   english_us_arpa \
   english_us_arpa \
   /home/jeffeuxmartin/ReUnits/NewPipeline/outout

# mfa validate /home/jeffeuxmartin/ReUnits/NewPipeline/wavs/ english_us_arpa english_us_arpa

