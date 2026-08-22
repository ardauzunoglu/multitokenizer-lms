cd /scratch/dkhasha1/auzunog1/multitokenizer-lms/eval

python plot_train_lm_loss.py \
  ../nanotron/train_dclm_stack_20m.log \
  ../nanotron/train_dclm_stack_20m_single_small.log \
  ../nanotron/train_dclm_stack_20m_single_large.log \
  --labels multi single-small single-large \
  --smooth-window 25 \
  --output train_lm_loss.png