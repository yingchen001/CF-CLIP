# for dog/cat
DESCRIPTION=$1
python scripts/train.py --exp_dir ../results/${DESCRIPTION// /_}  --description "$DESCRIPTION" \
--stylegan_size 512 --w_space --id_lambda 0 --perc_lambda 0.02 \
--latents_train_path ../latent_codes/afhqdog/W_afhqdog_train.pt \
--latents_test_path ../latent_codes/afhqdog/W_afhqdog_test.pt \
--stylegan_weights ../pretrained/afhqdog.pt \
# # for ffhq
# DESCRIPTION=$1
# python scripts/train.py --exp_dir ../results/${DESCRIPTION// /_}  --description "$DESCRIPTION"