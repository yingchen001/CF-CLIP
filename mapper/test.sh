# python scripts/inference.py --exp_dir ./test/green_lipstick \
# --checkpoint_path ../pretrained/ffhq/green_lipstick.pt \
# --description "green lipstick" \
# --latents_test_path ../latent_codes/test_faces.pt \
# --n_images 200;

python scripts/inference.py --w_space --stylegan_size 512 --n_images 200 \
--exp_dir ./test/green_tongue \
--checkpoint_path ../pretrained/dog/tiger_stripes.pt \
--description "tiger stripes" \
--latents_test_path /data/vdc/yingchen.yu/project/text-edit/StyleCLIP/global_directions/npy/afhqdog/W_afhqdog_test.pt \