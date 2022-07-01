#### Run depth evaluation ####
python main_depth.py --train 0 --model_type 'midas_v21' \
--test_usage 'edina_test' \
--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/depth_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00007-iter-42000.ckpt \
--dataset_pickle_file /mars/home/khiem/projects/ongoing_exps/EgoDepthNormal_Code/dataset_creators/scannet_edina_camready_final_clean.pkl \
--batch_size 8 --skip_every_n_image_test 40 \
--data_root /mars/mnt/oitstorage/EDINA_pruned \
--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_eval_release_depth

#### Run normal evaluation ####
python main_normal.py --train 0 --model_type 'midas_v21' \
--test_usage 'edina_test' \
--checkpoint /mars/mnt/oitstorage/khiem_storage/workspace/CVPR22/result/normal_camready_midasv21_baseline_midasloss_randomcropresize_fixdata/model-epoch-00011-iter-02000.ckpt \
--dataset_pickle_file /mars/home/khiem/projects/ongoing_exps/EgoDepthNormal_Code/dataset_creators/scannet_edina_camready_final_clean.pkl \
--batch_size 8 --skip_every_n_image_test 40 \
--data_root /mars/mnt/oitstorage/EDINA_pruned \
--save_visualization /mars/mnt/oitstorage/khiem_storage/output_midas/cvpr22/test_eval_release_normal