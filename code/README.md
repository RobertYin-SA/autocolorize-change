Robert Yin edited usage
-----------------------------
# Training model
python colornet_train.py --train_filenames test --original_graph_path vgg/vgg16-20160129.tfmodel --batch_size 1 --num_epochs 1000 --checkpoint_path model
# Testing model
python colornet_test.py --test_filenames test --original_graph_path vgg/vgg16-20160129.tfmodel --batch_size 1 --num_epochs 1 --training_checkpoint model --output_path pre_pic
-----------------------------

