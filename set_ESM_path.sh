user_path=$1

# download ESM models
python setup_ESM.py ${user_path}

# change path in template files
sed "s%#ESM_PATH#%${user_path}%g" tmp/augmentation.py > data_loader/augmentation.py
sed "s%#ESM_PATH#%${user_path}%g" tmp/predict.yaml > predict.yaml
sed "s%#ESM_PATH#%${user_path}%g" tmp/config.yaml > trained_model/config.yaml
